# Luminance Extractor for Photometric Stereo - Documentazione Tecnica Sintetica

## 1) Scopo del software
Luminance Extractor for Photometric Stereo converte immagini RAW/TIFF in mappe di luminanza lineare (grayscale a 3 canali uguali in output), ottimizzate per flussi di Photometric Stereo (RTI/dome), mantenendo coerenza radiometrica tra immagini.

Obiettivi principali:
- estrazione di intensita luminosa fisicamente coerente;
- controllo esplicito della calibrazione dark (metadata RAW o dark frames manuali);
- normalizzazione globale consistente su tutto il dataset;
- esportazione ad alta profondita (consigliato TIFF 16-bit, spazio Linear).

## 2) Input e vincoli
- RAW supportati: CR2, NEF, ARW, DNG, RAF, ORF, RW2, RAW.
- TIFF supportati: solo 16-bit lineari (vincolo esplicito nel software).

Per i RAW il decoding usa impostazioni lineari (gamma 1, no auto bright, no auto scale, white balance neutro) per evitare trasformazioni non lineari prima dell'estrazione della luminanza.

## 3) Pipeline passo per passo

### Step 0 - Configurazione modalita luminanza
Sono disponibili tre modalita:
1. RAW Bayer 2x2
2. Demosaic RGB mean (default)
3. Demosaic RGB weighted (con pesi R, G, B editabili)

Formula luminanza:
- media aritmetica:

L = (R + G + B) / 3

- media pesata:

L = (wR * R + wG * G + wB * B) / (wR + wG + wB)

Dopo il calcolo su dominio 16-bit, la mappa viene normalizzata in [0, +inf) dividendo per 65535 e clippando a zero il negativo.

### Step 1 - Calibrazione dark
La logica e mutuamente esclusiva:
- se carichi dark frames manuali, la calibrazione dark da metadata RAW viene disattivata;
- se non carichi dark frames manuali, puoi usare il dark level da metadata RAW.

E disponibile un parametro utente globale per la correzione dark:
- Lift coefficient (editabile, default 0.000000, range -0.99..0.99)

Il coefficiente modifica la sottrazione dark effettiva:

dark_eff = dark_metadata * (1 - coeff)

La stessa logica di scala viene applicata anche alla componente dark manuale (cap-on map e offset cap-off).

con coeff > 0 riduci la sottrazione (eviti over-subtraction), con coeff < 0 aumenti la sottrazione.

Modalita supportate:

#### 1. RAW metadata dark level
- sottrazione del black level per-canale CFA in dominio RAW (correzione per-canale prima della ricostruzione RGB).

#### 2. Cap-on dark frames (sensor dark per-pixel)
- stima master dark map come media pixel-wise delle pose cap-on:

D_map(x,y) = (1/N) * somma_i D_i(x,y)

- sottrazione per-pixel: I_corr = max(I - D_map, 0)

#### 3. Cap-off + dome-off dark frames (ambient bias residuo)

**Importante: Cap-off non è doppia sottrazione, ma misura del bias ambientale residuo.**

Processo:

**a) Calcolo dell'offset cap-off (residuo dopo rimozione cap-on):**

Per ogni frame cap-off, si rimuove prima la componente sensore (D_map), quindi si calcola il residuo:

residuo_i(x,y) = max( D_off_i(x,y) - D_map(x,y), 0 )

Il clipping a zero è cruciale: garantisce che il residuo sia sempre non-negativo e non accumuli correzioni fittizie.

offset_cap_off = media_i( media_{x,y}( residuo_i(x,y) ) )

**b) Applicazione sequenziale (non cumulativa):**

Sull'immagine:

I_step1 = max( I_raw - D_map * 65535, 0 )        # Sottrazione per-pixel cap-on

I_final = max( I_step1 - offset_cap_off * 65535, 0 )  # Sottrazione scalare cap-off

**c) Perché funziona (no sovrastima):**

| Componente | Natura | Variazione | Sottrazione |
|-----------|--------|-----------|-----------|
| **D_map (cap-on)** | Dark termico sensore | Per-pixel spaziale | pixel-wise |
| **offset_cap_off** | Bias luce parassita / fumo termico | Uniforme | scalare globale |

Cap-off misura **solo il residuo ambientale** dopo aver tolto il contributo sensore. Non è una doppia sottrazione perché:

1. Il cap-off viene calcolato su una base gia corretta (frame cap-off - D_map)
2. Il clipping nel calcolo garantisce che il residuo non possa essere negativo (nessuna sovra-correzione)
3. L'offset cap-off è uno scalare unico, non una mappa pixel-wise, quindi agisce come bias additivo uniforme

**d) Esempio numerico:**

```
Scenario: sensore con thermal dark, illuminazione con luce parassita

Frame cap-on misurato:  100 (dark termico sensore)
Frame cap-off misurato: 150 (dark sensore + luce parassita ambiente)

Calcolo D_map (media cap-on):
  D_map = 100

Calcolo residuo cap-off:
  residuo = frame_cap_off - D_map = 150 - 100 = 50 ✓ (solo bias ambiente)
  offset_cap_off = 50

Applicazione su immagine scientifica:
  foto_raw = 1000 conteggi
  
  passo 1 (cap-on per-pixel):
    foto_corretta = 1000 - 100 = 900
  
  passo 2 (cap-off scalare):
    foto_finale = 900 - 50 = 850
  
  Correzione totale: 1000 - 850 = 150 ✓
  (corrisponde a: 100 sensore + 50 ambiente, NO doppia sottrazione)

Scenario ERRATO (se fosse doppia sottrazione):
  foto_finale_sbagliata = 1000 - 100 - 100 - 50 = 750 ✗ (overcorrection!)
```

**e) Validazione nel codice:**

Nel metodo `_compute_ambient_dark_offset()`:

```python
rgb = self._read_linear_rgb_65535(path, demosaic)
if master_dark_map is not None:
    dark_map_m = self._match_dark_map_shape(master_dark_map, rgb.shape[0], rgb.shape[1])
    rgb = np.clip(rgb - dark_map_m, 0.0, None)  # ← Clipping a zero, no doppia sottrazione
per_frame_means.append(float(np.mean(rgb) / 65535.0))
```

Questo ordine evita sovra-correzione mediante separazione di componenti e clipping guardia.

### Step 2 - Preview clipping e stima robusta del picco (Phase 1)
Su ogni immagine viene calcolato un picco robusto, escludendo pixel oltre una soglia di saturazione impostata (burned threshold). Con undersampling 1/N (default 1/32):
- si campiona la mappa luminanza;
- si scartano i campioni oltre soglia t;
- si usa un percentile alto dei campioni validi (default 99.8) come picco robusto dell'immagine.

Picco globale dataset:

global_max = max_k(peak_k)

Serve come fattore unico di stretch per garantire comparabilita tra scatti.

Esclusione file con saturazioni:
- l'esclusione automatica di un frame non avviene su singolo pixel saturo;
- viene applicata una soglia di area saturata (default 0.05% dei pixel campionati oltre burned threshold).

Metriche QA salvate in histogram_stretch.txt:
- peak_percentile;
- burned_exclusion_ratio_pct;
- median_peak;
- p99_burned_high_pct;
- p99_burned_black_pct;
- frames_excluded_ratio_pct.

Nella preview clipping:
- clipping alte luci: evidenziato in rosso;
- clipping nei neri (valori prossimi a zero): evidenziato in blu.

La statistica preview mostra entrambe le percentuali separatamente.

### Step 2b - Dark level overestimation check
Nel box Dark Calibration e disponibile il controllo rapido:
- Dark level overestimation check

Funzione:
- analizza velocemente tutti i RAW selezionati;
- stima eventuale clipping nei neri legato alla calibrazione dark attiva (metadata RAW o manual dark frames);
- propone un coefficiente correttivo (lift) quando necessario.

Comportamento UI:
- analisi eseguita in background con avanzamento sulla progress bar principale;
- report finale in popup con pulsanti Set coefficient e Close;
- Set coefficient applica direttamente il valore al campo Lift coefficient e aggiorna subito la preview.

Tolleranza numerica:
- se il clipping globale e > 0 ma inferiore a 0.000001% (6 decimali), il report indica esplicitamente che il clipping e trascurabile (sotto soglia di tolleranza).

### Step 3 - Sviluppo finale (Phase 2)
Per ogni file:
1. ricostruzione mappa luminanza lineare con la modalita scelta;
2. eventuale sharpness (solo se Use sharpness e attivo; default OFF);
3. stretch assoluto condiviso:

L_stretch = clip(L / global_max, 0, 1)

4. eventuale trasferimento verso spazio colore scelto (Linear, sRGB, Display P3, ProPhoto);
5. replica su 3 canali identici (grayscale RGB);
6. eventuale rotazione export;
7. salvataggio nel formato/profondita scelti;
8. copia metadata (con controllo coerente dei tag colore).

## 4) Note matematiche di robustezza
- Linearita radiometrica: il software mantiene il dominio lineare prima del mapping finale, condizione fondamentale per metodi fotometrici.
- Correzione dark separata (per-canale/per-pixel + bias scalare): separa componente sensore da componente ambientale, riducendo bias sistematici.
- Stretch globale unico: evita normalizzazioni indipendenti per frame che altererebbero i rapporti di intensita inter-immagine.
- Clipping robusto in stima picco percentile: attenua l'influenza di hotspot/saturazioni locali e outlier residui.

## 5) Parametri consigliati per Photometric Stereo
- Output format: TIFF
- Bit depth: 16-bit
- Color space: Linear
- Use sharpness: OFF (attivare solo per test controllati)
- Luminance source:
  - default pratico: Demosaic RGB mean
  - alternativa piu "sensor-like": RAW Bayer 2x2
- Calibrazione dark:
  - usare cap-on per dark sensore;
  - aggiungere cap-off+dome-off per bias ambientale;
  - non combinare dark metadata e dark frames manuali.
- Parametri robustezza Step 1:
  - peak percentile: 99.8 (consigliato);
  - burned exclusion ratio: 0.05% (consigliato come default prudente).

## 6) Controlli aggiuntivi per calibrazione linearita
- Durante Analyze sensor linearity viene controllata la coerenza dei metadati di acquisizione.
- Se ISO o FNumber variano tra i frame di calibrazione, il software mostra warning espliciti nel report.
- Best practice: per stimare una LUT affidabile mantenere ISO e diaframma costanti e variare solo il tempo di esposizione.
- E disponibile una soglia dedicata Saturated-pixel threshold (%) per definire quando un pixel del ROI e considerato saturo durante la calibrazione linearita.
- Ordine operativo dei check linearita:
  1) classificazione pixel saturi nel ROI usando Saturated-pixel threshold;
  2) esclusione globale frame con Exclude burned ROI >= %;
  3) esclusione near-saturation tramite Near-sat margin from threshold;
  4) solo per i frame inclusi, CenterMean calcolato escludendo i pixel saturi.

## 7) Limiti e assunzioni
- Il modello fotometrico a valle (stima normali) assume tipicamente riflettanza Lambertiana o quasi-Lambertiana.
- Superfici altamente speculari, subsurface scattering marcato o saturazioni estese possono degradare la qualita della ricostruzione.
- Demosaicing puo introdurre correlazioni/interpolazioni cromatiche; la modalita RAW Bayer 2x2 riduce questo effetto al costo di risoluzione spaziale.

## 8) Riferimenti bibliografici essenziali
1. Woodham, R. J. (1980). Photometric method for determining surface orientation from multiple images. Optical Engineering, 19(1), 139-144.

2. Horn, B. K. P. (1986). Robot Vision. MIT Press.

3. Debevec, P. E., Malik, J. (1997). Recovering high dynamic range radiance maps from photographs. SIGGRAPH 1997.

4. Nayar, S. K., Ikeuchi, K., Kanade, T. (1991). Surface reflection: Physical and geometrical perspectives. IEEE TPAMI, 13(7), 611-634.

5. Mallick, S. P., Zickler, T., Belhumeur, P. N., Kriegman, D. J. (2005). Beyond Lambert: Reconstructing specular surfaces using color. CVPR 2005.

6. Gonzalez, R. C., Woods, R. E. (2008). Digital Image Processing (3rd ed.). Pearson. (Riferimento classico per filtraggio, sharpening e quantizzazione).

7. Lukac, R., Plataniotis, K. N. (2007). Color Filter Arrays: Design and Performance Analysis. IEEE Press/Wiley. (Contesto CFA/demosaicing).

## 9) Conclusione operativa
La pipeline implementata e coerente con buone pratiche radiometriche per acquisizioni multi-illuminazione: preserva linearita, separa correzioni dark, usa normalizzazione globale comune e mantiene output ad alta dinamica. Questo rende i dati adatti a successive stime di normali/albedo con maggiore stabilita numerica e comparabilita inter-frame.
