# Quick Start - Luminance Extractor for Photometric Stereo

## Obiettivo
Generare immagini di luminanza lineare, coerenti tra tutti gli scatti del dataset, pronte per pipeline RTI/Photometric Stereo.

## Setup consigliato (default robusto)
- Output Format: TIFF
- Bit Depth: 16-bit
- Color Space: Linear
- Luminance Source: Demosaic RGB mean
- Use sharpness: OFF
- Undersample 1/N (preview): 1/32
- Peak percentile (Step 1): 99.8
- Burned exclusion ratio (Step 1): 0.05%
- Overwrite existing files: a scelta operativa

## Dark calibration (regola pratica)
Usa una sola famiglia di calibrazione alla volta:
- Opzione A: Dark level calibration from RAW metadata
- Opzione B: Dark frames manuali (consigliato)
  - Cap on: per dark sensore per-pixel
  - Cap off + dome off: per bias ambientale

Nota: se carichi dark frames manuali, la calibrazione da metadata RAW viene disattivata automaticamente.

Con entrambe le opzioni (metadata RAW o manual dark frames) puoi regolare:
- Lift coefficient (default 0.000000, range -0.99..0.99)
- coeff positivo: riduce la sottrazione del dark metadata (utile se sovrastimato)
- coeff negativo: aumenta la sottrazione

Nota: con dark frames manuali il coefficiente scala la sottrazione della dark map cap-on e dell'offset cap-off.

## Workflow in 7 passi
1. Carica i file RAW/TIFF del dataset.
2. (Opzionale ma consigliato) Carica dark frames:
   - cap-on,
   - cap-off + dome-off.
3. Verifica il clipping preview:
   - Burned threshold (%): tipicamente 99.
   - Undersample: 1/32 per analisi veloce.
   - Colori preview:
     - rosso = clipping alte luci
     - blu = clipping nei neri
4. Premi Dark level overestimation check (valido sia con metadata RAW sia con dark frames manuali):
   - usa la progress bar principale durante l'analisi;
   - nel popup usa Set coefficient per applicare il valore suggerito;
   - usa Close per chiudere senza applicare.
   - se il clipping e sotto 0.000001%, viene indicato come trascurabile.
5. Se usi Analyze linearity and build inverse LUT, imposta i controlli nell'ordine operativo:
   - Saturated-pixel threshold (%): soglia usata per etichettare pixel saturi nel ROI (default 99.80; abbassare se la camera clippa sotto il massimo digitale);
   - Exclude burned ROI >= %: esclusione globale del frame di calibrazione;
   - Near-sat margin from threshold: esclusione frame troppo vicino alla saturazione.
   Nota: i pixel burned vengono rimossi dal calcolo del CenterMean solo dopo il check di esclusione globale del frame.
6. Premi Calculate Stretch (Step 1):
   - calcola il picco globale del dataset (global_max) con stima robusta percentile-based;
   - esclude i frame solo se la quota di pixel burned supera la soglia percentuale.
7. Premi Develop Luminance (Step 2):
   - applica stretch globale condiviso e genera output.
8. Apri la cartella di output e usa i TIFF lineari nel solver Photometric Stereo.

## Quando usare Demosaic Weighted
Attivalo solo se hai una motivazione fotometrica specifica (es. risposta spettrale/illuminanti noti).
- Pesi iniziali consigliati: R=1, G=2, B=1

## Quando usare RAW Bayer 2x2
Utile per ridurre artefatti da demosaicing.
- Atteso: risoluzione dimezzata in larghezza e altezza.

## Checklist prima dell'export
- [ ] TIFF 16-bit + Linear impostati
- [ ] Use sharpness OFF (se vuoi massima neutralita radiometrica)
- [ ] Dark workflow coerente (metadata oppure manuale)
- [ ] ISO e diaframma costanti se hai fatto calibrazione linearita (variare solo il tempo)
- [ ] Stretch Step 1 completato o valore salvato confermato

## Riferimento
Per dettagli completi, formule e bibliografia: vedere DOCUMENTAZIONE_SOFTWARE.md.
