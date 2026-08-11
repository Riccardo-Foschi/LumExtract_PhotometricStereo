# Luminance Extractor for Photometric Stereo
This software processes RAW images and implements a calibration pipeline to develop B&amp;W luminance files preserving relative lightness to be used in Photometric Stereo workflows

-----> [Download the beta version of the software from here!](https://github.com/Riccardo-Foschi/LumExtract_PhotometricStereo/releases/download/v1.2.0/v-1-2-0.zip) <-----  



***

**OPERATIVE CHECKLIST – Luminance Extractor (Photometric Stereo)**

1) **Acquisition setup**
- Set the camera to full manual mode.  
- Keep ISO and aperture constant across the entire dataset.  
- Keep focus, distance, and framing constant.  
- Disable any in‑camera non‑linear processing.

2) **Dataset preparation**
- Ensure all input files are RAW or 16‑bit linear TIFF.  
- Verify consistent naming between input frames, grey card, and light index.  
- Verify that ExposureTime metadata is present in the linearity calibration files.

3) **Recommended software settings**
- Output: TIFF  
- Bit depth: 16‑bit  
- Color space: Linear  
- Use sharpness: OFF  
- Luminance source: Demosaic RGB mean (or RAW Bayer 2×2 if you prefer to minimize demosaic effects)

4) **Dark calibration**
- Use only one mode per run:  
  A) RAW metadata dark level  
  B) Manual dark frames (cap‑on + cap‑off)  
- If using manual dark frames, acquire them in the same thermal session.  
- If black clipping appears, only in extreme cases, use the “Dark level overestimation check” and adjust the Lift coefficient.

5) **Flatfield (optional)**
- Enable flatfield only with a valid 1:1 pairing between input and grey frames.  
- Check the before/after preview to confirm reduction of non‑uniformity.

6) **Light intensity compensation (optional but recommended)**
- Load a .dome or .lp file consistent with your light setup.  
- Set the light's distance correctly (mm).  
- Save/use the LED intensity compensation file.  
- If you change the output rotation, recheck the compatibility of the compensation file.

7) **Linearity calibration (if needed)**
- Use at least 5–7 frames with different exposures.  
- Keep ISO and F‑number constant; vary only ExposureTime.  
- Exclude saturated or near‑saturated samples.  
- Apply the LUT only if the report indicates real non‑linearity (most cameras only manifest non linearity at very high exposure so in most cases this calibration is not necessary)

8) **Calculate Stretch**
- Set the burned threshold (to improve photometric stereo normal map creation for shiny metallic objects, stretch the histogram until specular highlights are clipped).  
- Set Undersample (typical: 1/32 for a quick preview).  
- The software also present an optional feature to set automatically the stretch to fit the histogram between 0 and 100%.

9) **Rotate image**
- in case of need rotate the output saved images (check compatibility with .dome afterwards)

10) **Develop Luminance**
- Run the final export with linear settings.  
- Verify the output and log (`export_log.txt`) in the destination folder.

***

<img width="683" height="638" alt="image" src="https://github.com/user-attachments/assets/0357795f-e588-42dd-b771-68c22e9cfc3f" />

***
