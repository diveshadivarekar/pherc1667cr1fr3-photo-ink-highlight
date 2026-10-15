# PHerc. 1667 Cr 1 Fr 3 - contrast enhancement of the fragment photograph to make its ink marks readable

**Vesuvius Challenge Progress Prize submission.** Object: **PHerc. 1667, Cr. 1, Fr. 3** (mount label `1667.01.03`).
This package concerns this one fragment only: no data, models or results from any other object.

## What this is - and is not

- **Input:** one colour photograph of the carbonised fragment in its mount (`images/input/PHerc1667Cr1Fr3_photo.jpg`, 1474 x 1723 px).
- **Method:** classical image processing (`code/highlight_characters.py`): isolate the fragment from the white mount and clips, percentile stretch + CLAHE inside the fragment, 3x upscale crop.
- **Output:** a contrast-enhanced crop in which the dark ink-like strokes on the upper and lower parts of the fragment stand out against the charred papyrus (`images/output/PHerc1667Cr1Fr3_photo_enhanced.png`).
- **Not done:** no CT/virtual unwrapping, no machine-learning model, **no transcription or letter identification**, no ground truth.

## Results

| item                            | file                                                  | note                                                                   |
| ------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| original photo                  | `images/input/PHerc1667Cr1Fr3_photo.jpg`              | as supplied                                                            |
| fragment mask outline           | `images/output/PHerc1667Cr1Fr3_photo_highlighted.png` | green outline of the segmented fragment (~`results/measurements.json`) |
| **enhanced crop (main result)** | `images/output/PHerc1667Cr1Fr3_photo_enhanced.png`    | 885 x ~1450 px, CLAHE, 3x                                              |
| original vs enhanced            | `images/output/side_by_side_original_vs_enhanced.png` | same crop, grey original left, enhanced right                          |
| debug masks                     | `images/output/*_dbg_*.png`                           | fragment mask, stroke response, binary, kept components                |

## Honest caveats

1. **The automatic stroke detector kept 0 components** with the default parameters (`results/run_log.txt`), so the "highlighted" overlay shows only the fragment outline. The improvement is in the **enhanced crop**, which a human reads by eye; the dark strokes there are _candidate_ ink/ cracks, not verified letters.
2. **Raising the threshold does not fix it.** At `--thresh 90` the detector keeps 31 components / 6 regions (`images/output/PHerc1667Cr1Fr3_photo_highlighted_thresh90.png`), but visual inspection shows they mostly sit on bright fibre highlights (top-hat response) and edge texture, not on the dark strokes. We therefore do **not** present them as ink detections; they are included only for transparency.
3. Some dark lines in the crop may be cracks, fibre gaps or carbonisation damage rather than ink; the photo alone cannot separate them.
4. Single photograph, single lighting condition, no scale bar: no physical measurements are claimed.
5. Method is generic and not tuned on any ground truth.

## Reproduce

```bash
pip install -r requirements.txt
python code/highlight_characters.py images/input/PHerc1667Cr1Fr3_photo.jpg -o images/output/PHerc1667Cr1Fr3_photo_highlighted.png --debug
sha256sum -c CHECKSUMS.sha256        # outputs are byte-identical on the tested environment (opencv 4.12.0, numpy 2.2.6)
```

## Licence and credits

Code: MIT (`LICENCE`). Photograph of the Herculaneum fragment: from the Vesuvius Challenge / Herculaneum papyri materials; the rights of the photograph's owner apply - confirm the source licence before redistribution.
