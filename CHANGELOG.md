# Changelog

## 1.6.0 — 2026-11-10

- **Slots: precise placement.** New *Slots* mode on the Pick defects page. Box one pocket and press Detect: every similar pocket is found, the grid is completed, and poorly matching positions (usually empty pockets) are marked. Click pockets to make them targets; move, add and remove slots with the mouse; set the slot size and match strength. Detect on all images reuses the same pocket.
- New placement **Into slots (exact centre)**: pieces are centred exactly on target slots, with only rotation and contrast varying. *Defects per image* sets how many slots are filled per image. The recipe preview shows the real result.
- **Per-piece label.** Each picked piece is labeled either as a defect (pasted copies get the defect class) or as a good part (pasted copies stay background). A piece placed into a slot replaces the pocket's content, including any labeled defect there.
- Slots, targets and labels are saved in the project.

## 1.5.2 — 2026-11-03

Same features as 1.5.1; packaging and compatibility fixes.

- **Self-contained Windows bundle.** The offline zip now contains its own Python 3.12 with every package installed. Nothing is installed on the target PC; `start.bat` only launches the app. This fixes the "Search for app in the Store?" dialog on PCs without Python, where Windows' Store shortcut for `python` was picked up.
- **Windows 10 1809 / LTSC 2019 support.** PySide6 is limited to < 6.10, because Qt 6.10+ needs `icuuc.dll`, which those Windows versions do not have. The bundle ships the exact versions that pass the full test suite on Windows Python 3.12 (`packaging/windows-bundle-constraints.txt`).
- Source-folder `start.bat` looks for Python in the standard install folders, then on PATH, and skips the Microsoft Store shortcut.

## 1.5.1 — 2026-11-02

Packaging update (2026-11-03): one-click `start.bat` in the source folder and the offline bundle, and `make_offline_bundle.bat`.


- **Real source images are no longer copied into the dataset or the database.** They can hold defects nobody labeled, which taught the model those defects are background. The dataset now contains only synthetic images (not good) and generated backgrounds (good). The "Include source images" option is removed.
- New repair command for workspaces saved with 1.4/1.5.0: `python -m segsynth prune "<workspace>\META\<name>.sqlite"` removes the `real_*` images and any sample whose image file is missing, from both the database and `DATA`.

## 1.5.0 — 2026-11-02

- **AI tool ready export.** With "Generate a .sqlite file" the dataset is saved as a workspace the AI tool can train from directly: images in `DATA/synthetic/` and `DATA/background/`, the database in `META/`, plus `DATA_THUMB/`. Sample paths are relative to `DATA`, as the tool expects.
- Images are split into TRAIN and VALIDATE (default 15 %, adjustable on the Save page), stratified over good / not good and image kind; the AI tool needs validation samples to train.
- Database file names no longer contain spaces.
- New `segsynth.core.export` module shared by the Save page and the command line (`segsynth generate` with the database option writes the same workspace).

## 1.4.0 — 2026-11-02

- **Generate a .sqlite file** checkbox on the Save dataset page (with defect class name and colour). The label tool database is written at save time, so it can be added without regenerating. The option and class settings moved there from the Recipe page.
- Dataset layout: `synthetic/` for generated defect images, `background/` for source images and generated backgrounds.
- Every generated image keeps its label mask, so all saved images are labeled: background areas good, defect pixels with the defect class; images without defects are good, images with defects are not good.
- Generate page statistics refresh whenever the page is shown.

## 1.3.0 — 2026-11-02

- New optional step **Background generating**, enabled with "Create background images" on the Source images page. Images are cut into 2–8 pieces (horizontal or vertical cuts), the user clicks the pieces that are not good, and the good pieces of different images are stitched into new backgrounds.
- Pieces keep their position; every source is aligned to the first image before stitching, and seams are cross-faded over an adjustable band so no seam line appears.
- All combinations of good pieces from at least two different images are used, up to a configurable maximum (random selection above it).
- Generated backgrounds receive synthetic defects like any other background and are added to the dataset (and label tool database) as good images; defects picked on a source follow its pieces and are labeled.
- Live preview with full-resolution viewer; settings and rejected pieces are saved in the project.

## 1.2.0 — 2026-10-23

- Label tool export now includes the source images: images without picked defects are labeled good (all background, anomaly "Normal"); images with picked defects are labeled bad with those defects as their mask. Option "Include source images", on by default.
- Synthetic images are labeled bad (anomaly "Anomalous"); their mask now also contains the defects picked on their background image.
- Synthetic images are spread evenly over all background images.

## 1.1.0 — 2026-10-23

- New output option: label tool database (`.sqlite`). Each synthetic image is added as a TRAIN sample with the pasted defects as its semantic-segmentation label, using the tool's run-length mask format with CRC32. Images go to an `Images` folder next to the database.
- Configurable defect class name and colour for the exported database.
- Image paths in `manifest.json` now always use forward slashes.

## 1.0.0 — 2026-10-23

First release.

- Five-step desktop workflow: source images, defect picking, recipe, generation, saving.
- Defect picker with brush, eraser, box, lasso, undo/redo, zoom/pan and snap-to-defect refinement.
- Texture-preserving copy-paste with directional background estimation; Poisson (normal/mixed) and soft-paste blending.
- Variation of size, elongation, rotation, bend, elastic wobble, contrast and optional flips, all relative to the defect's own axis.
- Placement modes (anywhere, same column, same row, near source), minimum gap, avoidance of picked defects, several defects per image.
- Live variation preview, threaded generation with live gallery and full-resolution viewer.
- Project save/open, recent projects, reproducible seeded runs with `manifest.json`.
- Headless CLI: `segsynth generate`.
- Offline wheel bundle builder and standalone Windows build via CI.
