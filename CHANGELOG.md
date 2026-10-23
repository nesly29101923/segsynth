# Changelog

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
