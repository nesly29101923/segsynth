# Changelog

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
