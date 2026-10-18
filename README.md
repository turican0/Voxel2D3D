# Voxel2D3D

Reconstructs the **most probable voxel model** from a handful of hand-drawn (or sprite-sheet)
2D views and exports it to **GLB**. Written in Rust, with an optional **SDL3** preview window.

* the viewing angles of the images are **not required** - yaw per image, camera pitch and
  vertical stretch are estimated;
* drawings never match exactly, so nothing is hard-carved: every voxel gets a probability and the
  model is the maximum-a-posteriori occupancy;
* every image may be **nudged by a few pixels** (continuously, sub-pixel) to fit the others; an
  image that has to move pays a score penalty, but the whole set fits better.

## Build

```sh
# Arch Linux
sudo pacman -S rust sdl3
cargo build --release

# core only (no SDL3 needed): reconstruction + GLB export
cargo build --release --no-default-features
```

## Use

```sh
voxel2d3d frames/*.png -o model.glb --report report/        # fully automatic
voxel2d3d frames/*.png --span 180 -o model.glb              # images cover a half circle evenly
voxel2d3d frames/*.png --yaw-step 25.714 -o model.glb       # step between images known
voxel2d3d frames/*.png --angles angles.txt --pitch 30       # yaws (and pitch) known
voxel2d3d frames/*.png --fine-y 0.1,0.55 -o model.glb       # legs (10-55 % of the height) at 2x resolution
voxel2d3d frames/*.png -o model.glb --view                  # open the SDL3 preview
```

`voxel2d3d --help` lists every option. Images need a transparent background (alpha) or a
roughly uniform background colour. Give them **in rotation order** - the order is the starting
guess for the angles.

`--report DIR` writes `report.json` (estimated yaw, pitch, shifts, IoU per image) and one overlay
PNG per input: grey = input and model agree, red = only in the input, blue = only in the model.

Viewer: drag = rotate, wheel = zoom, space = auto-rotate, `R` = reset, `Esc` = quit.

## Pipeline (stages)

1. **Angles first.** Colour regions (markers) are tracked from view to view; the tracks give a
   starting estimate of every yaw, the pitch and the position of the rotation axis in each cropped
   image. Then the angle of **every image is searched separately** (window `--yaw-dev`, default
   10 deg) with silhouettes, colour consistency and the marker residual as the score.
2. **Coarse model** is carved from the silhouettes.
3. **Deviations.** The model is rendered into every view; where it does not fit, parts of the
   image are nudged by the smallest displacement (`--nudge`, default 3 px, penalised, smooth).
4. **The same again with full colours** on the corrected images (stage 1 used 8 posterised colours).

5. **Reconstruction:** evidence on a tight grid, small holes closed (gaps wider than 2 voxels, e.g.
   between the legs, stay open), then a *coverage repair*: every silhouette pixel the model does
   not explain gets the most probable voxel along its ray (only near the existing model). Colours
   come from the best-facing view, not a blend. `--no-repair` switches the repair off.

`--no-stages` runs a single pass, `--no-markers` skips the marker seeding.

## How it works

1. **Silhouettes.** Alpha channel or background colour (Otsu), holes filled, specks removed.
   A signed distance field gives soft inside-probabilities `p = sigmoid(d / sigma)` at three
   sharpness levels (blurry -> sharp).
2. **Camera.** Orthographic, rotation around the vertical axis. Per image: yaw and a 2D shift
   (the crop means the position of the rotation axis in the image is unknown: nominal position =
   silhouette bbox centre, bounded + penalised shift on top). Shared: pitch and vertical stretch
   (oblique game sprites are not foreshortened by cos(pitch)).
3. **Evidence.** Each view votes for every voxel: inside the silhouette = weak evidence for
   "occupied", outside = strong evidence for "empty" (log-likelihood ratios from `--noise`,
   `--occlusion`, `--prior`). Occupied iff the sum is positive - a visual hull that tolerates
   an outlier drawing.
4. **Angle estimation.** Coordinate descent (yaw -> shift -> pitch + vertical stretch) maximising:
   mean IoU of the soft projected hull with each silhouette + colour consistency of the hull
   surface between views - shift penalty. The three best starting layouts each run through the
   whole pipeline and the best final score wins. Coarse-to-fine in grid size and silhouette softness, started from several
   evenly spaced layouts (both rotation directions).
5. **Final model.** Evidence on a tight grid, largest connected component, colours from the
   views that see the voxel best, surface mesh with vertex colours -> GLB (+Y up, +Z front,
   `--rotate` turns it). Vertex colours are written as normalized `UNSIGNED_BYTE` `COLOR_0`
   (linear), which Blender, three.js and Godot read; note that `trimesh` ignores it.

## Conventions

yaw 0 = camera on +Z looking at the origin, yaw 90 = camera on +X; positive pitch = camera above
the object. World unit = one pixel of the source images (scaled to `--height` on export).

## Limits (read this)

* Silhouettes alone are **ambiguous**: for blobby objects, a mirrored rotation direction with
  negative pitch fits almost as well. This is why pitch is limited to >= 0 by default
  (`--pitch-range`) and colour consistency is part of the score.
* The fully automatic angle search is **not reliable on small pixel-art sprites** (tested on an
  8-frame character sprite sheet): it needs a hint (`--span 180` / `--yaw-step`). If you know (or can read off the widths of the
  silhouettes) the step between frames, `--yaw-step` fixes it and the estimator only handles the
  per-image deviations (`--yaw-dev`), pitch, vertical stretch and shifts. `--angles` pins them completely.
* The estimated pitch of the sample sprite sheet came out lower (~9 deg) than it looks; the model
  is still coherent from novel angles, but treat the estimated pitch as approximate (`--pitch` fixes it).
* Thin parts (arms, cloth) shrink in a visual hull built from few views.
* The SDL3 viewer is a simple software renderer (surface voxels as squares); fine for a preview,
  not for huge grids.

## Testing

```sh
cargo test --no-default-features
python3 tools/gen_synthetic.py 30 0,25,50,75,100,125,150,175   # views of a known scene
voxel2d3d v?.png -o synth.glb --report report/
```

## License

MIT
