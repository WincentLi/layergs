# Rendering and Evaluation

`render_eval.py` renders the base surfel layer, residual-only layer, and
LayerGS-composed image. Providing a ground-truth manifest additionally
enables PSNR, MS-SSIM, and AlexNet LPIPS evaluation. It does not run SLAM
or train models, and currently supports the EGG-Fusion export format used
by the three sample packages.

### Installation

Use Linux with an NVIDIA GPU, a CUDA toolkit with `nvcc`, and a compatible
CUDA-enabled PyTorch and torchvision installation. The smoke tests used
Python 3.9, PyTorch 1.13.1+cu117, and CUDA 11.7. Install PyTorch and
torchvision first, then run from this repository:

```bash
python -m pip install -r requirements-render.txt
python -m pip install --no-build-isolation ./rasterizers/diff-gaussian-surfels
python -m pip install --no-build-isolation ./rasterizers/diff-gaussian-rasterization
```

The bundled rasterizers use the matching 2DGS and 3DGS interfaces. They
are not interchangeable with arbitrary packages of the same name.
Their upstream licenses and attribution notices are retained under
`rasterizers/`; see [third-party notices](../THIRD_PARTY_NOTICES.md).

### Render a sample

Extract a model package into `samples/room0/`, containing `base_map.ply`,
`residual_3dgs.ply`, and `keyframe_cameras.json`, then run:

```bash
python render_eval.py --model-dir samples/room0 --output outputs/room0 --frame-ids 0
```

Omit `--frame-ids` to render all exported keyframes. `--stride 2` selects
every second available camera, not every second original dataset frame.
For each selected camera, the script writes `*_base.png`, `*_residual.png`,
and `*_layered.png`. Both layers use a black background and compose as
`residual_color + (1 - residual_alpha) * base_color`.

To match the exported models, the script preserves their native projection
conventions: the surfel rasterizer takes explicit principal-point offsets,
while the residual rasterizer uses the centered-FOV projection used during
training. It does not recalibrate the pretrained residual layer.

To render all trajectory frames instead of only exported keyframes:

```bash
python render_eval.py --model-dir samples/room0 --trajectory samples/room0/trajectory_est_matrix.txt --output outputs/room0_all
```

The trajectory must contain one flattened, row-major camera-to-world
4-by-4 matrix per line. Its row index becomes the frame ID, and calibration
comes from the first exported keyframe. Trajectory and model coordinates
must be consistent.

### Evaluate against reference images

Prepare a CSV manifest with exact frame-to-reference correspondence:

```csv
frame,rgb,depth
0,rgb/frame000000.png,depth/depth000000.png
```

Paths are absolute or relative to the manifest directory. IDs refer to
the exported camera IDs, or trajectory row indices with `--trajectory`.
Reference images must already match the camera resolution and alignment.
For TUM, use the RGB/depth timestamp association from the corresponding
run rather than assuming raw file numbering matches camera IDs.

```bash
python render_eval.py --model-dir samples/room0 --output outputs/room0_eval --frame-ids 0 --gt-manifest /path/to/gt.csv --depth-scale 1000
```

The depth column is optional. With depth, valid pixels have finite,
positive depth. Without depth, all pixels are evaluated. `--depth-scale`
converts stored depth to metres; use `5000` for native TUM depth PNGs.
PSNR uses valid-pixel RGB MSE. MS-SSIM and LPIPS zero invalid pixels in
both images before evaluation. MS-SSIM requires both image dimensions
to exceed 160 pixels with its default settings. Metrics are computed on
floating-point renders before PNG quantization and averaged equally
over selected frames. Evaluation writes `metrics.csv` and `summary.json`
locally. No evaluation results are included in this repository.

LPIPS requires pretrained AlexNet weights, which may download on first
use. `--skip-lpips` explicitly disables it. `--no-save-images` suppresses
PNG export. `--depth-prefilter-eps 0.05` optionally culls residual centres
behind the rendered base depth by more than 5 cm; the default is disabled.
Keep this setting and the selected views consistent when comparing runs.

### Tests

```bash
python -m unittest discover -s tests -v
```
