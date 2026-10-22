# LayerGS: Layered 2D–3D Gaussian Mapping for Dense RGB-D SLAM

**Rendering and evaluation tools are available. The full SLAM implementation will be released soon.**

LayerGS is a compact layered 2D–3D Gaussian mapping framework for streaming RGB-D SLAM. It couples a 2D Gaussian surfel base with a lightweight residual 3D Gaussian layer through explicit bindings, structured online allocation, and multiview refinement to improve rendering quality.

## Qualitative Results

![Qualitative comparison of backbone renderings, residual-only renderings, and LayerGS results](assets/qualitative_comparison.png)

Qualitative comparison using the 2DGS-SLAM and EGG-Fusion backbones. For each backbone, the figure shows the original rendering, the residual-only rendering, and the LayerGS-enhanced result alongside the reference image.

## Release Status

This repository provides reconstruction samples and standalone rendering and evaluation tools for the EGG-Fusion integration. The full mapping and training implementation will follow.

## Reconstruction Samples

The final reconstruction results from three additional runs of the EGG-Fusion integration process are provided below. These results are **partial results**.

| Dataset | Scene | Download |
| --- | --- | --- |
| Replica | room0 | [Model package](https://github.com/WincentLi/layergs/releases/download/reconstruction-samples-20261021/replica_room0_layergs_eggfusion_sample.zip) |
| Replica | office0 | [Model package](https://github.com/WincentLi/layergs/releases/download/reconstruction-samples-20261021/replica_office0_layergs_eggfusion_sample.zip) |
| TUM RGB-D | fr3_office (freiburg3_long_office_household) | [Model package](https://github.com/WincentLi/layergs/releases/download/reconstruction-samples-20261021/tum_fr3_office_layergs_eggfusion_sample.zip) |

Each package contains the base surfel map, residual Gaussian map,
estimated trajectories, configuration, and keyframe camera metadata.
The two Gaussian layers retain their native export formats. No rendered
images, original datasets, evaluation scores, or evaluation logs are
included. Use the script below to render these models or evaluate them against your own reference data.

## Rendering and Evaluation

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
`rasterizers/`; see [third-party notices](THIRD_PARTY_NOTICES.md).

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
