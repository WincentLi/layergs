# LayerGS: Layered 2D–3D Gaussian Mapping for Dense RGB-D SLAM

LayerGS couples a 2D Gaussian surfel base with a lightweight residual 3D Gaussian layer to improve rendering quality in streaming RGB-D SLAM.

**Rendering and evaluation tools are available. Full implementation coming soon.**

![LayerGS qualitative comparison](assets/qualitative_comparison.png)

## Samples

Partial reconstruction results from the EGG-Fusion integration:

| Dataset | Scene | Download |
| --- | --- | --- |
| Replica | room0 | [Model](https://github.com/WincentLi/layergs/releases/download/reconstruction-samples-20261021/replica_room0_layergs_eggfusion_sample.zip) |
| Replica | office0 | [Model](https://github.com/WincentLi/layergs/releases/download/reconstruction-samples-20261021/replica_office0_layergs_eggfusion_sample.zip) |
| TUM RGB-D | fr3_office | [Model](https://github.com/WincentLi/layergs/releases/download/reconstruction-samples-20261021/tum_fr3_office_layergs_eggfusion_sample.zip) |

## Rendering and Evaluation

Requires Linux, an NVIDIA GPU, CUDA, and compatible PyTorch/torchvision.

```bash
pip install -r requirements-render.txt
pip install --no-build-isolation ./rasterizers/diff-gaussian-surfels ./rasterizers/diff-gaussian-rasterization
python render_eval.py --model-dir samples/room0 --output outputs/room0 --frame-ids 0
```

Extract a model package into `samples/room0` first. The script renders the base, residual-only, and fused images. Add `--gt-manifest /path/to/gt.csv` to evaluate PSNR, MS-SSIM, and LPIPS.

See [usage details](docs/rendering.md) for reference-data preparation and additional options.
