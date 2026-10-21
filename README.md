# LayerGS: Layered 2D–3D Gaussian Mapping for Dense RGB-D SLAM

**Code will be released soon.**

LayerGS is a compact layered 2D–3D Gaussian mapping framework for streaming RGB-D SLAM. It couples a 2D Gaussian surfel base with a lightweight residual 3D Gaussian layer through explicit bindings, structured online allocation, and multiview refinement to improve rendering quality.

## Qualitative Results

![Qualitative comparison of backbone renderings, residual-only renderings, and LayerGS results](assets/qualitative_comparison.png)

Qualitative comparison using the 2DGS-SLAM and EGG-Fusion backbones. For each backbone, the figure shows the original rendering, the residual-only rendering, and the LayerGS-enhanced result alongside the reference image.

## Release Status

This repository currently provides a preview of our work. The implementation and instructions will be added here soon.

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
included. The implementation and loading instructions will follow.
