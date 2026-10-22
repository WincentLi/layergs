# Third-party rasterizers

`rasterizers/diff-gaussian-surfels` is the surfel rasterizer used by
EGG-Fusion, distributed from https://github.com/panxkun/diff-gaussian-surfels.
Its copyright headers and `LICENSE.md` are preserved.

`rasterizers/diff-gaussian-rasterization` is the depth/alpha-capable 3D
Gaussian rasterizer used by the LayerGS EGG-Fusion integration. Its
upstream README credits the Graphdeco 3D Gaussian Splatting rasterizer,
https://github.com/slothfulxtx/diff-gaussian-rasterization for depth
rendering, and https://github.com/shakibakh/diff-gaussian-rasterization
for MCMC support. The local variant exposes the Python package as
`diff_gaussian_rasterization_3d`. Its copyright headers and `LICENSE.md`
are preserved.

Both rasterizers include GLM headers with their original attribution and
`third_party/glm/copying.txt`. Source files are supplied without build
products or repository metadata. These components remain subject to
their own licenses, including the Gaussian-Splatting research/evaluation,
non-commercial restrictions. No broader license is granted here.
