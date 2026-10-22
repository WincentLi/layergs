import math
import numpy as np
import torch
from plyfile import PlyData
from diff_gaussian_rasterization import GaussianRasterizationSettings as Settings2D, GaussianRasterizer as Rasterizer2D
from diff_gaussian_rasterization_3d import GaussianRasterizationSettings as Settings3D, GaussianRasterizer as Rasterizer3D

def load_base_ply(filepath):
    ply = PlyData.read(filepath)
    verts = ply.elements[0]
    N = verts.count
    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1)
    f_dc_vals = np.stack([verts['f_dc_0'], verts['f_dc_1'], verts['f_dc_2']], axis=1)
    features_dc = torch.tensor(f_dc_vals, dtype=torch.float32).reshape(N, 1, 3)
    rest_names = sorted([p.name for p in verts.properties if p.name.startswith('f_rest_')], key=lambda x: int(x.split('_')[-1]))
    n_rest_coeffs = len(rest_names) // 3
    f_rest = np.zeros((N, len(rest_names)), dtype=np.float32)
    for i, name in enumerate(rest_names):
        f_rest[:, i] = verts[name]
    features_rest = torch.tensor(f_rest, dtype=torch.float32).reshape(N, 3, n_rest_coeffs).transpose(1, 2).contiguous()
    shs = torch.cat([features_dc, features_rest], dim=1)
    scales_raw = np.stack([verts['scale_0'], verts['scale_1'], verts['scale_2']], axis=1)
    scales = torch.exp(torch.tensor(scales_raw, dtype=torch.float32))
    rots_raw = np.stack([verts['rot_0'], verts['rot_1'], verts['rot_2'], verts['rot_3']], axis=1)
    rotations = torch.nn.functional.normalize(torch.tensor(rots_raw, dtype=torch.float32))
    opacities_raw = verts['opacity']
    opacities = torch.sigmoid(torch.tensor(opacities_raw, dtype=torch.float32)).unsqueeze(-1)
    sh_degree = int(math.sqrt(n_rest_coeffs + 1)) - 1
    return ({'xyz': torch.tensor(xyz, dtype=torch.float32), 'opacity': opacities, 'scales': scales, 'rotations': rotations, 'shs': shs, 'sh_degree': sh_degree}, N)

def build_projection_matrix(fx, fy, cx, cy, width, height, znear, zfar):
    fovx = 2 * math.atan(width / (2 * fx))
    fovy = 2 * math.atan(height / (2 * fy))
    tanHalfFovX = math.tan(fovx / 2)
    tanHalfFovY = math.tan(fovy / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def render_2dgs(gaussian_data, c2w, fx, fy, cx, cy, W, H, bg, device):
    gs = {k: v.to(device) for k, v in gaussian_data.items() if isinstance(v, torch.Tensor)}
    w2c = torch.inverse(c2w)
    znear, zfar = (0.01, 100.0)
    fovx = 2 * math.atan(W / (2 * fx))
    fovy = 2 * math.atan(H / (2 * fy))
    projmat = build_projection_matrix(fx, fy, cx, cy, W, H, znear, zfar).t().to(device)
    world_view_transform = w2c.T
    full_proj_transform = world_view_transform @ projmat
    camera_center = c2w[:3, 3]
    raster_settings = Settings2D(image_height=H, image_width=W, tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5), cx=cx, cy=cy, bg=bg, scale_modifier=1.0, viewmatrix=world_view_transform, projmatrix=full_proj_transform, sh_degree=gaussian_data['sh_degree'], campos=camera_center, prefiltered=False, debug=False)
    rasterizer = Rasterizer2D(raster_settings=raster_settings)
    tile_mask = torch.ones((H + 15) // 16, (W + 15) // 16, dtype=torch.int32, device=device)
    result = rasterizer(means3D=gs['xyz'], opacities=gs['opacity'], shs=gs['shs'], colors_precomp=None, scales=gs['scales'], rotations=gs['rotations'], cov3D_precomp=None, tile_mask=tile_mask)
    return {'color': result[0], 'normal': result[1], 'depth': result[2], 'opacity': result[3]}

def render_3dgs(res_params, c2w, fx, fy, cx, cy, W, H, bg, sh_degree, device):
    N = res_params['xyz'].shape[0]
    if N == 0:
        return {'color': torch.zeros(3, H, W, device=device), 'depth': torch.zeros(1, H, W, device=device), 'alpha': torch.zeros(1, H, W, device=device)}
    w2c = torch.inverse(c2w)
    znear, zfar = (0.01, 100.0)
    fovx = 2 * math.atan(W / (2 * fx))
    fovy = 2 * math.atan(H / (2 * fy))
    projmat = build_projection_matrix(fx, fy, cx, cy, W, H, znear, zfar).t().to(device)
    world_view_transform = w2c.T
    full_proj_transform = world_view_transform @ projmat
    camera_center = c2w[:3, 3]
    raster_settings = Settings3D(image_height=H, image_width=W, tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5), bg=bg, scale_modifier=1.0, viewmatrix=world_view_transform, projmatrix=full_proj_transform, sh_degree=sh_degree, campos=camera_center, prefiltered=False, debug=False)
    rasterizer = Rasterizer3D(raster_settings=raster_settings)
    means2D = torch.zeros((N, 2), device=device)
    color, depth, alpha, radii, extra = rasterizer(means3D=res_params['xyz'].contiguous(), means2D=means2D, opacities=res_params['opacity'].contiguous(), shs=res_params['shs'].contiguous(), scales=res_params['scales'].contiguous(), rotations=res_params['rotations'].contiguous())
    return {'color': color, 'depth': depth, 'alpha': alpha}

def depth_prefilter_mask(xyz_world, base_depth, cam, w2c, eps):
    N = xyz_world.shape[0]
    device = xyz_world.device
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    ones = torch.ones(N, 1, device=device)
    xyz_h = torch.cat([xyz_world, ones], dim=-1)
    xyz_cam = (w2c @ xyz_h.T).T
    x_cam, y_cam, z_cam = (xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2])
    fx, fy = (cam['fx'], cam['fy'])
    cx, cy = (cam['cx'], cam['cy'])
    H, W = base_depth.shape
    u = fx * x_cam / z_cam.clamp(min=1e-06) + cx
    v = fy * y_cam / z_cam.clamp(min=1e-06) + cy
    inside = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (z_cam > 0.01)
    u_int = u.long().clamp(0, W - 1)
    v_int = v.long().clamp(0, H - 1)
    base_depth_sampled = base_depth[v_int, u_int]
    depth_diff = z_cam - base_depth_sampled
    cull = inside & (depth_diff > eps)
    return ~cull
