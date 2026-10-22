import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_msssim import ms_ssim


def select_ids(available, requested, stride):
    if stride < 1:
        raise ValueError("stride must be positive")
    ids = sorted(available)
    if requested:
        selected = sorted({int(value) for value in requested.split(",")})
        missing = set(selected) - set(ids)
        if missing:
            raise ValueError(f"Unavailable frame IDs: {sorted(missing)}")
        ids = selected
    if not ids:
        raise ValueError("No cameras selected")
    return ids[::stride]


def load_cameras(path):
    raw = json.loads(Path(path).read_text())
    frames = {}
    for key, camera in raw.items():
        pose = np.asarray(camera["c2w"], dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid camera pose: {key}")
        if camera["width"] <= 0 or camera["height"] <= 0 or camera["fx"] <= 0 or camera["fy"] <= 0:
            raise ValueError(f"Invalid camera calibration: {key}")
        frames[int(key)] = camera
    if not frames:
        raise ValueError("Camera file is empty")
    return frames


def load_manifest(path):
    path = Path(path).resolve()
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"frame", "rgb"}.issubset(reader.fieldnames or []):
            raise ValueError("Ground-truth manifest requires frame,rgb columns")
        for row in reader:
            index = int(row["frame"])
            if index in rows:
                raise ValueError(f"Duplicate ground-truth frame: {index}")
            if not row["rgb"].strip():
                raise ValueError(f"Missing RGB path: {index}")
            rows[index] = {
                field: path.parent / row[field].strip()
                for field in ["rgb", "depth"] if row.get(field, "").strip()
            }
    return rows


def compose(residual, alpha, base):
    return residual + (1.0 - alpha.clamp(0, 1)) * base


def compute_metrics(prediction, reference, valid, lpips):
    if prediction.shape != reference.shape or valid.shape != reference.shape[1:]:
        raise ValueError("RGB and depth dimensions must match camera dimensions")
    if not valid.any():
        raise ValueError("Frame has no valid pixels")
    mse = ((prediction - reference) ** 2)[:, valid].mean().item()
    psnr = 100.0 if mse < 1e-10 else -10.0 * math.log10(mse)
    prediction, reference = prediction.clone(), reference.clone()
    prediction[:, ~valid] = 0
    reference[:, ~valid] = 0
    prediction = prediction.unsqueeze(0).clamp(0, 1)
    reference = reference.unsqueeze(0).clamp(0, 1)
    scores = {"psnr": psnr, "ms_ssim": ms_ssim(prediction, reference, data_range=1.0, size_average=True).item()}
    if lpips is not None:
        lpips.reset()
        scores["lpips"] = lpips(prediction, reference).item()
    return scores


def read_reference(row, camera, depth_scale, device):
    image = cv2.imread(str(row["rgb"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(row["rgb"])
    height, width = camera["height"], camera["width"]
    if image.shape[:2] != (height, width):
        raise ValueError(f"Reference image resolution differs from camera: {row['rgb']}")
    rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()).to(device).permute(2, 0, 1).float() / 255.0
    valid = torch.ones(height, width, dtype=torch.bool, device=device)
    if "depth" in row:
        depth = cv2.imread(str(row["depth"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(row["depth"])
        if depth.shape != (height, width):
            raise ValueError(f"Depth resolution differs from camera: {row['depth']}")
        depth = depth.astype(np.float32) / depth_scale
        valid = torch.from_numpy(np.isfinite(depth) & (depth > 0)).to(device)
    return rgb, valid


def save_rgb(path, image):
    rgb = (image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description="Render and evaluate exported LayerGS EGG-Fusion models")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-ids", help="Comma-separated frame IDs; defaults to all exported keyframes")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--trajectory", type=Path, help="Optional camera-to-world trajectory: one flattened 4x4 matrix per line")
    parser.add_argument("--gt-manifest", type=Path, help="CSV with frame,rgb,depth columns; depth is optional")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-prefilter-eps", type=float, default=0.0)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    args = parser.parse_args()
    if args.depth_scale <= 0 or args.depth_prefilter_eps < 0:
        parser.error("depth-scale must be positive and depth-prefilter-eps nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Rendering requires an NVIDIA GPU and CUDA-enabled PyTorch")
    from layergs_render import load_base_ply, render_2dgs, render_3dgs, depth_prefilter_mask

    cameras = load_cameras(args.model_dir / "keyframe_cameras.json")
    if args.trajectory:
        matrices = np.loadtxt(args.trajectory, ndmin=2)
        if matrices.shape[1] != 16 or not np.isfinite(matrices).all():
            raise ValueError("Trajectory must contain 16 finite values per line")
        calibration = cameras[min(cameras)]
        cameras = {i: dict(calibration, c2w=matrix.reshape(4, 4).tolist()) for i, matrix in enumerate(matrices)}
    ids = select_ids(cameras, args.frame_ids, args.stride)
    references = load_manifest(args.gt_manifest) if args.gt_manifest else None
    if references is not None and set(ids) - set(references):
        raise ValueError(f"Ground-truth manifest is missing selected frames: {sorted(set(ids) - set(references))[:10]}")
    device = "cuda"
    base, _ = load_base_ply(args.model_dir / "base_map.ply")
    residual, _ = load_base_ply(args.model_dir / "residual_3dgs.ply")
    base = {key: value.to(device) if torch.is_tensor(value) else value for key, value in base.items()}
    residual = {key: value.to(device) if torch.is_tensor(value) else value for key, value in residual.items()}
    lpips = None
    if references is not None and not args.skip_lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    background = torch.zeros(3, device=device)
    with torch.no_grad():
        for position, frame in enumerate(ids):
            camera = cameras[frame]
            c2w = torch.tensor(camera["c2w"], dtype=torch.float32, device=device)
            fx, fy, cx, cy = (camera[key] for key in ("fx", "fy", "cx", "cy"))
            width, height = int(camera["width"]), int(camera["height"])
            base_out = render_2dgs(base, c2w, fx, fy, cx, cy, width, height, background, device)
            selected = residual
            if args.depth_prefilter_eps > 0:
                keep = depth_prefilter_mask(residual["xyz"], base_out["depth"].squeeze(0), camera, torch.inverse(c2w), args.depth_prefilter_eps)
                selected = {key: value[keep].contiguous() if torch.is_tensor(value) else value for key, value in residual.items()}
            residual_out = render_3dgs(selected, c2w, fx, fy, cx, cy, width, height, background, residual["sh_degree"], device)
            images = {
                "base": base_out["color"],
                "residual": residual_out["color"],
                "layered": compose(residual_out["color"], residual_out["alpha"], base_out["color"]),
            }
            if not args.no_save_images:
                for name, image in images.items():
                    save_rgb(args.output / f"{frame:06d}_{name}.png", image)
            if references is not None:
                reference, valid = read_reference(references[frame], camera, args.depth_scale, device)
                row = {"frame": frame, "valid_pixels": int(valid.sum().item())}
                for name in ("base", "layered"):
                    for metric, value in compute_metrics(images[name], reference, valid, lpips).items():
                        row[f"{name}_{metric}"] = value
                rows.append(row)
            print(f"[{position + 1}/{len(ids)}] frame {frame}", flush=True)
    if rows:
        with (args.output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in {"frame", "valid_pixels"}}
        summary["frames"] = len(rows)
        summary["depth_prefilter_eps"] = args.depth_prefilter_eps
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
