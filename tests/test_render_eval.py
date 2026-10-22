import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import render_eval as app


class RenderEvalTests(unittest.TestCase):
    def test_frame_ids_are_sorted_and_checked(self):
        self.assertEqual(app.select_ids([0, 30, 60], "60,0", 1), [0, 60])
        with self.assertRaises(ValueError):
            app.select_ids([0, 30], "1", 1)
        with self.assertRaises(ValueError):
            app.select_ids([0, 30], None, 0)

    def test_camera_poses_are_camera_to_world(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            pose = np.eye(4)
            pose[0, 3] = 2.0
            camera = dict(fx=300, fy=300, cx=160, cy=160,
                          width=320, height=320, c2w=pose.tolist())
            path.write_text(json.dumps({"30": camera}))
            frames = app.load_cameras(path)
            self.assertEqual(frames[30]["c2w"][0][3], 2.0)

    def test_manifest_paths_are_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.csv"
            path.write_text("frame,rgb,depth\n30,rgb/30.png,depth/30.png\n")
            rows = app.load_manifest(path)
            self.assertEqual(rows[30]["rgb"], path.parent / "rgb/30.png")
            path.write_text("frame,rgb,depth\n30,a.png,\n30,b.png,\n")
            with self.assertRaises(ValueError):
                app.load_manifest(path)

    def test_masked_psnr_ignores_invalid_pixels(self):
        reference = torch.zeros(3, 320, 320)
        prediction = torch.full_like(reference, 0.5)
        valid = torch.zeros(320, 320, dtype=torch.bool)
        valid[0, 0] = True
        prediction[:, ~valid] = 1.0
        metrics = app.compute_metrics(prediction, reference, valid, None)
        self.assertAlmostEqual(metrics["psnr"], 6.020599913, places=5)
        self.assertNotIn("lpips", metrics)

    def test_identical_images_have_unit_ms_ssim(self):
        image = torch.full((3, 320, 320), 0.5)
        metrics = app.compute_metrics(image, image, torch.ones(320, 320, dtype=torch.bool), None)
        self.assertAlmostEqual(metrics["ms_ssim"], 1.0, places=5)
        with self.assertRaises(ValueError):
            app.compute_metrics(image, image, torch.zeros(320, 320, dtype=torch.bool), None)

    def test_composition_uses_premultiplied_residual(self):
        base = torch.ones(3, 2, 2)
        residual = torch.full_like(base, 0.1)
        alpha = torch.full((1, 2, 2), 0.25)
        result = app.compose(residual, alpha, base)
        self.assertTrue(torch.allclose(result, torch.full_like(base, 0.85)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA rasterizers required")
    def test_ply_loader_applies_parameter_activations(self):
        from plyfile import PlyData, PlyElement
        from layergs_render import load_base_ply
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity"]
        vertices = np.zeros(1, dtype=[(name, "f4") for name in names])
        vertices["x"] = 2
        vertices["scale_0"] = np.log(2)
        vertices["rot_0"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.ply"
            PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
            data, count = load_base_ply(path)
            self.assertEqual(count, 1)
            self.assertEqual(data["xyz"][0, 0].item(), 2.0)
            self.assertTrue(torch.allclose(data["scales"], torch.tensor([[2.0, 1.0, 1.0]])))
            self.assertEqual(data["opacity"].item(), 0.5)
            self.assertTrue(torch.equal(data["rotations"], torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
            self.assertEqual(data["sh_degree"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA rasterizers required")
    def test_prefilter_removes_only_occluded_centres(self):
        from layergs_render import depth_prefilter_mask
        xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [100.0, 0.0, 2.0]])
        depth = torch.ones(4, 4)
        camera = dict(fx=1, fy=1, cx=1, cy=1)
        keep = depth_prefilter_mask(xyz, depth, camera, torch.eye(4), 0.05)
        self.assertEqual(keep.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
