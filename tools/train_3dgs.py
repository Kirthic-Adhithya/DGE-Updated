"""Minimal vanilla 3DGS trainer built on the gaussiansplatting package bundled with DGE.

The bundled gaussiansplatting/train.py is stale relative to DGE's modified OptimizationParams / GaussianModel,
and the official repo's rasterizer is incompatible with DGE's (DGE's returns an extra depth map). This script
uses DGE's own rasterizer + the vanilla (mask-free) GaussianModel so the resulting .ply loads directly in DGE.

Usage: python tools/train_3dgs.py -s data/tandt/truck -m outputs/truck_3dgs --iterations 15000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import os
import sys
from argparse import ArgumentParser
from random import randint

import torch
from tqdm import tqdm

from gaussiansplatting.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel
from gaussiansplatting.utils.image_utils import psnr
from gaussiansplatting.utils.loss_utils import l1_loss, ssim


def main():
    parser = ArgumentParser()
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=15000)
    # Higher threshold = fewer densification events = fewer Gaussians (default 2e-4). Needed to fit DGE editing in 8 GB.
    parser.add_argument("--densify_grad_threshold", type=float, default=2e-4)
    args = parser.parse_args(sys.argv[1:])
    dataset, pipe = mp.extract(args), pp.extract(args)
    # OptimizationParams needs max_steps up front; a throwaway parser keeps its flags from clashing with ours.
    opt = OptimizationParams(ArgumentParser(), args.iterations)
    opt.densify_grad_threshold = args.densify_grad_threshold

    os.makedirs(dataset.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    background = torch.zeros(3, device="cuda")
    stack = None
    ema = 0.0
    bar = tqdm(range(1, args.iterations + 1), desc="3DGS training")
    for it in bar:
        gaussians.update_learning_rate(it)
        if it % 1000 == 0:
            gaussians.oneupSHdegree()
        if not stack:
            stack = scene.getTrainCameras().copy()
        cam = stack.pop(randint(0, len(stack) - 1))

        pkg = render(cam, gaussians, pipe, background)
        image, vsp, vis, radii = pkg["render"], pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"]
        gt = cam.original_image.cuda()
        loss = (1 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim * (1 - ssim(image, gt))
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if it % 10 == 0:
                bar.set_postfix(loss=f"{ema:.5f}", n=gaussians.get_xyz.shape[0])
            if it < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(vsp, vis)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    size_thr = 20 if it > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_thr)
                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    scene.save(args.iterations)
    # Held-out evaluation (every 8th view because of --eval)
    with torch.no_grad():
        for name, cams in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras()[:20])):
            if not cams:
                continue
            vals = []
            for cam in cams:
                img = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0, 1)
                vals.append(psnr(img, cam.original_image.cuda()).mean().item())
            print(f"[{name}] PSNR {sum(vals) / len(vals):.2f} over {len(vals)} views, {gaussians.get_xyz.shape[0]} gaussians")


if __name__ == "__main__":
    main()
