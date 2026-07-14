#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


_lpips_loss_net = None  # lazy-init frozen VGG for --lambda_lpips fine-tune stage


def edge_aware_tv_loss(render, gt, beta=10.0):
    """Edge-preserving total-variation loss (DET-GS style). Penalizes gradients
    in the rendered image weighted by GT-edge-awareness: homogeneous GT regions
    (sky, panels) get high weight → floaters/speckle smoothed; GT edges (metal
    trusses) get near-zero weight → sharp boundaries preserved. C,H,W tensors."""
    dx_r = torch.abs(render[:, :, 1:] - render[:, :, :-1])
    dy_r = torch.abs(render[:, 1:, :] - render[:, :-1, :])
    dx_g = torch.abs(gt[:, :, 1:] - gt[:, :, :-1]).mean(0, keepdim=True)
    dy_g = torch.abs(gt[:, 1:, :] - gt[:, :-1, :]).mean(0, keepdim=True)
    wx = torch.exp(-beta * dx_g)
    wy = torch.exp(-beta * dy_g)
    return (wx * dx_r).mean() + (wy * dy_r).mean()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # per-training-image appearance affine (exposure/WB drift absorber):
    # render*(1+gain)+bias is compared to GT during TRAINING ONLY, so drift is
    # absorbed here instead of being baked into the model; test renders stay raw.
    app_gain = app_bias = app_optim = None
    if opt.appearance_affine:
        n_train = len(viewpoint_stack)
        app_gain = torch.zeros(n_train, 3, device="cuda", requires_grad=True)
        app_bias = torch.zeros(n_train, 3, device="cuda", requires_grad=True)
        app_optim = torch.optim.Adam([app_gain, app_bias], lr=opt.app_lr)
        print(f"Appearance affine enabled: {n_train} images, lr={opt.app_lr}, reg={opt.lambda_app}")

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if app_gain is not None:
            uid = viewpoint_cam.uid
            image_pm = image * (1.0 + app_gain[uid]).view(3, 1, 1) + app_bias[uid].view(3, 1, 1)
        else:
            image_pm = image
        Ll1 = l1_loss(image_pm, gt_image)
        ssim_value = fast_ssim(image_pm.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if app_gain is not None and opt.lambda_app > 0:
            loss = loss + opt.lambda_app * (app_gain.square().mean() + app_bias.square().mean())
        if opt.lambda_tv > 0:
            loss = loss + opt.lambda_tv * edge_aware_tv_loss(image_pm, gt_image, opt.tv_beta)
        if opt.lambda_opacity > 0 and iteration > opt.opacity_from_iter:
            o = gaussians.get_opacity.clamp(1e-6, 1 - 1e-6)
            opacity_entropy = -(o * torch.log(o) + (1 - o) * torch.log(1 - o)).mean()
            loss = loss + opt.lambda_opacity * opacity_entropy
        if opt.lambda_lpips > 0 and iteration > opt.lpips_from_iter:
            # perceptual fine-tune stage: directly optimize the competition's
            # 0.4-weight LPIPS term (frozen VGG backbone, same family as scoring)
            global _lpips_loss_net
            if _lpips_loss_net is None:
                import lpips as _lpips_pkg
                _lpips_loss_net = _lpips_pkg.LPIPS(net="vgg", verbose=False).cuda()
                for p in _lpips_loss_net.parameters():
                    p.requires_grad_(False)
            # clamp for exact parity with the scored artifact (renders can exceed 1)
            loss = loss + opt.lambda_lpips * _lpips_loss_net(
                image_pm.clamp(0, 1).unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).squeeze()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # The multiview consistent densification of fastgs
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            # Tie the aggressive final-prune phase to densify_until_iter instead
            # of a hardcoded 15k: densification and final-pruning must NOT overlap
            # (they collide and can collapse the model to 0 gaussians). This lets
            # densify_until_iter be raised past 15k to add capacity/detail.
            # (parameterized end: last prune 3k before the end — identical to the
            # original `< 30_000` at the default 30k schedule, scales with --iterations)
            if iteration % 3000 == 0 and iteration > opt.densify_until_iter and iteration <= opt.iterations - 3000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score,
                                             score_thresh = opt.final_prune_thresh)
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if app_optim is not None:
                    app_optim.step()
                    app_optim.zero_grad(set_to_none=True)
                    # keep the affine zero-mean per channel: only relative
                    # (per-image) drift is absorbed, global exposure stays in
                    # the model — otherwise raw test renders inherit a bias
                    app_gain -= app_gain.mean(0)
                    app_bias -= app_bias.mean(0)
                    if iteration % 5000 == 0:
                        print(f"[APP] |gain| mean {app_gain.abs().mean():.4f} max {app_gain.abs().max():.4f} "
                              f"|bias| mean {app_bias.abs().mean():.4f} max {app_bias.abs().max():.4f}")

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
