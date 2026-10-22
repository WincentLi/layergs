/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& render_mask,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const float cx,
	const float cy,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &tile_indices,
	const int tile_num,
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_opac,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

void PreprocessSurfelsFunctionCUDA(
    const torch::Tensor& points,
    const torch::Tensor& rotations,
	const torch::Tensor& scales,
	const torch::Tensor& colors,
    const torch::Tensor& confidence,
	const torch::Tensor& tic,
	const torch::Tensor& eta,
	const torch::Tensor& sigma2,
	const torch::Tensor& observe_count,
	const torch::Tensor& error_count,
	const torch::Tensor& stable_mask,
	const torch::Tensor& intrinsic,
    const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& frame_vmap,
    const torch::Tensor& frame_nmap,
	const torch::Tensor& frame_cmap,
    const torch::Tensor& frame_dmap,
    const torch::Tensor& frame_mask,
	const torch::Tensor& frame_imap,
	const torch::Tensor& depth_buff,
	const torch::Tensor& model_vmap,
	const torch::Tensor& model_nmap,
	const torch::Tensor& model_mask,
	const torch::Tensor& inview_mask,
	const torch::Tensor& surface_mask,
	const float fusion_dist_thres,
	const float alpha_p,
	const float alpha_n
	);

void ProjectSurfelsToFrameCUDA(
    const torch::Tensor& points,
    const torch::Tensor& rotations,
	const torch::Tensor& stable_mask,
    const torch::Tensor& intrinsic,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float& factor,
    const torch::Tensor& index_map,
    const torch::Tensor& depth_buffer
	);