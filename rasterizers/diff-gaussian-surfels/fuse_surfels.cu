#include <torch/extension.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <tuple>
#include <string>
#include "cuda_rasterizer/auxiliary.h"

namespace cg = cooperative_groups;

__device__ glm::mat3 quaternion2rotmat_(const glm::vec4 q) {
	// Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	return R;
}


__device__ __forceinline__ glm::vec4 rotmat2quaternion(const glm::mat3& R) {

    float trace = R[0][0] + R[1][1] + R[2][2];
    float w, x, y, z;

    if (trace > 0.0f) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        w = 0.25f / s;
        x = (R[2][1] - R[1][2]) * s;
        y = (R[0][2] - R[2][0]) * s;
        z = (R[1][0] - R[0][1]) * s;
    } else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
        float s = 2.0f * sqrtf(1.0f + R[0][0] - R[1][1] - R[2][2]);
        w = (R[2][1] - R[1][2]) / s;
        x = 0.25f * s;
        y = (R[0][1] + R[1][0]) / s;
        z = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        float s = 2.0f * sqrtf(1.0f + R[1][1] - R[0][0] - R[2][2]);
        w = (R[0][2] - R[2][0]) / s;
        x = (R[0][1] + R[1][0]) / s;
        y = 0.25f * s;
        z = (R[1][2] + R[2][1]) / s;
    } else {
        float s = 2.0f * sqrtf(1.0f + R[2][2] - R[0][0] - R[1][1]);
        w = (R[1][0] - R[0][1]) / s;
        x = (R[0][2] + R[2][0]) / s;
        y = (R[1][2] + R[2][1]) / s;
        z = 0.25f * s;
    }

    return glm::vec4(w, x, y, z);
}

__device__ __forceinline__ glm::mat3 rotvec2rotmatrix(const float3& v, const float angle) {

    float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    float ux = v.x / norm;
    float uy = v.y / norm;
    float uz = v.z / norm;

    float cosTheta = cosf(angle);
    float sinTheta = sinf(angle);

    glm::mat3 R;
    R[0][0] = cosTheta + ux * ux * (1 - cosTheta);
    R[0][1] = ux * uy * (1 - cosTheta) - uz * sinTheta;
    R[0][2] = ux * uz * (1 - cosTheta) + uy * sinTheta;

    R[1][0] = uy * ux * (1 - cosTheta) + uz * sinTheta;
    R[1][1] = cosTheta + uy * uy * (1 - cosTheta);
    R[1][2] = uy * uz * (1 - cosTheta) - ux * sinTheta;

    R[2][0] = uz * ux * (1 - cosTheta) - uy * sinTheta;
    R[2][1] = uz * uy * (1 - cosTheta) + ux * sinTheta;
    R[2][2] = cosTheta + uz * uz * (1 - cosTheta);

    return R;
}



__device__ __forceinline__ float norm(const float3& v){
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float3 normalize(const float3& v){
    float inv_norm = 1.0f / norm(v);
    return make_float3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
}

__device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ float3 cross(const float3& v1, const float3& v2) {
    return make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
}

__device__ __forceinline__ float clamp(const float a, const float min, float max){
    if(a > max) return max;
    else if(a < min) return min;
    else return a;
}

__device__ __host__ __forceinline__ float3 operator+(const float3& v1, const float3& v2){
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __host__ __forceinline__ float3 operator-(const float3& v1, const float3& v2){
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ __host__ __forceinline__ float3 operator*(const float3& v1, float s){
    return make_float3(v1.x * s, v1.y * s, v1.z * s);
}

__device__ __host__ __forceinline__ float3 operator/(const float3& v1, float s){
    return make_float3(v1.x / s, v1.y / s, v1.z / s);
}

__device__ __forceinline__ float3 nearest_interpolate(
    const float* __restrict__ map, 
    const int& wd, 
    const int& ht, 
    const float& x, 
    const float& y) {

    int ix = min(max(__float2int_rn(x), 0), wd - 1);
    int iy = min(max(__float2int_rn(y), 0), ht - 1);

    int idx = iy * wd + ix;
    
    return make_float3(map[idx * 3], map[idx * 3 + 1], map[idx * 3 + 2]);
}

__device__ __forceinline__ float nearest_interpolate_(
    const float* __restrict__ map, 
    const int& wd, 
    const int& ht, 
    const float& x, 
    const float& y) {

    int ix = min(max(__float2int_rn(x), 0), wd - 1);
    int iy = min(max(__float2int_rn(y), 0), ht - 1);

    int idx = iy * wd + ix;
    
    return map[idx];
}

__device__ __forceinline__ bool is_valid(const float2& coord, const float* map, const int& wd, const int& ht, const int& ch) {
    int x = __float2int_rd(coord.x);
    int y = __float2int_rd(coord.y);

    if (x < 0 || x + 1 >= wd || y < 0 || y + 1 >= ht) return false;

    bool valid = true;
    for (int c = 0; c < ch; c++) {
        if (map[(y * wd + x) * ch + c] == 0) {
            valid = false; break;
        }
        if (map[(y * wd + x + 1) * ch + c] == 0) {
            valid = false; break;
        }
        if (map[((y + 1) * wd + x) * ch + c] == 0) {
            valid = false; break;
        }
        if (map[((y + 1) * wd + x + 1) * ch + c] == 0) {
            valid = false; break;
        }
    }
    return valid;
}

__device__ __forceinline__ bool is_valid(const float2& coord, const bool* map, const int& wd, const int& ht, const int& ch) {
    int x = __float2int_rd(coord.x);
    int y = __float2int_rd(coord.y);

    if (x < 0 || x + 1 >= wd || y < 0 || y + 1 >= ht) return false;

    bool valid = true;
    for (int c = 0; c < ch; c++) {
        if (!map[(y * wd + x) * ch + c]) {
            valid = false; break;
        }
        if (!map[(y * wd + x + 1) * ch + c]) {
            valid = false; break;
        }
        if (!map[((y + 1) * wd + x) * ch + c]) {
            valid = false; break;
        }
        if (!map[((y + 1) * wd + x + 1) * ch + c]) {
            valid = false; break;
        }
    }
    return valid;
}

__global__ void preprocessSurfel(
    const int P,
    const int ht,
    const int wd,
    const float* intrinsic,
    const float* viewmatrix,
    const float* projmatrix,
    const float* frame_vmap,
    const float* frame_nmap,
    const float* frame_cmap,
    const float* frame_dmap,
    const bool* frame_mask,
    const int* frame_imap,
    const float* frame_depth,
    const float* model_vmap,
    const float* model_nmap,
    const bool* model_mask,
    float* points,
    float* rotations,
    float* scales,
    float* colors,
    float* confidence,
    int* tic,
    float* eta,
    float* sigma2,
    int* observe_count,
    int* error_count,
    bool* stable_mask,
    bool* inview_mask,
    bool* surface_mask,
    const float fusion_dist_thres,
    const float alpha_p,
    const float alpha_n){

    int idx = cg::this_grid().thread_rank();
    if(idx >= P) return;
    
    float3 p_w = { points[3 * idx], points[3 * idx + 1], points[3 * idx + 2] };
    float3 p_c = transformPoint4x3(p_w, viewmatrix);

    float4 p_hom = transformPoint4x4(p_w, projmatrix);
    float p_proj_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_proj_w, p_hom.y * p_proj_w, p_hom.z * p_proj_w };

    const float fx = intrinsic[0];
    const float fy = intrinsic[1];
    const float cx = intrinsic[2];
    const float cy = intrinsic[3];

    inview_mask[idx] = false;
    surface_mask[idx] = false;

    float2 coord = {ndc2Pix(p_proj.x, wd, cx), ndc2Pix(p_proj.y, ht, cy)};
    if (!in_frustum(p_c.z, coord, ht, wd)) return;

    glm::vec4 rot = {rotations[idx * 4 + 0], rotations[idx * 4 + 1], rotations[idx * 4 + 2], rotations[idx * 4 + 3]};
    glm::mat3 R = quaternion2rotmat_(rot);
    float3 x_axis = transformVec4x3({R[0][0], R[1][0], R[2][0]}, viewmatrix);
    float3 y_axis = transformVec4x3({R[0][1], R[1][1], R[2][1]}, viewmatrix);
    float3 z_axis = transformVec4x3({R[0][2], R[1][2], R[2][2]}, viewmatrix);

    float viewCos[3];
    if (!front_facing(z_axis, p_c, &viewCos[idx], false)) return;

    inview_mask[idx] = true;

    if (!is_valid(coord, frame_nmap, wd, ht, 3)) return;
    if (!is_valid(coord, frame_mask, wd, ht, 1)) return;

    float3 normal_w = normalize(make_float3(R[0][2], R[1][2], R[2][2]));
    float3 normal_c = normalize(nearest_interpolate(frame_nmap, wd, ht, coord.x, coord.y));
    float3 vertex_c = nearest_interpolate(frame_vmap, wd, ht, coord.x, coord.y);

    float3 camera_center = make_float3(viewmatrix[12], viewmatrix[13], viewmatrix[14]);
    float3 vertex_dist = p_w - vertex_c;

    int surfel_id = frame_imap[__float2int_rn(coord.y) * wd + __float2int_rn(coord.x)];

    if (surfel_id >= 0 && surfel_id == idx){
        surface_mask[idx] = true;
    }    
    
    float cos_angle = dot(vertex_dist, z_axis);
    if(norm(vertex_dist) > fusion_dist_thres){
        return;
    }

    // // Fusion
    // float3 eta_p_last = {eta[idx * 6 + 0], eta[idx * 6 + 1], eta[idx * 6 + 2]};
    // float3 eta_n_last = {eta[idx * 6 + 3], eta[idx * 6 + 4], eta[idx * 6 + 5]};

    float3 eta_p_last = {
        points[idx * 3 + 0] / sigma2[idx * 2 + 0],
        points[idx * 3 + 1] / sigma2[idx * 2 + 0],
        points[idx * 3 + 2] / sigma2[idx * 2 + 0]
    };

    float3 eta_n_last = {
        normal_w.x / sigma2[idx * 2 + 1],
        normal_w.y / sigma2[idx * 2 + 1],
        normal_w.z / sigma2[idx * 2 + 1]
    };

    float curr_depth = nearest_interpolate_(frame_dmap, wd, ht, coord.x, coord.y);
    float sigma2_p_z = (alpha_p * curr_depth) * (alpha_p * curr_depth);
    float sigma2_n_z = (alpha_n * curr_depth) * (alpha_n * curr_depth);

    float sigma2_p_last = sigma2[idx * 2 + 0];
    float sigma2_n_last = sigma2[idx * 2 + 1];

    float Lambda_p_curr = 1.f / sigma2_p_z + 1.f / sigma2_p_last;
    float Lambda_n_curr = 1.f / sigma2_n_z + 1.f / sigma2_n_last;

    float sigma2_p_curr = 1.f / Lambda_p_curr;
    float sigma2_n_curr = 1.f / Lambda_n_curr;

    float3 x_p_obs = vertex_c;
    float3 x_n_obs = normal_c;
    
    float Lambda_p_obs = 1.f / sigma2_p_z;
    float Lambda_n_obs = 1.f / sigma2_n_z;

    float3 eta_p_curr = {
        eta_p_last.x + Lambda_p_obs * x_p_obs.x,
        eta_p_last.y + Lambda_p_obs * x_p_obs.y,
        eta_p_last.z + Lambda_p_obs * x_p_obs.z
    };
    float3 eta_n_curr = {
        eta_n_last.x + Lambda_n_obs * x_n_obs.x,
        eta_n_last.y + Lambda_n_obs * x_n_obs.y,
        eta_n_last.z + Lambda_n_obs * x_n_obs.z
    };

    float3 x_p_curr = {
        sigma2_p_curr * eta_p_curr.x,
        sigma2_p_curr * eta_p_curr.y,
        sigma2_p_curr * eta_p_curr.z
    };
    float3 x_n_curr = {
        sigma2_n_curr * eta_n_curr.x,
        sigma2_n_curr * eta_n_curr.y,
        sigma2_n_curr * eta_n_curr.z
    };

    points[idx * 3 + 0] = x_p_curr.x;
    points[idx * 3 + 1] = x_p_curr.y;
    points[idx * 3 + 2] = x_p_curr.z;

    sigma2[idx * 2 + 0] = sigma2_p_curr;

    // eta[idx * 6 + 0] = eta_p_curr.x;
    // eta[idx * 6 + 1] = eta_p_curr.y;
    // eta[idx * 6 + 2] = eta_p_curr.z;

    // normal fusion
    float angle = acosf(clamp(dot(normal_w, normal_c), -1, 1)) * 180 / 3.1415926;
    if(angle < 60){
        float3 normal_new = normalize(x_n_curr);
        float3 n12 = normalize(cross(normal_w, normal_new));
        float cos_theta = clamp(dot(normal_w, normal_new), -1, 1);

        float theta = acosf(cos_theta) * 180 / 3.1415926;
        if(theta < 1) return;

        glm::mat3 R1 = rotvec2rotmatrix(n12, acosf(cos_theta));
        glm::mat3 R2 = R * R1; // ColMajor!!!
        glm::vec4 rot_new = rotmat2quaternion(R2);

        rotations[idx * 4 + 0] = rot_new.x;
        rotations[idx * 4 + 1] = rot_new.y;
        rotations[idx * 4 + 2] = rot_new.z;
        rotations[idx * 4 + 3] = rot_new.w;

        sigma2[idx * 2 + 1] = sigma2_n_curr;

        // eta[idx * 6 + 3] = eta_n_curr.x;
        // eta[idx * 6 + 4] = eta_n_curr.y;
        // eta[idx * 6 + 5] = eta_n_curr.z;

    }
}

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
){

    const int P = points.size(0);

    preprocessSurfel<<<(P + 255) / 256, 256>>>(
        P,
        frame_nmap.size(0),
        frame_nmap.size(1),
        intrinsic.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        frame_vmap.contiguous().data_ptr<float>(),
        frame_nmap.contiguous().data_ptr<float>(),
        frame_cmap.contiguous().data_ptr<float>(),
        frame_dmap.contiguous().data_ptr<float>(),
        frame_mask.contiguous().data_ptr<bool>(),
        frame_imap.contiguous().data_ptr<int>(),
        depth_buff.contiguous().data_ptr<float>(),
        model_vmap.contiguous().data_ptr<float>(),
        model_nmap.contiguous().data_ptr<float>(),
        model_mask.contiguous().data_ptr<bool>(),
        points.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        confidence.contiguous().data_ptr<float>(),
        tic.contiguous().data_ptr<int>(),
        eta.contiguous().data_ptr<float>(),
        sigma2.contiguous().data_ptr<float>(),
        observe_count.contiguous().data_ptr<int>(),
        error_count.contiguous().data_ptr<int>(),
        stable_mask.contiguous().data_ptr<bool>(),
        inview_mask.contiguous().data_ptr<bool>(),
        surface_mask.contiguous().data_ptr<bool>(),
        fusion_dist_thres,
        alpha_p,
        alpha_n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << " CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

}


__global__ void projectSurfelsToFrame(
    const int P,
    const int image_height,
    const int image_width,
    const float* points,
    const float* rotations,
    const bool* stable_mask,
    const float* intrinsic,
    const float* viewmatrix,
    const float* projmatrix,
    const float& factor,
    int* index_map,
    float* min_depth_buffer
){
    int idx = cg::this_grid().thread_rank();
    if(idx >= P) return;

    if(!stable_mask[idx]) return;

    float3 p_w = { points[idx * 3 + 0], points[idx * 3 + 1], points[idx * 3 + 2] };
    float3 p_c = transformPoint4x3(p_w, viewmatrix);
    float4 p_hom = transformPoint4x4(p_w, projmatrix);
    float p_proj_w = 1.0f / (p_hom.w + 1e-7f);
    float3 p_proj = { p_hom.x * p_proj_w, p_hom.y * p_proj_w, p_hom.z * p_proj_w };

    const float fx = intrinsic[0];
    const float fy = intrinsic[1];
    const float cx = intrinsic[2];
    const float cy = intrinsic[3];
    float2 coord = {ndc2Pix(p_proj.x, image_width, cx), ndc2Pix(p_proj.y, image_height, cy)};
    if (!in_frustum(p_c.z, coord, image_height, image_width)) return;

    glm::vec4 rot = {rotations[idx * 4 + 0], rotations[idx * 4 + 1], rotations[idx * 4 + 2], rotations[idx * 4 + 3]};
    glm::mat3 R = quaternion2rotmat_(rot);
    float3 x_axis = transformVec4x3({R[0][0], R[1][0], R[2][0]}, viewmatrix);
    float3 y_axis = transformVec4x3({R[0][1], R[1][1], R[2][1]}, viewmatrix);
    float3 z_axis = transformVec4x3({R[0][2], R[1][2], R[2][2]}, viewmatrix);

    float viewCos[3];
    if (!front_facing(z_axis, p_c, &viewCos[idx], false)) return;

    int x = (int)coord.x;
    int y = (int)coord.y;
    if (x < 0 || x >= image_width || y < 0 || y >= image_height) return;

    float depth = p_c.z;
    const int kernel_radius = 1;
    for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
        for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= image_width || ny < 0 || ny >= image_height) continue;
            int n_pixel_idx = ny * image_width + nx;
            unsigned int depth_int = __float_as_uint(depth);
            unsigned int old = atomicMin((unsigned int*)&min_depth_buffer[n_pixel_idx], depth_int);
            float old_depth = __uint_as_float(old);
            if (depth < old_depth) {
                index_map[n_pixel_idx] = idx;
            }
        }
    }
}


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
){
    int P = points.size(0);

    projectSurfelsToFrame<<<(P + 255) / 256, 256>>>(
        P,
        index_map.size(0),
        index_map.size(1),
        points.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        stable_mask.contiguous().data_ptr<bool>(),
        intrinsic.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        factor,
        index_map.contiguous().data_ptr<int>(),
        depth_buffer.contiguous().data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << " CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}