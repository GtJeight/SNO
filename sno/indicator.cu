#include "indicator.h"

#define SPATIAL_DIM 3
#define M_PI 3.14159265358979323846
#define THREADS_PER_BLOCK 1024


// Vector operation helper functions
__forceinline__ __device__ void add_vec_(double* dst, const double* src, int n_dims) {
    for (int d = 0; d < n_dims; d++) { dst[d] += src[d]; }
}

__forceinline__ __device__ void add_vec_(double* dst, const double* src, double scale, int n_dims) {
    for (int d = 0; d < n_dims; d++) { dst[d] += (src[d] * scale); }
}

__forceinline__ __device__ void subtract_vec(double* dst, const double* src1, const double* src2, int n_dims) {
    for (int d = 0; d < n_dims; d++) { dst[d] = src1[d] - src2[d]; }
}

__forceinline__ __device__ void assign_vec(double* dst, const double* src, int n_dims) {
    for (int d = 0; d < n_dims; d++) { dst[d] = src[d]; }
}

__forceinline__ __device__ void assign_vec(double* dst, const double* src, double scale, int n_dims) {
    for (int d = 0; d < n_dims; d++) { dst[d] = (src[d] * scale); }
}

__forceinline__ __device__ double inner_prod(const double* vec1, const double* vec2, int n_dims) {
    double result = 0;
    for (int d = 0; d < n_dims; d++) { result += vec1[d] * vec2[d]; }
    return result;
}

__forceinline__ __device__ double S(const double r) {
    double r_ = r / 0.75;
    return erf(r_) - 2 / sqrt(M_PI) * r_ * exp(-r_ * r_);
}

__forceinline__ __device__ double eval_A_mu(const double* diff, const double* n, double width)
{
	double dist = sqrt(inner_prod(diff, diff, SPATIAL_DIM));
    if (dist < 1e-12) return 0.0;
    double res = inner_prod(diff, n, SPATIAL_DIM);
    return res / (dist * dist * dist) * S(dist / width) * pow(width, 3);
}

__forceinline__ __device__ void eval_AT_s_add(double* out, const double* diff, const double* s, double density , double width)
{
    double dist = sqrt(inner_prod(diff, diff, SPATIAL_DIM));
	if (dist < 1e-12) return;
    for (int k = 0; k < SPATIAL_DIM; ++k)
    {
        out[k] += (*s) * diff[k] / (4 * M_PI * density * dist * dist * dist) * S(dist / width);
    }
}

void multiply_by_A_cuda_kernel_launcher(
    const double* d_query_points, 
    const double* d_points, 
    const double* d_normals, 
    const double* d_density, 
    const double* d_node_pos, 
    const double* d_node_width, 
    const int* d_ngbr_list, 
    const int* d_ngbr_list_startid, 
    const int* d_ngbr_size_list, 
    int num_points, 
    double* d_out_attr
)
{
    int num_blocks = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    multiply_by_A_cuda_kernel << <num_blocks, THREADS_PER_BLOCK >> > (
        d_query_points,
        d_points,
        d_normals,
        d_density,
        d_node_pos,
        d_node_width,
        d_ngbr_list,
        d_ngbr_list_startid,
        d_ngbr_size_list,
        num_points,
        d_out_attr
        );
}

void multiply_by_AT_cuda_kernel_launcher(
    const double* d_query_points, 
    const double* d_points, 
    const double* d_query_attrs, 
    const double* d_density, 
    const double* d_node_pos, 
    const double* d_node_width, 
    const int* d_ngbr_list, 
    const int* d_ngbr_list_startid, 
    const int* d_ngbr_size_list, 
    int num_querys, 
    double* d_out_attr
)
{
    int num_blocks = (num_querys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    multiply_by_AT_cuda_kernel << <num_blocks, THREADS_PER_BLOCK >> > (
        d_query_points,
        d_points,
        d_query_attrs,
        d_density,
        d_node_pos,
        d_node_width,
        d_ngbr_list,
        d_ngbr_list_startid,
        d_ngbr_size_list,
        num_querys,
        d_out_attr
        );
}

__global__ void multiply_by_A_cuda_kernel(
    const double* query_points, 
    const double* points,
    const double* normals,
    const double* density,
    const double* node_pos, 
    const double* node_width, 
    const int* ngbr_list, 
    const int* ngbr_list_startid, 
    const int* ngbr_size_list,
    int num_points,
    double* out_attr
)
{
    int query_index = blockDim.x * blockIdx.x + threadIdx.x;

    double out_val = 0.;
    for (int point_idx = 0; point_idx < num_points; ++point_idx)
    {
        for (int j = 0; j < ngbr_size_list[point_idx]; ++j)
        {
			int node_idx = ngbr_list[ngbr_list_startid[point_idx] + j];
            double diff[3];
            for (int k = 0; k < 3; k++)
            {
                diff[k] = node_pos[SPATIAL_DIM * node_idx + k] - query_points[SPATIAL_DIM * point_idx + k];
            }

            out_val += eval_A_mu(diff, normals + SPATIAL_DIM * point_idx, node_width[node_idx]);
        }

        out_val /= (4 * M_PI * density[point_idx]);
    }

    out_attr[query_index] = out_val;
}

__global__ void multiply_by_AT_cuda_kernel(
    const double* query_points,
    const double* points,
    const double* query_attrs,
    const double* density,
    const double* node_pos,
    const double* node_width,
    const int* ngbr_list,
    const int* ngbr_list_startid,
    const int* ngbr_size_list,
    int num_querys,
    double* out_attr
)
{
    int point_index = blockDim.x * blockIdx.x + threadIdx.x;

    double out_vec[SPATIAL_DIM] = { 0.,0.,0. };
    for (int query_idx = 0; query_idx < num_querys; ++query_idx)
    {

        for (int j = 0; j < ngbr_size_list[point_index]; ++j)
        {
            int node_idx = ngbr_list[ngbr_list_startid[point_index] + j];
            double diff[3];
            for (int k = 0; k < 3; k++)
            {
                diff[k] = node_pos[SPATIAL_DIM * node_idx + k] - query_points[SPATIAL_DIM * point_index + k];
            }

            eval_AT_s_add(out_vec, diff, query_attrs + query_idx, density[point_index], node_width[node_idx]);
        }
    }

    assign_vec(out_attr + point_index * SPATIAL_DIM, out_vec, SPATIAL_DIM);
}

