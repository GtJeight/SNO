#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// Vector operation helper functions
__forceinline__ __device__ void add_vec_(double* dst, const double* src, int n_dims);
__forceinline__ __device__ void add_vec_(double* dst, const double* src, double scale, int n_dims);
__forceinline__ __device__ void subtract_vec(double* dst, const double* src1, const double* src2, int n_dims);
__forceinline__ __device__ void assign_vec(double* dst, const double* src, int n_dims);
__forceinline__ __device__ void assign_vec(double* dst, const double* src, double scale, int n_dims);
__forceinline__ __device__ double inner_prod(const double* vec1, const double* vec2, int n_dims);
__forceinline__ __device__ double S(const double r);
__forceinline__ __device__ double eval_A_mu(const double* diff, const double* n, double width);
__forceinline__ __device__ void eval_AT_s_add(double* out, const double* diff, const double* s, double density, double width);

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
);

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
);

__global__ void multiply_by_A_cuda_kernel(
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
);

__global__ void multiply_by_AT_cuda_kernel(
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
);
