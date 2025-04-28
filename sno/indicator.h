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
);

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
);
