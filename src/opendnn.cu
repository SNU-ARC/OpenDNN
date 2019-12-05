#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#include "opendnn.h"
#include "opendnn_kernel.cuh"

#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>

using namespace std;

// Context management
#ifdef __cuBLAS_ENGINE__
#include <cublas_v2.h>
cublasHandle_t cublas_handle;
#endif

CUstream global_stream;

struct opendnnContext {
    int driver_num_;
    int temp;
    #ifdef __cuBLAS_ENGINE__
    cublasHandle_t cublas_handle_;
    #endif
};

void opendnnCreate (opendnnHandle_t* handle) {
    *handle = new opendnnContext;

    #ifdef __cuBLAS_ENGINE__
    cublasCreate(&((*handle)->cublas_handle_));
    cublas_handle = (*handle)->cublas_handle_;
    #endif
}

void opendnnDestroy (opendnnHandle_t handle){
    // delete handle;
    #ifdef __cuBLAS_ENGINE__
    cublasDestroy(handle->cublas_handle_);
    #endif
}

void opendnnSetStream (CUstream stream) {
    global_stream = stream;
    #ifdef __cuBLAS_ENGINE__
    cublasSetStream(cublas_handle, global_stream);
    #endif
}


// Tensor management
void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t* tensor_desc) {
    *tensor_desc = new opendnnTensorStruct;
}

void opendnnSetTensor4dDescriptor (opendnnTensorDescriptor_t tens, int n, int c, int h, int w) {
    int w_str = 1;
    int h_str = w * w_str;
    int c_str = h * h_str;
    int n_str = c * c_str;
    opendnnSetTensor4dDescriptorEx (tens, n, c, h, w, n_str, c_str, h_str, w_str);
}

void opendnnSetTensor4dDescriptorEx (opendnnTensorDescriptor_t tens, int n, int c, int h, int w,
  int nStride, int cStride, int hStride, int wStride) {
    tens->number_ = n;
    tens->channel_ = c;
    tens->height_ = h;
    tens->width_ = w;
    tens->stride_n = nStride;
    tens->stride_c = cStride;
    tens->stride_h = hStride;
    tens->stride_w = wStride;
}

void opendnnGetTensor4dDescriptor (opendnnTensorDescriptor_t tens, int* n, int* c, int* h, int* w,
  int* nStride, int* cStride, int* hStride, int* wStride) {
    *n = tens->number_;
    *c = tens->channel_;
    *h = tens->height_;
    *w = tens->width_;
    *nStride = tens->stride_n;
    *cStride = tens->stride_c;
    *hStride = tens->stride_h;
    *wStride = tens->stride_w;
}

// Filter
void opendnnCreateFilterDescriptor (opendnnFilterDescriptor_t* filter) {
    *filter = new opendnnFilterStruct;
}

void opendnnSetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int out, int in, int h, int w) {
    filter->output_ = out;
    filter->input_ = in;
    filter->height_ = h;
    filter->width_ = w;
}

void opendnnGetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int* out, int* in, int* h, int* w) {
    *out = filter->output_;
    *in = filter->input_;
    *h = filter->height_;
    *w = filter->width_;
}

// Convolution
void opendnnCreateConvolutionDescriptor (opendnnConvolutionDescriptor_t* conv_desc) {
    *conv_desc = new opendnnConvolutionStruct;
    (*conv_desc)->group = 1;
}

void opendnnSetConvolution2dDescriptor (opendnnConvolutionDescriptor_t conv_desc, int ph, int pw, int sh, int sw, int ux, int uy) {
    conv_desc->pad_h = ph;
    conv_desc->pad_w = pw;
    conv_desc->vertical_stride = sw;
    conv_desc->horizon_stride = sh;
    conv_desc->upscale_x = ux;
    conv_desc->upscale_y = uy;
}

void opendnnGetConvolution2dDescriptor (opendnnConvolutionDescriptor_t conv_desc,
  int* ph, int* pw, int* sh, int* sw, int* ux, int* uy) {
    *ph = conv_desc->pad_h;
    *pw = conv_desc->pad_w;
    *sw = conv_desc->vertical_stride;
    *sh = conv_desc->horizon_stride;
    *ux = conv_desc->upscale_x;
    *uy = conv_desc->upscale_y;
}

void opendnnSetConvolutionGroupCount(opendnnConvolutionDescriptor_t conv_desc, int group) {
    conv_desc->group = group;
}

void opendnnGetConvolutionGroupCount(opendnnConvolutionDescriptor_t conv_desc, int* group) {
    *group = conv_desc->group;
}
void opendnnGetConvolutionForwardWorkspaceSize (opendnnHandle_t handle,
                                                opendnnTensorDescriptor_t bottom_desc,
                                                opendnnFilterDescriptor_t filter_desc,
                                                opendnnConvolutionDescriptor_t conv_desc,
                                                opendnnTensorDescriptor_t top_desc, size_t* size_in_bytes){
    int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    int fil_out, fil_in, fil_h, fil_w;
    opendnnGetTensor4dDescriptor (bottom_desc, &bot_n, &bot_c, &bot_h, &bot_w,
        &bot_nst, &bot_cst, &bot_hst, &bot_wst);
    opendnnGetTensor4dDescriptor (top_desc, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);
    opendnnGetFilter4dDescriptor (filter_desc, &fil_out, &fil_in, &fil_h, &fil_w);

    *size_in_bytes = bot_n*top_h*top_w*bot_c*fil_h*fil_w*sizeof(float);
}

// void opendnnGetConvolutionForwardWorkspaceSize_cu (cudnnHandle_t handle,
//                                                 cudnnTensorDescriptor_t bottom_desc,
//                                                 cudnnFilterDescriptor_t filter_desc,
//                                                 cudnnConvolutionDescriptor_t conv_desc,
//                                                 cudnnTensorDescriptor_t top_desc, size_t* size_in_bytes){
//     int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
//     int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
//     int fil_out, fil_in, fil_h, fil_w;
//     cudnnDataType_t dataType;
//     cudnnTensorFormat_t format;
//
//     cudnnGetTensor4dDescriptor (bottom_desc, &dataType, &bot_n, &bot_c, &bot_h, &bot_w,
//         &bot_nst, &bot_cst, &bot_hst, &bot_wst);
//     cudnnGetTensor4dDescriptor (top_desc, &dataType, &top_n, &top_c, &top_h, &top_w,
//         &top_nst, &top_cst, &top_hst, &top_wst);
//     cudnnGetFilter4dDescriptor (filter_desc, &dataType, &format, &fil_out, &fil_in, &fil_h, &fil_w);
//
//     *size_in_bytes = bot_n*top_h*top_w*bot_c*fil_h*fil_w*sizeof(float);
// }

void opendnnAddTensor (opendnnHandle_t handle,
  opendnnTensorDescriptor_t bias_desc, const float* bias_data,
  opendnnTensorDescriptor_t output_desc, float* output_data) {
    int bias_n, bias_c, bias_h, bias_w, bias_nst, bias_cst, bias_hst, bias_wst;
    int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;
    opendnnGetTensor4dDescriptor (bias_desc, &bias_n, &bias_c, &bias_h, &bias_w,
        &bias_nst, &bias_cst, &bias_hst, &bias_wst);
    opendnnGetTensor4dDescriptor (output_desc, &out_n, &out_c, &out_h, &out_w,
        &out_nst, &out_cst, &out_hst, &out_wst);

    for (int i = 0; i < out_n; ++i) {
      for (int j = 0; j < out_c; ++j) {
        for (int k = 0; k < out_h; ++k) {
          for (int l = 0; l < out_w; ++l) {
            output_data[l*out_wst + k*out_hst + j*out_cst + i*out_nst] +=
              bias_data[j*bias_cst];
              // TODO: AddTensor method is not appropriate for Caffe bias computation
              // Braodcasting operations are needed like NumbPy lib in Python
              // Currently this hard-codes repeatedly accumulating over channel dim

              // This is original implementation of cudnnAddTensor
              // bias_data[l*bias_wst + k*bias_hst + j*bias_cst + i*bias_nst];
          }
        }
      }
    }
}

void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  
  const int MAX_THREADS=1024;
  im2col_gpu_kernel<<<(num_kernels+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS, 0, global_stream>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
}

void opendnnInnerProductForward(opendnnHandle_t handle,
    opendnnTensorDescriptor_t input_desc, bool TransA, float* input,
    opendnnTensorDescriptor_t weight_desc, bool TransB, float* weight,
    opendnnTensorDescriptor_t output_desc, float* output){
  int in_n, in_c, in_h, in_w, in_nst, in_cst, in_hst, in_wst;
  int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;
  // int wei_n, wei_c, wei_h, wei_w;
  opendnnGetTensor4dDescriptor (input_desc, &in_n, &in_c, &in_h, &in_w,
      &in_nst, &in_cst, &in_hst, &in_wst);
  opendnnGetTensor4dDescriptor (output_desc, &out_n, &out_c, &out_h, &out_w,
      &out_nst, &out_cst, &out_hst, &out_wst);
  // Weight is transposed, do Input (A:M by K) x Weight^T (B^T: K by N)
  // instead of A x B
  int M = in_n;
  int K = in_c * in_h * in_w;
  int N = out_c;

  dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);

  if (TransA == false & TransB == false)
    matmul_block_lin_shared<<<grid,block,0,global_stream>>>
      (input,weight,output,M,K,K,N,M,N);
  else if (TransA == false & TransB == true)
    matmul_block_lin_shared_trans<<<grid,block,0,global_stream>>>
      (input,weight,output,M,K,N,K,M,N);
  else if (TransB == true & TransB == false){
    cout << "Error! InnerProduct: transB, not supported yet" << endl;
    exit(-1);
  } else if (TransB == true & TransB == true) {
    cout << "Error! InnerProduct: transA & B, not supported yet" << endl;
    exit(-1);
  }
}

// InnerProduct computation API
#ifdef __cuBLAS_ENGINE__
void caffe_gpu_gemm(cublasHandle_t handle, const bool TransA,
    const bool TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemm(handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}
#endif

void opendnnConvolutionForward (opendnnHandle_t handle,
  const opendnnTensorDescriptor_t bottom_desc, const float* bottom,
  const opendnnFilterDescriptor_t filter_desc, const float* filter,
  const opendnnConvolutionDescriptor_t conv_desc, float* workspace, size_t workSpaceSizeInBytes,
  const opendnnTensorDescriptor_t top_desc, float* top) {
    int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    int pad_h, pad_w, str_h, str_w, ups_x, ups_y;
    int fil_out, fil_in, fil_h, fil_w;
    int group;
    opendnnGetTensor4dDescriptor (bottom_desc, &bot_n, &bot_c, &bot_h, &bot_w,
        &bot_nst, &bot_cst, &bot_hst, &bot_wst);
    opendnnGetTensor4dDescriptor (top_desc, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);
    opendnnGetFilter4dDescriptor (filter_desc, &fil_out, &fil_in, &fil_h, &fil_w);
    opendnnGetConvolution2dDescriptor (conv_desc, &pad_h, &pad_w, &str_h, &str_w, &ups_x, &ups_y);
    opendnnGetConvolutionGroupCount(conv_desc, &group);

    float *col_buf = workspace;
    size_t col_nst = top_h*top_w*bot_c*fil_h*fil_w;

    for (int n = 0; n < bot_n; ++n) {
        const float* input = bottom + n*bot_nst;
        float* col_in_batch = col_buf + n*col_nst;
        cudaMemset(col_in_batch, 0, sizeof(float)*col_nst);

        im2col_gpu(input,
            bot_c, bot_w, bot_h, fil_h, fil_w,
            pad_h, pad_w, str_h, str_w,
            1, 1, col_in_batch);
    }

    int fil_out_ = fil_out / group;
    int fil_in_  = fil_in / group;
    int bot_c_   = bot_c / group;
    int top_c_   = top_c / group;

    // Forward through cuDNN in parallel over groups.
    float* output = top;
    for (int g = 0; g < group; g++) {
        const int M = top_c_;
        const int N = top_h*top_w;
        const int K = bot_c_*fil_h*fil_w;
        const int weight_offset = fil_out_*fil_in_*fil_h*fil_w;
        const int col_offset = bot_c_*fil_h*fil_w*top_h*top_w;
        const int output_offset = top_c_*top_h*top_w;

        const float *A = filter + weight_offset * g;
        const float *B = col_buf + col_offset * g;
        float *C = output + output_offset * g;

        const int N_BATCH = bot_n;
        dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE, N_BATCH);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        #ifdef __cuBLAS_ENGINE__
        float a = 1.0f, b=0.0f;
        // cuBLAS is column-major GEMM
        cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N,M,K,
          /*alpha=*/&a, B, /*lda1=*/N, /*lda2=*/col_nst,  A, /*ldb1=*/K, /*ldb2=*/0,
          /*beta=*/&b,  C, /*ldc1=*/N, /*ldc2=*/top_nst, /*batch_count=*/N_BATCH
        );
        #else
        // Row-major GEMM
        gemmStridedBatched<<<grid,block,0,global_stream>>>(M,N,K,
          A,M,0, B,K,col_nst, C,M,top_nst
        );
        #endif
    }
}

// void opendnnConvolutionForward_cu (cudnnHandle_t handle,
//   const cudnnTensorDescriptor_t bottom_desc, const float* bottom,
//   const cudnnFilterDescriptor_t filter_desc, const float* filter,
//   const cudnnConvolutionDescriptor_t conv_desc, float* workspace, size_t workSpaceSizeInBytes,
//   const cudnnTensorDescriptor_t top_desc, float* top, const string name) {
//     int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
//     int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
//     int pad_h, pad_w, str_h, str_w, ups_x, ups_y;
//     int fil_out, fil_in, fil_h, fil_w;
//     int group;
//
//     cudnnDataType_t dataType;
//     cudnnTensorFormat_t format;
//     cudnnConvolutionMode_t mode;
//
//     cudnnGetTensor4dDescriptor (bottom_desc, &dataType, &bot_n, &bot_c, &bot_h, &bot_w,
//         &bot_nst, &bot_cst, &bot_hst, &bot_wst);
//     cudnnGetTensor4dDescriptor (top_desc, &dataType, &top_n, &top_c, &top_h, &top_w,
//         &top_nst, &top_cst, &top_hst, &top_wst);
//     cudnnGetFilter4dDescriptor (filter_desc, &dataType, &format, &fil_out, &fil_in, &fil_h, &fil_w);
//     cudnnGetConvolution2dDescriptor (conv_desc, &pad_h, &pad_w, &str_h, &str_w, &ups_x, &ups_y, &mode, &dataType);
//     cudnnGetConvolutionGroupCount(conv_desc, &group);
//
//     float *col_buf = workspace;
//     size_t col_nst = top_h*top_w*bot_c*fil_h*fil_w;
//
//     for (int n = 0; n < bot_n; ++n) {
//         const float* input = bottom + n*bot_nst;
//         float* col_in_batch = col_buf + n*col_nst;
//         cudaMemset(col_in_batch, 0, sizeof(float)*col_nst);
//
//         im2col_gpu(input,
//             bot_c, bot_w, bot_h, fil_h, fil_w,
//             pad_h, pad_w, str_h, str_w,
//             1, 1, col_in_batch);
//     }
//
//     int fil_out_ = fil_out / group;
//     int fil_in_  = fil_in / group;
//     int bot_c_   = bot_c / group;
//     int top_c_   = top_c / group;
//
//     // Forward through cuDNN in parallel over groups.
//     float* output = top;
//     for (int g = 0; g < group; g++) {
//         const int M = top_c_;
//         const int N = top_h*top_w;
//         const int K = bot_c_*fil_h*fil_w;
//         // const float alpha = 1.0f;
//         // const float beta = 0.0f;
//         const int weight_offset = fil_out_*fil_in_*fil_h*fil_w;
//         const int col_offset = bot_c_*fil_h*fil_w*top_h*top_w;
//         const int output_offset = top_c_*top_h*top_w;
//
//         const float *A = filter + weight_offset * g;
//         const float *B = col_buf + col_offset * g;
//         float *C = output + output_offset * g;
//
//         const int N_BATCH = bot_n;
//         dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE, N_BATCH);
//         dim3 block(BLOCK_SIZE, BLOCK_SIZE);
//         #ifdef __cuBLAS_ENGINE__
//         float a = 1.0f, b=0.0f;
//         // cuBLAS is column-major GEMM
//         cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N,M,K,
//           /*alpha=*/&a, B, /*lda1=*/N, /*lda2=*/col_nst,  A, /*ldb1=*/K, /*ldb2=*/0,
//           /*beta=*/&b,  C, /*ldc1=*/N, /*ldc2=*/top_nst, /*batch_count=*/N_BATCH
//         );
//         #else
//         // Row-major GEMM
//         number_gemmStridedBatched<<<grid,block,0,global_stream>>>(M,N,K,
//           A,M,0, B,K,col_nst, C,M,top_nst,
//           Number::cfg[name+".weight"], Number::cfg[name+".input"]);
//         #endif
//     }
//
// }


