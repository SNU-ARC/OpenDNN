#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

#include <opendnn.h>
#include <CL/cl.hpp>

#include <sys/time.h>
#include <unistd.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6*tv.tv_usec;
}

#include <stdlib.h>

const int BLOCK_SIZE = 24;
const int LIN_BLOCK_SIZE = 128;

using namespace std;

struct opendnnContext {
    int driver_num;
    vector<cl::Platform> all_platforms;
    vector<cl::Device> all_devices;
    cl::Context context;
    cl::CommandQueue* cmdq;
    // TODO: create program in handler?
    cl::Program* program;
};

// Context create / destroy
void opendnnCreate (opendnnHandle_t* handle) {
    *handle = new opendnnContext;
    cl::Platform::get(&((*handle)->all_platforms));
    if ((*handle)->all_platforms.size() == 0) {
        cerr << "No Platforms Found!!" << endl;
        exit(1);
    }
    cl::Platform default_platform = ((*handle)->all_platforms)[0];
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &((*handle)->all_devices));
    if ((*handle)->all_devices.size() == 0) {
        cerr << "No Devices Found!!" << endl;
        exit(1);
    }
    cl::Device device = ((*handle)->all_devices)[0];
    cl::Context context(device);
    (*handle)->context = context;

    (*handle)->cmdq = new cl::CommandQueue(context, device);
    // TODO: create program in handler?
    // (*handle)->program = new cl::Program(
    //     context, (*handle)->devices, xcl::import_binary_file("kernel.xclbin"));
    // Read source file
    std::ifstream sourceFile("/usr/local/cuda/lib64/opendnn_kernel.cl");
    std::string sourceCode(std::istreambuf_iterator<char>{sourceFile}, {});
    cl::Program::Sources sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size()+1));

    // Program build
    (*handle)->program = new cl::Program(context, sources);
    if ((*handle)->program->build({device}) != CL_SUCCESS) {
        if ((*handle)->program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_ERROR)
        std::cerr << "Build log for " << device.getInfo<CL_DEVICE_NAME>() << ":" << std::endl
            << (*handle)->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
}

void opendnnDestroy (opendnnHandle_t handle){
    delete handle->program;
    delete handle->cmdq;
    delete handle;
}

// Tensor management
void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t* tensor_desc) {
    *tensor_desc = new opendnnTensorStruct;
}

void opendnnSetTensor2dDescriptor (opendnnTensorDescriptor_t tens, int h, int w) {
    int w_str = 1;
    int h_str = w * w_str;
    opendnnSetTensor4dDescriptorEx (tens, 1, 1, h, w, -1, -1, h_str, w_str);
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
    tens->count = n*c*h*w;
}

void opendnnGetTensor2dDescriptor (opendnnTensorDescriptor_t tens, int* h, int* w, int* hStride, int* wStride) {
    *h = tens->height_;
    *w = tens->width_;
    *hStride = tens->stride_h;
    *wStride = tens->stride_w;
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
    filter->count = out*in*h*w;
}

void opendnnGetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int* out, int* in, int* h, int* w) {
    *out = filter->output_;
    *in = filter->input_;
    *h = filter->height_;
    *w = filter->width_;
}


// Convolution methods
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

// Pooling methods
void opendnnCreatePoolingDescriptor (opendnnPoolingDescriptor_t* pool) {
    *pool = new (struct opendnnPoolingStruct);
}

void opendnnSetPooling2dDescriptor (opendnnPoolingDescriptor_t pool, opendnnPoolingMode_t mode,
  int wh, int ww, int vp, int hp, int vs, int hs) {
    pool->pooling_mode = mode;
    pool->w_height = wh;
    pool->w_width = ww;
    pool->vertical_padding = vp;
    pool->horizon_padding = hp;
    pool->vertical_stride = vs;
    pool->horizon_stride = hs;
}

void opendnnGetPooling2dDescriptor (opendnnPoolingDescriptor_t pool, opendnnPoolingMode_t* mode,
  int* wh, int* ww, int* vp, int* hp, int* vs, int* hs) {
    *mode = pool->pooling_mode;
    *wh = pool->w_height;
    *ww = pool->w_width;
    *vp = pool->vertical_padding;
    *hp = pool->horizon_padding;
    *vs = pool->vertical_stride;
    *hs = pool->horizon_stride;
}

// Normalization methods
void opendnnCreateNormDescriptor (opendnnNormDescriptor_t* nm) {
    *nm = new (struct opendnnNormStruct);
}

// Activation methods
void opendnnCreateActivationDescriptor (opendnnActivationDescriptor_t* act){
    *act = new (struct opendnnActivationStruct);
}
void opendnnSetActivationDescriptor (opendnnActivationDescriptor_t act,
  opendnnActivationMode_t mode){
  act->activation_mode = mode;
  // act->relu_ceiling = relu_ceiling;
}
void opendnnGetActivationDescriptor (opendnnActivationDescriptor_t act,
  opendnnActivationMode_t* mode){
  *mode = act->activation_mode;
  // *relu_ceiling = act->relu_ceiling;
}

void opendnnSetNormDescriptor (opendnnNormDescriptor_t nm, int N, double a, double b, double K, opendnnNormMode_t mode) {
    nm->normN = N;
    nm->normAlpha = a;
    nm->normBeta = b;
    nm->normK = K;
    nm->normMode = mode;
}

void opendnnGetNormDescriptor (opendnnNormDescriptor_t nm, int *N, double *a, double *b, double *K, opendnnNormMode_t* mode) {
    *N = nm->normN;
    *a = nm->normAlpha;
    *b = nm->normBeta;
    *K = nm->normK;
    *mode = nm->normMode;
}

// Actual computation
void opendnnActivationForward (opendnnHandle_t handle, opendnnActivationDescriptor_t act,
  opendnnTensorDescriptor_t input_desc, float* input,
  opendnnTensorDescriptor_t output_desc, float* output) {
    int in_n, in_c, in_h, in_w, in_nst, in_cst, in_hst, in_wst;
    int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;
    opendnnActivationMode_t mode;
    opendnnGetTensor4dDescriptor(input_desc, &in_n, &in_c, &in_h, &in_w,
        &in_nst, &in_cst, &in_hst, &in_wst);
    opendnnGetTensor4dDescriptor(output_desc, &out_n, &out_c, &out_h, &out_w,
        &out_nst, &out_cst, &out_hst, &out_wst);
    opendnnGetActivationDescriptor(act, &mode);

    // Retrieve OpenCL context
    cl::Context context = handle->context;
    cl::CommandQueue* q = handle->cmdq;
    cl::Device default_device = handle->all_devices[0];
    cl::Program* program = handle->program;

    string acts;

    switch (mode) {
        case 0: acts = "sigmoid";
        case 1: acts = "ReLU";
        case 2: acts = "htan";
    }

    // Extract a kernel function out of the program (.cl)
    cl::Kernel ReLU = cl::Kernel (*program, acts.c_str());
    const int N = in_n*in_c*in_h*in_w;
    size_t local = LIN_BLOCK_SIZE;
    size_t global = (N + LIN_BLOCK_SIZE-1)/LIN_BLOCK_SIZE * LIN_BLOCK_SIZE;

    cl::Buffer input_dev(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*N, input);
    cl::Buffer output_dev(context, CL_MEM_WRITE_ONLY, sizeof(float)*N);

    int narg = 0;
    ReLU.setArg(narg++, input_dev);
    ReLU.setArg(narg++, output_dev);
    ReLU.setArg(narg++, N);
    q->enqueueNDRangeKernel(
        ReLU,
        cl::NullRange,
        cl::NDRange(global),
        cl::NDRange(local)
    );
    q->enqueueReadBuffer(output_dev, CL_TRUE, 0, sizeof(float)*N, output);
    q->finish();
}

void opendnnPoolingForward (opendnnHandle_t handle, opendnnPoolingDescriptor_t pool,
  opendnnTensorDescriptor_t input_desc, float* input, opendnnTensorDescriptor_t output_desc, float* output) {
    opendnnPoolingMode_t mode;
    int in_n, in_c, in_h, in_w, in_nst, in_cst, in_hst, in_wst;
    int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;
    int kernel_h, kernel_w, v_pad, h_pad, v_str, h_str;
    opendnnGetPooling2dDescriptor(pool, &mode, &kernel_h, &kernel_w, &v_pad, &h_pad, &v_str, &h_str);
    opendnnGetTensor4dDescriptor(input_desc, &in_n, &in_c, &in_h, &in_w,
        &in_nst, &in_cst, &in_hst, &in_wst);
    opendnnGetTensor4dDescriptor(output_desc, &out_n, &out_c, &out_h, &out_w,
        &out_nst, &out_cst, &out_hst, &out_wst);

    // Retrieve OpenCL context
    cl::Context context = handle->context;
    cl::CommandQueue* q = handle->cmdq;
    cl::Device default_device = handle->all_devices[0];
    cl::Program* program = handle->program;

    int in = in_n * in_c * in_h * in_w;
    int out = out_n * out_c * out_h * out_w;
    cl::Buffer input_buf(context, CL_MEM_READ_ONLY, sizeof(float)*in);
    cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, sizeof(float)*out);
    if (mode == POOLING_MAX) {
        q->enqueueWriteBuffer(input_buf, CL_FALSE, 0, sizeof(float)*in, input);
        // Extract a kernel function out of the program (.cl)
        cl::Kernel Pooling = cl::Kernel (*program, "MaxPoolKernel");

        int narg = 0;
        Pooling.setArg(narg++, out);
        Pooling.setArg(narg++, input_buf);
        Pooling.setArg(narg++, in_n);
        Pooling.setArg(narg++, in_c);
        Pooling.setArg(narg++, in_h);
        Pooling.setArg(narg++, in_w);
        Pooling.setArg(narg++, out_h);
        Pooling.setArg(narg++, out_w);
        Pooling.setArg(narg++, kernel_h);
        Pooling.setArg(narg++, kernel_w);
        Pooling.setArg(narg++, h_str);
        Pooling.setArg(narg++, v_str);
        Pooling.setArg(narg++, h_pad);
        Pooling.setArg(narg++, v_pad);
        Pooling.setArg(narg++, output_buf);
        q->enqueueNDRangeKernel(
            Pooling,
            cl::NullRange,
            cl::NDRange((out + LIN_BLOCK_SIZE-1)/LIN_BLOCK_SIZE*LIN_BLOCK_SIZE),
            cl::NDRange(LIN_BLOCK_SIZE)
        );
        q->enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float)*out, output);
    }
    else if (mode == POOLING_AVG) {
        q->enqueueWriteBuffer(input_buf, CL_FALSE, 0, sizeof(float)*in, input);
        cl::Kernel Pooling = cl::Kernel (*program, "AvgPoolKernel");

        int narg = 0;
        Pooling.setArg(narg++, out);
        Pooling.setArg(narg++, input_buf);
        Pooling.setArg(narg++, in_n);
        Pooling.setArg(narg++, in_c);
        Pooling.setArg(narg++, in_h);
        Pooling.setArg(narg++, in_w);
        Pooling.setArg(narg++, out_h);
        Pooling.setArg(narg++, out_w);
        Pooling.setArg(narg++, kernel_h);
        Pooling.setArg(narg++, kernel_w);
        Pooling.setArg(narg++, h_str);
        Pooling.setArg(narg++, v_str);
        Pooling.setArg(narg++, h_pad);
        Pooling.setArg(narg++, v_pad);
        Pooling.setArg(narg++, output_buf);
         
        q->enqueueNDRangeKernel(
            Pooling,
            cl::NullRange,
            cl::NDRange((out + LIN_BLOCK_SIZE-1)/LIN_BLOCK_SIZE*LIN_BLOCK_SIZE),
            cl::NDRange(LIN_BLOCK_SIZE)
        );
        q->enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float)*out, output);
    }
    q->finish();
}

void opendnnNormForward (opendnnHandle_t, opendnnNormDescriptor_t norm,
  opendnnTensorDescriptor_t input_desc, const float* input,
  opendnnTensorDescriptor_t output_desc, float* output) {
    int local_size;
    double alpha, beta, K;
    int in_n, in_c, in_h, in_w, in_nst, in_cst, in_hst, in_wst;
    int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;
    opendnnNormMode_t mode;
    opendnnGetNormDescriptor(norm, &local_size, &alpha, &beta, &K, &mode);
    opendnnGetTensor4dDescriptor(input_desc, &in_n, &in_c, &in_h, &in_w,
        &in_nst, &in_cst, &in_hst, &in_wst);
    opendnnGetTensor4dDescriptor(output_desc, &out_n, &out_c, &out_h, &out_w,
        &out_nst, &out_cst, &out_hst, &out_wst);

    if (mode == CROSS_CHANNEL) {
      for (int n = 0; n < in_n; ++n) {
        for (int c = 0; c < in_c; ++c) {
          for (int h = 0; h < in_h; ++h) {
            for (int w = 0; w < in_w; ++w) {
              int start = c - (local_size - 1) / 2;
              int end = std::min (start + local_size, in_c);
              start = std::max (start, 0);
              float scale = K;
              for (int i = start; i < end; ++i) {
                  float value = *(input + w * in_wst + h * in_hst + i * in_cst + n * in_nst);
                  scale += (value * value * alpha) / local_size;
              }
              *(output + w * out_wst + h * out_hst + c * out_cst + n * out_nst) = 
                  *(input + w * in_wst + h * in_hst + c * in_cst + n * in_nst) / pow (scale, beta);
            }
          }
        }
      }
    }
    else if (mode == WITHIN_CHANNEL) {
      for (int n = 0; n < in_n; ++n) {
        for (int c = 0; c < in_c; ++c) {
          for (int h = 0; h < in_h; ++h) {
            int h_start = h - (local_size - 1) / 2;
            int h_end = std::min (h_start + local_size, in_h);
            h_start = std::max (h_start, 0);
            for (int w = 0; w < in_w; ++w) {
              float scale = K;
              int w_start = w - (local_size - 1) / 2;
              int w_end = std::min(w_start + local_size, in_w);
              w_start = std::max(w_start, 0);
              for (int nh = h_start; nh < h_end; ++nh) {
                for (int nw = w_start; nw < w_end; ++nw) {
                  float value = input[nw*in_wst + nh*in_hst + c*in_cst + n*in_nst];
                  scale += (value * value * alpha) / (local_size * local_size);
                }
              }
              output[w*out_wst + h*out_hst + c*out_cst + n*out_nst] =
                  input[w*in_wst + h*in_hst + c*in_cst + n*in_nst] /
                  std::pow(scale, beta);
            }
          }
        }
      }
    }
}

// Convolution computation API
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

void opendnnConvolutionForward (opendnnHandle_t handle,
  const opendnnTensorDescriptor_t input_desc, const float* input_cst,
  const opendnnFilterDescriptor_t filter_desc, const float* filter_cst,
  const opendnnConvolutionDescriptor_t conv_desc, float* workspace, size_t size_in_bytes,
  const opendnnTensorDescriptor_t output_desc, float* output) {
    int in_n, in_c, in_h, in_w, in_nst, in_cst, in_hst, in_wst;
    int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;
    int pad_h, pad_w, str_h, str_w, ups_x, ups_y;
    int fil_out, fil_in, fil_h, fil_w;
    int group;
    opendnnGetTensor4dDescriptor (input_desc, &in_n, &in_c, &in_h, &in_w,
        &in_nst, &in_cst, &in_hst, &in_wst);
    opendnnGetTensor4dDescriptor (output_desc, &out_n, &out_c, &out_h, &out_w,
        &out_nst, &out_cst, &out_hst, &out_wst);
    opendnnGetFilter4dDescriptor (filter_desc, &fil_out, &fil_in, &fil_h, &fil_w);
    opendnnGetConvolution2dDescriptor (conv_desc, &pad_h, &pad_w, &str_h, &str_w, &ups_x, &ups_y);
    opendnnGetConvolutionGroupCount(conv_desc, &group);
    float* input = (float*) input_cst;
    float* filter = (float*) filter_cst;

    // cout << "Group: " << group << '\n';
    // Retrieve OpenCL context
    cl::Context context = handle->context;
    cl::CommandQueue* q = handle->cmdq;
    cl::Device default_device = handle->all_devices[0];
    cl::Program* program = handle->program;

    // Extract a kernel function out of the program (.cl)
    cl::Kernel im2col=cl::Kernel(*program, "im2col_gpu_kernel");

    // im2col
    size_t col_cnt_in_batch = out_h*out_w*in_c*fil_h*fil_w;
    size_t col_cnt = in_n*col_cnt_in_batch;
    float *col_buf;
    col_buf = (float*) malloc(sizeof(float)*col_cnt);
    cl::Buffer col_buf_device(context, CL_MEM_READ_WRITE, sizeof(float)*col_cnt);

    for (int n = 0; n < in_n; ++n) {
        float* input_in_batch = input + n*in_nst;
        float* col_in_batch = col_buf + n*col_cnt_in_batch;
        memset((void*)col_in_batch, 0, sizeof(float)*col_cnt_in_batch);
        cl_int ch;
        cl::Buffer input_in_batch_device(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                            sizeof(float)*in_nst, input_in_batch);
        cl::Buffer col_in_batch_device(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
                            sizeof(float)*col_cnt_in_batch, col_in_batch);

        // Parameter setting
        int height_col = (in_h + 2 * pad_h - fil_h) / str_h + 1;
        int width_col = (in_w + 2 * pad_w - fil_w) / str_w + 1;
        int num_kernels = in_c * height_col * width_col;
        size_t global = (num_kernels+LIN_BLOCK_SIZE-1)/LIN_BLOCK_SIZE * LIN_BLOCK_SIZE;

        int narg = 0;
        im2col.setArg(narg++, num_kernels);
        im2col.setArg(narg++, input_in_batch_device);
        im2col.setArg(narg++, in_h);
        im2col.setArg(narg++, in_w);
        im2col.setArg(narg++, fil_h);
        im2col.setArg(narg++, fil_w);
        im2col.setArg(narg++, pad_h);
        im2col.setArg(narg++, pad_w);
        im2col.setArg(narg++, str_h);
        im2col.setArg(narg++, str_w);
        im2col.setArg(narg++, 1);
        im2col.setArg(narg++, 1);
        im2col.setArg(narg++, height_col);
        im2col.setArg(narg++, width_col);
        im2col.setArg(narg++, col_in_batch_device);

        // Launch the kernel
        ch = q->enqueueNDRangeKernel(
            im2col,
            cl::NullRange,
            cl::NDRange(global),
            cl::NDRange(LIN_BLOCK_SIZE)
        );
        q->enqueueCopyBuffer(col_in_batch_device, col_buf_device, 
            0, sizeof(float)*n*col_cnt_in_batch, sizeof(float)*col_cnt_in_batch);
    }
    q->finish();
    // double start = get_time();
    int fil_out_ = fil_out / group;
    int fil_in_  = fil_in / group;
    int in_c_   = in_c / group;
    int out_c_   = out_c / group;

    // Extract a kernel function out of the program (.cl)
    cl::Kernel krnl_matmul=cl::Kernel(*program, "matmul_block_lin_shared_batch");

    // Forward through cuDNN in parallel over groups.
    float* col_in_batch = col_buf;
    float* out_in_batch = output;

    //TODO!!!: Group Issue
    for (int g = 0; g < group; g++) {
        const int M = out_c_;
        const int N = out_h*out_w;
        const int K = in_c_*fil_h*fil_w;
        const int N_BATCH = in_n;

        const int weight_offset = fil_out_*fil_in_*fil_h*fil_w;
        const int col_offset = in_c_*fil_h*fil_w*out_h*out_w;
        const int output_offset = out_c_*out_h*out_w;
        float *A = filter + weight_offset * g;
        float *B = col_in_batch + col_offset * g;
        float *C = out_in_batch + output_offset * g;

        // Allocate memory on FPGA, cl::Buffer object is the reference for device
        cl::Buffer buf_filter(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            M*K*sizeof(float), A);
        cl::Buffer buf_input(context, CL_MEM_READ_ONLY,
            group*N_BATCH*K*N*sizeof(float));
        cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY,
            group*N_BATCH*M*N*sizeof(float));

        cl_int check = q->enqueueCopyBuffer(col_buf_device, buf_input, sizeof(float)*col_offset*g, 0, sizeof(float)*(N_BATCH*K*N*group - col_offset*g));
        // cout << "pre: " << check << endl;
        // Set kernel arguments
        int narg = 0;
        krnl_matmul.setArg(narg++, buf_filter);
        krnl_matmul.setArg(narg++, buf_input);
        krnl_matmul.setArg(narg++, buf_output);
        krnl_matmul.setArg(narg++, M);
        krnl_matmul.setArg(narg++, K);
        krnl_matmul.setArg(narg++, K);
        krnl_matmul.setArg(narg++, N);
        krnl_matmul.setArg(narg++, M);
        krnl_matmul.setArg(narg++, N);
        krnl_matmul.setArg(narg++, group);

        // Launch the kernel
        size_t global[3] = {N, M, N_BATCH};
        size_t local[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
        for (int i = 0 ; i < 3; i++) {
            global[i] = (global[i] + local[i] - 1) / local[i] * local[i];
        }

        check = q->enqueueNDRangeKernel(
            krnl_matmul,
            cl::NullRange,
            cl::NDRange(global[0], global[1], global[2]),
            cl::NDRange(local[0], local[1], local[2])
        );
        // cout << "kernel: " << check << endl;
        check = q->enqueueReadBuffer(buf_output, CL_TRUE, 0, sizeof(float)*out_n*out_c*out_h*out_w, C);
        // cout << "post: " << check << endl;
    }
    q->finish();
    // double end = get_time();
    // cerr << end-start << '\n';
}


void opendnnInnerProductForward(opendnnHandle_t handle,
    opendnnTensorDescriptor_t input_desc, bool TransA, float* input,
    opendnnTensorDescriptor_t weight_desc, bool TransB, float* weight,
    opendnnTensorDescriptor_t output_desc, float* output) {
    int in_n, in_c, in_h, in_w, in_nst, in_cst, in_hst, in_wst;
    int out_n, out_c, out_h, out_w, out_nst, out_cst, out_hst, out_wst;

    opendnnGetTensor4dDescriptor (input_desc, &in_n, &in_c, &in_h, &in_w, &in_nst, &in_cst, &in_hst, &in_wst);
    opendnnGetTensor4dDescriptor (output_desc, &out_n, &out_c, &out_h, &out_w, &out_nst, &out_cst, &out_hst, &out_wst);

    int M = in_n;
    int K = in_c * in_h * in_w;
    int N = out_c;

    // Retrieve OpenCL context
    cl::Context context = handle->context;
    cl::CommandQueue* q = handle->cmdq;
    cl::Device default_device = handle->all_devices[0];
    cl::Program* program = handle->program;

    cl::Buffer in_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*M*K, input);
    cl::Buffer w_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*K*N, weight);
    cl::Buffer out_buf(context, CL_MEM_WRITE_ONLY, sizeof(float)*M*N);

    size_t global[2] = {N, M};
    size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE};
    for (int i = 0 ; i < 2; i++) {
        global[i] = (global[i] + local[i] - 1) / local[i] * local[i];
    }


    if (TransA == false & TransB == false) {
        cl::Kernel matmul = cl::Kernel(*program, "matmul_block_lin_shared");
        int narg = 0;
        matmul.setArg(narg++, in_buf);
        matmul.setArg(narg++, w_buf);
        matmul.setArg(narg++, out_buf);
        matmul.setArg(narg++, M);
        matmul.setArg(narg++, K);
        matmul.setArg(narg++, K);
        matmul.setArg(narg++, N);
        matmul.setArg(narg++, M);
        matmul.setArg(narg++, N);
        q->enqueueNDRangeKernel(
            matmul,
            cl::NullRange,
            cl::NDRange(global[0], global[1]),
            cl::NDRange(local[0], local[1])
        );
    } else if (TransA == false & TransB == true) {
        cl::Kernel matmul = cl::Kernel(*program, "matmul_block_lin_shared_trans");
        int narg = 0;
        matmul.setArg(narg++, in_buf);
        matmul.setArg(narg++, w_buf);
        matmul.setArg(narg++, out_buf);
        matmul.setArg(narg++, M);
        matmul.setArg(narg++, K);
        matmul.setArg(narg++, N);
        matmul.setArg(narg++, K);
        matmul.setArg(narg++, M);
        matmul.setArg(narg++, N);
        q->enqueueNDRangeKernel(
            matmul,
            cl::NullRange,
            cl::NDRange(global[0], global[1]),
            cl::NDRange(local[0], local[1])
        );
    } else {
        exit(-1);
    }

    // Extract a kernel function out of the program (.cl)
    q->enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(float)*M*N, output);
    q->finish();
}
