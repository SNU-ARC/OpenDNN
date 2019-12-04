#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>
#include <opendnn.h>

const int BLOCK_SIZE=24;

#define CHECK_ERR(err)  \
    if (err != CL_SUCCESS) {    \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);   \
        exit(EXIT_FAILURE); \
    }

struct opendnnContext {
    cl_uint num_platforms;
    cl_platform_id * platforms;
    cl_uint num_devices;
    cl_device_id * devices;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    size_t max_work_group_size;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_ulong max_mem_alloc_size;
};

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

char * get_source_code (const char * file_name, size_t * len) {
    char * source_code;
    char buf[1024];
    size_t length, temp;
    FILE * file = fopen(file_name, "r");
    if (!file) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        getcwd(buf, 1024);
        printf("Working Directory: %s\n", buf);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char *)malloc(length+1);
    temp = fread(source_code, length, 1, file);
    temp = length;
    source_code[temp] = '\0';
    fclose(file);
    *len = length;
    return source_code;
}


// Current version targets the number 1 heterogen architecture (In this example, GPU)
void opendnnCreate (opendnnHandle_t* handle) {
    *handle = malloc(sizeof(struct opendnnContext));
    cl_uint num_platforms;
    cl_uint num_devices;
    cl_int err;
    char* kernel_source;
    size_t kernel_source_size;
    clGetPlatformIDs(0, NULL, &num_platforms);
    (*handle)->platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, (*handle)->platforms, NULL);
    clGetDeviceIDs((*handle)->platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    (*handle)->devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs((*handle)->platforms[1], CL_DEVICE_TYPE_ALL, num_devices, (*handle)->devices, NULL);
    clGetDeviceInfo((*handle)->devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &(*handle)->max_work_group_size, NULL);
    clGetDeviceInfo((*handle)->devices[0], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &(*handle)->global_mem_size, NULL);
    clGetDeviceInfo((*handle)->devices[0], CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong), &(*handle)->local_mem_size, NULL);
    clGetDeviceInfo((*handle)->devices[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_ulong), &(*handle)->max_mem_alloc_size, NULL);
    // (*handle)->max_work_group_size = max_work_group_size;
    // (*handle)->global_mem_size = global_mem_size;
    // (*handle)->local_mem_size = local_mem_size;
    // (*handle)->max_mem_alloc_size = max_mem_alloc_size;
    (*handle)->num_platforms = num_platforms;
    (*handle)->num_devices = num_devices;
    (*handle)->context = clCreateContext(NULL, 1, (*handle)->devices, NULL, NULL, &err);
    CHECK_ERR(err);
    (*handle)->queue = clCreateCommandQueue((*handle)->context, (*handle)->devices[0], 0, &err);
    CHECK_ERR(err);
    kernel_source = get_source_code("opendnn_kernel.cl", &kernel_source_size);
    (*handle)->program = clCreateProgramWithSource((*handle)->context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    err = clBuildProgram((*handle)->program, 1, &(*handle)->devices[0], "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;
        err = clGetProgramBuildInfo((*handle)->program, (*handle)->devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERR(err);
        log = (char*)malloc(log_size+1);
        err = clGetProgramBuildInfo((*handle)->program, (*handle)->devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERR(err);
        log[log_size] = '\0';
        printf("Compiler error: %s\n", log);
        free(log);
        exit(0);
    }
}

void opendnnDestroy (opendnnHandle_t handle) {
    clReleaseProgram(handle->program);
    clReleaseCommandQueue(handle->queue);
    clReleaseContext(handle->context);
    free(handle->devices);
    free(handle->platforms);
    free(handle);
}

// Tensor management
void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t* tensor_desc) {
    *tensor_desc = malloc(sizeof(struct opendnnTensorStruct));
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
    *filter = malloc(sizeof(struct opendnnFilterStruct));
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
    *conv_desc = malloc(sizeof(struct opendnnConvolutionStruct));
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
    *pool = malloc(sizeof(opendnnPoolingDescriptor_t));
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
    *nm = malloc(sizeof(struct opendnnNormStruct));
}

// Activation methods
void opendnnCreateActivationDescriptor (opendnnActivationDescriptor_t* act){
    *act = malloc(sizeof(struct opendnnActivationStruct));
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
    cl_context context = handle->context;
    cl_command_queue q = handle->queue;
    cl_program program = handle->program;
    char acts[8];
    cl_int err;

    switch (mode) {
        case 0: strcpy(acts,"sigmoid");
        case 1: strcpy(acts,"ReLU");
        case 2: strcpy(acts,"htan");
    }

    // Extract a kernel function out of the program (.cl)
    cl_kernel kernel = clCreateKernel (program, acts, &err);
    const int N = in_n*in_c*in_h*in_w;
    size_t local = BLOCK_SIZE*BLOCK_SIZE;
    size_t global = (N + local-1)/local * local;

    cl_mem input_dev, output_dev;

    input_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*N, input, &err);
    output_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N, NULL, &err);
    CHECK_ERR(err);

    int narg = 0;
    clSetKernelArg(kernel, narg++, sizeof(cl_mem), &input_dev);
    clSetKernelArg(kernel, narg++, sizeof(cl_mem), &output_dev);
    clSetKernelArg(kernel, narg++, sizeof(int), &N);
    clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clEnqueueReadBuffer(q, output_dev, CL_TRUE, 0, sizeof(float) * N, output, 0, NULL, NULL);
    clReleaseMemObject(input_dev);
    clReleaseMemObject(output_dev);
    clReleaseKernel(kernel);
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
    cl_context context = handle->context;
    cl_command_queue q = handle->queue;
    cl_program program = handle->program;

    int in = in_n * in_c * in_h * in_w;
    int out = out_n * out_c * out_h * out_w;
    cl_mem input_buf, output_buf;
    cl_int err;
    cl_kernel kernel = NULL;
    input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*in, input, &err);
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*out, NULL, &err);
    if (mode == POOLING_MAX) {
        // Extract a kernel function out of the program (.cl)
        kernel = clCreateKernel (program, "MaxPoolKernel", &err);
    } else if (mode == POOLING_AVG) {
        kernel = clCreateKernel (program, "AvgPoolKernel", &err);
    }
    size_t local = BLOCK_SIZE*BLOCK_SIZE;
    size_t global = (out + local-1)/local * local;

    int narg = 0;
    clSetKernelArg(kernel, narg++, sizeof(int), &out);
    clSetKernelArg(kernel, narg++, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, narg++, sizeof(int), &in_n);
    clSetKernelArg(kernel, narg++, sizeof(int), &in_c);
    clSetKernelArg(kernel, narg++, sizeof(int), &in_h);
    clSetKernelArg(kernel, narg++, sizeof(int), &in_w);
    clSetKernelArg(kernel, narg++, sizeof(int), &out_h);
    clSetKernelArg(kernel, narg++, sizeof(int), &out_w);
    clSetKernelArg(kernel, narg++, sizeof(int), &kernel_h);
    clSetKernelArg(kernel, narg++, sizeof(int), &kernel_w);
    clSetKernelArg(kernel, narg++, sizeof(int), &h_str);
    clSetKernelArg(kernel, narg++, sizeof(int), &v_str);
    clSetKernelArg(kernel, narg++, sizeof(int), &h_pad);
    clSetKernelArg(kernel, narg++, sizeof(int), &v_pad);
    clSetKernelArg(kernel, narg++, sizeof(cl_mem), &output_buf);
    clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clEnqueueReadBuffer(q, output_buf, CL_TRUE, 0, sizeof(float)*out, output, 0, NULL, NULL);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    DEBUG(input);
    DEBUG(output);
    printf("Pooling Fin\n");
}

void opendnnNormForward (opendnnHandle_t handle, opendnnNormDescriptor_t norm,
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
              int end = min (start + local_size, in_c);
              start = max (start, 0);
              float scale = K;
              for (int i = start; i < end; ++i) {
                  float value = *(input + w * in_wst + h * in_hst + i * in_cst + n * in_nst);
                  scale += (value * value * alpha) / local_size;
              }
              *(output + w * out_wst + h * out_hst + c * out_cst + n * out_nst) = 
                  *(input + w * in_wst + h * in_hst + c * in_cst + n * in_nst) / pow ((double)scale, beta);
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
            int h_end = min (h_start + local_size, in_h);
            h_start = max (h_start, 0);
            for (int w = 0; w < in_w; ++w) {
              float scale = K;
              int w_start = w - (local_size - 1) / 2;
              int w_end = min(w_start + local_size, in_w);
              w_start = max(w_start, 0);
              for (int nh = h_start; nh < h_end; ++nh) {
                for (int nw = w_start; nw < w_end; ++nw) {
                  float value = input[nw*in_wst + nh*in_hst + c*in_cst + n*in_nst];
                  scale += (value * value * alpha) / (local_size * local_size);
                }
              }
              output[w*out_wst + h*out_hst + c*out_cst + n*out_nst] =
                  input[w*in_wst + h*in_hst + c*in_cst + n*in_nst] /
                  pow((double)scale, beta);
            }
          }
        }
      }
    }
    DEBUG(input);
    DEBUG(output);
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

    DEBUG(output_data);
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
    DEBUG(output_data);
}

void opendnnConvolutionForward (opendnnHandle_t handle,
  opendnnTensorDescriptor_t input_desc, float* input,
  opendnnFilterDescriptor_t filter_desc, float* filter,
  opendnnConvolutionDescriptor_t conv_desc,
  opendnnTensorDescriptor_t output_desc, float* output) {
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

    // cout << "Group: " << group << '\n';
    // Retrieve OpenCL context
    cl_context context = handle->context;
    cl_command_queue q = handle->queue;
    cl_program program = handle->program;
    cl_int err;

    cl_mem in_buf, w_buf=NULL, out_buf=NULL;
    in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*in_n*in_c*in_h*in_w, input, &err);
    CHECK_ERR(err);
    w_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*fil_h*fil_w*fil_in*fil_out, filter, &err);
    out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*out_n*out_c*out_h*out_w, NULL, &err);
    printf("Buffer size: %d %d %d\n", sizeof(in_buf), sizeof(w_buf), sizeof(out_buf));

    // Extract a kernel function out of the program (.cl)
    cl_kernel kernel;

    // im2col
    size_t col_cnt_in_batch = out_h*out_w*in_c*fil_h*fil_w;
    size_t col_cnt = in_n*col_cnt_in_batch;
    float *col_buf;
    col_buf = (float*) malloc(sizeof(float)*col_cnt);
    memset(col_buf, 0, sizeof(float)*col_cnt);
    cl_mem col_buf_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*col_cnt, NULL, &err);
    CHECK_ERR(err);
    err = clEnqueueWriteBuffer(q, col_buf_device, CL_FALSE, 0, sizeof(float)*col_cnt, col_buf, 0, NULL, NULL);
    CHECK_ERR(err);
    kernel = clCreateKernel(program, "im2col_gpu_kernel", &err);
    CHECK_ERR(err);

    for (int n = 0; n < in_n; ++n) {
        cl_buffer_region im_region;
        cl_buffer_region col_region;
        im_region.origin = n*in_nst;
        im_region.size = in_nst;
        col_region.origin = n*col_cnt_in_batch;
        col_region.size = col_cnt_in_batch;
        cl_mem input_in_batch_device = clCreateSubBuffer(in_buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &im_region, &err);
        CHECK_ERR(err);
        cl_mem col_in_batch_device = clCreateSubBuffer(col_buf_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &col_region, &err);
        CHECK_ERR(err);

        // Parameter setting
        int height_col = (in_h + 2 * pad_h - fil_h) / str_h + 1;
        int width_col = (in_w + 2 * pad_w - fil_w) / str_w + 1;
        int num_kernels = in_c * height_col * width_col;
        int d = 1;
        size_t local = BLOCK_SIZE*BLOCK_SIZE;
        size_t global = (num_kernels+local-1/local) * local;

        int narg = 0;
        clSetKernelArg(kernel, narg++, sizeof(int), &num_kernels);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &input_in_batch_device);
        clSetKernelArg(kernel, narg++, sizeof(int), &in_h);
        clSetKernelArg(kernel, narg++, sizeof(int), &in_w);
        clSetKernelArg(kernel, narg++, sizeof(int), &fil_h);
        clSetKernelArg(kernel, narg++, sizeof(int), &fil_w);
        clSetKernelArg(kernel, narg++, sizeof(int), &pad_h);
        clSetKernelArg(kernel, narg++, sizeof(int), &pad_w);
        clSetKernelArg(kernel, narg++, sizeof(int), &str_h);
        clSetKernelArg(kernel, narg++, sizeof(int), &str_w);
        clSetKernelArg(kernel, narg++, sizeof(int), &d);
        clSetKernelArg(kernel, narg++, sizeof(int), &d);
        clSetKernelArg(kernel, narg++, sizeof(int), &height_col);
        clSetKernelArg(kernel, narg++, sizeof(int), &width_col);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &col_in_batch_device);
        clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        clReleaseMemObject(input_in_batch_device);
        clReleaseMemObject(col_in_batch_device);
    }
    clReleaseKernel(kernel);
    int fil_out_ = fil_out / group;
    int fil_in_  = fil_in / group;
    int in_c_   = in_c / group;
    int out_c_   = out_c / group;

    // Extract a kernel function out of the program (.cl)
    kernel = clCreateKernel(program, "matmul_block_lin_shared_batch", &err);

    // Forward through cuDNN in parallel over groups.
//     float* col_in_batch = col_buf;
//     float* out_in_batch = output;
//
//     //TODO!!!: Group Issue
    for (int n = 0; n < out_n; n++) {
    for (int g = 0; g < group; g++) {
        const int M = out_c_;
        const int N = out_h*out_w;
        const int K = in_c_*fil_h*fil_w;

        const int weight_offset = fil_out_*fil_in_*fil_h*fil_w;
        const int col_offset = in_c_*fil_h*fil_w*out_h*out_w;
        const int output_offset = out_c_*out_h*out_w;
        cl_buffer_region a_region, b_region, c_region;
        a_region.origin = weight_offset * (n * group + g);
        a_region.size = weight_offset;
        b_region.origin = col_offset * (n * group + g);
        b_region.size = col_offset;
        c_region.origin = output_offset * (n * group + g);
        c_region.size = output_offset;
        cl_mem A = clCreateSubBuffer(w_buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &a_region, &err);
        cl_mem B = clCreateSubBuffer(col_buf_device, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &b_region, &err);
        cl_mem C = clCreateSubBuffer(out_buf, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &c_region, &err);

        // Set kernel arguments
        // Launch the kernel
        size_t global[2] = {N, M};
        size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE};
        for (int i = 0 ; i < 2; i++) {
            global[i] = (global[i] + local[i] - 1) / local[i] * local[i];
        }
        int narg = 0;
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &A);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &B);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &C);
        clSetKernelArg(kernel, narg++, sizeof(int), &M);
        clSetKernelArg(kernel, narg++, sizeof(int), &K);
        clSetKernelArg(kernel, narg++, sizeof(int), &K);
        clSetKernelArg(kernel, narg++, sizeof(int), &N);
        clSetKernelArg(kernel, narg++, sizeof(int), &M);
        clSetKernelArg(kernel, narg++, sizeof(int), &N);
        clSetKernelArg(kernel, narg++, sizeof(int), &group);
        clEnqueueNDRangeKernel(q, kernel, 2, NULL, global, local, 0, NULL, NULL);
        clReleaseMemObject(A);
        clReleaseMemObject(B);
        clReleaseMemObject(C);
    }
    }
    clReleaseMemObject(col_buf_device);
    clReleaseMemObject(in_buf);
    clReleaseMemObject(w_buf);
    clReleaseMemObject(out_buf);
    clReleaseKernel(kernel);
    free(col_buf);
    DEBUG(input);
    DEBUG(output);
    printf("Conv Fin\n");
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
    cl_context context = handle->context;
    cl_command_queue q = handle->queue;
    cl_program program = handle->program;
    cl_int err;

    cl_mem in_buf, w_buf, out_buf;
    in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*M*K, input, &err);
    w_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*K*N, weight, &err);
    out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*M*N, NULL, &err);

    size_t global[2] = {N, M};
    size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE};
    for (int i = 0 ; i < 2; i++) {
        global[i] = (global[i] + local[i] - 1) / local[i] * local[i];
    }
    cl_kernel kernel;

    if (TransA == false && TransB == false) {
        kernel = clCreateKernel(program, "matmul_block_lin_shared", &err);
        int narg = 0;
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &w_buf);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, narg++, sizeof(int), &M);
        clSetKernelArg(kernel, narg++, sizeof(int), &K);
        clSetKernelArg(kernel, narg++, sizeof(int), &K);
        clSetKernelArg(kernel, narg++, sizeof(int), &N);
        clSetKernelArg(kernel, narg++, sizeof(int), &M);
        clSetKernelArg(kernel, narg++, sizeof(int), &N);
        clEnqueueNDRangeKernel(q, kernel, 2, NULL, global, local, 0, NULL, NULL);
    } else if (TransA == false && TransB == true) {
        kernel = clCreateKernel(program, "matmul_block_lin_shared_trans", &err);
        int narg = 0;
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &w_buf);
        clSetKernelArg(kernel, narg++, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel, narg++, sizeof(int), &M);
        clSetKernelArg(kernel, narg++, sizeof(int), &K);
        clSetKernelArg(kernel, narg++, sizeof(int), &N);
        clSetKernelArg(kernel, narg++, sizeof(int), &K);
        clSetKernelArg(kernel, narg++, sizeof(int), &M);
        clSetKernelArg(kernel, narg++, sizeof(int), &N);
        clEnqueueNDRangeKernel(q, kernel, 2, NULL, global, local, 0, NULL, NULL);
    } else {
        exit(-1);
    }
    clEnqueueReadBuffer(q, out_buf, CL_TRUE, 0, sizeof(float)*M*N, output, 0, NULL, NULL);
    clReleaseMemObject(in_buf);
    clReleaseMemObject(w_buf);
    clReleaseMemObject(out_buf);
    clReleaseKernel(kernel);
    printf("FC Fin\n");
}

