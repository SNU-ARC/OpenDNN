#ifndef openDNN_H_
#define openDNN_H_

// #define CL_HPP_CL_1_2_DEFAULT_BUILD
// #define CL_HPP_TARGET_OPENCL_VERSION 120
// #define CL_HPP_MINIMUM_OPENCL_VERSION 120
// #define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/opencl.h>
#include <stdbool.h>

#define DEBUG_DATA_CNT 8
// #define __OPENDNN_DEBUG__

#ifdef __OPENDNN_DEBUG__
  #define stdform std::cerr << std::setw(12)
  #define DEBUG(_data) for(int i=0; i<DEBUG_DATA_CNT; i++) stdform << _data[i] << " "; \
    std::cerr << std::endl
#else
  #define DEBUG(_data)
#endif

struct opendnnContext;

typedef struct opendnnContext* opendnnHandle_t;

typedef float Dtype;

typedef enum {
    OPENDNN_ACTIVATION_SIGMOID,
    OPENDNN_ACTIVATION_RELU,
    OPENDNN_ACTIVATION_TANH,
//     OPENDNN_ACTIVATION_CLIPPED_RELU,
} opendnnActivationMode_t;

typedef enum {
    POOLING_MAX,
    POOLING_AVG
} opendnnPoolingMode_t;

// typedef enum {
//     CONVOLUTION,
//     CROSS_CORRELATION
// } opendnnConvolutionMode_t;

// TODO: Fix normalization API like corresponding cuDNN
typedef enum {
    CROSS_CHANNEL,
    WITHIN_CHANNEL
} opendnnNormMode_t;

typedef struct opendnnTensorStruct {
    int number_;
    int channel_;
    int height_;
    int width_;

    int count;

    int stride_n;
    int stride_c;
    int stride_h;
    int stride_w;
} *opendnnTensorDescriptor_t;

typedef struct opendnnFilterStruct {
    int output_;
    int input_;
    int height_;
    int width_;

    int count;
} *opendnnFilterDescriptor_t;

typedef struct opendnnConvolutionStruct {
    int pad_h;
    int pad_w;
    int vertical_stride;
    int horizon_stride;
    int upscale_x;
    int upscale_y;
    int group;
} *opendnnConvolutionDescriptor_t;

typedef struct opendnnActivationStruct {
    // double relu_ceiling;
    opendnnActivationMode_t activation_mode;
} *opendnnActivationDescriptor_t;

typedef struct opendnnPoolingStruct {
    opendnnPoolingMode_t pooling_mode;
    int w_height;
    int w_width;
    int vertical_padding;
    int horizon_padding;
    int vertical_stride;
    int horizon_stride;
} *opendnnPoolingDescriptor_t;

typedef struct opendnnNormStruct {
    int normN;
    double normAlpha;
    double normBeta;
    double normK;
    opendnnNormMode_t normMode;
} *opendnnNormDescriptor_t;


// OpenDNN API
void opendnnCreate (opendnnHandle_t*);
void opendnnDestroy (opendnnHandle_t);

// Tensor management
void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t*);
void opendnnSetTensor4dDescriptorEx (opendnnTensorDescriptor_t, int, int, int, int, int, int, int, int);
void opendnnSetTensor4dDescriptor (opendnnTensorDescriptor_t, int, int, int, int);
void opendnnGetTensor4dDescriptor (opendnnTensorDescriptor_t, int*, int*, int*, int*, int*, int*, int*, int*);
void opendnnSetTensor2dDescriptor (opendnnTensorDescriptor_t, int, int);
void opendnnGetTensor2dDescriptor (opendnnTensorDescriptor_t, int*, int*, int*, int*);

// Filter
void opendnnCreateFilterDescriptor (opendnnFilterDescriptor_t*);
void opendnnSetFilter4dDescriptor (opendnnFilterDescriptor_t, int, int, int, int);
void opendnnGetFilter4dDescriptor (opendnnFilterDescriptor_t, int*, int*, int*, int*);

// Convolution
void opendnnCreateConvolutionDescriptor (opendnnConvolutionDescriptor_t*);
void opendnnSetConvolution2dDescriptor (opendnnConvolutionDescriptor_t, int, int, int, int, int, int);
void opendnnGetConvolution2dDescriptor (opendnnConvolutionDescriptor_t, int*, int*, int*, int*, int*, int*);
void opendnnSetConvolutionGroupCount(opendnnConvolutionDescriptor_t, int);
void opendnnGetConvolutionGroupCount(opendnnConvolutionDescriptor_t, int*);

// Pooling
void opendnnCreatePoolingDescriptor (opendnnPoolingDescriptor_t*);
void opendnnSetPooling2dDescriptor (opendnnPoolingDescriptor_t, opendnnPoolingMode_t, int, int, int, int, int, int);
void opendnnGetPooling2dDescriptor (opendnnPoolingDescriptor_t, opendnnPoolingMode_t*, int*, int*, int*, int*, int*, int*);

// Activation
void opendnnCreateActivationDescriptor (opendnnActivationDescriptor_t*);
void opendnnSetActivationDescriptor (opendnnActivationDescriptor_t,
                                     opendnnActivationMode_t);
void opendnnGetActivationDescriptor (opendnnActivationDescriptor_t,
                                     opendnnActivationMode_t*);

// Normalization
// TODO: this is different from cuDNN API
void opendnnCreateNormDescriptor (opendnnNormDescriptor_t*);
void opendnnSetNormDescriptor (opendnnNormDescriptor_t, int, double, double, double, opendnnNormMode_t);
void opendnnGetNormDescriptor (opendnnNormDescriptor_t, int*, double*, double*, double*, opendnnNormMode_t*);

// Softmax
// TODO: Softmax is not implemented now
// void opendnnSoftmaxForward (opendnnTensor, const Dtype*, OpenTensor, Dtype*);

// Actual computation methods
void opendnnAddTensor (opendnnHandle_t, opendnnTensorDescriptor_t,
                       const float*, opendnnTensorDescriptor_t, float*);
void opendnnInnerProductForward(opendnnHandle_t handle,
    opendnnTensorDescriptor_t input_desc, bool TransA, float* input,
    opendnnTensorDescriptor_t weight_desc, bool TransB, float* weight,
    opendnnTensorDescriptor_t output_desc, float* output);
void opendnnConvolutionForward (opendnnHandle_t,
                                opendnnTensorDescriptor_t, float*,
                                opendnnFilterDescriptor_t, float*,
                                opendnnConvolutionDescriptor_t,
                                opendnnTensorDescriptor_t, float*);
void opendnnConvolutionForward_gold (opendnnHandle_t,
                                opendnnTensorDescriptor_t, float*,
                                opendnnFilterDescriptor_t, float*,
                                opendnnConvolutionDescriptor_t,
                                opendnnTensorDescriptor_t, float*);
void opendnnConvolutionForward_fpga (opendnnHandle_t,
                                opendnnTensorDescriptor_t, Dtype*,
                                opendnnFilterDescriptor_t, Dtype*,
                                opendnnConvolutionDescriptor_t,
                                opendnnTensorDescriptor_t, Dtype*);
void opendnnPoolingForward (opendnnHandle_t, opendnnPoolingDescriptor_t,
                            opendnnTensorDescriptor_t, float*,
                            opendnnTensorDescriptor_t, float*);
void opendnnNormForward (opendnnHandle_t, opendnnNormDescriptor_t,
                         opendnnTensorDescriptor_t, const float*,
                         opendnnTensorDescriptor_t, float*);
void opendnnActivationForward (opendnnHandle_t, opendnnActivationDescriptor_t,
                         opendnnTensorDescriptor_t, float*,
                         opendnnTensorDescriptor_t, float*);

void top_accel (
    const int top_n, const int top_c, const int top_w, const int top_h,
    const int bot_n, const int bot_c, const int bot_w, const int bot_h,
    const int fil_w, const int fil_h, const int fil_in,
    const int pad_w, const int pad_h,
    const int str_w, const int str_h,
    const int row, const int col, const int to,
    Dtype* top, const Dtype* bottom, Dtype* filter
    );

#endif // OPEN_CNN_H_
