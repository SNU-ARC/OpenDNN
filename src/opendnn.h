#ifndef openDNN_H_
#define openDNN_H_

#include <string>
#include <cuda.h>
#include <cudnn.h>

// ======================================================================
//
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

// typedef enum {
//     SIGMOID,
//     RELU,
//     TANH
// } opendnnActivationMode_t;

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

// typedef struct activation {
//     double relu_ceiling;
//     opendnnActivationMode_t activation_mode;
// } *opendnnActivationDescriptor_t;

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
void opendnnSetStream (CUstream);

// Tensor management
void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t*);
void opendnnSetTensor4dDescriptorEx (opendnnTensorDescriptor_t, int, int, int, int, int, int, int, int);
void opendnnSetTensor4dDescriptor (opendnnTensorDescriptor_t, int, int, int, int);
void opendnnGetTensor4dDescriptor (opendnnTensorDescriptor_t, int*, int*, int*, int*, int*, int*, int*, int*);

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
void opendnnGetConvolutionForwardWorkspaceSize (opendnnHandle_t,
                                                opendnnTensorDescriptor_t,
                                                opendnnFilterDescriptor_t,
                                                opendnnConvolutionDescriptor_t,
                                                opendnnTensorDescriptor_t, size_t*);
// void opendnnGetConvolutionForwardWorkspaceSize_cu (cudnnHandle_t,
//                                                 cudnnTensorDescriptor_t,
//                                                 cudnnFilterDescriptor_t,
//                                                 cudnnConvolutionDescriptor_t,
//                                                 cudnnTensorDescriptor_t, size_t*);


// Pooling
void opendnnCreatePoolingDescriptor (opendnnPoolingDescriptor_t*);
void opendnnSetPooling2dDescriptor (opendnnPoolingDescriptor_t, opendnnPoolingMode_t, int, int, int, int, int, int);
void opendnnGetPooling2dDescriptor (opendnnPoolingDescriptor_t, opendnnPoolingMode_t*, int*, int*, int*, int*, int*, int*);

// Activation
// TODO: ReLU is default, thus we don't use these API now. Fix this for portability
// void opendnnCreateActivationDescriptor (opendnnActivationDescriptor_t*);
// void opendnnSetActivationDesc (opendnnActivationDescriptor_t, double, opendnnActivationMode_t);
// void opendnnGetActivationDesc (opendnnActivationDescriptor_t, double, opendnnActivationMode_t);

// Normalization
// TODO: this is different from cuDNN API
void opendnnCreateNormDescriptor (opendnnNormDescriptor_t*);
void opendnnSetNormDescriptor (opendnnNormDescriptor_t, int, double, double, double, opendnnNormMode_t);
void opendnnGetNormDescriptor (opendnnNormDescriptor_t, int*, double*, double*, double*, opendnnNormMode_t*);

// Softmax
// TODO: Softmax is not implemented now
// void opendnnSoftmaxForward (opendnnTensor, const Dtype*, OpenTensor, Dtype*);

// InnerProduct (FC)
// TODO: There are no InnerProduct methods in cuDNN but with OpenDNN?
// void opendnnCreateInnerProductDescriptor (opendnnInnerProduct*);
// void opendnnSetInnerProductDescriptor (opendnnInnerProduct, );
// void opendnnGetInnerProductDescriptor (opendnnInnerProduct, *);

// Actual computation methods
void opendnnAddTensor (opendnnHandle_t, opendnnTensorDescriptor_t,
                       const float*, opendnnTensorDescriptor_t, float*);
void opendnnConvolutionForward (opendnnHandle_t,
                                const opendnnTensorDescriptor_t, const float*,
                                const opendnnFilterDescriptor_t, const float*,
                                const opendnnConvolutionDescriptor_t, float*, size_t,
                                const opendnnTensorDescriptor_t, float*);
// void opendnnConvolutionForward_cu (cudnnHandle_t,
//                                 const cudnnTensorDescriptor_t, const float*,
//                                 const cudnnFilterDescriptor_t, const float*,
//                                 const cudnnConvolutionDescriptor_t, float*, size_t,
//                                 const cudnnTensorDescriptor_t, float*, const std::string name);
// void opendnnPoolingForward (opendnnHandle_t, opendnnPoolingDescriptor_t,
//                             opendnnTensorDescriptor_t, const Dtype*,
//                             opendnnTensorDescriptor_t, Dtype*);
// void opendnnNormForward (opendnnHandle_t, opendnnNormDescriptor_t,
//                          opendnnTensorDescriptor_t, const Dtype*,
//                          opendnnTensorDescriptor_t, Dtype*);
void opendnnInnerProductForward(opendnnHandle_t handle,
    opendnnTensorDescriptor_t input_desc, bool TransA, float* input,
    opendnnTensorDescriptor_t weight_desc, bool TransB, float* weight,
    opendnnTensorDescriptor_t output_desc, float* output);

#endif // OPEN_CNN_H_
