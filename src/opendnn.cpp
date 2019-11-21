#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#include "opendnn.h"

#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>

using namespace std;

struct opendnnContext {
    int driver_num_;
    int temp;
};

void opendnnCreate (opendnnHandle_t* handle) {
    *handle = new opendnnContext;
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
  const opendnnTensorDescriptor_t input_desc, const float* input,
  const opendnnFilterDescriptor_t filter_desc, const float* filter,
  const opendnnConvolutionDescriptor_t conv_desc, float* workspace, size_t workSpaceSizeInBytes,
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
    int fil_out_ = fil_out / group;
    int fil_in_  = fil_in / group;
    int in_c_   = in_c / group;
    int out_c_   = out_c / group;

    const int w_offset = fil_out*fil_in*fil_h*fil_w;
    const int i_offset = in_c*in_h*in_w;
    const int o_offset = out_c*out_h*out_w;

    for (int g = 0; g < group; ++g) {
      for (int n = 0; n < out_n; ++n) {  // TODO: batch processing
        for (int c = 0; c < out_c; ++c) {
          for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
              float sum = 0.0f;
              for (int k = 0; k < in_c; ++k) {
                for (int fh = 0; fh < fil_h; ++fh) {
                  for (int fw = 0; fw < fil_w; fw++) {
                    int ih = h*str_h - pad_h + fh;
                    int iw = w*str_w - pad_w + fw;
                    if (iw >= 0 && iw < in_w && ih >= 0 && ih < in_h) {
                        sum += input[iw*in_wst + ih*in_hst + k*in_cst + n*in_nst] *
                               filter[fw + fh*fil_w + k*fil_w*fil_h + c*fil_in*fil_w*fil_h];
                    }
                  }
                }
              }
              output[w*out_wst + h*out_hst + c*out_cst + n*out_nst] = sum;
            }
          }
        }
      }
      filter += w_offset;
      input  += i_offset;
      output += o_offset;
    }
    DEBUG(output);
    DEBUG(input);
}

