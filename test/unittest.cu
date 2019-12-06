#include <opendnn.h>
#include <iostream>
#include <cstring>

using namespace std;

int main () {
  // Declaration
  opendnnHandle_t handle;
  opendnnTensorDescriptor_t input_desc, output_desc, bias_desc;
  opendnnFilterDescriptor_t filter_desc;
  opendnnConvolutionDescriptor_t conv_desc;

  // Test dimensions
  int n=1; // batch
  int c=1, h=4, w=4;
  int oc=1, oh=3, ow=3;
  int kh=2, kw=2;
  int ngroup = 1; // Grouped conv_descolution for AlexNet, MobileNet, etc.

  // Initialization
  opendnnCreate(&handle);
  opendnnCreateTensorDescriptor(&input_desc); 
  opendnnCreateFilterDescriptor(&filter_desc);
  opendnnCreateTensorDescriptor(&output_desc);
  opendnnCreateTensorDescriptor(&bias_desc);
  opendnnCreateConvolutionDescriptor(&conv_desc);
  opendnnSetTensor4dDescriptor(input_desc, n, c, h, w);
  opendnnSetFilter4dDescriptor(filter_desc, oc, c, kh, kw);
  opendnnSetTensor4dDescriptor(output_desc, n, oc, oh, ow);
  opendnnSetTensor4dDescriptor(bias_desc, 1, oc, 1, 1);
  opendnnSetConvolution2dDescriptor(conv_desc, /*pad_h=*/0,/*pad_w=*/0,
                                               /*str_h=*/1,/*str_w=*/1,
                                               /*dir_h=*/0,/*dir_w=*/0);
  opendnnSetConvolutionGroupCount(conv_desc, ngroup); 
  const float input[16] = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
  const float filter[4] = {1,0,0,1};
  float output[9] = {0,};
  float bias[1] = {1};

  // GPU device memory allocation and init
  float *input_dev, *filter_dev, *output_dev, *bias_dev = NULL;
  cudaMalloc(&input_dev, sizeof(float)*n*c*h*w);
  cudaMalloc(&filter_dev, sizeof(float)*oc*c*kh*kw);
  cudaMalloc(&output_dev, sizeof(float)*n*oc*oh*ow);
  cudaMalloc(&bias_dev, sizeof(float)*c);
  cudaMemcpy(input_dev, input, sizeof(float)*n*c*h*w, cudaMemcpyHostToDevice);
  cudaMemcpy(filter_dev, filter, sizeof(float)*oc*c*kh*kw, cudaMemcpyHostToDevice);
  cudaMemcpy(output_dev, output, sizeof(float)*n*oc*oh*ow, cudaMemcpyHostToDevice);
  cudaMemcpy(bias_dev, bias, sizeof(float)*c, cudaMemcpyHostToDevice);
  
  // For now, workspace is needed to save im2col transposed input tensor
  // opendnnGetConvolutionForwardWorkspaceSize returns just n*oc*kh*kw*oh*ow*sizeof(float)
  size_t size_in_bytes;
  float* workspace;
  opendnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_desc,
                                            filter_desc, conv_desc,
                                            output_desc, &size_in_bytes);
  cudaMalloc(&workspace, size_in_bytes);

  // Perform convolution
  opendnnConvolutionForward(handle,
     input_desc, input_dev,
     filter_desc, filter_dev, conv_desc,
     workspace, size_in_bytes,
     output_desc, output_dev
  );

  // Perform bias addition
  opendnnAddTensor(handle, bias_desc, bias_dev, output_desc, output_dev);

  // Get results
  cudaMemcpy(output, output_dev, sizeof(float)*n*oc*oh*ow, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 9; i++) {
    cout << output[i] << '\n';
  }
  cout << "Done" << endl;
  opendnnDestroy(handle);
  return 0;
}
