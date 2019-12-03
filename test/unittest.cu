#include <opendnn.h>
#include <iostream>
#include <cstring>

using namespace std;

int main () {
  // Declaration
  opendnnHandle_t handle;
  opendnnTensorDescriptor_t input_desc, output_desc;
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
  opendnnCreateTensorDescriptor(&output_desc);
  opendnnCreateFilterDescriptor(&filter_desc);
  opendnnCreateConvolutionDescriptor(&conv_desc);
  opendnnSetTensor4dDescriptor(input_desc, n, c, h, w);
  opendnnSetTensor4dDescriptor(output_desc, n, oc, oh, ow);
  opendnnSetFilter4dDescriptor(filter_desc, oc, c, kh, kw);
  opendnnSetConvolution2dDescriptor(conv_desc, /*pad_h=*/0,/*pad_w=*/0,
                                               /*str_h=*/1,/*str_w=*/1,
                                               /*dir_h=*/0,/*dir_w=*/0);
  opendnnSetConvolutionGroupCount(conv_desc, ngroup); 
  const float b[16] = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
  const float f[4] = {1,0,0,1};
  float t[9] = {0,};

  // GPU device memory allocation and init
  float *b_dev, *f_dev, *t_dev = NULL;
  cudaMalloc(&b_dev, sizeof(float)*n*c*h*w);
  cudaMalloc(&f_dev, sizeof(float)*oc*c*kh*kw);
  cudaMalloc(&t_dev, sizeof(float)*n*oc*oh*ow);
  cudaMemcpy(b_dev, b, sizeof(float)*n*c*h*w, cudaMemcpyHostToDevice);
  cudaMemcpy(f_dev, f, sizeof(float)*oc*c*kh*kw, cudaMemcpyHostToDevice);
  cudaMemcpy(t_dev, t, sizeof(float)*n*oc*oh*ow, cudaMemcpyHostToDevice);
  
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
     input_desc, b_dev,
     filter_desc, f_dev, conv_desc,
     workspace, size_in_bytes,
     output_desc, t_dev
  );

  // Get results
  cudaMemcpy(t, t_dev, sizeof(float)*n*oc*oh*ow, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 9; i++) {
    cout << t[i] << '\n';
  }
  cout << "Done" << endl;
  opendnnDestroy(handle);
  return 0;
}
