#include <iostream>
#include <cstring>

#include <opendnn.h>

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
  float input[16] = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
  float filter[4] = {1,0,0,1};
  float output[9] = {0,};
  float bias[1] = {1};

  // For CPU version, workspace is not needed
  size_t size_in_bytes = 0;
  float* workspace = NULL;

  // Perform convolution
  opendnnConvolutionForward(handle,
     input_desc, input,
     filter_desc, filter, conv_desc,
     workspace, size_in_bytes,
     output_desc, output
  );

  // Perform bias addition
  opendnnAddTensor(handle, bias_desc, bias, output_desc, output);

  for (int i = 0; i < 9; i++) {
    cout << output[i] << '\n';
  }

  opendnnDestroy(handle);

  cout << "Done" << endl;
  cout << "If outputs are all zero, check libopendnn is correctly built with a TARGET option in common.mk" << endl;
  cout << "Or you should check LD_LIBRARY_PATH directs the right version of libopendnn.so" << endl;
  return 0;
}
