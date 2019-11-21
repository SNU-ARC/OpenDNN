#include <opendnn.h>
#include <iostream>
#include <cstring>

using namespace std;

int main () {
	// Declaration
	opendnnHandle_t handle;
	opendnnTensorDescriptor_t bottom, top;
	opendnnFilterDescriptor_t weight;
	opendnnConvolutionDescriptor_t conv;
	float workspace;
	size_t workspace_byte;

	// Init
	opendnnCreate(&handle);
	opendnnCreateTensorDescriptor(&bottom); 
	opendnnCreateTensorDescriptor(&top);
	opendnnCreateFilterDescriptor(&weight);
	opendnnCreateConvolutionDescriptor(&conv);
	opendnnSetTensor4dDescriptor(bottom, 1, 1, 4, 4);
	opendnnSetTensor4dDescriptor(top, 1, 1, 3, 3);
	opendnnSetFilter4dDescriptor(weight, 1, 1, 2, 2);
	opendnnSetConvolution2dDescriptor(conv, 0, 0, 1, 1, 1, 1);
	opendnnSetConvolutionGroupCount(conv, 1); 
	const float b[16] = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
	const float w[4] = {1,0,0,1};
	float t[9] = {0,};
	
	// Test
	opendnnConvolutionForward(handle, bottom, b, weight, w, conv, &workspace, workspace_byte, top, t);
	for (int i = 0; i < 9; i++) {
		cout << t[i] << '\n';
	}
	cout << "Done" << endl;
	return 0;
}
