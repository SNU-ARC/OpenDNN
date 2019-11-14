const int BLOCK_SIZE = 24;

// im2col kernel
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* data_col) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < (n);
       index += blockDim.x * gridDim.x){
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

// Naive matrix multiply kernel
__global__ void naive_matmul( int M, int N, int K, 
    const float* A, const float* B, float* C )
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if( 0 <= i && i < N && 0 <= j && j < M)
    {
        float sum = 0.f;
        for(int k = 0 ; k  < K ; ++k)
        {
            sum += A[j*K + k] * B[k*N + i];
        }

        C[j*N + i] = sum;
    }
}

// Allocation and Unbox kernel
__global__ void allocDataUnbox(float* output, float* input, const int maxsize, const DataType type, const int bwTotal, const int bwInt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Number temp(type, bwTotal, bwInt);
    if (i < maxsize) {
        temp = input[i];
        output[i] = temp.asFloat();
    }
}

// Blocked (Tiled) version
__global__ void matmul_block(float* A, float* B, float* C,
                             int ARows, int ACols, int BRows,
                             int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < BLOCK_SIZE; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols){
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
}



// Blocked (Tiled) & linearized shared memory
__global__ void matmul_block_lin_shared(float* A, float* B, float* C,
                             int ARows, int ACols, int BRows,
                             int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
         else
             As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0;

         if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < BLOCK_SIZE; ++n)
             CValue += As[threadIdx.y*BLOCK_SIZE+n] * Bs[n*BLOCK_SIZE+threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols){
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
}

__global__ void matmul_block_lin_shared_trans(float* A, float* B, float* C,
                             int ARows, int ACols, int BRows,
                             int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
         else
             As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0;

         if (k*BLOCK_SIZE + threadIdx.y < BCols && Col < BRows)
             Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = B[Col*BCols + k*BLOCK_SIZE + threadIdx.y];
         else
             Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < BLOCK_SIZE; ++n)
             CValue += As[threadIdx.y*BLOCK_SIZE+n] * Bs[n*BLOCK_SIZE+threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols){
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
}

// Blocked (Tiled) & linearized shared memory with batch parallel
__global__ void matmul_block_lin_shared_batch(const float* A, const float* B, float* C,
                             int ARows, int ACols,
                             int BRows, int BCols,
                             int CRows, int CCols, int group)
{
    float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0f;

        if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = B[blockIdx.z*BCols*BRows*group + (k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0f;

        __syncthreads();

        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += (As[threadIdx.y*BLOCK_SIZE+n] * Bs[n*BLOCK_SIZE+threadIdx.x]);

        __syncthreads();
    }

    if (Row < CRows && Col < CCols){
        C[blockIdx.z*CCols*CRows*group + ((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
}


// A: MxK, B: KxN, C: MxN
__global__ void gemmStridedBatched(
           const int M, const int N, const int K,
           const float* A, const int ARows, const int nPerBatchA,
           const float* B, const int BRows, const int nPerBatchB,
                 float* C, const int CRows, const int nPerBatchC)
{
    float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    const int ACols = K;
    const int BCols = N;
    const int CCols = N;

    __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
        if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows) {
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x] =
              A[Row*ACols + k*BLOCK_SIZE + threadIdx.x]; // nPerBatchA will be zero, thus not use it
        } else {
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0f;
        }
        if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols){
            Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] =
              B[blockIdx.z*nPerBatchB + (k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        } else {
            Bs[threadIdx.y*BLOCK_SIZE+threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += As[threadIdx.y*BLOCK_SIZE+n] * Bs[n*BLOCK_SIZE+threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols){
        C[blockIdx.z*nPerBatchC + ((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
}

