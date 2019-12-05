__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void im2col_gpu_kernel(const int n , global const float *data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    global float *data_col) {
    int index = get_global_id(0);
    if (index >= n) return;
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    __global float * data_col_ptr;
    data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    __global const float * data_im_ptr = &data_im[0];
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            int h_im = h_offset + i * dilation_h;
            int w_im = w_offset + j * dilation_w;
            *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
            data_col_ptr += height_col * width_col;
        }
    }
}

__kernel
void matmul_block_lin_shared(
    global const float *A, global const float *B, global float *C,
    const int ARows, const int ACols,
    const int BRows, const int BCols,
    const int CRows, const int CCols) {

    float CValue = 0;
    int thx = get_local_id(0);
    int thy = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int dimx = get_local_size(0);
    int dimy = get_local_size(1);

    int Row = by*32 + thy;
    int Col = bx*32 + thx;

    local float As[32*32];
    local float Bs[32*32];

    for (int k = 0; k < (ACols + 31)/32; k++) {
        if (k*32 + thx < ACols && Row < ARows)
            As[thy*32 + thx] = A[Row*ACols + k*32 + thx];
        else
            As[thy*32 + thx] = 0.0f;

        if (k*32 + thy < BRows && Col < BCols)
            Bs[thy*32 + thx] = B[Col + (k*32 + thy)*BCols];
        else
            Bs[thy*32 + thx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int n = 0; n < 32; ++n)
            CValue += (As[thy*32 + n] * Bs[n*32 + thx]);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (Row < CRows && Col < CCols) {
        C[get_global_id(1)*CCols + get_global_id(0)] = CValue;
    }
}

__kernel __attribute__((reqd_work_group_size(32, 32, 1)))
void matmul_block_lin_shared_trans(
    global const float *A, global const float *B, global float *C,
    const int ARows, const int ACols,
    const int BRows, const int BCols,
    const int CRows, const int CCols) {

    float CValue = 0;
    int thx = get_local_id(0);
    int thy = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int dimx = get_local_size(0);
    int dimy = get_local_size(1);

    int Row = by*32 + thy;
    int Col = bx*32 + thx;

    local float As[32*32];
    local float Bs[32*32];

    for (int k = 0; k < (ACols + 31)/32; k++) {
        if (k*32 + thx < ACols && Row < ARows)
            As[thy*32 + thx] = A[Row*ACols + k*32 + thx];
        else
            As[thy*32 + thx] = 0.0f;

        if (k*32 + thy < BCols && Col < BRows)
            Bs[thy*32 + thx] = B[BCols*Col + k*32 + thy];
        else
            Bs[thy*32 + thx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int n = 0; n < 32; ++n)
            CValue += (As[thy*32 + n] * Bs[n*32 + thx]);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (Row < CRows && Col < CCols) {
        C[get_global_id(1)*CCols + get_global_id(0)] = CValue;
    }
}

__kernel __attribute__((reqd_work_group_size(32, 32, 1)))
void matmul_block_lin_shared_batch(
    global const float *A, global const float *B, global float *C,
    const int ARows, const int ACols,
    const int BRows, const int BCols,
    const int CRows, const int CCols, const int group) {

    float CValue = 0;
    int thx = get_local_id(0);
    int thy = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int bz = get_group_id(2);
    int dimx = get_local_size(0);
    int dimy = get_local_size(1);

    int Row = by*32 + thy;
    int Col = bx*32 + thx;

    local float As[32*32];
    local float Bs[32*32];

    for (int k = 0; k < (ACols + 31)/32; k++) {
        if (k*32 + thx < ACols && Row < ARows)
            As[thy*32 + thx] = A[Row*ACols + k*32 + thx];
        else
            As[thy*32 + thx] = 0.0f;

        if (k*32 + thy < BRows && Col < BCols)
            Bs[thy*32 + thx] = B[bz*BCols*BRows*group + (k*32 + thy)*BCols + Col];
        else
            Bs[thy*32 + thx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int n = 0; n < 32; ++n)
            CValue += (As[thy*32 + n] * Bs[n*32 + thx]);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (Row < CRows && Col < CCols) {
        C[bz*CCols*CRows*group + (get_global_id(1)*CCols) + get_global_id(0)] = CValue;
    }
}

__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void ReLU (global const float *input, global float *output, const int N) {
    int i = get_global_id(0);
    if (i < N) output[i] = fmax(input[i], 0.f);
}

__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void sigmoid (global const float *input, global float *output, const int N) {
    int i = get_global_id(0);
    if (i < N) output[i] = 1. / (1. + exp(-input[i]));
}

__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void htan (global const float *input, global float *output, const int N) {
    int i = get_global_id(0);
    if (i < N) output[i] = tanh(input[i]);
}

__kernel __attribute__((reqd_work_group_size(32, 32, 1)))
void normCross (global const float *input, global float *output, const int A) {

}

__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void MaxPoolKernel (const int nthreads,
    const global float* bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    global float *top_data) {
    for (int i = get_global_id(0); i < nthreads; i += get_global_size(0)) {
        const int pw = i % pooled_width;
        const int ph = (i / pooled_width) % pooled_height;
        const int c = (i / pooled_width / pooled_height) % channels;
        const int n = i / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float maxval = -FLT_MAX;
        int maxidx = -1;
        const global float * bottom_slice = bottom_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (bottom_slice[h * width + w] > maxval) {
                    maxidx = h * width + w;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        top_data[i] = maxval;
        // if (mask) 
        //     mask[i] = maxidx;
        // else
        //     top_mask[i] = maxidx;
    }
}

__kernel __attribute__((reqd_work_group_size(1024, 1, 1)))
void AvgPoolKernel (const int nthreads,
    const global float* bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    global float *top_data) {
    for (int i = get_global_id(0); i < nthreads; i += get_global_size(0)) {
        const int pw = i % pooled_width;
        const int ph = (i / pooled_width) % pooled_height;
        const int c = (i / pooled_width / pooled_height) % channels;
        const int n = i / pooled_width / pooled_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        float aveval = 0;
        const global float* bottom_slice = bottom_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottom_slice[h * width + w];
            }
        }
        top_data[i] = aveval / pool_size;
    }
}

__kernel
void kernel_channel_max (const int num, const int channels,
    const int spatial_dim, global const float* data, global float* out) {
    for (int i = get_global_id(0); i < num * spatial_dim; i += get_global_size(0)) {
        int n = i / spatial_dim;
        int s = i % spatial_dim;
        float maxval = -FLT_MAX;
        for (int c = 0; c < channels; ++c) {
            maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
        }
        out[i] = maxval;
    }
}

__kernel
void kernel_channel_subtract (const int count,
    const int num, const int channels,
    const int spatial_dim, global const float* channel_max, global float* data) {
    for (int i = get_global_id(0); i < count; i += get_global_size(0)) {
        int n = i / channels / spatial_dim;
        int s = i % spatial_dim;
        data[i] -= channel_max[n * spatial_dim +s];
    }
}

__kernel
void kernel_exp (const int count, global const float* data, global float* out) {
    int i = get_global_id(0);
    if (i < count)
        out[i] = exp(data[i]);
}

__kernel
void kernel_channel_sum (const int num, const int channels,
    const int spatial_dim, global const float* data, global float* out) {
    int i = get_global_id(0);
    if (i < num * spatial_dim) {
        int n = i / spatial_dim;
        int s = i % spatial_dim;
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += data[(n * channels + c) * spatial_dim + s];
        }
        out[i] = sum;
    }
}

__kernel
void kernel_channel_div (const int count, const int num, const int channels,
    const int spatial_dim, global const float* channel_sum, global float* data) {
    int i = get_global_id(0);
    if (i < count) {
        int n = i / channels / spatial_dim;
        int s = i % spatial_dim;
        float v = data[i] / channel_sum[n * spatial_dim + s];
        data[i] = v;
    }
}

__kernel
void normalization () {

}
