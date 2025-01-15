#include<fn.h>

/* Module layers */
//! Only useable for NDArrays on GPU

void fc_fwd(NDArray& input, // (N, C1)
    NDArray& weight, // (C1, C2)
    NDArray& bias, // (C2)
    NDArray& output) // (N, C2)
{
    shape_t shape_tmp({input.shape[0], 1}), shape_bias({1, bias.size});
    NDArray tmp(shape_tmp, 1.0f, input.device); // (N, 1)
    output = input * weight + tmp * bias.reshape(shape_bias);
}

void fc_bwd(NDArray& input, // (N, C1)
    NDArray& weight, // (C1, C2)
    NDArray& bias,  // (C2)
    NDArray& grad_output, // (N, C2)
    NDArray& grad_input,  // (N, C1)
    NDArray& grad_weight, // (C1, C2)
    NDArray& grad_bias) // (C2)
{
    grad_input = grad_output * weight.T2d();
    grad_weight = input.T2d() * grad_output;
    shape_t shape_tmp = {1, grad_output.shape[0]};
    NDArray tmp(shape_tmp, 1, grad_output.device);
    grad_bias = tmp * grad_output;
    grad_bias = grad_bias.reshape({grad_bias.size});
}

__global__ void im2col_kernel(
    scalar_t* img, // (N, C1, H, W)
    scalar_t* pad_img, // (N, C1, H+2, W+2)
    scalar_t* tgt, // (C1, 9, N, H, W)
    int N, int C, int H, int W
) {
    int n = blockIdx.x, 
        c = blockIdx.y, 
        h = threadIdx.x, // [0, H-1]
        w = threadIdx.y; // [0, W-1]
    pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+1) * (W+2) + w+1] = img[n * C * H * W + c * H * W + h * W + w];
    __syncthreads();
    tgt[c * 9 * N * H * W + 0 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + h * (W+2) + w];
    tgt[c * 9 * N * H * W + 1 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + h * (W+2) + w + 1];
    tgt[c * 9 * N * H * W + 2 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + h * (W+2) + w + 2];
    tgt[c * 9 * N * H * W + 3 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+1) * (W+2) + w];
    tgt[c * 9 * N * H * W + 4 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+1) * (W+2) + w + 1];
    tgt[c * 9 * N * H * W + 5 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+1) * (W+2) + w + 2];
    tgt[c * 9 * N * H * W + 6 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+2) * (W+2) + w];
    tgt[c * 9 * N * H * W + 7 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+2) * (W+2) + w + 1];
    tgt[c * 9 * N * H * W + 8 * N * H * W + n * H * W + h * W + w] = pad_img[n * C * (H+2) * (W+2) + c * (H+2) * (W+2) + (h+2) * (W+2) + w + 2];
}

NDArray& im2col(    //return (C1*3*3, N*H*W)
    NDArray& img  // (N, C1, H, W)
) {
    int N = img.shape[0], C1 = img.shape[1], H = img.shape[2], W = img.shape[3];
    NDArray pad_img({N, C1, H+2, W+2}, 0, img.device); // zero padding
    NDArray* tgt = new NDArray({C1, 9, N, H, W}, 0, img.device);
    im2col_kernel<<<dim3(N, C1), dim3(H, W)>>>(img.p, pad_img.p, tgt->p, N, C1, H, W);
    cudaDeviceSynchronize();
    return tgt->reshape({C1*9, N*H*W});
}

void conv2d_k33p1s1_fwd(
    NDArray& input, // (N, C1, H, W)
    NDArray& kernel, // (C2, C1, 3, 3)
    NDArray& bias, // (C2)
    NDArray& output // (N, C2, H, W)
) {
    int N = input.shape[0], C1 = input.shape[1], H = input.shape[2], W = input.shape[3];
    int C2 = kernel.shape[0];
    NDArray col = im2col(input); // (C1*3*3, N*H*W)
    NDArray tmp({1, N*H*W}, 1, input.device);
    output = kernel.reshape({C2, C1*9}) * col + bias.reshape({C2, 1}) * tmp; // (C2, N*H*W)
    output = output.reshape({C2, N, H, W}).swap(0, 1);
}

__global__ void col2im_kernel(
    scalar_t* grad_col, // (C1*3*3, N*H*W)
    scalar_t* pad_grad_col, // (C1*3*3, N*(H+2)*(W+2))
    scalar_t* tgt, // (N, C1, H, W)
    int N, int C, int H, int W
) {
    int n = blockIdx.x, 
        c = blockIdx.y, 
        h = threadIdx.x, // [0, H-1]
        w = threadIdx.y; // [0, W-1]
    for (int i = 0; i < 9; i++) {
        pad_grad_col[c * 9 * N * (H+2) * (W+2) + i * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+1) * (W+2) + w+1] = grad_col[c * 9 * N * H * W + i * N * H * W + n * H * W + h * W + w];
    }
    __syncthreads();
    tgt[n * C * H * W + c * H * W + h * W + w] = 
        pad_grad_col[c * 9 * N * (H+2) * (W+2) + 0 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+2) * (W+2) + w+2]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 1 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+2) * (W+2) + w+1]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 2 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+2) * (W+2) + w]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 3 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+1) * (W+2) + w+2]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 4 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+1) * (W+2) + w+1]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 5 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h+1) * (W+2) + w]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 6 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h) * (W+2) + w+2]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 7 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h) * (W+2) + w+1]
        + pad_grad_col[c * 9 * N * (H+2) * (W+2) + 8 * N * (H+2) * (W+2) + n * (H+2) * (W+2) + (h) * (W+2) + w];
}

NDArray& col2im(NDArray& grad_col, // (C1*3*3, N*H*W)
    int N, int C1, int H, int W) {
    NDArray* dst = new NDArray({N, C1, H, W}, 0, grad_col.device);
    NDArray* pad_grad_col = new NDArray({C1*9, N*(H+2)*(W+2)}, 0, grad_col.device);
    col2im_kernel<<<dim3(N, C1), dim3(H, W)>>>(grad_col.p, pad_grad_col->p, dst->p, N, C1, H, W);
    cudaDeviceSynchronize();
    return *dst;
}

void conv2d_k33p1s1_bwd(
    NDArray& input, // (N, C1, H, W)
    NDArray& kernel, // (C2, C1, 3, 3)
    NDArray& grad_output,// (N, C2, H, W)
    NDArray& grad_input, // (N, C1, H, W)
    NDArray& grad_kernel, // (C2, C1, 3, 3)
    NDArray& grad_bias) // (C2)
{
    NDArray col = im2col(input); // (C1*3*3, N*H*W)
    int N = input.shape[0], C1 = input.shape[1], H = input.shape[2], W = input.shape[3];
    int C2 = grad_output.shape[1];
    NDArray grad_col = kernel.reshape({C2, C1*9}).T2d() * grad_output.swap(0, 1).reshape({C2, N*H*W}); // (C1*3*3, N*H*W)
    grad_input = col2im(grad_col, N, C1, H, W);
    grad_kernel = grad_output.swap(0, 1).reshape({C2, N*H*W}) * col.T2d(); // (C2, C1*3*3)
    grad_kernel = grad_kernel.reshape({C2, C1, 3, 3});
    NDArray tmp2({N*H*W, 1}, 1, input.device);
    grad_bias = grad_output.swap(0, 1).reshape({C2, N*H*W}) * tmp2; // (C2, 1)
    grad_bias = grad_bias.reshape({C2});
}

__global__ void maxpool2d_k22s2_fwd_kernel(
    scalar_t* i, 
    scalar_t* o, 
    int N, int C, int H, int W) 
{
    int n = blockIdx.x, c = blockIdx.y, h = threadIdx.x, w = threadIdx.y;
    int o_idx = n * C * H/2 * W/2 + c * H/2 * W/2 + h * W/2 + w;
    int i_idx = n * C * H * W + c * H * W + 2 * h * W + 2 * w;
    o[o_idx] = fmax(fmax(i[i_idx], i[i_idx + 1]), fmax(i[i_idx + W], i[i_idx + W + 1]));
}

void maxpool2d_k22s2_fwd(
    NDArray& input, // (N, C, H, W)
    NDArray& output // (N, C, H//2, W//2)
) {
    int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
    assert(2*output.shape[2] == H && 2*output.shape[3] == W);
    maxpool2d_k22s2_fwd_kernel<<<dim3(N, C), dim3(H/2, W/2)>>>(input.p, output.p, N, C, H, W);
}


__global__ void maxpool2d_k22s2_bwd_kernel(
    scalar_t* i, 
    scalar_t* o,
    scalar_t* grad_o,
    scalar_t* grad_i, 
    int N, int C, int H, int W) 
{
    int n = blockIdx.x, c = blockIdx.y, h = threadIdx.x, w = threadIdx.y;
    int o_idx = n * C * H/2 * W/2 + c * H/2 * W/2 + h * W/2 + w;
    int i_idx = n * C * H * W + c * H * W + 2 * h * W + 2 * w;
    grad_i[i_idx] = (i[i_idx] == o[o_idx]) * grad_o[o_idx];
    grad_i[i_idx + 1] = (i[i_idx + 1] == o[o_idx]) * grad_o[o_idx];
    grad_i[i_idx + W] = (i[i_idx + W] == o[o_idx]) * grad_o[o_idx];
    grad_i[i_idx + W + 1] = (i[i_idx + W + 1] == o[o_idx]) * grad_o[o_idx];
}

void maxpool2d_k22s2_bwd(
    NDArray& input, // (N, C, H, W)
    NDArray& output, // (N, C, H//2, W//2)
    NDArray& grad_output, // (N, C, H//2, W//2)
    NDArray& grad_input // (N, C, H, W)
) {
    thrust::device_ptr<scalar_t> grad_input_ptr(grad_input.p);
    thrust::fill(grad_input_ptr, grad_input_ptr + grad_input.size, 0);
    int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
    maxpool2d_k22s2_bwd_kernel<<<dim3(N, C), dim3(H/2, W/2)>>>(input.p, output.p, grad_output.p, grad_input.p, N, C, H, W);
}

__global__ void row_max_kernel(
    scalar_t* input, // (m, n)
    scalar_t* output, // (m, n) keep dims
    int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        scalar_t max_val = input[idx * n];
        for (int i = 1; i < n; i++) {
            max_val = fmaxf(max_val, input[idx * n + i]);
        }
        for (int i = 0; i < n; i++) {
            output[idx * n + i] = max_val;
        }
    }
}

__global__ void row_sum_kernel(
    scalar_t* input, // (m, n)
    scalar_t* output, // (m, n) keep dims
    int m, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        scalar_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum += input[idx * n + i];
        }
        for (int i = 0; i < n; i++) {
            output[idx * n + i] = sum;
        }
    }
}

void softmax_fwd(NDArray& input, NDArray& output) {
    row_max_kernel<<<(input.shape[0] + 255) / 256, 256>>>(input.p, output.p, input.shape[0], input.shape[1]);
    NDArray i = input - output; // for numerical stability
    thrust::device_ptr<scalar_t> i_ptr(i.p);
    thrust::for_each(i_ptr, i_ptr + i.size, [] __device__ (scalar_t& x) { x = expf(x); });
    NDArray si(i);
    row_sum_kernel<<<(input.shape[0] + 255) / 256, 256>>>(i.p, si.p, input.shape[0], input.shape[1]);
    output = i / si;
}


__global__ void celoss_kernel(
    scalar_t* prob, // (N, C)
    scalar_t* tgt,  // (N)
    scalar_t* losses, // (N)
    int N, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        losses[idx] = -logf(prob[idx * C + (int)tgt[idx]]);
    }

}

NDArray& celoss_fwd(
    NDArray& prob, // (N, C)
    NDArray& tgt   // (N)
) {
    scalar_t* losses;
    cudaMalloc(&losses, prob.shape[0] * sizeof(scalar_t));
    celoss_kernel<<<(prob.shape[0] + 255) / 256, 256>>>(prob.p, tgt.p, losses, prob.shape[0], prob.shape[1]);
    cudaDeviceSynchronize();
    thrust::device_ptr<scalar_t> losses_ptr(losses);
    scalar_t loss = thrust::reduce(losses_ptr, losses_ptr + prob.shape[0]) / prob.shape[0];
    cudaFree(losses);
    NDArray* loss_ptr = new NDArray({1}, loss, prob.device);
    return *loss_ptr;
}

__global__ void softmax_ce_bwd_kernel(
    scalar_t* prob, // (N, C)
    scalar_t* tgt,  // (N)
    scalar_t grad_loss,
    scalar_t* grad_input, // (N, C)
    int N, int C
) {
    int i = blockIdx.x, j = threadIdx.x;
    grad_input[i * C + j] = prob[i * C + j] - (j == (int)tgt[i]);
    grad_input[i * C + j] *= grad_loss / N;
}

void softmax_ce_bwd(
    NDArray& prob, // (N, C)
    NDArray& tgt,  // (N)
    NDArray& grad_loss, // (1)
    NDArray& grad_input // (N, C)
) {
    
    scalar_t grad_loss_val = grad_loss.tolist()[0];
    softmax_ce_bwd_kernel<<<prob.shape[0], prob.shape[1]>>>(
        prob.p, tgt.p, grad_loss_val, grad_input.p, prob.shape[0], prob.shape[1]);
}

void relu_fwd(
    NDArray& input, // (N, C)
    NDArray& output // (N, C)
) {
    thrust::device_ptr<scalar_t> input_ptr(input.p), output_ptr(output.p);
    thrust::transform(input_ptr, input_ptr + input.size, output_ptr, [] __device__ (scalar_t x) { return fmaxf(0, x); });
}

void relu_bwd(
    NDArray& input, // (N, C)
    NDArray& grad_output, // (N, C)
    NDArray& grad_input // (N, C)
) {
    thrust::device_ptr<scalar_t> input_ptr(input.p), grad_output_ptr(grad_output.p), grad_input_ptr(grad_input.p);
    thrust::transform(input_ptr, input_ptr + input.size, grad_output_ptr, grad_input_ptr, [] __device__ (scalar_t x, scalar_t y) { return x >= 0 ? y : 0; });
}

void sigmoid_fwd(
    NDArray& input, // (N, C)
    NDArray& output // (N, C)
) {
    thrust::device_ptr<scalar_t> input_ptr(input.p), output_ptr(output.p);
    thrust::transform(input_ptr, input_ptr + input.size, output_ptr, [] __device__ (scalar_t x) { return 1 / (1 + expf(-x)); });
}

void sigmoid_bwd(
    NDArray& input, // (N, C)
    NDArray& grad_output, // (N, C)
    NDArray& grad_input // (N, C)
) {
    thrust::device_ptr<scalar_t> input_ptr(input.p), grad_output_ptr(grad_output.p), grad_input_ptr(grad_input.p);
    thrust::transform(input_ptr, input_ptr + input.size, grad_output_ptr, grad_input_ptr, [] __device__ (scalar_t x, scalar_t y) { return y  / (1+expf(-x)) * (1-(1/(1+expf(-x)))); });
}