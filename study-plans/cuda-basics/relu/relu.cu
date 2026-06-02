#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    const unsigned int idx{blockDim.x * blockIdx.x + threadIdx.x};

    if (idx >= N) {
        return;
    }

    output[idx] = fmaxf(0.0, input[idx]);
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}