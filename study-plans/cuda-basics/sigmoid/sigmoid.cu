#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    const unsigned int idx{blockDim.x * blockIdx.x + threadIdx.x};

    if (idx >= N) {
        return;
    }

    output[idx] = 1.0 / (1.0 + expf(-1.0 * input[idx]));
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}