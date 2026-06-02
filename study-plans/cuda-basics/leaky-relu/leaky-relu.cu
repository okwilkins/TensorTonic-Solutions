#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    const unsigned int idx{blockDim.x * blockIdx.x + threadIdx.x};

    if (idx >= N) {
        return;
    }

    // Don't use condtionals to not cause control divergence
    float pos{fmaxf(0.0, input[idx])};
    float neg{fminf(alpha * input[idx], 0.0)};
    output[idx] = pos + neg;
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}