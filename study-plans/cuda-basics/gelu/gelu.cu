#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    const unsigned int idx{blockDim.x * blockIdx.x + threadIdx.x};

    if (idx >= N) {
        return;
    }

    output[idx] = 0.5 * input[idx] * (1 + erff(input[idx] / sqrtf(2)));
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
