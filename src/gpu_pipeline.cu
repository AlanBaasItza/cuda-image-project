#include "gpu_pipeline.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void blurKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) return;
    int sum = 0;
    for (int ky=-1; ky<=1; ++ky)
        for (int kx=-1; kx<=1; ++kx)
            sum += in[(y+ky)*w + (x+kx)];
    out[y*w + x] = (unsigned char)(sum / 9);
}

__global__ void sobelThresholdKernel(const unsigned char* in, unsigned char* out, int w, int h, int thr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) return;

    int gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int sx=0, sy=0;

    for (int ky=-1; ky<=1; ++ky) {
        for (int kx=-1; kx<=1; ++kx) {
            int p = in[(y+ky)*w + (x+kx)];
            sx += p * gx[ky+1][kx+1];
            sy += p * gy[ky+1][kx+1];
        }
    }

    int mag2 = sx*sx + sy*sy;
    int thr2 = thr*thr;
    out[y*w + x] = (mag2 >= thr2) ? 255 : 0;
}

GrayImage runGpuPipeline(const GrayImage& in, int threshold) {
    GrayImage out{in.width, in.height, std::vector<unsigned char>(in.width*in.height, 0)};
    size_t bytes = in.data.size() * sizeof(unsigned char);

    unsigned char *d_in=nullptr, *d_blur=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_blur, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, in.data.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_blur, 0, bytes);
    cudaMemset(d_out, 0, bytes);

    dim3 block(16,16);
    dim3 grid((in.width + block.x - 1)/block.x, (in.height + block.y - 1)/block.y);

    blurKernel<<<grid, block>>>(d_in, d_blur, in.width, in.height);
    sobelThresholdKernel<<<grid, block>>>(d_blur, d_out, in.width, in.height, threshold);

    cudaDeviceSynchronize();
    cudaMemcpy(out.data.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_blur);
    cudaFree(d_out);
    return out;
}