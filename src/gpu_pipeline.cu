#include "gpu_pipeline.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

// =============================================================================
// gpu_pipeline.cu
// Pipeline de procesamiento de imagen en GPU usando CUDA.
// Etapas: Blur 3x3 -> Sobel (deteccion de bordes) -> Umbral binario
// Cada kernel usa shared memory para reducir accesos a memoria global.
// =============================================================================

#define BLOCK_SIZE 16
#define TILE_W  (BLOCK_SIZE + 2)
#define TILE_H  (BLOCK_SIZE + 2)

// -----------------------------------------------------------------------------
// warmupGpu: fuerza la inicializacion del contexto CUDA antes de medir tiempos.
// La primera llamada a cualquier funcion CUDA tarda ~200ms extra por el setup
// del driver. Se llama una sola vez al inicio para que no afecte las mediciones.
// -----------------------------------------------------------------------------
static void warmupGpu() {
    static bool done = false;
    if (done) return;
    unsigned char* tmp = nullptr;
    cudaMalloc(&tmp, 1);
    cudaFree(tmp);
    cudaDeviceSynchronize();
    done = true;
}

// -----------------------------------------------------------------------------
// Macro para verificar errores en llamadas CUDA.
// Imprime archivo y linea donde ocurrio el error.
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// blurKernelShared
// Aplica un filtro de suavizado 3x3 (promedio de 9 vecinos).
// Cada bloque carga un tile de (BLOCK_SIZE+2)^2 pixeles en shared memory,
// incluyendo el halo de 1 pixel en cada borde, para evitar lecturas repetidas
// de memoria global.
// -----------------------------------------------------------------------------
__global__ void blurKernelShared(
    const unsigned char* __restrict__ in,
    unsigned char*       __restrict__ out,
    int w, int h)
{
    __shared__ unsigned char smem[TILE_H][TILE_W];

    const int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Cada hilo carga el pixel correspondiente al tile con halo (offset -1,-1)
    const int lx = (int)gx - 1;
    const int ly = (int)gy - 1;
    int cx = max(0, min(w - 1, lx));
    int cy = max(0, min(h - 1, ly));
    smem[threadIdx.y][threadIdx.x] = in[cy * w + cx];
    __syncthreads();

    // Solo los hilos del interior del tile (sin halo) producen salida
    if (threadIdx.x == 0 || threadIdx.y == 0) return;
    if (threadIdx.x >= TILE_W - 1 || threadIdx.y >= TILE_H - 1) return;
    if (gx - 1 <= 0 || gy - 1 <= 0 || gx - 1 >= w - 1 || gy - 1 >= h - 1) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = gx - 1;
    const int oy = gy - 1;

    int sum = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            sum += smem[ty + dy][tx + dx];
    out[oy * w + ox] = (unsigned char)(sum / 9);
}

// -----------------------------------------------------------------------------
// sobelKernelShared
// Calcula la magnitud del gradiente usando las mascaras de Sobel (Gx, Gy).
// Usa shared memory con halo igual que blurKernelShared.
// La salida es la magnitud normalizada a [0, 255], sin umbralizar,
// para poder guardar esta etapa intermedia.
// -----------------------------------------------------------------------------
__global__ void sobelKernelShared(
    const unsigned char* __restrict__ in,
    unsigned char*       __restrict__ out,
    int w, int h)
{
    __shared__ unsigned char smem[TILE_H][TILE_W];

    const int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int lx = (int)gx - 1;
    const int ly = (int)gy - 1;
    int cx = max(0, min(w - 1, lx));
    int cy = max(0, min(h - 1, ly));
    smem[threadIdx.y][threadIdx.x] = in[cy * w + cx];
    __syncthreads();

    if (threadIdx.x == 0 || threadIdx.y == 0) return;
    if (threadIdx.x >= TILE_W - 1 || threadIdx.y >= TILE_H - 1) return;
    if (gx - 1 <= 0 || gy - 1 <= 0 || gx - 1 >= w - 1 || gy - 1 >= h - 1) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = gx - 1;
    const int oy = gy - 1;

    // Mascaras de Sobel para gradiente horizontal (GX) y vertical (GY)
    const int GX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int GY[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    int sx = 0, sy = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int p = smem[ty + dy][tx + dx];
            sx += p * GX[dy + 1][dx + 1];
            sy += p * GY[dy + 1][dx + 1];
        }
    }
    int mag = __float2int_rn(sqrtf((float)(sx * sx + sy * sy)));
    out[oy * w + ox] = (unsigned char)min(255, mag);
}

// -----------------------------------------------------------------------------
// thresholdKernel
// Umbral binario: pixeles con magnitud >= thr se marcan como 255 (borde),
// el resto como 0. Kernel 1D simple, no requiere shared memory.
// -----------------------------------------------------------------------------
__global__ void thresholdKernel(
    const unsigned char* __restrict__ in,
    unsigned char*       __restrict__ out,
    int n, int thr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (in[i] >= thr) ? 255 : 0;
}

// -----------------------------------------------------------------------------
// runGpuPipeline
// Ejecuta los tres kernels en secuencia y devuelve las tres etapas intermedias.
// Llama a warmupGpu() antes de iniciar para que el tiempo medido sea del
// computo real, sin incluir la inicializacion del contexto CUDA.
// -----------------------------------------------------------------------------
PipelineResult runGpuPipeline(const GrayImage& in, int threshold) {
    warmupGpu();

    const int W = in.width, H = in.height;
    const int N = W * H;
    const size_t bytes = N * sizeof(unsigned char);

    PipelineResult res;
    res.blur      = {W, H, std::vector<unsigned char>(N, 0)};
    res.sobel     = {W, H, std::vector<unsigned char>(N, 0)};
    res.threshold = {W, H, std::vector<unsigned char>(N, 0)};

    // Reservar buffers en GPU
    unsigned char *d_in = nullptr, *d_blur = nullptr,
                  *d_sobel = nullptr, *d_thr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,    bytes));
    CUDA_CHECK(cudaMalloc(&d_blur,  bytes));
    CUDA_CHECK(cudaMalloc(&d_sobel, bytes));
    CUDA_CHECK(cudaMalloc(&d_thr,   bytes));

    CUDA_CHECK(cudaMemcpy(d_in, in.data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_blur,  0, bytes));
    CUDA_CHECK(cudaMemset(d_sobel, 0, bytes));
    CUDA_CHECK(cudaMemset(d_thr,   0, bytes));

    // Configuracion de grid para kernels 2D (blur y sobel)
    dim3 blockTile(TILE_W, TILE_H);
    dim3 gridTile((W + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Configuracion de grid para kernel 1D (threshold)
    const int BLOCK1D = 256;
    dim3 grid1D((N + BLOCK1D - 1) / BLOCK1D);

    // Lanzar kernels en secuencia
    blurKernelShared  <<<gridTile, blockTile>>>(d_in,    d_blur,  W, H);
    sobelKernelShared <<<gridTile, blockTile>>>(d_blur,  d_sobel, W, H);
    thresholdKernel   <<<grid1D,   BLOCK1D  >>>(d_sobel, d_thr,   N, threshold);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copiar resultados al host
    CUDA_CHECK(cudaMemcpy(res.blur.data.data(),      d_blur,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.sobel.data.data(),     d_sobel, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res.threshold.data.data(), d_thr,   bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_blur);
    cudaFree(d_sobel);
    cudaFree(d_thr);

    return res;
}
