#include "cpu_pipeline.hpp"
// =============================================================================
// cpu_pipeline.cpp  —  Implementación secuencial CPU
//
// Etapas:
//   1. Blur 3×3  (promedio simple, kernel de caja)
//   2. Sobel     (gradiente en X e Y, magnitud euclidiana)
//   3. Threshold (umbralización binaria: píxel ≥ umbral → 255, si no → 0)
//
// Devuelve PipelineResult con las tres etapas intermedias guardadas.
// =============================================================================
#include <cmath>
#include <algorithm>
#include <vector>

// Calcula el índice lineal de un píxel (x,y) en una imagen de ancho w
static inline int idx(int x, int y, int w) { return y * w + x; }

PipelineResult runCpuPipeline(const GrayImage& in, int threshold) {
    const int W = in.width, H = in.height;
    const size_t N = static_cast<size_t>(W) * H;

    PipelineResult res;
    res.blur      = {W, H, std::vector<unsigned char>(N, 0)};
    res.sobel     = {W, H, std::vector<unsigned char>(N, 0)};
    res.threshold = {W, H, std::vector<unsigned char>(N, 0)};

    // ---- Etapa 1: Blur 3×3 (promedio de los 9 vecinos) ---------------------
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    sum += in.data[idx(x + kx, y + ky, W)];
            res.blur.data[idx(x, y, W)] = static_cast<unsigned char>(sum / 9);
        }
    }

    // ---- Etapa 2: Sobel (detección de bordes) -------------------------------
    // Máscaras de Sobel en X e Y
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int sx = 0, sy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int p = res.blur.data[idx(x + kx, y + ky, W)];
                    sx += p * Gx[ky + 1][kx + 1];
                    sy += p * Gy[ky + 1][kx + 1];
                }
            }
            int mag = static_cast<int>(std::sqrt(float(sx * sx + sy * sy)));
            res.sobel.data[idx(x, y, W)] =
                static_cast<unsigned char>(std::min(255, mag));
        }
    }

    // ---- Etapa 3: Umbralización binaria -------------------------------------
    for (size_t i = 0; i < N; ++i)
        res.threshold.data[i] = (res.sobel.data[i] >= threshold) ? 255 : 0;

    return res;
}
