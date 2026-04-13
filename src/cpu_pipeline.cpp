#include "cpu_pipeline.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

static inline int idx(int x, int y, int w) { return y * w + x; }

GrayImage runCpuPipeline(const GrayImage& in, int threshold) {
    GrayImage blur{in.width, in.height, std::vector<unsigned char>(in.width * in.height, 0)};
    GrayImage sobel{in.width, in.height, std::vector<unsigned char>(in.width * in.height, 0)};
    GrayImage out{in.width, in.height, std::vector<unsigned char>(in.width * in.height, 0)};

    // Blur 3x3 (promedio simple)
    for (int y = 1; y < in.height - 1; ++y) {
        for (int x = 1; x < in.width - 1; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    sum += in.data[idx(x + kx, y + ky, in.width)];
            blur.data[idx(x, y, in.width)] = static_cast<unsigned char>(sum / 9);
        }
    }

    // Sobel
    int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    for (int y = 1; y < in.height - 1; ++y) {
        for (int x = 1; x < in.width - 1; ++x) {
            int sx = 0, sy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int p = blur.data[idx(x + kx, y + ky, in.width)];
                    sx += p * Gx[ky + 1][kx + 1];
                    sy += p * Gy[ky + 1][kx + 1];
                }
            }
            int mag = static_cast<int>(std::sqrt(float(sx*sx + sy*sy)));
            sobel.data[idx(x, y, in.width)] = static_cast<unsigned char>(std::min(255, mag));
        }
    }

    // Umbral
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = (sobel.data[i] >= threshold) ? 255 : 0;

    return out;
}