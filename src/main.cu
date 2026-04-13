#include <iostream>
#include <string>
#include <cmath>
#include "image_io.hpp"
#include "cpu_pipeline.hpp"
#include "gpu_pipeline.cuh"
#include "timer.hpp"

double mae(const GrayImage& a, const GrayImage& b) {
    if (a.data.size() != b.data.size()) return -1.0;
    double s = 0.0;
    for (size_t i=0;i<a.data.size();++i) s += std::abs((int)a.data[i] - (int)b.data[i]);
    return s / a.data.size();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: ./cuda_image_app <input.pgm> [threshold]\n";
        return 1;
    }

    std::string input = argv[1];
    int threshold = (argc >= 3) ? std::stoi(argv[2]) : 120;

    GrayImage img;
    if (!loadPGM(input, img)) {
        std::cerr << "Error: no se pudo cargar la imagen PGM: " << input << "\n";
        return 1;
    }

    Timer t;
    t.start();
    GrayImage cpuOut = runCpuPipeline(img, threshold);
    double cpuMs = t.stopMs();

    t.start();
    GrayImage gpuOut = runGpuPipeline(img, threshold);
    double gpuMs = t.stopMs();

    savePGM("output/cpu_out.pgm", cpuOut);
    savePGM("output/gpu_out.pgm", gpuOut);

    double err = mae(cpuOut, gpuOut);
    std::cout << "CPU ms: " << cpuMs << "\n";
    std::cout << "GPU ms: " << gpuMs << "\n";
    std::cout << "Speedup: " << (cpuMs / gpuMs) << "x\n";
    std::cout << "MAE CPU vs GPU: " << err << "\n";
    std::cout << "Salida: output/cpu_out.pgm, output/gpu_out.pgm\n";
    return 0;
}