#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <algorithm>

// =============================================================================
// main.cu
// Punto de entrada de la aplicacion.
//
// Modos de uso:
//   Imagen unica:  ./cuda_image_app <input.pgm> [threshold]
//   Batch:         ./cuda_image_app --batch <directorio> [threshold]
//
// Por cada imagen procesada se generan 6 archivos en output/:
//   <nombre>_cpu_blur.pgm   <nombre>_gpu_blur.pgm
//   <nombre>_cpu_sobel.pgm  <nombre>_gpu_sobel.pgm
//   <nombre>_cpu_final.pgm  <nombre>_gpu_final.pgm
//
// Al finalizar se imprime una tabla comparativa CPU vs GPU.
// =============================================================================

#include "image_io.hpp"
#include "cpu_pipeline.hpp"
#include "gpu_pipeline.cuh"
#include "timer.hpp"

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// mae: calcula el Error Absoluto Medio entre dos imagenes del mismo tamano.
// Retorna -1 si los tamanos no coinciden.
// -----------------------------------------------------------------------------
static double mae(const GrayImage& a, const GrayImage& b) {
    if (a.data.size() != b.data.size()) return -1.0;
    double s = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        s += std::abs((int)a.data[i] - (int)b.data[i]);
    return s / a.data.size();
}

// Retorna el nombre del archivo sin extension a partir de una ruta
static std::string stem(const std::string& path) {
    return fs::path(path).stem().string();
}

// Imprime una barra de progreso en la misma linea: [=====>    ] N/total
static void printProgress(int current, int total) {
    const int W = 30;
    int filled = (total > 0) ? (current * W / total) : 0;
    std::cout << "\r  [";
    for (int i = 0; i < W; ++i)
        std::cout << (i < filled ? '=' : (i == filled ? '>' : ' '));
    std::cout << "] " << current << "/" << total << std::flush;
}

// -----------------------------------------------------------------------------
// BatchEntry: almacena las metricas de rendimiento de una imagen procesada.
// -----------------------------------------------------------------------------
struct BatchEntry {
    std::string name;
    int width, height;
    double cpuMs, gpuMs, speedup, maeVal;
};

// -----------------------------------------------------------------------------
// processImage: ejecuta ambos pipelines sobre una imagen y guarda los
// resultados intermedios en outDir. Devuelve las metricas de rendimiento.
// -----------------------------------------------------------------------------
static BatchEntry processImage(
    const std::string& inputPath,
    const std::string& outDir,
    int threshold)
{
    GrayImage img;
    if (!loadPGM(inputPath, img)) {
        std::cerr << "\n  [ERROR] Could not load: " << inputPath << "\n";
        return {inputPath, 0, 0, -1, -1, -1, -1};
    }

    fs::create_directories(outDir);

    // Pipeline CPU
    Timer t;
    t.start();
    PipelineResult cpuRes = runCpuPipeline(img, threshold);
    double cpuMs = t.stopMs();
    cpuRes.timeMs = cpuMs;

    // Pipeline GPU (warmup ya fue llamado dentro de runGpuPipeline)
    t.start();
    PipelineResult gpuRes = runGpuPipeline(img, threshold);
    double gpuMs = t.stopMs();
    gpuRes.timeMs = gpuMs;

    // Guardar las tres etapas intermedias de cada pipeline
    const std::string base = outDir + "/" + stem(inputPath);
    savePGM(base + "_cpu_blur.pgm",  cpuRes.blur);
    savePGM(base + "_cpu_sobel.pgm", cpuRes.sobel);
    savePGM(base + "_cpu_final.pgm", cpuRes.threshold);
    savePGM(base + "_gpu_blur.pgm",  gpuRes.blur);
    savePGM(base + "_gpu_sobel.pgm", gpuRes.sobel);
    savePGM(base + "_gpu_final.pgm", gpuRes.threshold);

    double maeVal  = mae(cpuRes.threshold, gpuRes.threshold);
    double speedup = (gpuMs > 0.0) ? (cpuMs / gpuMs) : 0.0;

    return {stem(inputPath), img.width, img.height,
            cpuMs, gpuMs, speedup, maeVal};
}

// -----------------------------------------------------------------------------
// printReport: imprime la tabla comparativa de rendimiento CPU vs GPU.
// Muestra tiempos, speedup con color ANSI y barra proporcional, y MAE.
// -----------------------------------------------------------------------------
static void printReport(const std::vector<BatchEntry>& results) {
    const int cName = 20, cDim = 12, cMs = 10, cSpd = 9, cMae = 8;
    const int totalW = cName + cDim + cMs + cMs + cSpd + cMae + 7;

    auto hline = [&](char c = '-') {
        std::cout << "  " << std::string(totalW, c) << "\n";
    };
    auto col = [](int w, const std::string& s) {
        std::cout << std::left << std::setw(w) << s;
    };

    std::cout << "\n";
    hline('=');
    std::cout << "  PERFORMANCE REPORT -- CPU vs GPU (CUDA Shared Memory)\n";
    hline('=');

    std::cout << "  ";
    col(cName, "Image");
    col(cDim,  "Size");
    col(cMs,   "CPU (ms)");
    col(cMs,   "GPU (ms)");
    col(cSpd,  "Speedup");
    col(cMae,  "MAE");
    std::cout << "\n";
    hline('-');

    double totalCpu = 0, totalGpu = 0;
    double maxSpeedup = 0, minSpeedup = 1e9;

    for (const auto& r : results) {
        if (r.cpuMs < 0) continue;

        std::string dims = std::to_string(r.width) + "x" + std::to_string(r.height);

        // Barra de asteriscos proporcional al speedup (max 20 caracteres)
        int bars = (r.speedup > 0 && r.speedup < 50)
            ? static_cast<int>(r.speedup * 2) : 0;
        std::string bar(std::min(bars, 20), '*');

        std::cout << "  ";
        col(cName, r.name.substr(0, cName - 1));
        col(cDim,  dims);

        std::cout << std::right << std::fixed << std::setprecision(2)
                  << std::setw(cMs - 1) << r.cpuMs << " "
                  << std::setw(cMs - 1) << r.gpuMs << " ";

        // Verde si GPU gana (speedup > 1), rojo si CPU es mas rapida
        std::cout << (r.speedup > 1.0 ? "\033[32m" : "\033[31m");
        std::cout << std::setw(cSpd - 1) << std::fixed << std::setprecision(2)
                  << r.speedup << "x \033[0m";

        std::cout << std::setw(cMae - 1) << std::fixed << std::setprecision(3)
                  << r.maeVal << "\n";

        // Mini barra visual de speedup alineada debajo de la columna Speedup
        if (!bar.empty())
            std::cout << "  " << std::string(cName + cDim + cMs * 2, ' ')
                      << " |" << bar << "\n";

        totalCpu  += r.cpuMs;
        totalGpu  += r.gpuMs;
        maxSpeedup = std::max(maxSpeedup, r.speedup);
        minSpeedup = std::min(minSpeedup, r.speedup);
    }

    hline('-');

    if (!results.empty()) {
        double avgSpeedup = (totalGpu > 0) ? (totalCpu / totalGpu) : 0.0;
        std::cout << "  ";
        col(cName, "TOTAL / AVG");
        col(cDim,  "");
        std::cout << std::right << std::fixed << std::setprecision(2)
                  << std::setw(cMs - 1) << totalCpu << " "
                  << std::setw(cMs - 1) << totalGpu << " ";
        std::cout << "\033[1;36m"
                  << std::setw(cSpd - 1) << avgSpeedup << "x\033[0m\n";

        hline('=');
        std::cout << "  Max speedup: " << std::fixed << std::setprecision(2)
                  << maxSpeedup << "x  |  Min: " << minSpeedup << "x\n";
        std::cout << "  Total CPU: " << totalCpu << " ms"
                  << "  |  GPU: " << totalGpu << " ms\n";
    }

    hline('=');

    std::cout << "\n  Intermediate stages saved per image:\n";
    std::cout << "    *_cpu_blur.pgm   - blur stage (CPU)\n";
    std::cout << "    *_cpu_sobel.pgm  - Sobel gradient (CPU)\n";
    std::cout << "    *_cpu_final.pgm  - binary edges (CPU)\n";
    std::cout << "    *_gpu_blur.pgm   - blur stage (GPU)\n";
    std::cout << "    *_gpu_sobel.pgm  - Sobel gradient (GPU)\n";
    std::cout << "    *_gpu_final.pgm  - binary edges (GPU)\n";
    std::cout << "\n";
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  Single image: " << argv[0] << " <input.pgm> [threshold=120]\n"
                  << "  Batch:        " << argv[0] << " --batch <directory> [threshold=120]\n";
        return 1;
    }

    int threshold = 120;
    std::vector<BatchEntry> results;

    if (std::string(argv[1]) == "--batch") {
        // Modo batch: procesa todos los .pgm de un directorio
        if (argc < 3) {
            std::cerr << "Batch mode requires an input directory.\n";
            return 1;
        }
        if (argc >= 4) threshold = std::stoi(argv[3]);

        std::string dir = argv[2];
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(dir))
            if (entry.path().extension() == ".pgm")
                files.push_back(entry.path().string());
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            std::cerr << "No .pgm files found in: " << dir << "\n";
            return 1;
        }

        std::cout << "Batch mode: " << files.size() << " file(s) in '"
                  << dir << "', threshold=" << threshold << "\n";

        for (int i = 0; i < (int)files.size(); ++i) {
            printProgress(i, (int)files.size());
            results.push_back(processImage(files[i], "output", threshold));
        }
        printProgress((int)files.size(), (int)files.size());
        std::cout << "\n";

    } else {
        // Modo imagen unica
        if (argc >= 3) threshold = std::stoi(argv[2]);
        std::cout << "Processing: " << argv[1]
                  << "  threshold=" << threshold << "\n";
        results.push_back(processImage(argv[1], "output", threshold));
    }

    printReport(results);
    return 0;
}
