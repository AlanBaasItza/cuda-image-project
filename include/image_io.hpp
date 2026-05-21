#pragma once
// =============================================================================
// image_io.hpp  —  Estructuras de datos y carga/guardado de imágenes PGM
// =============================================================================
#include <string>
#include <vector>

// Imagen en escala de grises (un byte por píxel)
struct GrayImage {
    int width  = 0;
    int height = 0;
    std::vector<unsigned char> data;
};

// Resultado completo de un pipeline (CPU o GPU), incluye etapas intermedias
struct PipelineResult {
    GrayImage blur;       // Etapa 1: suavizado
    GrayImage sobel;      // Etapa 2: magnitud de gradiente
    GrayImage threshold;  // Etapa 3: umbralización binaria (= salida final)
    double timeMs = 0.0;  // Tiempo de ejecución en milisegundos
};

bool loadPGM(const std::string& path, GrayImage& img);
bool savePGM(const std::string& path, const GrayImage& img);
