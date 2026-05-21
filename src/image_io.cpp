#include "image_io.hpp"
// =============================================================================
// image_io.cpp  —  Lectura y escritura de imágenes en formato PGM binario (P5)
// =============================================================================
#include <fstream>
#include <iostream>

// Salta líneas de comentario que empiezan con '#'
static void skipComments(std::ifstream& f) {
    while (f.peek() == '#') {
        std::string line;
        std::getline(f, line);
    }
}

bool loadPGM(const std::string& path, GrayImage& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    std::string magic;
    f >> magic;
    if (magic != "P5") return false;   // Solo PGM binario

    skipComments(f);
    f >> img.width;
    skipComments(f);
    f >> img.height;
    skipComments(f);
    int maxv;
    f >> maxv;
    f.get();  // consume el salto de línea tras el encabezado

    if (img.width <= 0 || img.height <= 0 || maxv != 255) return false;

    img.data.resize(static_cast<size_t>(img.width) * img.height);
    f.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    return f.good();
}

bool savePGM(const std::string& path, const GrayImage& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return f.good();
}
