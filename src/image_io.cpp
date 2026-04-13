#include "image_io.hpp"
#include <fstream>
#include <iostream>

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
    if (magic != "P5") return false;

    skipComments(f);
    f >> img.width;
    skipComments(f);
    f >> img.height;
    skipComments(f);
    int maxv;
    f >> maxv;
    f.get(); // consume newline

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