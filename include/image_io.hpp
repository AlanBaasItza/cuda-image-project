#pragma once
#include <string>
#include <vector>

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> data;
};

bool loadPGM(const std::string& path, GrayImage& img);
bool savePGM(const std::string& path, const GrayImage& img);