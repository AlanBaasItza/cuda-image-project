#pragma once
// =============================================================================
// gpu_pipeline.cuh  —  Pipeline paralelo en GPU (blur → sobel → threshold)
//                       con memoria compartida (shared memory) en cada kernel
// =============================================================================
#include "image_io.hpp"

// Ejecuta el pipeline completo en GPU y devuelve todas las etapas intermedias.
PipelineResult runGpuPipeline(const GrayImage& in, int threshold);
