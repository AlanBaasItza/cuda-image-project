#pragma once
// =============================================================================
// cpu_pipeline.hpp  —  Pipeline secuencial en CPU (blur → sobel → threshold)
// =============================================================================
#include "image_io.hpp"

// Ejecuta el pipeline completo en CPU y devuelve todas las etapas intermedias.
PipelineResult runCpuPipeline(const GrayImage& in, int threshold);
