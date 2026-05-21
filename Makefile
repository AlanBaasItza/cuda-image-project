APP     = cuda_image_app
NVCC    = nvcc
# -std=c++17 necesario para std::filesystem
# -O2 optimizaciones de velocidad
CXXFLAGS = -O2 -std=c++17 -Iinclude

SRC = src/main.cu src/image_io.cpp src/cpu_pipeline.cpp src/gpu_pipeline.cu

all: $(APP)

$(APP):
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(APP)

# Ejecutar con imagen única (threshold por defecto 120)
run:
	./$(APP) data/input/test.pgm 120

# Procesar todas las imágenes de data/input en modo batch
batch:
	./$(APP) --batch data/input 120

clean:
	rm -f $(APP)
	rm -f output/*.pgm

.PHONY: all run batch clean
