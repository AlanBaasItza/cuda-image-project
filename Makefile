APP=cuda_image_app
NVCC=nvcc
CXXFLAGS=-O2 -Iinclude

SRC=src/main.cu src/image_io.cpp src/cpu_pipeline.cpp src/gpu_pipeline.cu
all:
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(APP)

run:
	./$(APP) data/input/test.pgm 120

clean:
	rm -f $(APP)