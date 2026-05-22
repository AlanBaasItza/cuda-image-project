# CUDA Image Processing — Detección de Bordes CPU vs GPU

**Asignatura:** Sistemas Distribuidos  
**Carrera:** Ingeniería de Software

**Integrantes del equipo:**
- Alan Baas Itza
- Jesus Oswaldo Chan Uicab
- JESUS EVERARDO JIMENEZ RIVERA
- DANIEL MENDEZ SIERRA
- BUNTAROU EMILIANO ORDUÑO HARA
- ARANDRY AZAEL RABANALES ANDRADE

---

Video demostrativo del uso de este proyecto: [Demostracion](https://drive.google.com/file/d/1yA-_M2DAR-2BObyQ1KwH5-XuIHU_sNzE/view?usp=drive_link)

---
## ¿Qué hace este proyecto?

Este proyecto implementa un **pipeline de procesamiento digital de imágenes** enfocado en la **detección de bordes**, y lo ejecuta en paralelo de dos formas:

- **CPU:** procesamiento secuencial, un píxel a la vez.
- **GPU (CUDA):** procesamiento masivamente paralelo, miles de píxeles al mismo tiempo.

El objetivo es medir y comparar el rendimiento de ambos enfoques: cuándo vale la pena paralelizar y cuánto se puede ganar.

### ¿Qué es la detección de bordes?

En procesamiento digital de imágenes, un **borde** es una zona donde el nivel de brillo de la imagen cambia abruptamente, lo que generalmente corresponde a los contornos de objetos. Detectar bordes es el primer paso en muchos sistemas de visión por computadora: reconocimiento de objetos, seguimiento de movimiento, inspección industrial, diagnóstico médico por imagen, etc.
Cuando el volumen de datos crece (más resolución, más imágenes, más frecuencia), la CPU puede convertirse en cuello de botella. Aquí es donde la GPU ofrece ventajas claras gracias al paralelismo masivo.
### Pipeline de procesamiento

El programa aplica tres etapas en secuencia sobre cada imagen en escala de grises (formato PGM):

```
Imagen original
      │
      ▼
┌─────────────────────────────────────────┐
│  Etapa 1: Blur (suavizado 3×3)          │
│  Promedia cada píxel con sus 8 vecinos  │
│  → Reduce el ruido antes de derivar     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Etapa 2: Sobel (detección de bordes)   │
│  Aplica dos máscaras de convolución:    │
│    Gx = gradiente horizontal            │
│    Gy = gradiente vertical              │
│    magnitud = sqrt(Gx² + Gy²)           │
│  → Resalta zonas de cambio brusco       │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Etapa 3: Umbralización (Threshold)     │
│  Si magnitud ≥ umbral → píxel blanco    │
│  Si magnitud < umbral → píxel negro     │
│  → Imagen binaria de bordes             │
└─────────────────────────────────────────┘
      │
      ▼
  Imagen de bordes detectados
```

Cada etapa se guarda por separado, tanto en la versión CPU como en la GPU.

---

## Estructura del proyecto

```
cuda-image-project/
├── include/
│   ├── image_io.hpp        # Estructuras GrayImage y PipelineResult
│   ├── cpu_pipeline.hpp    # Interfaz del pipeline CPU
│   └── gpu_pipeline.cuh    # Interfaz del pipeline GPU (CUDA)
├── src/
│   ├── main.cu             # Punto de entrada, modo batch, reporte
│   ├── image_io.cpp        # Lectura/escritura de archivos PGM
│   ├── cpu_pipeline.cpp    # Blur + Sobel + Threshold en CPU
│   ├── gpu_pipeline.cu     # Blur + Sobel + Threshold en GPU (shared memory)
│   └── timer.hpp           # Temporizador de alta resolución
├── scripts/
│   └── compare.py          # Comparación CPU vs GPU (MAE, PSNR)
├── data/
│   └── input/              # Imágenes de entrada (.pgm)
├── output/                 # Resultados generados automáticamente
└── Makefile
```

---

## Requisitos

| Componente | Versión mínima |
|---|---|
| GPU NVIDIA | Compute Capability 3.5+ |
| CUDA Toolkit | 11.0+ |
| Driver NVIDIA | Compatible con el toolkit |
| Compilador C++ | g++ con C++17 (`std::filesystem`) |
| Python (opcional) | 3.7+ para `compare.py` |
|x64 Native Tools Command Prompt for VS|


Verificar instalación:
```bash
nvcc --version    # Debe mostrar versión del toolkit
nvidia-smi        # Debe mostrar la GPU disponible
```
## Sitios utiles para conseguir recursos y convertirlos a pgm
- https://convertio.co/es/jpg-pgm/
- https://www.pexels.com/

---

## Instalación y compilación

```bash
# Clonar el repositorio
git clone https://github.com/AlanBaasItza/cuda-image-project
cd cuda-image-project

# Compilar (genera el ejecutable cuda_image_app)
make
```

Si el sistema tiene una arquitectura de GPU específica, se puede indicar:
```bash
nvcc -O2 -std=c++17 -Iinclude -arch=sm_75 src/main.cu src/image_io.cpp src/cpu_pipeline.cpp src/gpu_pipeline.cu -o cuda_image_app
```

### Elegir el valor de `-arch`

El argumento `-arch` de `nvcc` indica la arquitectura CUDA objetivo para tu GPU.  
Para saber cuál debes usar, identifica primero la **compute capability** de tu tarjeta en la tabla oficial de NVIDIA:

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

La regla es simple: la compute capability se traduce al formato `sm_XY`.

Ejemplos comunes:

- Compute Capability **8.9** → `-arch=sm_89`
- Compute Capability **8.6** → `-arch=sm_86`
- Compute Capability **7.5** → `-arch=sm_75`
- Compute Capability **6.1** → `-arch=sm_61`

Por ejemplo, si tu GPU es una **RTX 4070**, NVIDIA la clasifica con compute capability **8.9**, así que puedes compilar con:

```bash
nvcc -O2 -std=c++17 -Iinclude -arch=sm_89 src/main.cu src/image_io.cpp src/cpu_pipeline.cpp src/gpu_pipeline.cu -o cuda_image_app
```

(Si no estás seguro de tu GPU, puedes verificarla con: `nvidia-smi`)

---

## Uso

### Modo imagen única

Procesa un solo archivo `.pgm`:

```bash
./cuda_image_app <imagen.pgm> [threshold]
```

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `<imagen.pgm>` | Ruta a la imagen de entrada en escala de grises | (requerido) |
| `[threshold]` | Umbral de detección de bordes (0–255). Menor = más bordes detectados | `120` |

**Ejemplo:**
```bash
./cuda_image_app data/input/test.pgm 120
```

**Ejemplo con umbral más sensible (detecta más bordes):**
```bash
./cuda_image_app data/input/test.pgm 60
```

### Modo batch (procesamiento por lotes)

Procesa todas las imágenes `.pgm` de un directorio:

```bash
./cuda_image_app --batch <directorio> [threshold]
```

**Ejemplo:**
```bash
./cuda_image_app --batch data/input 120
```

Se mostrará una barra de progreso y al final un reporte comparativo completo.

También disponible como atajo:
```bash
make batch
```

---

## Archivos de salida

Por cada imagen procesada (llamémosla `foto`) se generan **6 archivos** en `output/`:

| Archivo | Descripción |
|---|---|
| `foto_cpu_blur.pgm` | Imagen suavizada — etapa 1 CPU |
| `foto_cpu_sobel.pgm` | Magnitud del gradiente Sobel — etapa 2 CPU |
| `foto_cpu_final.pgm` | Bordes binarios (resultado final CPU) |
| `foto_gpu_blur.pgm` | Imagen suavizada — etapa 1 GPU |
| `foto_gpu_sobel.pgm` | Magnitud del gradiente Sobel — etapa 2 GPU |
| `foto_gpu_final.pgm` | Bordes binarios (resultado final GPU) |

Las imágenes PGM se pueden abrir con GIMP, IrfanView, o cualquier visor que soporte el formato.

---

## Reporte de rendimiento

Al finalizar, el programa imprime una tabla como esta:

```
============================================================================
  PERFORMANCE REPORT -- CPU vs GPU (CUDA Shared Memory)
  ============================================================================
  Image               Size        CPU (ms)  GPU (ms)  Speedup  MAE     
  ----------------------------------------------------------------------------
  pexels-dejana-popov 8192x6144      393.41    178.35     2.21x   0.325
                                                       |****
  pexels-felix-mitter 8192x1967      133.96     13.74     9.75x   0.145
                                                       |*******************
  pexels-myeong-rae-j 8192x5461      331.70     34.65     9.57x   0.121
                                                       |*******************
  test2               1536x2048       30.37      4.49     6.76x   0.078
                                                       |*************
  test3               3376x6000      149.01     16.28     9.15x   0.213
                                                       |******************
                                                       |***********
  test5               8192x3955      253.29     25.20    10.05x   0.150
                                                       |********************
  test6               6079x8192      367.38     75.35     4.88x   0.247
                                                       |*********
  v2osk-1Z2niiBPg5A-u 7372x4392      244.18    113.34     2.15x   0.233
                                                       |****
  ----------------------------------------------------------------------------
  TOTAL / AVG                       1920.85    464.55     4.13x
  ============================================================================
  Max speedup: 10.05x  |  Min: 2.15x
  Total CPU: 1920.85 ms  |  GPU: 464.55 ms
  ============================================================================
```

- **Speedup** aparece en verde si la GPU es más rápida (>1×), en rojo si no.
- La barra de asteriscos es proporcional al speedup para visualizarlo de un vistazo.
- **MAE** (Mean Absolute Error) mide la diferencia entre la salida CPU y GPU; valores cercanos a 0 indican equivalencia numérica.

---

## Comparación con script Python

Para comparar un par específico:
```bash
python scripts/compare.py output/test_cpu_final.pgm output/test_gpu_final.pgm
```

Para comparar automáticamente todos los pares en `output/`:
```bash
python scripts/compare.py --dir output
```

Salida de ejemplo:
```
  ──────────────────────────────────────────────────
  Imagen : test  (512×512  262,144 px)
  CPU    : output/test_cpu_final.pgm
  GPU    : output/test_gpu_final.pgm

  MAE    :  0.0000  [                    ]
  PSNR   :    ∞ dB  (salidas idénticas)
  Estado : ✓ Idénticas
```

---

## Detalle técnico: optimización con memoria compartida

La versión GPU usa **shared memory** (memoria compartida por hilos dentro de un bloque CUDA), lo que reduce significativamente los accesos a memoria global.

### Sin shared memory (versión original)
Cada hilo lee los 9 píxeles vecinos directamente de memoria global. Para una imagen de 512×512 con bloques de 16×16, eso significa que cada píxel es leído hasta **9 veces** desde memoria global (latencia ~600 ciclos).

### Con shared memory (versión actual)
Cada bloque carga primero un **tile de (16+2)×(16+2) = 324 píxeles** en shared memory (latencia ~4 ciclos), luego los 256 hilos del bloque hacen sus cálculos de blur o Sobel completamente desde ahí.

```
Memoria global   →  [tile en shared memory]  →  cálculo por hilo
 (lenta, 1 vez)        (rápida, muchas veces)
```

Esto reduce los accesos a memoria global en aproximadamente **8×** para los kernels de convolución.

---

## Formato de imagen soportado

El proyecto trabaja con imágenes en formato **PGM binario (P5)**:
- Un canal (escala de grises), 8 bits por píxel (0–255).
- Se pueden convertir imágenes desde otros formatos con ImageMagick:

## Problemas comunes y solución

## `nvcc` no reconocido
- Verificar instalación de CUDA Toolkit
- Abrir terminal correcta (x64 Native Tools Command Prompt)

## Error con compilador `cl.exe`
- Asegurar instalación de “Desarrollo de escritorio con C++” en Visual Studio

## No se genera salida
- Verificar que exista `data/input/test.pgm`
- Confirmar formato PGM válido (P5, max value 255)

## MAE muy alto
- Revisar implementación de kernels y bordes
- Verificar umbral y consistencia en CPU/GPU
