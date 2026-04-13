# Proyecto Final — Aceleración de Procesamiento de Imágenes con CUDA  
**Asignatura:** Sistemas Distribuidos  
**Equipo:** Equipo Torrent  
**Profesor:** Francisco Moo Mena  

---

## 1. Descripción general del proyecto

Este proyecto implementa un prototipo de **procesamiento de imágenes acelerado con GPU** usando **CUDA**, y compara su desempeño contra una versión secuencial en CPU.

El flujo implementado es:

1. Filtro de suavizado (Blur 3x3)  
2. Detección de bordes (Sobel)  
3. Umbralización binaria (Threshold)

La salida se genera en dos versiones:

- `cpu_out.pgm` (resultado por CPU)
- `gpu_out.pgm` (resultado por GPU)

Además, el programa reporta:

- Tiempo CPU (ms)
- Tiempo GPU (ms)
- Speedup (CPU/GPU)
- MAE (error promedio absoluto entre ambas salidas)

---

## 2. Importancia y visión del proyecto

## ¿Por qué es importante?
Este proyecto demuestra una necesidad real en ingeniería: **procesar datos visuales de forma rápida y eficiente**.  
Muchas aplicaciones actuales dependen de esto:

- inspección industrial por visión
- análisis de imágenes médicas
- monitoreo urbano y seguridad
- robótica y drones
- automatización en tiempo casi real

Cuando el volumen de datos crece (más resolución, más imágenes, más frecuencia), la CPU puede convertirse en cuello de botella. Aquí es donde la GPU ofrece ventajas claras gracias al paralelismo masivo.

## Visión del proyecto
La visión es construir una base técnica robusta para evolucionar hacia sistemas más avanzados:

- de imagen estática a video en tiempo real
- de detección de bordes a segmentación inteligente
- de ejecución local a despliegues híbridos (edge/cloud)
- de un pipeline académico a una arquitectura aplicable en industria

Este trabajo no solo busca “hacer correr CUDA”, sino **entender y medir** el impacto de usar cómputo paralelo en problemas prácticos.

---

## 3. Requisitos del sistema

## Hardware
- GPU NVIDIA compatible con CUDA
- Memoria suficiente para imágenes de prueba

## Software
- Windows + Visual Studio Community 2026 (con C++)
- CUDA Toolkit instalado (incluye `nvcc`)
- Driver NVIDIA actualizado
- (Opcional) Python 3 para script de comparación

## Verificación rápida del entorno
En terminal (x64 Native Tools Command Prompt for VS 2026):

```bat
nvcc --version
nvidia-smi
```

Si ambos comandos responden correctamente, el entorno está listo.

---

## 4. Estructura del proyecto

```text
cuda-image-project/
├─ include/
│  ├─ image_io.hpp
│  ├─ cpu_pipeline.hpp
│  └─ gpu_pipeline.cuh
├─ src/
│  ├─ main.cu
│  ├─ image_io.cpp
│  ├─ cpu_pipeline.cpp
│  ├─ gpu_pipeline.cu
│  └─ timer.hpp
├─ data/
│  └─ input/
│     └─ test.pgm
├─ output/
│  ├─ cpu_out.pgm
│  └─ gpu_out.pgm
├─ scripts/
│  ��─ compare.py
├─ Makefile
└─ README.md
```

---

## 5. Compilación

Desde la carpeta raíz del proyecto:

```bat
nvcc -O2 -Iinclude src/main.cu src/image_io.cpp src/cpu_pipeline.cpp src/gpu_pipeline.cu -o cuda_image_app.exe
```

Si no hay errores, se generará:

```text
cuda_image_app.exe
```

---

## 6. Ejecución

```bat
cuda_image_app.exe data/input/test.pgm 120
```

Parámetros:

- `data/input/test.pgm` → imagen de entrada en formato PGM (P5)
- `120` → umbral (threshold), opcional (si no se indica, usa valor por defecto)

---

## 7. Salidas esperadas

Después de ejecutar, se deben generar:

- `output/cpu_out.pgm`
- `output/gpu_out.pgm`

Y en consola se muestran métricas como:

- CPU ms
- GPU ms
- Speedup
- MAE CPU vs GPU

---

## 8. Cómo validar funcionamiento correcto

Se considera funcionamiento correcto cuando:

1. El ejecutable corre sin errores.
2. Se generan ambos archivos de salida en `output/`.
3. Se imprimen tiempos y speedup en consola.
4. El MAE es bajo (salidas similares entre CPU y GPU).

Validación opcional con Python:

```bat
python scripts/compare.py output/cpu_out.pgm output/gpu_out.pgm
```

---

## 9. Interpretación de resultados

## Speedup
Se calcula como:

```text
Speedup = Tiempo_CPU / Tiempo_GPU
```

- `> 1` significa que GPU fue más rápida
- mientras mayor sea, mejor ganancia de rendimiento

## MAE (Error Promedio Absoluto)
Mide diferencia promedio entre salida CPU y GPU:

- cercano a `0` = resultados muy parecidos
- un valor pequeño es esperable por detalles numéricos

---

## 10. Buenas prácticas de pruebas

Para un análisis más sólido:

- Probar distintos umbrales (ej. 80, 120, 160)
- Repetir varias veces y promediar tiempos
- Probar imágenes con diferentes tamaños/resoluciones
- Registrar resultados en tabla para el reporte final

---

## 11. Problemas comunes y solución

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

---

## 12. Alcance actual y mejoras futuras

## Alcance actual
- Pipeline base de procesamiento de imagen
- Comparación de rendimiento CPU vs GPU
- Validación cuantitativa con MAE

## Mejoras futuras
- Guardar etapas intermedias (blur, sobel, threshold)
- Optimización con memoria compartida en CUDA
- Procesamiento por lotes y/o video
- Interfaz gráfica para pruebas
- Integración con casos de uso reales (industrial, médico, monitoreo)

---

## 13. Créditos del equipo

- ALAN RUBEN BAAS ITZA  
- Jesus Oswaldo Chan Uicab  
- JESUS EVERARDO JIMENEZ RIVERA  
- DANIEL MENDEZ SIERRA  
- BUNTAROU EMILIANO ORDUÑO HARA  
- ARANDRY AZAEL RABANALES ANDRADE  

---

## 14. Conclusión

Este proyecto establece una base funcional y medible para aplicar cómputo paralelo con CUDA a un problema práctico de visión computacional.  
Su principal aporte es demostrar, con evidencia de ejecución, que la aceleración por
