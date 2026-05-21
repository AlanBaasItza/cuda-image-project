#!/usr/bin/env python3
"""
compare.py — Comparación visual de salidas CPU vs GPU
Uso:
    python scripts/compare.py output/test_cpu_final.pgm output/test_gpu_final.pgm
    python scripts/compare.py --dir output               (compara todos los pares)
"""
import sys
import os
import pathlib

# ---------------------------------------------------------------------------
def read_pgm(path: str):
    """Lee un PGM binario (P5) y devuelve (width, height, bytes)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5", f"{path} no es PGM binario (P5)"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        _maxv = int(f.readline())
        data = f.read()
    return w, h, data

def mae(a: bytes, b: bytes) -> float:
    """Error Absoluto Medio entre dos buffers de píxeles."""
    assert len(a) == len(b), "Las imágenes tienen distinto tamaño"
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

def psnr(a: bytes, b: bytes) -> float:
    """Peak Signal-to-Noise Ratio (dB). Infinito si son idénticas."""
    import math
    mse = sum((x - y)**2 for x, y in zip(a, b)) / len(a)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255**2 / mse)

def bar(value: float, max_val: float = 10.0, width: int = 20) -> str:
    """Barra ASCII proporcional."""
    filled = int(min(value / max_val, 1.0) * width)
    return "[" + "=" * filled + " " * (width - filled) + "]"

def compare_pair(cpu_path: str, gpu_path: str):
    """Compara un par CPU/GPU e imprime métricas."""
    w1, h1, d1 = read_pgm(cpu_path)
    w2, h2, d2 = read_pgm(gpu_path)

    name = pathlib.Path(cpu_path).stem.replace("_cpu_final", "")
    print(f"\n  {'─'*50}")
    print(f"  Imagen : {name}  ({w1}×{h1}  {w1*h1:,} px)")
    print(f"  CPU    : {cpu_path}")
    print(f"  GPU    : {gpu_path}")

    if w1 != w2 or h1 != h2:
        print("  ⚠ Dimensiones distintas — no comparable")
        return

    m   = mae(d1, d2)
    p   = psnr(d1, d2)
    identical = (m == 0.0)

    print(f"\n  MAE    : {m:7.4f}  {bar(m, 5.0)}")
    if p == float('inf'):
        print(f"  PSNR   :    ∞ dB  (salidas idénticas)")
    else:
        print(f"  PSNR   : {p:7.2f} dB  {bar(p, 60.0)}")
    print(f"  Estado : {'✓ Idénticas' if identical else '~ Diferencia mínima esperada'}")

def compare_dir(directory: str):
    """Encuentra todos los pares *_cpu_final.pgm / *_gpu_final.pgm en un dir."""
    d = pathlib.Path(directory)
    cpu_files = sorted(d.glob("*_cpu_final.pgm"))

    if not cpu_files:
        print(f"No se encontraron archivos *_cpu_final.pgm en '{directory}'")
        sys.exit(1)

    print(f"\n{'='*54}")
    print(f"  COMPARACIÓN CPU vs GPU — {len(cpu_files)} imagen(es)")
    print(f"{'='*54}")

    for cpu_path in cpu_files:
        gpu_path = pathlib.Path(str(cpu_path).replace("_cpu_final", "_gpu_final"))
        if gpu_path.exists():
            compare_pair(str(cpu_path), str(gpu_path))
        else:
            print(f"\n  ⚠ Sin par GPU para: {cpu_path.name}")

    print(f"\n{'='*54}\n")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dir":
        compare_dir(sys.argv[2])
    elif len(sys.argv) == 3:
        compare_pair(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
