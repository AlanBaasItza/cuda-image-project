# Uso: python scripts/compare.py output/cpu_out.pgm output/gpu_out.pgm
import sys

def read_pgm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxv = int(f.readline())
        data = f.read()
    return w, h, data

a = read_pgm(sys.argv[1])
b = read_pgm(sys.argv[2])
assert a[0]==b[0] and a[1]==b[1]
err = sum(abs(x-y) for x,y in zip(a[2], b[2])) / len(a[2])
print("MAE:", err)