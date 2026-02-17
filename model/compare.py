import numpy as np
import argparse


def load_hex_matrix(path, rows, cols):
    tokens = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # split by space or comma
            parts = line.replace(",", " ").split()
            tokens.extend(parts)

    if len(tokens) < rows * cols:
        raise ValueError("Not enough hex values in file.")

    values = []
    for t in tokens[:rows*cols]:
        # remove 0x if present
        t = t.lower().replace("0x", "")
        # interpret as signed 16-bit integer
        val = int(t, 16)
        if val >= 0x8000:   # convert from two's complement
            val -= 0x10000
        values.append(val)

    return np.array(values, dtype=np.int32).reshape(rows, cols)

def main():
	parser = argparse.ArgumentParser(description="compare matrices")
	parser.add_argument("--n", type=int, required=True, default=3, help="")
	parser.add_argument("--ref", type=str, required=True, default="../memories/ref_output.hex", help="")
	parser.add_argument("--sim", type=str, required=True, default="../memories/sim_output.hex", help="")
	args = parser.parse_args()
	print(f"\n [INFO] - Reference Matrix {args.ref} :")
	ref = load_hex_matrix(args.ref, args.n, args.n)
	print(ref)
	print(f"\n [INFO] - Simulation Matrix {args.sim} :")
	sim = load_hex_matrix(args.sim, args.n, args.n)
	print(sim)
	

if __name__ == "__main__":
    main()
