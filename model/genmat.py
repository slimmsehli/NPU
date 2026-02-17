#!/usr/bin/env python3
"""
Generate two N x N int8 matrices and write them to hex files (one value per line, no '0x').

- Values are stored as signed int8 in memory (-128..127).
- Output lines are two-digit lowercase hex of the underlying 8-bit two's complement (00..ff).
- Each matrix is written to its own file.

Usage:
  python make_int8_hex_mats.py --n 1024 \
      --seedA 123 --seedB 456 \
      --outA A.hex --outB B.hex

If you prefer reproducible but different matrices from a single seed, you can set only --seedA
and omit --seedB; the script will derive the second sequence from seedA.
"""

import argparse
import numpy as np
import os

def write_matrix_to_hex_lines(mat: np.ndarray, path: str) -> None:
    """
    Write each element of an int8 matrix to a file as two-digit hex (00..ff), one per line.
    """
    if mat.dtype != np.int8:
        raise ValueError("Matrix dtype must be int8.")
    # Flatten in row-major order
    flat = mat.ravel()
    # Convert to unsigned bytes for two's complement representation, then to hex without '0x'
    # Ensure lowercase, two digits.
    hex_lines = (f"{(int(v) & 0xFF):02x}" for v in flat)

    # Write efficiently
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(hex_lines))
        f.write("\n")  # final newline (optional but common)

def main():
    parser = argparse.ArgumentParser(description="Generate two N x N int8 matrices and write hex files.")
    parser.add_argument("--n", type=int, required=True, help="Matrix dimension N (creates N x N).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for matrix A.")
    parser.add_argument("--out", type=str, default="A.hex", help="Output filename for matrix A.")
    parser.add_argument("--min", type=str, default="-20", help="minimum value.")
    parser.add_argument("--max", type=str, default="20", help="maximum value.")
    parser.add_argument("--dist", type=str, default="uniform",
                        choices=["uniform", "normal", "zeros", "ones"],
                        help="Distribution for values: uniform(-128..127), normal(μ=0,σ≈42), zeros, ones.")
    args = parser.parse_args()
    high = args.max
    low = args.min
    N = args.n
    if N <= 0:
        raise ValueError("N must be positive.")

    # Prepare RNGs
    if args.seed is not None:
        rngA = np.random.default_rng(args.seed)
    else:
        rngA = np.random.default_rng()

    # Generate matrices
    def gen_mat(rng):
        if args.dist == "uniform":
            # Uniform over all int8 values
            return rng.integers(low=low, high=high, size=(N, N), dtype=np.int8)
        elif args.dist == "normal":
            # Normal centered at 0; clip to int8 range
            # std chosen so that most values fall within -128..127
            x = rng.normal(loc=0.0, scale=42.0, size=(N, N))
            x = np.clip(np.rint(x), -128, 127).astype(np.int8)
            return x
        elif args.dist == "zeros":
            return np.zeros((N, N), dtype=np.int8)
        elif args.dist == "ones":
            return np.ones((N, N), dtype=np.int8)
        else:
            raise ValueError("Unknown distribution.")

    A = gen_mat(rngA)

    # Ensure output directory exists
    for path in (args.out):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    write_matrix_to_hex_lines(A, args.out)

    # Also print a small confirmation
    total = N * N
    print(f"Wrote {total} values to {args.out} (two's complement hex, no '0x').")

if __name__ == "__main__":
    main()


### usage

#python make_int8_hex_mats.py --n 1024 --seedA 1 --seedB 2 --outA A.hex --outB B.hex

# 2) Use one seed; the second matrix seed is derived automatically
#python make_int8_hex_mats.py --n 256 --seedA 1234

# 3) Normal distribution (centered 0), different output names
#python make_int8_hex_mats.py --n 512 --dist normal --outA matA.hex --outB matB.hex

# 4) Deterministic test files
#python make_int8_hex_mats.py --n 8 --dist ones --outA ones.hex --dist zeros --outB zeros.hex




