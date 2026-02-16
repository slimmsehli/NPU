import numpy as np

############# Load matrix layer

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

############# Multiply Layer

def multiply(A, B):
	return A.dot(B)


############# Activate layer

def activate(matrix, rows, cols, activation="relu"):
    # reshape matrix into the given dimensions
    mat = np.array(matrix, dtype=np.float32).reshape(rows, cols)

    # choose activation
    if activation.lower() == "relu":
        activated = np.maximum(mat, 0)
    elif activation.lower() == "sigmoid":
        activated = 1 / (1 + np.exp(-mat))
    elif activation.lower() == "tanh":
        activated = np.tanh(mat)
    else:
        raise ValueError("Unsupported activation function")

    return  np.array(activated, dtype=np.int32)


############# TEST
	
A = load_hex_matrix("./memories/inputs.hex", rows=3, cols=3)
B = load_hex_matrix("./memories/weights_L0.hex", rows=3, cols=3)
C = multiply(A,B)
D = activate(C, rows=3, cols=3)


print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nA x B =")
print(C)

print("\nRelu(AxB) =")
print(D)
