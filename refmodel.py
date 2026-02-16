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

############# Bias layer

import numpy as np

def bias(matrix, rows, cols, bias):
    """
    Add a bias vector to a 2D matrix along the columns.

    Parameters:
        matrix : list/1D array of length rows*cols (float or int)
        rows   : int
        cols   : int
        bias   : list/1D array of length 'cols'

    Returns:
        New numpy array of shape (rows, cols) with bias added.
    """
    mat = np.array(matrix, dtype=np.float32).reshape(rows, cols)
    b = np.array(bias, dtype=np.float32).reshape(1, cols)

    if b.shape[1] != cols:
        raise ValueError(f"Bias length ({b.shape[1]}) must equal cols ({cols}).")

    return np.array(mat + b, dtype=np.int32)  # broadcast along rows

############# Scale layer
import numpy as np

def scale(matrix, rows, cols, scale):
    """
    Apply NPU-style scaling to a matrix.

    Parameters:
        matrix : list or 1D array of numbers
        rows   : number of rows
        cols   : number of columns
        scale  : scaling factor (float)

    Returns:
        A new scaled matrix (numpy array)
    """

    # reshape input into matrix form
    mat = np.array(matrix, dtype=np.float32).reshape(rows, cols)

    # apply scaling
    scaled = mat * scale

    return np.array(scaled, dtype=np.int32)

############# quantize
def requantize(matrix, rows, cols, scale, zero_point=0, dtype='int8',
               rounding='nearest', saturate=True):
    """
    Affine quantization to int8/int16 with optional saturation.

    q = clip( round(x/scale) + zero_point, qmin, qmax )

    Parameters:
        matrix      : list/1D array, length rows*cols (float or int)
        rows, cols  : int
        scale       : float > 0
        zero_point  : int (offset)
        dtype       : 'int8' or 'int16'
        rounding    : 'nearest' | 'floor' | 'ceil' | 'trunc'
        saturate    : if True, clamp to dtype range

    Returns:
        Quantized numpy array of dtype np.int8 or np.int16, shape (rows, cols).
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")

    mat = np.array(matrix, dtype=np.float32).reshape(rows, cols)

    # scale to quant domain
    q = mat / float(scale)
    q = q + float(zero_point)

    # rounding
    if rounding == 'nearest':
        q = np.rint(q)         # round to nearest, ties to even
    elif rounding == 'floor':
        q = np.floor(q)
    elif rounding == 'ceil':
        q = np.ceil(q)
    elif rounding == 'trunc':
        q = np.trunc(q)
    else:
        raise ValueError("rounding must be one of: 'nearest', 'floor', 'ceil', 'trunc'")

    # choose dtype range
    if dtype == 'int8':
        qmin, qmax = -128, 127
        out_dtype = np.int8
    elif dtype == 'int16':
        qmin, qmax = -32768, 32767
        out_dtype = np.int16
    else:
        raise ValueError("dtype must be 'int8' or 'int16'")

    # saturation (clipping)
    if saturate:
        q = np.clip(q, qmin, qmax)

    return q.astype(out_dtype)

############# TEST

# load matrices
A = load_hex_matrix("./memories/inputs.hex", rows=3, cols=3)
B = load_hex_matrix("./memories/weights_L0.hex", rows=3, cols=3)
# multiply
C = multiply(A,B)
# add bias
D = bias(C, rows=3, cols=3, bias=[0,0,0])
# scale
E = scale(D, rows=3, cols=3, scale=30)
# activate
F = activate(E, rows=3, cols=3)
# quantize
G = requantize(F, rows=3, cols=3, scale=1, zero_point=0, dtype='int8',rounding='nearest', saturate=True)

# quantize

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nA x B =")
print(C)

print("\n(AxB)+c =")
print(D)

print("\n Scale (AxB + c) =")
print(E)

print("\nRelu( Scale(AxB + c)) =")
print(F)

print("\n Quantize Relu( Scale(AxB + c)) =")
print(G)
