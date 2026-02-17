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

############# Layer
def layer( name, matA, matB, rows=3, cols=3, en_bias=0, en_scale=0, en_activation=0, en_quantize=0,
           bias_col=[0,0,0], scale_factor=1, quantize_type="int8", DEBUG=0):
  print(f"\n [INFO] - Layer [{name}] started ... ")
  if (DEBUG):
    print(f"\n [INFO] - Layer [{name}] Input Matrix:")
    print(matA)
    print(f"\n [INFO] - Layer [{name}] Weights Matrix:")
    print(matB)
  
  C = multiply(matA,matB)
  if (DEBUG):
      print(f"\n [INFO] - Layer [{name}] Post Multiplication:")
      print(C)
      
  if en_bias:
    C = bias(C, rows=rows, cols=cols, bias=bias_col)
    if (DEBUG):
      print(f"\n [INFO] - Layer [{name}] Post Bias:")
      print(C)
  
  if en_scale:
    C = scale(C, rows=rows, cols=cols, scale=scale_factor)
    if (DEBUG):
      print(f"\n [INFO] - Layer [{name}] Post Scale:")
      print(C)
  
  if en_activation:
    C =  activate(C, rows=rows, cols=cols),
    if (DEBUG):
      print(f"\n [INFO] - Layer [{name}] Post Activation:")
      print(C)
  
  if en_quantize:
    C = requantize(C, rows=rows, cols=cols, scale=1, zero_point=0, dtype=quantize_type,rounding='nearest', saturate=True)
    if (DEBUG):
      print(f"\n [INFO] - Layer [{name}] Post quantization:")
      print(C)
  
  print(f"\n [INFO] - Layer [{name}] ended ... ")
  return C

############# save output 

import numpy as np

def save_output(matrix, path, bits=16, signed=True):
    """
    Save a numeric matrix as integer HEX tokens (two's complement if signed=True).
    Each row is written on a new line, tokens separated by spaces.
    
    Parameters:
        matrix : array-like (2D). Can be float or int; will be rounded and cast.
        path   : output file path (str)
        bits   : 8, 16, or 32
        signed : if True, two's complement signed range; else unsigned
    """
    mat = np.array(matrix, dtype=np.float64)  # work in float; we'll quantize
    # Round to nearest integer before casting
    mat_int = np.rint(mat).astype(np.int64)

    if bits not in (8, 16, 32):
        raise ValueError("bits must be one of: 8, 16, 32")

    if signed:
        minv = -(1 << (bits - 1))
        maxv = (1 << (bits - 1)) - 1
    else:
        minv = 0
        maxv = (1 << bits) - 1

    # Saturate to valid range
    mat_int = np.clip(mat_int, minv, maxv)

    # Convert to two's complement representation if signed
    mask = (1 << bits) - 1
    mat_twos = (mat_int.astype(np.int64) & mask)

    with open(path, "w") as f:
        for r in range(mat_twos.shape[0]):
            row_tokens = []
            for v in mat_twos[r]:
                row_tokens.append(format(int(v), f"0{bits//16}x"))
            f.write("\n".join(row_tokens) + "\n")


############# TEST for 3 layers model
cols = 20
rows = 20
DEBUG = 0
output_file = "output.hex" #"../memories/ref_output.hex"

# load matrices
print(f"\n [INFO] - Loading input Matrix ... \n ")
inp = load_hex_matrix("A.hex", rows=rows, cols=cols) #"../memories/inputs.hex"
print(f"\n [INFO] - Loading Weights Matrices L0,L1 and L2... \n ")
w1 = load_hex_matrix("B.hex", rows=rows, cols=cols) #"../memories/weights_L0.hex"
w2 = load_hex_matrix("B.hex", rows=rows, cols=cols) #"../memories/weights_L1.hex"
w3 = load_hex_matrix("B.hex", rows=rows, cols=cols) #"../memories/weights_L2.hex"

# run the 3 layers
print(f"\n [INFO] - Running Layers... \n ")
L1 = layer("L1", inp, w1, rows=rows, cols=cols, en_bias=0, en_scale=1, en_activation=1, en_quantize=1, bias_col=[0,0,0], scale_factor=1, quantize_type="int8", DEBUG=DEBUG)
L2 = layer("L2", L1, w2, rows=rows, cols=cols, en_bias=0, en_scale=1, en_activation=1, en_quantize=1, bias_col=[0,0,0], scale_factor=1, quantize_type="int8", DEBUG=DEBUG)
L3 = layer("L3", L2, w3, rows=rows, cols=cols, en_bias=0, en_scale=1, en_activation=1, en_quantize=1, bias_col=[0,0,0], scale_factor=1, quantize_type="int8", DEBUG=DEBUG)

# save output matrix to local file
print(f"\n [INFO] - Saving output reference Matrix to {output_file} ... \n ")
save_output(matrix=L3, path=output_file, bits=16, signed=True)





