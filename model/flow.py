import numpy as np
import struct
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ==========================================
# PART 1: THE COMPILER (Your Software Stack)
# ==========================================
class AfricanNPUCompiler:
    def __init__(self):
        self.byte_stream = bytearray()
        
    def add_layer(self, weights, biases):
        # 1. Quantize Weights (Float -> INT8)
        # We scale 0.5 to 64, -0.5 to -64, etc.
        w_min, w_max = weights.min(), weights.max()
        scale = 127.0 / max(abs(w_min), abs(w_max))
        
        q_weights = (weights * scale).astype(np.int8)
        
        rows, cols = q_weights.shape
        print(f"[Compiler] Packing Layer: {cols} Inputs -> {rows} Outputs")
        
        # 2. Create Header [MAGIC, ROWS, COLS]
        self.byte_stream.append(0xAA) # Magic "Start of Layer" byte
        self.byte_stream.append(rows)
        self.byte_stream.append(cols)
        
        # 3. Add Weights Payload (Row by Row)
        # We flatten the matrix into a list of bytes
        flat_weights = q_weights.flatten().tobytes()
        self.byte_stream.extend(flat_weights)
        
        # Note: For this simple MVP, we are ignoring Bias to keep the hardware simple.
        # Real NPUs usually fuse bias into the next step.

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            f.write(self.byte_stream)
        print(f"[Compiler] Saved binary model to {filename} ({len(self.byte_stream)} bytes)\n")

# ==========================================
# PART 2: THE SIMULATOR (Your Hardware)
# ==========================================
class NPUSimulator:
    def __init__(self):
        self.memory = None # This simulates the NPU's internal instruction memory
        
    def load_hex(self, filename):
        with open(filename, 'rb') as f:
            self.memory = f.read()
            
    def run_inference(self, input_vector):
        # Quantize Input Image (0..16 -> 0..127)
        # Scikit-learn digits are 0-16, we scale to match INT8 range
        current_data = (input_vector / 16.0 * 127).astype(np.int8)
        
        ptr = 0
        layer_idx = 1
        
        while ptr < len(self.memory):
            # --- FETCH & DECODE ---
            if self.memory[ptr] != 0xAA: break # Stop if no magic byte
            
            rows = self.memory[ptr+1]
            cols = self.memory[ptr+2]
            ptr += 3 
            
            # --- LOAD WEIGHTS ---
            count = rows * cols
            weight_data = self.memory[ptr : ptr+count]
            ptr += count
            
            # Reconstruct Matrix from bytes
            weights = np.frombuffer(weight_data, dtype=np.int8).reshape(rows, cols)
            
            # --- EXECUTE (The Matrix Multiply) ---
            # This is the step your Verilog hardware will do in parallel
            acc = np.dot(weights, current_data)
            
            # --- ACTIVATION (ReLU) ---
            # Hardware sets all negative numbers to 0
            acc = np.maximum(0, acc)
            
            # --- RE-QUANTIZE ---
            # Keep numbers in 8-bit range for the next layer
            current_data = np.clip(acc // 64, -127, 127).astype(np.int8)
            
            print(f"[NPU Hardware] Finished Layer {layer_idx}. Output shape: {current_data.shape}")
            layer_idx += 1
            
        return current_data

# ==========================================
# PART 3: REAL WORLD USAGE
# ==========================================

# A. Download & Train a Tiny Model (using Scikit-Learn)
print("--- 1. Training AI Model ---")
digits = load_digits() # Load MNIST (8x8 images)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Create a small Neural Net: 64 inputs -> 16 neurons -> 10 outputs (digits 0-9)
# We turn off bias just to simplify our custom NPU for now
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000) 
mlp.fit(X_train, y_train)
print(f"Model Trained. Accuracy: {mlp.score(X_test, y_test)*100:.1f}%")

# B. Extract Weights & Compile
print("--- 2. Compiling for NPU ---")
compiler = AfricanNPUCompiler()

# Scikit-learn stores weights in a list of matrices called 'coefs_'
# Layer 1 (Input -> Hidden)
# We transpose (.T) because sklearn stores (In, Out) but we want (Out, In) for dot product
compiler.add_layer(mlp.coefs_[0].T, mlp.intercepts_[0]) 

# Layer 2 (Hidden -> Output)
compiler.add_layer(mlp.coefs_[1].T, mlp.intercepts_[1])

compiler.save_to_file("mnist_model.bin")

# C. Run on NPU Simulator
print("--- 3. Running on NPU Simulator ---")
npu = NPUSimulator()
npu.load_hex("mnist_model.bin")

# Pick a random test image
test_idx = 42
input_image = X_test[test_idx]
true_label = y_test[test_idx]

# Run!
npu_output = npu.run_inference(input_image)
predicted_label = np.argmax(npu_output)

print(f"\n[RESULT]")
print(f"True Digit:      {true_label}")
print(f"NPU Prediction:  {predicted_label}")

if true_label == predicted_label:
    print("SUCCESS: The NPU works!")
else:
    print("FAIL: Quantization error might be too high.")
