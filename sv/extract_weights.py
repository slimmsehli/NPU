import torch
from transformers import GPT2LMHeadModel
import numpy as np

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Access state dictionary (all weights)
state_dict = model.state_dict()

# Explore what's inside
for name, param in state_dict.items():
    print(f"{name}: shape={param.shape}, dtype={param.dtype}")
    
# Example output:
# transformer.wte.weight: shape=torch.Size([50257, 768]), dtype=torch.float32
# transformer.h.0.attn.c_attn.weight: shape=torch.Size([768, 2304]), dtype=torch.float32
# ~/.cache/huggingface/


import os
import numpy as np

# Create directory for weights
os.makedirs('gpt2_weights', exist_ok=True)

# Save each layer's weights
for name, param in state_dict.items():
    # Convert to numpy
    weight_np = param.detach().cpu().numpy()
    
    # Save as binary file
    filename = name.replace('.', '_') + '.npy'
    np.save(f'gpt2_weights/{filename}', weight_np)
    
    print(f"Saved {filename}: shape={weight_np.shape}, size={weight_np.nbytes/1024:.2f} KB")
    
    
########### 1. Post-Training Quantization (Simplest)
def quantize_weight(weight_fp32, bits=8):
    """
    Quantize floating point weights to integer representation
    Returns: quantized weights, scale factor, zero point
    """
    # Determine quantization range
    if bits == 8:
        qmin, qmax = -128, 127  # INT8
    elif bits == 4:
        qmin, qmax = -8, 7      # INT4
    
    # Calculate scale and zero point
    min_val = weight_fp32.min()
    max_val = weight_fp32.max()
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    
    # Quantize
    weight_q = np.clip(np.round(weight_fp32 / scale + zero_point), qmin, qmax)
    weight_q = weight_q.astype(np.int8)
    
    return weight_q, scale, zero_point

# Example usage
for name, param in state_dict.items():
    weight_fp32 = param.detach().cpu().numpy()
    weight_q, scale, zero_point = quantize_weight(weight_fp32, bits=8)
    
    # Save quantized weights
    np.savez(f'gpt2_weights_quantized/{name}.npz',
             weight=weight_q,
             scale=scale,
             zero_point=zero_point)

#################### 2. Symmetric Quantization (Hardware-Friendly)
def symmetric_quantize(weight_fp32, bits=8):
    """
    Symmetric quantization: simpler, no zero-point
    Good for hardware implementation
    """
    if bits == 8:
        qmax = 127  # INT8 symmetric range [-127, 127]
    
    # Scale based on maximum absolute value
    scale = np.abs(weight_fp32).max() / qmax
    
    # Quantize
    weight_q = np.clip(np.round(weight_fp32 / scale), -qmax, qmax)
    weight_q = weight_q.astype(np.int8)
    
    return weight_q, scale

# Dequantization (for verification):
def dequantize(weight_q, scale):
    return weight_q.astype(np.float32) * scale


#############################   3. Per-Channel Quantization (Better Accuracy)
def per_channel_quantize(weight_fp32, axis=0, bits=8):
    """
    Quantize each channel independently for better accuracy
    Commonly used for Conv/Linear layer weights
    """
    qmax = 127
    
    # Calculate scale per channel
    scales = np.abs(weight_fp32).max(axis=tuple(i for i in range(len(weight_fp32.shape)) if i != axis), keepdims=True) / qmax
    scales = np.squeeze(scales)
    
    # Quantize
    weight_q = np.clip(np.round(weight_fp32 / scales.reshape([-1] + [1]*(len(weight_fp32.shape)-1))), -qmax, qmax)
    weight_q = weight_q.astype(np.int8)
    
    return weight_q, scales


############################### Complete Quantization Example
import torch
from transformers import GPT2LMHeadModel
import numpy as np
import os

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

os.makedirs('gpt2_quantized', exist_ok=True)

# Quantize and save all layers
quantization_info = {}

for name, param in model.state_dict().items():
    weight_fp32 = param.detach().cpu().numpy()
    
    # Use symmetric quantization
    weight_q, scale = symmetric_quantize(weight_fp32, bits=8)
    
    # Save quantized weights
    np.savez_compressed(f'gpt2_quantized/{name.replace(".", "_")}.npz',
                       weight=weight_q,
                       scale=scale,
                       original_shape=weight_fp32.shape)
    
    # Track quantization info
    quantization_info[name] = {
        'shape': weight_fp32.shape,
        'scale': float(scale) if np.isscalar(scale) else scale.tolist(),
        'bits': 8,
        'original_size_mb': weight_fp32.nbytes / (1024**2),
        'quantized_size_mb': weight_q.nbytes / (1024**2)
    }

# Save metadata
import json
with open('gpt2_quantized/quantization_info.json', 'w') as f:
    json.dump(quantization_info, f, indent=2)

print(f"Original model size: {sum(info['original_size_mb'] for info in quantization_info.values()):.2f} MB")
print(f"Quantized model size: {sum(info['quantized_size_mb'] for info in quantization_info.values()):.2f} MB")
```

## What Your NPU Needs to Support

### **1. Operations/Layers**

For GPT-2, your NPU must implement:
- **Matrix Multiplication** (GEMM): Most critical operation
- **Layer Normalization**: x_norm = (x - mean) / sqrt(variance + epsilon)
- **Activation Functions**: GELU (Gaussian Error Linear Unit)
- **Softmax**: For attention scores
- **Element-wise operations**: Add, multiply

### **2. Data Flow**
```
Input Token IDs (batch_size × seq_len)
    ↓
Embedding Lookup (vocab_size × hidden_dim)
    ↓
For each Transformer Block:
    ↓
    Multi-Head Attention:
        - Q, K, V projections (3 matrix multiplications)
        - Scaled dot-product attention
        - Output projection
    ↓
    Layer Norm + Residual
    ↓
    Feed-Forward Network:
        - Linear layer 1: hidden_dim → 4*hidden_dim (GELU)
        - Linear layer 2: 4*hidden_dim → hidden_dim
    ↓
    Layer Norm + Residual
    ↓
Final Layer Norm
    ↓
Output Projection (hidden_dim → vocab_size)
    ↓
Softmax → Token Probabilities


############################# 3. Memory Layout for NPU
 #Your NPU needs to understand how to load weights:
# Create a binary format your NPU can parse
def create_npu_format(state_dict, output_file='model.bin'):
    """
    Custom binary format for NPU
    Format:
    - Header: magic number, version, num_layers
    - For each layer:
        - Layer type (uint32)
        - Dimensions (uint32 array)
        - Quantization params (scale, zero_point)
        - Weight data (int8 array)
    """
    with open(output_file, 'wb') as f:
        # Magic number and version
        f.write(np.uint32(0xDEADBEEF).tobytes())  # Magic
        f.write(np.uint32(1).tobytes())           # Version
        f.write(np.uint32(len(state_dict)).tobytes())  # Num layers
        
        for name, param in state_dict.items():
            weight_fp32 = param.detach().cpu().numpy()
            weight_q, scale = symmetric_quantize(weight_fp32)
            
            # Write layer metadata
            f.write(np.uint32(len(name)).tobytes())
            f.write(name.encode('utf-8'))
            
            # Write dimensions
            f.write(np.uint32(len(weight_q.shape)).tobytes())
            for dim in weight_q.shape:
                f.write(np.uint32(dim).tobytes())
            
            # Write quantization scale
            f.write(np.float32(scale).tobytes())
            
            # Write quantized weights
            f.write(weight_q.tobytes())

# Usage
create_npu_format(model.state_dict(), 'gpt2_npu.bin')


############### Testing Your Quantized Model
def test_quantization_accuracy():
    """Compare FP32 vs quantized inference"""
    model_fp32 = GPT2LMHeadModel.from_pretrained('gpt2')
    model_fp32.eval()
    
    # Test input
    input_ids = torch.tensor([[15496, 11, 616, 1438, 318]])  # "Hello, my name is"
    
    # FP32 inference
    with torch.no_grad():
        output_fp32 = model_fp32(input_ids).logits
    
    # Simulate quantized inference (pseudo-code)
    # You'll implement this in your NPU
    # output_quantized = run_quantized_model(input_ids)
    
    # Calculate difference
    # diff = torch.abs(output_fp32 - output_quantized).mean()
    # print(f"Mean absolute difference: {diff}")

test_quantization_accuracy()


######################################
Key Files Your NPU Should Load

Weights: *.npy or custom binary format
Architecture config: JSON describing layer types, dimensions
Quantization params: Scales, zero-points per layer

{
  "model_type": "gpt2",
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "vocab_size": 50257,
  "layers": [
    {
      "name": "transformer.wte",
      "type": "embedding",
      "shape": [50257, 768],
      "quantized": true,
      "scale": 0.0234
    }
  ]
}


