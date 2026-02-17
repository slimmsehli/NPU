
from transformers import AutoModel, AutoTokenizer
import torch
import json
import numpy as np

step1 = 1
step2 = 1

############################################################
############### Download & Extract Weights
############################################################
if (step1==1):
	# ── 1. Download a model ──────────────────────────────────────────────────────
	model_name = "bert-base-uncased"  # swap for any HF model

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
	model.eval()

	# ── 2. Inspect layers ────────────────────────────────────────────────────────
	print("=== Model Architecture ===")
	print(model)

	print("\n=== Named Layers & Shapes ===")
	for name, param in model.named_parameters():
		print(f"{name:60s} | shape: {str(param.shape):30s} | dtype: {param.dtype}")


############### Extract & Save Weights
if (step2==1):
	# ── 3. Extract all weights as numpy arrays ───────────────────────────────────
	weights = {}
	for name, param in model.named_parameters():
		weights[name] = param.detach().cpu().numpy()

	print(f"\nExtracted {len(weights)} weight tensors")

	# ── 4. Save in multiple formats ──────────────────────────────────────────────

	# Option A: NPZ (numpy, great for custom hardware loaders)
	np.savez("model_weights.npz", **weights)
	print("Saved: model_weights.npz")

	# Option B: Raw binary per layer (useful for direct memory-mapped loading on NPU)
	import os
	os.makedirs("weights_bin", exist_ok=True)
	metadata = {}

	for name, arr in weights.items():
		safe_name = name.replace("/", "_").replace(".", "_")
		filepath = f"weights_bin/{safe_name}.bin"
		arr.astype(np.int8).tofile(filepath)  # raw float32 binary -> int8
		metadata[name] = {
		    "file": filepath,
		    "shape": list(arr.shape),
		    "dtype": str(arr.dtype),
		    "size_bytes": arr.nbytes,
		}

	# Save metadata manifest (your NPU driver can read this)
	with open("weights_bin/manifest.json", "w") as f:
		  json.dump(metadata, f, indent=2)

	print("Saved raw binaries + manifest.json")


############################################################
############### Quantize Before Export (Optional but common for NPUs)
############################################################
# Convert float32 → int8 for NPU efficiency
def quantize_to_int8(arr):
  scale = np.max(np.abs(arr)) / 127.0
  quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
  return quantized, scale

os.makedirs("weights_int8", exist_ok=True)
quant_metadata = {}

for name, arr in weights.items():
  safe_name = name.replace("/", "_").replace(".", "_")
  q_arr, scale = quantize_to_int8(arr)
  q_arr.tofile(f"weights_int8/{safe_name}.bin")
  quant_metadata[name] = {
      "shape": list(arr.shape),
      "scale": float(scale),
      "dtype": "int8",
  }

with open("weights_int8/manifest.json", "w") as f:
	json.dump(quant_metadata, f, indent=2)

print("Saved INT8 quantized weights")

############################################################
############### Load a Specific Layer (for verification)
############################################################

# Reload and verify a specific layer
data = np.load("model_weights.npz")
layer = data["encoder.layer.0.attention.self.query.weight"]
print(f"Query weight shape: {layer.shape}")
print(f"Sample values: {layer[0, :5]}")






