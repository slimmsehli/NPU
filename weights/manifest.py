

import json
import numpy as np

# Load the manifest
with open("weights_bin/manifest.json", "r") as f:
    manifest = json.load(f)

# Browse all layers
for layer_name, info in manifest.items():
    print(f"Layer: {layer_name}")
    print(f"  File  : {info['file']}")
    print(f"  Shape : {info['shape']}")
    print(f"  Dtype : {info['dtype']}")
    print(f"  Bytes : {info['size_bytes']}")
    print()

# Load a specific layer's weights back from binary
def load_layer(layer_name):
    info = manifest[layer_name]
    arr = np.fromfile(info["file"], dtype=info["dtype"])
    arr = arr.reshape(info["shape"])
    return arr

# Example: load the first attention query weights
layer = load_layer("encoder.layer.0.attention.self.query.weight")
print(layer.shape)
print(layer)


