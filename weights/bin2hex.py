import numpy as np
import struct
import json

def bin_to_hex(bin_path, hex_path, dtype=np.float32):
    """
    Convert a raw binary weight file to a hex file readable by $readmemh() in Verilog.
    Each line = one value in hex (32-bit for float32, 16-bit for float16, 8-bit for int8)
    """
    arr = np.fromfile(bin_path, dtype=dtype)
    
    with open(hex_path, "w") as f:
        for val in arr.flatten():
            if dtype == np.float32:
                # Pack float32 as 4 bytes, format as 8 hex chars
                hex_val = struct.pack(">f", float(val)).hex()  # big-endian
            elif dtype == np.float16:
                hex_val = struct.pack(">e", float(val)).hex()  # 4 hex chars
            elif dtype == np.int8:
                hex_val = f"{val & 0xFF:02x}"                  # 2 hex chars
            f.write(hex_val + "\n")
    
    print(f"Written {len(arr)} values → {hex_path}")

# Convert all layers using the manifest
import os
os.makedirs("weights_hex", exist_ok=True)

with open("weights_bin/manifest.json", "r") as f:
    manifest = json.load(f)

for layer_name, info in manifest.items():
    safe_name = layer_name.replace("/", "_").replace(".", "_")
    bin_path = info["file"]
    hex_path = f"weights_hex/{safe_name}.hex"
    bin_to_hex(bin_path, hex_path, dtype=np.float32)
    print(f"  {layer_name} → {hex_path}")
