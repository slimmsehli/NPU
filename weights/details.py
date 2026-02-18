
#######################
#### 3. How many layers does the model have?
#######################

from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("bert-base-uncased")

# ── Count all named parameter tensors ──
all_params = list(model.named_parameters())
print(f"Total weight tensors : {len(all_params)}")

# ── Count logical transformer blocks ──
# For BERT, each block is model.encoder.layer[i]
num_blocks = len(model.encoder.layer)
print(f"Transformer blocks   : {num_blocks}")

# ── Print a clean numbered list ──
for i, (name, param) in enumerate(all_params):
    print(f"[{i:03d}] {name:60s} {list(param.shape)}")

"""
This will print something like:

Total weight tensors : 199
Transformer blocks   : 12

[000] embeddings.word_embeddings.weight          [30522, 768]
[001] embeddings.position_embeddings.weight      [512, 768]
[002] encoder.layer.0.attention.self.query.weight [768, 768]
...
"""
#######################
#### 4. Dimensions of each layer
#######################

from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

print(f"{'Index':<6} {'Layer Name':<65} {'Shape':<25} {'# Params'}")
print("-" * 115)

total = 0
for i, (name, param) in enumerate(model.named_parameters()):
    n = param.numel()
    total += n
    print(f"[{i:<4}] {name:<65} {str(list(param.shape)):<25} {n:,}")

print("-" * 115)
print(f"Total parameters: {total:,}")

#######################
#### 5. What operations are performed per layer?
#######################
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

# ── Print every sub-module with its type ──
print(f"{'Module Name':<65} {'Type'}")
print("-" * 95)
for name, module in model.named_modules():
    if name:  # skip the root
        print(f"{name:<65} {type(module).__name__}")

"""
This will show you something like:

embeddings.word_embeddings                      Embedding
embeddings.LayerNorm                            LayerNorm
encoder.layer.0.attention.self.query            Linear
encoder.layer.0.attention.self.key              Linear
encoder.layer.0.attention.self.value            Linear
encoder.layer.0.attention.output.dense          Linear
encoder.layer.0.attention.output.LayerNorm      LayerNorm
encoder.layer.0.intermediate.dense              Linear        ← GELU activation after this
encoder.layer.0.output.dense                    Linear
encoder.layer.0.output.LayerNorm                LayerNorm
...
"""


