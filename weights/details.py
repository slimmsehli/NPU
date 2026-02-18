from transformers import AutoModel
import torch
model_name = "openai-community/gpt2"

model = AutoModel.from_pretrained(model_name)

print(f"{'Index':<6} {'Layer Name':<50} {'Operation':<20} {'Shape':<20} {'Dtype'}")
print("-" * 115)

total_params = sum(p.numel() for p in model.parameters())

for i, (name, module) in enumerate(model.named_modules()):
    # Filter for leaf modules to see individual operations
    if len(list(module.children())) == 0:
        
        # 1. Get Shape
        if hasattr(module, 'weight') and module.weight is not None:
            shape = str(list(module.weight.shape))
        elif hasattr(module, 'normalized_shape'): 
            shape = str(list(module.normalized_shape))
        else:
            shape = "--"

        # 2. Get Dtype (Precision)
        # We check the first parameter of the module to see its type
        params = list(module.parameters())
        if params:
            dtype = str(params[0].dtype).replace("torch.", "")
        else:
            dtype = "--"

        # 3. Get Operation Name
        operation = module.__class__.__name__
        
        print(f"[{i:<4}] {name:<50} {operation:<20} {shape:<20} {dtype}")
    if i > 20:
      break

print("-" * 115)
print(f"Total parameters: {total_params:,}")
