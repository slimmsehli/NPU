import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
import math

# ============================================================
# LOAD REAL GPT-2 WEIGHTS
# ============================================================
model_name = "openai-community/gpt2"
print("Loading GPT-2 model and tokenizer...")
model = GPT2Model.from_pretrained(model_name)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Extract weights as numpy arrays for clarity
W = model.state_dict()

# Embedding tables
wte = W['wte.weight'].numpy()          # [50257, 768]
wpe = W['wpe.weight'].numpy()          # [1024,  768]

# Block 0 - LayerNorm 1
ln1_gamma = W['h.0.ln_1.weight'].numpy()   # [768]
ln1_beta  = W['h.0.ln_1.bias'].numpy()     # [768]

# Block 0 - Attention
# NOTE: GPT-2 Conv1D stores weights TRANSPOSED vs PyTorch Linear
# Conv1D weight shape is [in, out] not [out, in]
c_attn_w = W['h.0.attn.c_attn.weight'].numpy()   # [768, 2304]
c_attn_b = W['h.0.attn.c_attn.bias'].numpy()     # [2304]
c_proj_w = W['h.0.attn.c_proj.weight'].numpy()   # [768, 768]
c_proj_b = W['h.0.attn.c_proj.bias'].numpy()     # [768]

# Block 0 - LayerNorm 2
ln2_gamma = W['h.0.ln_2.weight'].numpy()   # [768]
ln2_beta  = W['h.0.ln_2.bias'].numpy()     # [768]

# Block 0 - MLP
c_fc_w   = W['h.0.mlp.c_fc.weight'].numpy()     # [768,  3072]
c_fc_b   = W['h.0.mlp.c_fc.bias'].numpy()       # [3072]
c_proj2_w = W['h.0.mlp.c_proj.weight'].numpy()  # [3072, 768]
c_proj2_b = W['h.0.mlp.c_proj.bias'].numpy()    # [768]

print("Weights loaded.\n")

# ============================================================
# INPUT: TOKENIZE A SENTENCE
# ============================================================

input_text = "Hello, I am a language model"
token_ids = tokenizer.encode(input_text)
seq_len = len(token_ids)
print(f"Input text   : '{input_text}'")
print(f"Token IDs    : {token_ids}")
print(f"Sequence len : {seq_len}\n")

# ============================================================
# HELPER FUNCTIONS
# (Each one maps directly to one NPU hardware block)
# ============================================================

def hw_embedding_lookup(table, ids):
    """
    EMBEDDING LOOKUP — BRAM read
    For each id, fetch the corresponding row from the table.
    No arithmetic, pure memory lookup.
    table : [vocab_size, dim]
    ids   : list of integers
    return: [len(ids), dim]
    """
    return table[ids]   # numpy fancy indexing = row fetch


def hw_elementwise_add(a, b):
    """
    RESIDUAL ADD UNIT
    Elementwise addition of two tensors of identical shape.
    Used for: embedding sum, residual connections.
    a, b : [seq_len, dim]
    return: [seq_len, dim]
    """
    return a + b


def hw_layernorm(x, gamma, eps=1e-5):
    """
    LAYER NORM UNIT — two-pass operation
    Pass 1: compute mean and variance across last dimension
    Pass 2: normalize, then apply learned gamma and beta (via VPU)
    x     : [seq_len, dim]
    gamma, beta: [dim]  — learned parameters
    return: [seq_len, dim]

    RTL note: beta is handled by VPU bias add after this block.
              Split into LayerNorm unit (stats+normalize) + VPU (gamma scale + beta bias)
    """
    # --- Pass 1: stats computation ---
    mean = np.mean(x, axis=-1, keepdims=True)          # [seq_len, 1]
    var  = np.var(x,  axis=-1, keepdims=True)          # [seq_len, 1]
    inv_std = 1.0 / np.sqrt(var + eps)                 # [seq_len, 1]  ← reciprocal sqrt LUT in HW

    print(f"  [LayerNorm] mean range    : {mean.min():.4f} to {mean.max():.4f}")
    print(f"  [LayerNorm] var range     : {var.min():.4f} to {var.max():.4f}")
    print(f"  [LayerNorm] inv_std range : {inv_std.min():.4f} to {inv_std.max():.4f}")

    # --- Pass 2: normalize + scale (gamma) ---
    x_norm = (x - mean) * inv_std                      # [seq_len, dim]
    return gamma * x_norm                              # gamma applied here, beta added by VPU next


def hw_vpu_bias(x, beta):
    """
    VPU — BIAS ADD
    Elementwise add of bias vector broadcast across seq_len.
    x    : [seq_len, dim]
    beta : [dim]
    return: [seq_len, dim]
    """
    return x + beta


def hw_mac_matmul(x, weight, bias):
    """
    MAC ARRAY — matrix multiply + bias
    This is the core compute block. All Conv1D layers map here.

    GPT-2 Conv1D weight layout is [in_features, out_features]
    so the matmul is x @ weight (no transpose needed, unlike nn.Linear)

    x      : [seq_len, in_features]
    weight : [in_features, out_features]
    bias   : [out_features]
    return : [seq_len, out_features]

    RTL note: your MAC array computes x @ weight row by row,
              then VPU adds bias elementwise.
    """
    result = x @ weight    # [seq_len, out_features]
    return result + bias   # bias broadcast across seq_len via VPU


def hw_split_qkv(qkv, n_heads=12, head_dim=64):
    """
    QKV SPLIT — address generation only, no compute
    Split the fused [seq_len, 2304] output into Q, K, V
    then reshape each into [n_heads, seq_len, head_dim]

    RTL note: this is purely how you address your activation BRAM.
              First 768 columns = Q, next 768 = K, last 768 = V.
              Then stride by head_dim to get each head's slice.
    """
    seq_len = qkv.shape[0]
    d_model = n_heads * head_dim   # 768

    Q = qkv[:, :d_model]            # [seq_len, 768]
    K = qkv[:, d_model:2*d_model]   # [seq_len, 768]
    V = qkv[:, 2*d_model:]          # [seq_len, 768]

    # Reshape to multi-head: [n_heads, seq_len, head_dim]
    Q = Q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)

    return Q, K, V   # each [n_heads, seq_len, head_dim]


def hw_attention_scores(Q, K, head_dim=64):
    """
    MAC ARRAY — QK^T + scaling
    For each head compute Q @ K^T then scale by 1/sqrt(head_dim)

    Q : [n_heads, seq_len, head_dim]
    K : [n_heads, seq_len, head_dim]
    return: [n_heads, seq_len, seq_len]

    RTL note: K is fed to MAC array with transposed addressing.
              Scaling (multiply by 0.125) is done by VPU scale.
    """
    scale = 1.0 / math.sqrt(head_dim)   # 0.125 for head_dim=64
    scores = Q @ K.transpose(0, 2, 1)   # [n_heads, seq_len, seq_len]
    scores = scores * scale

    print(f"  [Attn scores] shape    : {scores.shape}")
    print(f"  [Attn scores] range    : {scores.min():.4f} to {scores.max():.4f}")
    return scores


def hw_causal_mask(scores):
    """
    VPU ADD — causal mask application
    Add -inf to all positions where future tokens would attend.
    Upper triangle (excluding diagonal) gets -1e9 (large negative).

    scores : [n_heads, seq_len, seq_len]
    return : [n_heads, seq_len, seq_len]

    RTL note: the mask matrix is constant, precomputed and stored
              in a dedicated BRAM. VPU adds it elementwise to scores.
              In fixed-point use the most negative representable value.
    """
    seq_len = scores.shape[-1]
    # Upper triangle mask: position [i,j] is masked if j > i
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)   # [seq_len, seq_len]
    mask = mask * -1e9                                  # large negative = -inf after softmax
    return scores + mask   # broadcast across heads


def hw_softmax(x):
    """
    SOFTMAX UNIT — numerically stable, row-wise
    Step 1: find row max
    Step 2: subtract max (prevents exp overflow)
    Step 3: compute exp via LUT
    Step 4: sum exp values
    Step 5: divide (multiply by reciprocal)

    x      : [n_heads, seq_len, seq_len]
    return : [n_heads, seq_len, seq_len]  values in [0, 1]

    RTL note: processes one row at a time.
              Needs line buffer to store exp values between step 3 and step 5.
              exp() implemented as LUT with linear interpolation.
              Division implemented as reciprocal LUT * multiply.
    """
    # Step 1+2: subtract max per row for numerical stability
    x_max  = np.max(x, axis=-1, keepdims=True)    # [n_heads, seq_len, 1]
    x_shift = x - x_max

    # Step 3: exp
    exp_x  = np.exp(x_shift)                       # [n_heads, seq_len, seq_len]

    # Step 4+5: normalize
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)  # [n_heads, seq_len, 1]
    attn_w  = exp_x / sum_exp                         # [n_heads, seq_len, seq_len]

    print(f"  [Softmax] attn weights range: {attn_w.min():.4f} to {attn_w.max():.4f}")
    print(f"  [Softmax] row sums (should be 1.0): {attn_w[0].sum(axis=-1)[:3]}")
    return attn_w


def hw_attention_output(attn_weights, V):
    """
    MAC ARRAY — weighted sum of values
    attn_weights : [n_heads, seq_len, seq_len]
    V            : [n_heads, seq_len, head_dim]
    return       : [seq_len, n_heads*head_dim] = [seq_len, 768]

    RTL note: one matmul per head, results concatenated by address layout.
    """
    n_heads  = attn_weights.shape[0]
    seq_len  = attn_weights.shape[1]
    head_dim = V.shape[2]

    out = attn_weights @ V    # [n_heads, seq_len, head_dim]

    # Concatenate heads: transpose back and reshape
    out = out.transpose(1, 0, 2)              # [seq_len, n_heads, head_dim]
    out = out.reshape(seq_len, n_heads * head_dim)   # [seq_len, 768]

    print(f"  [Attn output] shape: {out.shape}")
    return out


def hw_gelu(x):
    """
    VPU — GELU ACTIVATION (NewGELU variant used by GPT-2)
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    RTL note: implement as piecewise linear LUT in BRAM.
              Precompute ~512 entries in Python covering range [-4, +4].
              Outside this range: clamp to 0 (x << 0) or x (x >> 4).
              Index LUT using upper bits of fixed-point input.
    """
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# ============================================================
# FULL GPT-2 BLOCK 0 FORWARD PASS — STEP BY STEP
# ============================================================

print("=" * 60)
print("STEP 1: Embedding Lookup")
print("=" * 60)
positions = np.arange(seq_len)                          # [0, 1, 2, ...]
tok_emb = hw_embedding_lookup(wte, token_ids)           # [seq_len, 768]
pos_emb = hw_embedding_lookup(wpe, positions)           # [seq_len, 768]
print(f"  tok_emb shape : {tok_emb.shape}")
print(f"  pos_emb shape : {pos_emb.shape}")

print("\nSTEP 2: Embedding Add (Residual Add unit)")
print("=" * 60)
x = hw_elementwise_add(tok_emb, pos_emb)               # [seq_len, 768]
print(f"  x shape : {x.shape}")
print(f"  x range : {x.min():.4f} to {x.max():.4f}")

# ---- save block input for first residual connection ----
residual_1 = x.copy()   # this stays alive until after c_proj

print("\nSTEP 3: LayerNorm 1 (ln_1) — pre-attention norm")
print("=" * 60)
x = hw_layernorm(x, ln1_gamma)    # gamma applied inside
x = hw_vpu_bias(x, ln1_beta)      # beta added by VPU bias
print(f"  After ln_1 shape : {x.shape}")
print(f"  After ln_1 range : {x.min():.4f} to {x.max():.4f}")

print("\nSTEP 4: Fused QKV Projection (c_attn) — MAC array")
print("=" * 60)
qkv = hw_mac_matmul(x, c_attn_w, c_attn_b)            # [seq_len, 2304]
print(f"  QKV shape : {qkv.shape}   (will be split into 3 x [seq_len, 768])")

print("\nSTEP 5: Split QKV + reshape to heads — address generation")
print("=" * 60)
Q, K, V = hw_split_qkv(qkv, n_heads=12, head_dim=64)
print(f"  Q shape : {Q.shape}   [n_heads, seq_len, head_dim]")
print(f"  K shape : {K.shape}")
print(f"  V shape : {V.shape}")

print("\nSTEP 6: Attention Scores QK^T + Scale — MAC array + VPU scale")
print("=" * 60)
scores = hw_attention_scores(Q, K, head_dim=64)        # [12, seq_len, seq_len]

print("\nSTEP 7: Causal Mask — VPU add with mask from BRAM")
print("=" * 60)
scores = hw_causal_mask(scores)
print(f"  Masked scores range : {scores[scores > -1e8].min():.4f} to {scores.max():.4f}")
print(f"  Masked positions    : {(scores < -1e8).sum()} elements set to -inf")

print("\nSTEP 8: Softmax — Softmax unit")
print("=" * 60)
attn_weights = hw_softmax(scores)                      # [12, seq_len, seq_len]

print("\nSTEP 9: Attention Output (attn_weights @ V) — MAC array")
print("=" * 60)
attn_out = hw_attention_output(attn_weights, V)        # [seq_len, 768]

print("\nSTEP 10: Output Projection (c_proj) — MAC array + VPU bias")
print("=" * 60)
attn_out = hw_mac_matmul(attn_out, c_proj_w, c_proj_b)  # [seq_len, 768]
print(f"  c_proj output shape : {attn_out.shape}")
print(f"  c_proj output range : {attn_out.min():.4f} to {attn_out.max():.4f}")

print("\nSTEP 11: First Residual Add — Residual Add unit")
print("=" * 60)
x = hw_elementwise_add(attn_out, residual_1)           # [seq_len, 768]
print(f"  After residual 1 range : {x.min():.4f} to {x.max():.4f}")

# ---- save for second residual connection ----
residual_2 = x.copy()   # stays alive until after mlp c_proj

print("\nSTEP 12: LayerNorm 2 (ln_2) — pre-MLP norm")
print("=" * 60)
x = hw_layernorm(x, ln2_gamma)
x = hw_vpu_bias(x, ln2_beta)
print(f"  After ln_2 range : {x.min():.4f} to {x.max():.4f}")

print("\nSTEP 13: MLP FC Layer (c_fc) expand 768->3072 — MAC array + VPU bias")
print("=" * 60)
x = hw_mac_matmul(x, c_fc_w, c_fc_b)                  # [seq_len, 3072]
print(f"  After c_fc shape : {x.shape}")
print(f"  After c_fc range : {x.min():.4f} to {x.max():.4f}")

print("\nSTEP 14: GELU Activation — VPU GELU LUT")
print("=" * 60)
x = hw_gelu(x)                                          # [seq_len, 3072]
print(f"  After GELU range : {x.min():.4f} to {x.max():.4f}")

print("\nSTEP 15: MLP Proj Layer (c_proj) contract 3072->768 — MAC array + VPU bias")
print("=" * 60)
x = hw_mac_matmul(x, c_proj2_w, c_proj2_b)            # [seq_len, 768]
print(f"  After c_proj shape : {x.shape}")
print(f"  After c_proj range : {x.min():.4f} to {x.max():.4f}")

print("\nSTEP 16: Second Residual Add — Residual Add unit")
print("=" * 60)
x = hw_elementwise_add(x, residual_2)                  # [seq_len, 768]
print(f"  Final block output shape : {x.shape}")
print(f"  Final block output range : {x.min():.4f} to {x.max():.4f}")

# ============================================================
# VERIFICATION — compare against HuggingFace reference
# ============================================================

print("\n" + "=" * 60)
print("VERIFICATION against HuggingFace reference output")
print("=" * 60)

input_ids = torch.tensor([token_ids])
with torch.no_grad():
    ref_out = model(input_ids, output_hidden_states=True)

# HuggingFace output after block 0 is hidden_states[1]
ref_block0 = ref_out.hidden_states[1].squeeze(0).numpy()   # [seq_len, 768]

max_diff = np.max(np.abs(x - ref_block0))
mean_diff = np.mean(np.abs(x - ref_block0))
print(f"  Max absolute difference  : {max_diff:.6f}")
print(f"  Mean absolute difference : {mean_diff:.6f}")
print(f"  Result: {'PASS ✓' if max_diff < 1e-3 else 'FAIL ✗ — check implementation'}")
