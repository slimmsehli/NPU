import torch
import numpy as np
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import math

# ============================================================
# LOAD REAL GPT-2 WEIGHTS
# ============================================================
model_name = "openai-community/gpt2"


import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
import math

print("Loading GPT-2 model and tokenizer...")
model = GPT2Model.from_pretrained(model_name)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
W = model.state_dict()
print("Weights loaded.\n")


print("State dict keys (first 20):")
for i, k in enumerate(W.keys()):
    print(f"  {k}")
    if i > 20:
        break

# ============================================================
# HELPER FUNCTIONS — each maps to one RTL block
# ============================================================
# ============================================================
# AUTO-DETECT KEY PREFIX
# Handles different HuggingFace versions which use different
# key naming conventions:
#   GPT2LMHeadModel  ->  'transformer.wte.weight'
#   GPT2Model        ->  'wte.weight'
# ============================================================

def get_prefix(W):
    """
    Detect whether state dict keys use 'transformer.' prefix or not.
    Returns the correct prefix string to use for all key lookups.
    """
    if 'transformer.wte.weight' in W:
        prefix = 'transformer.'
        print("  Key format detected: GPT2LMHeadModel  (prefix='transformer.')")
    elif 'wte.weight' in W:
        prefix = ''
        print("  Key format detected: GPT2Model  (prefix='')")
    else:
        # print available keys to help debug
        print("ERROR: Could not detect key format. Available keys:")
        for k in list(W.keys())[:30]:
            print(f"  {k}")
        raise KeyError("Unknown state dict format")
    return prefix

PFX = get_prefix(W)   # global prefix, used everywhere



def hw_embedding_lookup(table, ids):
    """BRAM lookup — no compute"""
    return table[ids]

def hw_elementwise_add(a, b):
    """Residual Add unit"""
    return a + b

def hw_layernorm(x, gamma, beta, eps=1e-5):
    """
    LayerNorm unit — float64 accumulators to match PyTorch exactly.
    RTL note: accumulator must be wider than data path.
    """
    x64     = x.astype(np.float64)
    mean    = np.mean(x64, axis=-1, keepdims=True)
    var     = np.mean((x64 - mean)**2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm  = (x64 - mean) * inv_std
    return (gamma * x_norm.astype(np.float32)) + beta

def hw_mac_matmul(x, weight, bias=None):
    """
    MAC array + optional VPU bias.
    float64 accumulation matches PyTorch matmul precision.
    """
    result = (x.astype(np.float64) @ weight.astype(np.float64)).astype(np.float32)
    if bias is not None:
        result = result + bias
    return result

def hw_split_qkv(qkv, n_heads=12, head_dim=64):
    """Address generation only — no compute"""
    seq_len  = qkv.shape[0]
    d_model  = n_heads * head_dim
    Q = qkv[:, :d_model].reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    K = qkv[:, d_model:2*d_model].reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    V = qkv[:, 2*d_model:].reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    return Q, K, V

def hw_attention_scores(Q, K, head_dim=64):
    """MAC array (K transposed) + VPU scale"""
    return (Q @ K.transpose(0, 2, 1)) * (1.0 / math.sqrt(head_dim))

def hw_causal_mask(scores):
    """VPU add — mask from BRAM"""
    seq_len = scores.shape[-1]
    mask    = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return scores + mask

def hw_softmax(x):
    """Softmax unit — numerically stable row-wise"""
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x   = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def hw_attention_output(attn_weights, V):
    """MAC array — weighted sum of V + head concatenation"""
    n_heads  = V.shape[0]
    seq_len  = attn_weights.shape[1]
    head_dim = V.shape[2]
    return (attn_weights @ V).transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)

def hw_gelu(x):
    """VPU GELU LUT — NewGELU variant"""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# ============================================================
# SINGLE BLOCK FORWARD PASS
# ============================================================

def run_gpt2_block(x, layer_idx, W, verbose=False):

    def vprint(*args):
        if verbose:
            print(*args)

    log    = {}
    prefix = f'{PFX}h.{layer_idx}'    # e.g. 'transformer.h.0' or 'h.0'

    ln1_g   = W[f'{prefix}.ln_1.weight'].numpy()
    ln1_b   = W[f'{prefix}.ln_1.bias'].numpy()
    cattn_w = W[f'{prefix}.attn.c_attn.weight'].numpy()
    cattn_b = W[f'{prefix}.attn.c_attn.bias'].numpy()
    cproj_w = W[f'{prefix}.attn.c_proj.weight'].numpy()
    cproj_b = W[f'{prefix}.attn.c_proj.bias'].numpy()
    ln2_g   = W[f'{prefix}.ln_2.weight'].numpy()
    ln2_b   = W[f'{prefix}.ln_2.bias'].numpy()
    cfc_w   = W[f'{prefix}.mlp.c_fc.weight'].numpy()
    cfc_b   = W[f'{prefix}.mlp.c_fc.bias'].numpy()
    cp2_w   = W[f'{prefix}.mlp.c_proj.weight'].numpy()
    cp2_b   = W[f'{prefix}.mlp.c_proj.bias'].numpy()

    log['input'] = x.copy()

    # --- Attention sub-block ---
    residual_1        = x.copy()
    x_norm            = hw_layernorm(x, ln1_g, ln1_b)
    log['ln1_out']    = x_norm.copy()
    vprint(f"  [L{layer_idx}] ln_1     : {x_norm.shape}  [{x_norm.min():.4f}, {x_norm.max():.4f}]")

    qkv               = hw_mac_matmul(x_norm, cattn_w, cattn_b)
    log['qkv']        = qkv.copy()

    Q, K, V           = hw_split_qkv(qkv)
    scores            = hw_attention_scores(Q, K)
    scores            = hw_causal_mask(scores)
    attn_w            = hw_softmax(scores)
    log['attn_weights'] = attn_w.copy()

    attn_out          = hw_attention_output(attn_w, V)
    attn_out          = hw_mac_matmul(attn_out, cproj_w, cproj_b)
    log['c_proj_out'] = attn_out.copy()

    x                 = hw_elementwise_add(attn_out, residual_1)
    log['after_residual_1'] = x.copy()

    # --- MLP sub-block ---
    residual_2        = x.copy()
    x_norm            = hw_layernorm(x, ln2_g, ln2_b)
    log['ln2_out']    = x_norm.copy()
    vprint(f"  [L{layer_idx}] ln_2     : {x_norm.shape}  [{x_norm.min():.4f}, {x_norm.max():.4f}]")

    x_fc              = hw_mac_matmul(x_norm, cfc_w, cfc_b)
    x_fc              = hw_gelu(x_fc)
    log['gelu_out']   = x_fc.copy()

    x_fc              = hw_mac_matmul(x_fc, cp2_w, cp2_b)
    log['c_proj2_out'] = x_fc.copy()

    x                 = hw_elementwise_add(x_fc, residual_2)
    log['output']     = x.copy()
    vprint(f"  [L{layer_idx}] output   : {x.shape}  [{x.min():.4f}, {x.max():.4f}]")

    return x, log


# ============================================================
# POST-TRANSFORMER BLOCKS
# These run once after all 12 layers — maps to final NPU stages
# ============================================================

def hw_final_layernorm(x, W):
    """
    Final LayerNorm after all transformer blocks.
    GPT-2 key: transformer.ln_f
    RTL: same LayerNorm unit, loaded with ln_f weights.
    """
    gamma = W[f'{PFX}ln_f.weight'].numpy()
    beta  = W[f'{PFX}ln_f.bias'].numpy()
    return hw_layernorm(x, gamma, beta)


def hw_lm_head(x, W):
    """
    Language model head — projects hidden state to vocabulary logits.
    Shape: [seq_len, 768] -> [seq_len, 50257]

    GPT-2 key: lm_head.weight  [50257, 768]
    Note: GPT-2 TIES lm_head weights to wte (same matrix, transposed).
    So lm_head.weight == wte.weight and the matmul is x @ wte.T

    RTL: MAC array. No bias for lm_head in GPT-2.
    This is the largest single matmul: [seq_len x 768] @ [768 x 50257]
    """
    if 'lm_head.weight' in W:
        lm_head_w = W['lm_head.weight'].numpy()
    else:
        # tied to wte — use embedding table transposed
        lm_head_w = W[f'{PFX}wte.weight'].numpy()
    return hw_mac_matmul(x, lm_head_w.T, bias=None)


# ============================================================
# TOKEN SAMPLING STRATEGIES
# These run on host CPU after NPU produces logits
# RTL note: sampling is typically done in software on the host,
#           the NPU just produces the logits vector
# ============================================================

def sample_greedy(logits):
    """
    Greedy decoding — always pick the highest probability token.
    Deterministic, but tends to produce repetitive text.
    RTL: just argmax over the last row of logits. Simple comparator tree.
    """
    return int(np.argmax(logits[-1]))   # only last token's logits matter for next token


def sample_temperature(logits, temperature=1.0):
    """
    Temperature sampling — divide logits by temperature before softmax.
    temperature < 1.0 = more focused/deterministic
    temperature > 1.0 = more random/creative
    """
    last_logits = logits[-1].astype(np.float64)
    last_logits = last_logits / temperature
    probs       = hw_softmax(last_logits.reshape(1, -1)).flatten()
    return int(np.random.choice(len(probs), p=probs))


def sample_topk(logits, k=50, temperature=1.0):
    """
    Top-k sampling — only sample from the k most likely tokens.
    Balances quality and diversity. k=1 is greedy.
    """
    last_logits = logits[-1].astype(np.float64) / temperature
    # zero out everything outside top-k
    top_k_idx   = np.argsort(last_logits)[-k:]
    masked      = np.full_like(last_logits, -1e9)
    masked[top_k_idx] = last_logits[top_k_idx]
    probs       = hw_softmax(masked.reshape(1, -1)).flatten()
    return int(np.random.choice(len(probs), p=probs))


# ============================================================
# FULL MODEL FORWARD PASS — one token step
# Runs embedding + 12 blocks + final LN + lm_head
# Returns logits for next token prediction
# ============================================================

def run_gpt2_forward(token_ids, W, verbose=False):
    """
    Full forward pass for a sequence of token_ids.
    Returns logits [seq_len, vocab_size] and all intermediate logs.
    """

    wte = W[f'{PFX}wte.weight'].numpy()
    wpe = W[f'{PFX}wpe.weight'].numpy()

    positions = np.arange(len(token_ids))
    x         = hw_elementwise_add(
                    hw_embedding_lookup(wte, token_ids),
                    hw_embedding_lookup(wpe, positions)
                )

    if verbose:
        print(f"  [Embedding] {x.shape}  [{x.min():.4f}, {x.max():.4f}]")

    all_logs = {}
    for layer_idx in range(12):
        x, log             = run_gpt2_block(x, layer_idx, W, verbose=verbose)
        all_logs[layer_idx] = log

    x      = hw_final_layernorm(x, W)
    logits = hw_lm_head(x, W)

    if verbose:
        print(f"  [LM Head]   {logits.shape}  [{logits.min():.4f}, {logits.max():.4f}]")

    return logits, all_logs


# ============================================================
# TEXT GENERATION LOOP
# Autoregressively generates tokens one at a time
# Each iteration: run full forward pass, sample next token, append
# ============================================================

def generate_text(prompt, W, model, tokenizer,
                  max_new_tokens=30,
                  mode='greedy',
                  temperature=0.8,
                  top_k=50,
                  verbose_first=False):
    """
    Full autoregressive text generation.

    Flow for each new token:
      1. Run full forward pass on current token sequence  [NPU]
      2. Extract logits for the last token position       [NPU]
      3. Apply sampling strategy                          [Host CPU]
      4. Append new token id to sequence                  [Host CPU]
      5. Repeat until max_new_tokens or EOS token

    Parameters:
      mode: 'greedy' | 'temperature' | 'topk'
    """

    token_ids = tokenizer.encode(prompt)
    print(f"\nPrompt       : '{prompt}'")
    print(f"Prompt tokens: {token_ids}")
    print(f"Sampling mode: {mode}  |  max_new_tokens: {max_new_tokens}")
    print(f"{'─'*60}")

    generated = []

    for step in range(max_new_tokens):

        # --- NPU forward pass ---
        verbose = (step == 0 and verbose_first)
        logits, _ = run_gpt2_forward(token_ids, W, verbose=verbose)

        # --- Sample next token (host side) ---
        if mode == 'greedy':
            next_token = sample_greedy(logits)
        elif mode == 'temperature':
            next_token = sample_temperature(logits, temperature=temperature)
        elif mode == 'topk':
            next_token = sample_topk(logits, k=top_k, temperature=temperature)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        generated.append(next_token)
        token_ids = token_ids + [next_token]

        # decode just the new token for live display
        new_word = tokenizer.decode([next_token])
        print(f"  Step {step+1:3d}  |  token_id: {next_token:6d}  |  '{new_word}'")

        # stop at end-of-sequence token
        if next_token == tokenizer.eos_token_id:
            print("  [EOS token reached]")
            break

    full_output = tokenizer.decode(token_ids)
    print(f"\n{'='*60}")
    print(f"FULL OUTPUT:\n{full_output}")
    print(f"{'='*60}\n")

    return full_output, token_ids


# ============================================================
# VERIFICATION — compare our logits against HuggingFace
# ============================================================

def verify_logits(prompt, W, model, tokenizer):
    """
    Verify our full model output (logits) against HuggingFace.
    This is the ultimate golden reference check.
    """
    print(f"\n{'='*60}")
    print(f"LOGIT VERIFICATION — '{prompt}'")
    print(f"{'='*60}")

    token_ids = tokenizer.encode(prompt)

    # our implementation
    our_logits, _ = run_gpt2_forward(token_ids, W, verbose=False)

    # HuggingFace reference
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        ref_logits = model(input_ids).logits.squeeze(0).numpy()

    max_diff  = np.max(np.abs(our_logits - ref_logits))
    mean_diff = np.mean(np.abs(our_logits - ref_logits))

    # check top-5 predicted next tokens match
    our_top5 = np.argsort(our_logits[-1])[-5:][::-1]
    ref_top5 = np.argsort(ref_logits[-1])[-5:][::-1]

    our_top5_words = [tokenizer.decode([t]) for t in our_top5]
    ref_top5_words = [tokenizer.decode([t]) for t in ref_top5]

    passed = max_diff < 0.1   # logits can have slightly larger abs diff but top-k should match

    print(f"  Logit max_diff   : {max_diff:.4f}")
    print(f"  Logit mean_diff  : {mean_diff:.6f}")
    print(f"  Our  top-5 next tokens: {our_top5_words}")
    print(f"  HF   top-5 next tokens: {ref_top5_words}")
    print(f"  Top-1 match: {'YES ✓' if our_top5[0] == ref_top5[0] else 'NO ✗'}")
    print(f"  Top-5 match: {'YES ✓' if list(our_top5) == list(ref_top5) else 'PARTIAL'}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    return passed


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    np.random.seed(42)

    # --- Verification first ---
    verify_logits(
        "Hello, I am a language model and I will run on an FPGA",
        W, model, tokenizer
    )

    # --- Generation: greedy (deterministic, good for testing) ---
    generate_text(
        prompt          = "Hello, I am a language model and I will run on an FPGA",
        W               = W,
        model           = model,
        tokenizer       = tokenizer,
        max_new_tokens  = 20,
        mode            = 'greedy'
    )

    # --- Generation: top-k sampling (more natural text) ---
    generate_text(
        prompt          = "Hello, I am a language model and I will run on an FPGA",
        W               = W,
        model           = model,
        tokenizer       = tokenizer,
        max_new_tokens  = 20,
        mode            = 'topk',
        temperature     = 0.8,
        top_k           = 50
    )

    # --- Generation: show verbose first step to see full dataflow ---
    generate_text(
        prompt          = "The FPGA implementation of this transformer",
        W               = W,
        model           = model,
        tokenizer       = tokenizer,
        max_new_tokens  = 15,
        mode            = 'greedy',
        verbose_first   = True
    )
