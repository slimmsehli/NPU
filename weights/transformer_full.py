import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def hw_embedding_lookup(table, ids):
    return table[ids]

def hw_elementwise_add(a, b):
    return a + b

def hw_layernorm(x, gamma, beta, eps=1e-5):
    """
    LayerNorm — matches PyTorch exactly.
    Uses float64 for stats accumulation to avoid precision drift.
    This is important: error in layernorm compounds across 12 layers.
    
    RTL note: your accumulator for mean/variance must be wider than
              your data path. If data is INT16, use INT32 or INT48
              accumulators for the stats computation.
    """
    x64     = x.astype(np.float64)
    mean    = np.mean(x64, axis=-1, keepdims=True)
    var     = np.mean((x64 - mean)**2, axis=-1, keepdims=True)   # biased, matches PyTorch
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm  = (x64 - mean) * inv_std
    return (gamma * x_norm.astype(np.float32)) + beta

def hw_mac_matmul(x, weight, bias):
    """
    MAC array + VPU bias.
    Uses float64 for accumulation to match PyTorch matmul precision.
    
    RTL note: accumulator width is critical here too. INT8 inputs
              with 768-length dot products need at least INT26 accumulators
              (8+8+log2(768)=26) before scaling back down.
    """
    # float64 accumulation matches PyTorch's internal precision
    result = x.astype(np.float64) @ weight.astype(np.float64)
    return result.astype(np.float32) + bias

def hw_split_qkv(qkv, n_heads=12, head_dim=64):
    seq_len  = qkv.shape[0]
    d_model  = n_heads * head_dim
    Q = qkv[:, :d_model].reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    K = qkv[:, d_model:2*d_model].reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    V = qkv[:, 2*d_model:].reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    return Q, K, V

def hw_attention_scores(Q, K, head_dim=64):
    scale = 1.0 / math.sqrt(head_dim)
    return (Q @ K.transpose(0, 2, 1)) * scale

def hw_causal_mask(scores):
    seq_len = scores.shape[-1]
    mask    = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return scores + mask

def hw_softmax(x):
    """
    Softmax — numerically stable.
    RTL note: subtract max before exp to prevent overflow in fixed-point.
    """
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x   = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def hw_attention_output(attn_weights, V):
    seq_len  = attn_weights.shape[1]
    head_dim = V.shape[2]
    n_heads  = V.shape[0]
    return (attn_weights @ V).transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)

def hw_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# ============================================================
# SINGLE BLOCK FORWARD PASS
# ============================================================

def run_gpt2_block(x, layer_idx, W, verbose=False):

    def vprint(*args):
        if verbose:
            print(*args)

    log    = {}
    prefix = f'h.{layer_idx}'

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
    log['residual_1'] = residual_1

    x_norm           = hw_layernorm(x, ln1_g, ln1_b)
    log['ln1_out']   = x_norm.copy()
    vprint(f"  [L{layer_idx}] ln_1     : {x_norm.shape}  [{x_norm.min():.4f}, {x_norm.max():.4f}]")

    qkv              = hw_mac_matmul(x_norm, cattn_w, cattn_b)
    log['qkv']       = qkv.copy()
    vprint(f"  [L{layer_idx}] qkv      : {qkv.shape}")

    Q, K, V          = hw_split_qkv(qkv)
    log['Q'] = Q.copy(); log['K'] = K.copy(); log['V'] = V.copy()

    scores           = hw_attention_scores(Q, K)
    log['scores_raw']    = scores.copy()

    scores           = hw_causal_mask(scores)
    log['scores_masked'] = scores.copy()
    vprint(f"  [L{layer_idx}] scores   : {scores.shape}  [{scores[scores>-1e8].min():.4f}, {scores.max():.4f}]")

    attn_w           = hw_softmax(scores)
    log['attn_weights']  = attn_w.copy()
    vprint(f"  [L{layer_idx}] attn_w   : {attn_w.shape}  sum[0,0]={attn_w[0,0].sum():.6f}")

    attn_out         = hw_attention_output(attn_w, V)
    log['attn_out']  = attn_out.copy()

    attn_out         = hw_mac_matmul(attn_out, cproj_w, cproj_b)
    log['c_proj_out'] = attn_out.copy()
    vprint(f"  [L{layer_idx}] c_proj   : {attn_out.shape}  [{attn_out.min():.4f}, {attn_out.max():.4f}]")

    x                = hw_elementwise_add(attn_out, residual_1)
    log['after_residual_1'] = x.copy()

    # --- MLP sub-block ---
    residual_2        = x.copy()
    log['residual_2'] = residual_2

    x_norm            = hw_layernorm(x, ln2_g, ln2_b)
    log['ln2_out']    = x_norm.copy()
    vprint(f"  [L{layer_idx}] ln_2     : {x_norm.shape}  [{x_norm.min():.4f}, {x_norm.max():.4f}]")

    x_fc              = hw_mac_matmul(x_norm, cfc_w, cfc_b)
    log['c_fc_out']   = x_fc.copy()
    vprint(f"  [L{layer_idx}] c_fc     : {x_fc.shape}  [{x_fc.min():.4f}, {x_fc.max():.4f}]")

    x_fc              = hw_gelu(x_fc)
    log['gelu_out']   = x_fc.copy()
    vprint(f"  [L{layer_idx}] gelu     : {x_fc.shape}  [{x_fc.min():.4f}, {x_fc.max():.4f}]")

    x_fc              = hw_mac_matmul(x_fc, cp2_w, cp2_b)
    log['c_proj2_out'] = x_fc.copy()
    vprint(f"  [L{layer_idx}] c_proj2  : {x_fc.shape}  [{x_fc.min():.4f}, {x_fc.max():.4f}]")

    x                 = hw_elementwise_add(x_fc, residual_2)
    log['output']     = x.copy()
    vprint(f"  [L{layer_idx}] output   : {x.shape}  [{x.min():.4f}, {x.max():.4f}]")

    return x, log


# ============================================================
# FULL MODEL FORWARD PASS
# ============================================================

def run_gpt2_full(token_ids, W, model, verbose=False, tolerance=1e-3):

    wte = W['wte.weight'].numpy()
    wpe = W['wpe.weight'].numpy()

    x   = hw_elementwise_add(
            hw_embedding_lookup(wte, token_ids),
            hw_embedding_lookup(wpe, np.arange(len(token_ids)))
          )

    input_ids  = torch.tensor([token_ids])
    with torch.no_grad():
        ref_out = model(input_ids, output_hidden_states=True)
    ref_hidden = [h.squeeze(0).numpy() for h in ref_out.hidden_states]

    all_logs   = {}
    all_passes = True

    emb_diff = np.max(np.abs(x - ref_hidden[0]))
    emb_pass = emb_diff < tolerance
    print(f"\n{'='*65}")
    print(f"  Embedding  |  max_diff: {emb_diff:.2e}  |  {'PASS ✓' if emb_pass else 'FAIL ✗'}")
    print(f"{'='*65}")
    if not emb_pass:
        all_passes = False

    for layer_idx in range(12):
        if verbose:
            print(f"\n{'─'*65}  BLOCK {layer_idx}")

        x, log = run_gpt2_block(x, layer_idx, W, verbose=verbose)
        all_logs[layer_idx] = log

        ref       = ref_hidden[layer_idx + 1]
        max_diff  = np.max(np.abs(x - ref))
        mean_diff = np.mean(np.abs(x - ref))
        passed    = max_diff < tolerance

        print(f"  Block {layer_idx:2d}  |  "
              f"max_diff: {max_diff:.2e}  |  "
              f"mean_diff: {mean_diff:.2e}  |  "
              f"{'PASS ✓' if passed else 'FAIL ✗'}")

        if not passed:
            all_passes = False

    print(f"\n{'='*65}")
    print(f"  FULL MODEL : {'ALL BLOCKS PASS ✓' if all_passes else 'SOME BLOCKS FAILED ✗'}")
    print(f"{'='*65}\n")

    return x, all_logs, ref_hidden


# ============================================================
# DETAILED INTERNAL TEST
# ============================================================

def test_block_internals(layer_idx, token_ids, W, model, tolerance=1e-3):

    print(f"\n{'='*65}")
    print(f"  DETAILED INTERNAL TEST — Block {layer_idx}")
    print(f"{'='*65}")

    hf_internals = {}

    def make_hook(name):
        def hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            hf_internals[name] = t.detach().squeeze(0).numpy()
        return hook

    block = model.h[layer_idx]
    hooks = [
        block.ln_1.register_forward_hook(make_hook('ln_1')),
        block.attn.c_attn.register_forward_hook(make_hook('c_attn')),
        block.attn.c_proj.register_forward_hook(make_hook('c_proj')),
        block.ln_2.register_forward_hook(make_hook('ln_2')),
        block.mlp.c_fc.register_forward_hook(make_hook('c_fc')),
        block.mlp.act.register_forward_hook(make_hook('gelu')),
        block.mlp.c_proj.register_forward_hook(make_hook('c_proj2')),
    ]

    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        model(input_ids, output_hidden_states=True)
    for h in hooks:
        h.remove()

    wte = W['wte.weight'].numpy()
    wpe = W['wpe.weight'].numpy()
    x   = hw_elementwise_add(
            hw_embedding_lookup(wte, token_ids),
            hw_embedding_lookup(wpe, np.arange(len(token_ids)))
          )
    for i in range(layer_idx):
        x, _ = run_gpt2_block(x, i, W, verbose=False)

    _, log = run_gpt2_block(x, layer_idx, W, verbose=False)

    checks = [
        ('ln_1 output',    log['ln1_out'],      hf_internals.get('ln_1')),
        ('QKV projection', log['qkv'],           hf_internals.get('c_attn')),
        ('c_proj output',  log['c_proj_out'],    hf_internals.get('c_proj')),
        ('ln_2 output',    log['ln2_out'],       hf_internals.get('ln_2')),
        ('c_fc output',    log['c_fc_out'],      hf_internals.get('c_fc')),
        ('GELU output',    log['gelu_out'],      hf_internals.get('gelu')),
        ('c_proj2 output', log['c_proj2_out'],   hf_internals.get('c_proj2')),
    ]

    all_pass = True
    for name, ours, ref in checks:
        if ref is None:
            print(f"  {name:25s} |  ref not captured")
            continue
        diff   = np.max(np.abs(ours - ref))
        passed = diff < tolerance
        print(f"  {name:25s} |  max_diff: {diff:.2e}  |  {'PASS ✓' if passed else 'FAIL ✗'}")
        if not passed:
            all_pass = False

    print(f"\n  Block {layer_idx} internals: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    return all_pass


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    input_text = "Hello, I am a language model and I will run on an FPGA"
    token_ids  = tokenizer.encode(input_text)
    print(f"Input : '{input_text}'")
    print(f"Tokens: {token_ids}  (seq_len={len(token_ids)})\n")

    print("TEST 1: FULL MODEL — all 12 blocks vs HuggingFace")
    final_out, all_logs, ref_hidden = run_gpt2_full(token_ids, W, model, verbose=False)

    print("\nTEST 2: DETAILED INTERNALS — Block 0")
    test_block_internals(0, token_ids, W, model)

    print("\nTEST 3: DETAILED INTERNALS — Block 11")
    test_block_internals(11, token_ids, W, model)

    print("\nTEST 4: VERBOSE PASS — Block 0")
    wte = W['wte.weight'].numpy()
    wpe = W['wpe.weight'].numpy()
    x0  = hw_elementwise_add(
            hw_embedding_lookup(wte, token_ids),
            hw_embedding_lookup(wpe, np.arange(len(token_ids)))
          )
    run_gpt2_block(x0, 0, W, verbose=True)

    print("\nTEST 5: MULTIPLE INPUTS")
    test_sentences = [
        "The quick brown fox",
        "In the beginning there was",
        "Neural networks are",
        "FPGA implementation of transformers",
    ]
    for sentence in test_sentences:
        tids = tokenizer.encode(sentence)
        out, _, _ = run_gpt2_full(tids, W, model, verbose=False)
        ref  = model(torch.tensor([tids]),
                     output_hidden_states=True).hidden_states[-1].squeeze(0).detach().numpy()
        diff = np.max(np.abs(out - ref))
        print(f"  '{sentence}'")
        print(f"    max_diff: {diff:.2e}  {'PASS ✓' if diff < 1e-3 else 'FAIL ✗'}\n")

