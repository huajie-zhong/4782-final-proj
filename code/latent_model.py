"""
latent_model.py

MLA-inspired Latent Medusa heads: low-rank factorization of the original
MedusaHead's hidden-space transform, inspired by DeepSeek's Multi-head
Latent Attention (MLA) bottleneck principle.

The original MedusaHead does:
    output = W2(SiLU(W1(h)) + h)        # W1: d→d, W2: d→V

LatentMedusaHead replaces W1 (d→d) with a low-rank bottleneck W_down @ W_up:
    output = W_out(W_up(SiLU(W_down(h))) + h)   # W_down: d→dl, W_up: dl→d, W_out: d→V

Same pattern — residual add in hidden space, single vocab projection.
Zero-init of W_down gives identity at init (same as original's zero-init of W1).

The bottleneck forces h through a compressed latent (d_latent ≪ d_model),
analogous to how MLA compresses KV representations through a low-rank latent.
Hypothesis: this bottleneck distills the most predictive features for
future-token speculation, potentially improving head accuracy.

Supports two modes:
    - uniform:   all K heads share the same d_latent
    - per-head:  head k uses d_latent_k = base_latent // (2^k)
                 e.g. TinyLlama (d=2048): [512, 256, 128, 64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class LatentMedusaHead(nn.Module):
    """A single Medusa Head with an MLA-inspired latent bottleneck,
    sharing the original backbone's LM head (Lite EAGLE style).

    Exactly like the original MedusaHead but with W1 (d→d) factored into
    a low-rank bottleneck, and W2 (d→V) shared with the backbone's lm_head.

    Architecture:
        output = shared_lm_head(W_up(SiLU(W_down(h))) + h)

    Parameters
    ----------
    hidden_size : int
        Backbone hidden dimension (d_model).
    shared_lm_head : nn.Module
        The backbone's original LM head (frozen).
    d_latent : int
        Latent bottleneck dimension.
    head_dtype : torch.dtype or None
        Dtype for head parameters.
    """

    def __init__(self, hidden_size, shared_lm_head, d_latent, head_dtype=None):
        super().__init__()
        # We assume shared_lm_head is already on the correct device/dtype
        device = next(shared_lm_head.parameters()).device
        dtype = head_dtype if head_dtype is not None else next(shared_lm_head.parameters()).dtype

        self.d_latent = d_latent
        self.shared_lm_head = shared_lm_head

        # Low-rank bottleneck (the ONLY trainable part):
        # W_down: d → d_latent  (zero-init → identity at step 0)
        self.W_down = nn.Linear(hidden_size, d_latent, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.W_down.weight)

        # W_up: d_latent → d
        self.W_up = nn.Linear(d_latent, hidden_size, bias=False, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(self.W_up.weight)

    def forward(self, h):
        """Forward pass.  Uses shared_lm_head for the vocab projection."""
        # Bottleneck transform + residual add in hidden space
        h_transformed = self.W_up(F.silu(self.W_down(h))) + h
        # Project to logits using the shared (frozen) backbone head
        # Cast back to the shared head's dtype (e.g. BF16) if needed
        head_weight_dtype = next(self.shared_lm_head.parameters()).dtype
        return self.shared_lm_head(h_transformed.to(head_weight_dtype))

    def extra_repr(self):
        return f"d_latent={self.d_latent}"


class LatentMedusaModel(nn.Module):
    """Wraps a base causal LM with K latent-bottleneck Medusa heads.

    Drop-in replacement for MedusaModel — identical forward() signature.

    Parameters
    ----------
    backbone : PreTrainedModel
        Frozen base causal language model.
    num_heads : int
        Number of Medusa heads (K).
    d_latent : int or None
        Base latent dimension.  None → hidden_size // 4.
    per_head_latent : bool
        If True, head k uses d_latent // (2^k).  Otherwise all heads
        share the same d_latent.
    head_dtype : torch.dtype or None
        Dtype for head parameters.
    """

    def __init__(self, backbone, num_heads=4, d_latent=None, per_head_latent=False,
                 head_dtype=None):
        super().__init__()
        self.backbone = backbone
        self.num_heads = num_heads

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size

        # Default latent dim: d_model // 4
        if d_latent is None:
            d_latent = hidden_size // 4
        self.base_d_latent = d_latent
        self.per_head_latent = per_head_latent

        # Extract original LM head weights
        if hasattr(self.backbone, "lm_head"):
            lm_head = self.backbone.lm_head
        elif hasattr(self.backbone, "embed_out"):
            lm_head = self.backbone.embed_out
        else:
            raise ValueError("Could not find the language modeling head in the backbone.")

        # Compute per-head latent dimensions
        if per_head_latent:
            latent_dims = [max(d_latent // (2 ** k), 32) for k in range(num_heads)]
        else:
            latent_dims = [d_latent] * num_heads

        self.latent_dims = latent_dims

        # Instantiate K Latent Medusa heads
        self.heads = nn.ModuleList([
            LatentMedusaHead(
                hidden_size, lm_head,
                d_latent=latent_dims[k], head_dtype=head_dtype,
            )
            for k in range(num_heads)
        ])

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, compute_original_logits=True, **kwargs):
        """Forward pass.  Identical signature to MedusaModel.forward().

        Returns original logits, head logits, and optionally past_key_values.
        """
        base_model = getattr(
            self.backbone, "model",
            getattr(self.backbone, "transformer", self.backbone.base_model),
        )

        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        backbone_ctx = torch.no_grad() if backbone_frozen else torch.enable_grad()
        with backbone_ctx:
            base_model_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_dict=True,
                **kwargs,
            )

        hidden_states = base_model_outputs.last_hidden_state

        # Original model logits (skip in training where unused)
        if compute_original_logits:
            if hasattr(self.backbone, "lm_head"):
                original_logits = self.backbone.lm_head(hidden_states)
            else:
                original_logits = self.backbone.embed_out(hidden_states)
        else:
            original_logits = None

        # Latent Medusa head logits
        head_logits = [head(hidden_states.to(head.W_down.weight.dtype)) for head in self.heads]

        if "use_cache" in kwargs and kwargs["use_cache"] or past_key_values is not None:
            return original_logits, head_logits, base_model_outputs.past_key_values

        return original_logits, head_logits

    def count_head_params(self):
        """Return total trainable parameters in the heads, and a per-head breakdown."""
        per_head = []
        for k, head in enumerate(self.heads):
            n = sum(p.numel() for p in head.parameters() if p.requires_grad)
            per_head.append(n)
        total = sum(per_head)
        return total, per_head

    def estimate_head_flops_per_token(self):
        """Rough FLOPs estimate per token for the head forward pass (no backbone).

        Returns (total_flops, per_head_flops).  Counts only matmuls (2 * M * N per
        Linear(M, N) forward), ignoring SiLU and addition.
        """
        d = self.backbone.config.hidden_size
        V = self.backbone.config.vocab_size
        per_head = []
        for k, head in enumerate(self.heads):
            dl = head.d_latent
            # W_down: d → dl, W_up: dl → d, W_out: d → V
            flops = 2 * d * dl + 2 * dl * d + 2 * d * V
            per_head.append(flops)
        return sum(per_head), per_head


# ---- FLOP-equivalent training budget ------------------------------------

def compute_flop_ratio(hidden_size, vocab_size, d_latent, per_head_latent=False, num_heads=4):
    """Compute the ratio original_head_flops / latent_head_flops.

    Returns ratio > 1 if latent is cheaper per step → train for more steps.

    Original head FLOPs per token:
        W1 (d→d): 2d²   +   W2 (d→V): 2dV   =   2d(d + V)

    Latent head FLOPs per token:
        W_down (d→dl): 2d·dl  +  W_up (dl→d): 2dl·d  +  W_out (d→V): 2dV
        = 2d(2·dl + V)

    Since dl < d, the latent head replaces d² with 2·d·dl = d²/2 (for dl=d/4),
    so latent heads are always CHEAPER than original: ratio > 1.
    """
    d = hidden_size
    V = vocab_size

    # Original: K heads × (2d² + 2dV) per token
    orig_per_head = 2 * d * (d + V)
    orig_total = num_heads * orig_per_head

    # Latent: sum over heads
    latent_total = 0
    for k in range(num_heads):
        dl = max(d_latent // (2 ** k), 32) if per_head_latent else d_latent
        # W_down (d→dl) + W_up (dl→d) + W_out (d→V)
        latent_per_head = 2 * d * (2 * dl + V)
        latent_total += latent_per_head

    return orig_total / latent_total if latent_total > 0 else 1.0


# ---- Self-tests ----------------------------------------------------------

def test_latent_head():
    """Verify LatentMedusaHead at init reproduces lm_head(h) exactly."""
    print("Testing LatentMedusaHead initialization and forward pass...")
    batch_size, seq_len, hidden_size, vocab_size = 1, 10, 2048, 32000
    d_latent = hidden_size // 4

    h = torch.randn(batch_size, seq_len, hidden_size)
    dummy_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    head = LatentMedusaHead(hidden_size, dummy_lm_head, d_latent)

    out = head(h)

    # Shape check
    assert out.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape ({batch_size}, {seq_len}, {vocab_size}), got {out.shape}"
    print(f"  Shape test passed: {out.shape}")

    # Init-identity check: W_down=0 → SiLU(0)=0 → W_up(0)=0 → output = W_out(0+h) = lm_head(h)
    original_out = dummy_lm_head(h)
    assert torch.allclose(out, original_out, atol=1e-5), \
        f"LatentMedusaHead output does not match lm_head at init! Max diff: {(out - original_out).abs().max():.6f}"
    print("  Init-identity test passed: output matches lm_head(h).")
    print("All LatentMedusaHead tests passed!\n")


def test_latent_model():
    """Unit test for LatentMedusaModel using TinyLlama."""
    from transformers import AutoTokenizer

    print("Testing LatentMedusaModel (uniform latent dims)...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModelForCausalLM.from_pretrained(model_name)

    num_heads = 4
    d_latent = backbone.config.hidden_size // 4  # 512 for TinyLlama

    # --- Uniform latent dims ---
    medusa_model = LatentMedusaModel(
        backbone, num_heads=num_heads, d_latent=d_latent, per_head_latent=False,
    )

    # Verify backbone frozen
    assert all(not p.requires_grad for p in medusa_model.backbone.parameters()), \
        "Backbone parameters are not properly frozen!"
    print("  Backbone freezing test passed.")

    # Verify bottleneck is trainable (but shared lm_head is not)
    for head in medusa_model.heads:
        assert head.W_down.weight.requires_grad, "W_down should be trainable!"
        assert head.W_up.weight.requires_grad, "W_up should be trainable!"
        # shared_lm_head should still be frozen
        assert not next(head.shared_lm_head.parameters()).requires_grad, "Shared LM head should be frozen!"
    print("  Parameter grad status test passed.")

    # Count params
    total, per_head = medusa_model.count_head_params()
    print(f"  Head params: total={total:,}, per_head={[f'{n:,}' for n in per_head]}")
    print(f"  Latent dims: {medusa_model.latent_dims}")

    prompt = "Hello, Latent Medusa!"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        original_logits, head_logits = medusa_model(**inputs)

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]
    vocab_size = backbone.config.vocab_size

    expected_shape = (batch_size, seq_len, vocab_size)
    assert original_logits.shape == expected_shape, \
        f"Original logits shape {original_logits.shape} != {expected_shape}"
    assert len(head_logits) == num_heads
    for i, hl in enumerate(head_logits):
        assert hl.shape == expected_shape, \
            f"Head {i} logits shape {hl.shape} != {expected_shape}"

    print(f"  Shape test passed: {len(head_logits) + 1} tensors of shape {expected_shape}.")

    # --- Per-head latent dims ---
    print("\nTesting LatentMedusaModel (per-head latent dims)...")
    medusa_model_ph = LatentMedusaModel(
        backbone, num_heads=num_heads, d_latent=d_latent, per_head_latent=True,
    )
    print(f"  Per-head latent dims: {medusa_model_ph.latent_dims}")
    total_ph, per_head_ph = medusa_model_ph.count_head_params()
    print(f"  Head params: total={total_ph:,}, per_head={[f'{n:,}' for n in per_head_ph]}")

    with torch.no_grad():
        orig2, heads2 = medusa_model_ph(**inputs)
    assert orig2.shape == expected_shape
    assert all(h.shape == expected_shape for h in heads2)
    print("  Per-head model test passed.")

    # --- FLOP ratio ---
    ratio = compute_flop_ratio(
        backbone.config.hidden_size, backbone.config.vocab_size,
        d_latent, per_head_latent=False,
    )
    print(f"\n  FLOP ratio (original / latent, uniform): {ratio:.2f}x")
    print(f"  → Latent is cheaper: train for ~{ratio:.1f}x more steps to FLOP-match")

    ratio_ph = compute_flop_ratio(
        backbone.config.hidden_size, backbone.config.vocab_size,
        d_latent, per_head_latent=True,
    )
    print(f"  FLOP ratio (original / latent, per-head): {ratio_ph:.2f}x")

    print("\nAll LatentMedusaModel tests passed!\n")


if __name__ == "__main__":
    test_latent_head()
    test_latent_model()
