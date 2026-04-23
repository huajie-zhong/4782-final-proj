"""
model.py

MedusaHead and MedusaModel wrapper, along with KV cache management logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class MedusaHead(nn.Module):
    """
    A single Medusa Head for predicting future tokens.
    It takes the hidden state of the backbone model, applies a linear layer,
    a SiLU activation, a residual connection, and a final linear layer to
    produce logits over the vocabulary.
    """
    def __init__(self, hidden_size, vocab_size, lm_head_weights):
        super().__init__()
        dtype = lm_head_weights.dtype
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)  # W1
        self.linear2 = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)   # W2
        
        # CRITICAL INITIALIZATION (paper §2.1.1):
        nn.init.zeros_(self.linear1.weight)         # W1 -> zero
        self.linear2.weight.data = lm_head_weights.clone()  # W2 -> copy of original LM head
        
    def forward(self, h):
        # residual connection passes h unchanged initially because linear1 is zero
        return self.linear2(F.silu(self.linear1(h)) + h)

class MedusaModel(nn.Module):
    """
    Wraps a base causal language model with K Medusa heads.
    """
    def __init__(self, backbone, num_heads=4):
        super().__init__()
        self.backbone = backbone
        self.num_heads = num_heads
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get hidden size and vocab size from the backbone config
        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size
        
        # Extract the original LM head weights.
        # This assumes a standard HuggingFace causal LM like LlamaForCausalLM.
        if hasattr(self.backbone, "lm_head"):
            lm_head = self.backbone.lm_head
        elif hasattr(self.backbone, "embed_out"):
            lm_head = self.backbone.embed_out
        else:
            raise ValueError("Could not find the language modeling head in the backbone.")
            
        # Instantiate K Medusa heads
        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, lm_head.weight.data)
            for _ in range(num_heads)
        ])
        
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        """
        Forward pass.
        Returns original logits, head logits, and optionally past_key_values.
        """
        # Run the base model to get hidden states.
        # We use the internal .model or .base_model to avoid the LM head computation of the backbone.
        # Note: LlamaForCausalLM has .model, GPTNeoX has .gpt_neox, etc.
        base_model = getattr(self.backbone, "model", getattr(self.backbone, "transformer", self.backbone.base_model))
        
        base_model_outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs
        )
        
        # For Llama, last_hidden_state is already normalized by the final LayerNorm
        hidden_states = base_model_outputs.last_hidden_state
        
        # Get original model logits
        if hasattr(self.backbone, "lm_head"):
            original_logits = self.backbone.lm_head(hidden_states)
        else:
            original_logits = self.backbone.embed_out(hidden_states)
            
        # Get Medusa head logits
        head_logits = [head(hidden_states) for head in self.heads]
        
        # Usually, during generation we want past_key_values
        if "use_cache" in kwargs and kwargs["use_cache"] or past_key_values is not None:
            return original_logits, head_logits, base_model_outputs.past_key_values
            
        return original_logits, head_logits

def test_medusa_head():
    """Unit test for MedusaHead."""
    print("Testing MedusaHead initialization and forward pass...")
    batch_size = 1
    seq_len = 10
    hidden_size = 2048
    vocab_size = 32000
    
    # Dummy hidden states and LM head weights
    h = torch.randn(batch_size, seq_len, hidden_size)
    dummy_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    head = MedusaHead(hidden_size, vocab_size, dummy_lm_head.weight.data)
    
    out = head(h)
    
    # Verify shape
    assert out.shape == (batch_size, seq_len, vocab_size), f"Expected shape ({batch_size}, {seq_len}, {vocab_size}), got {out.shape}"
    print(f"Shape test passed: {out.shape}")
    
    # Verify initialization matches original LM head
    original_out = dummy_lm_head(h)
    assert torch.allclose(out, original_out, atol=1e-6), "MedusaHead output does not match original LM head at init."
    print("Initialization test passed: Output matches original LM head.")
    print("All MedusaHead tests passed!\n")

def test_medusa_model():
    """Unit test for MedusaModel wrapper using TinyLlama."""
    from transformers import AutoTokenizer
    
    print("Testing MedusaModel initialization and forward pass...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModelForCausalLM.from_pretrained(model_name)
    
    num_heads = 4
    medusa_model = MedusaModel(backbone, num_heads=num_heads)
    
    # Verify backbone gradients are frozen
    frozen = all(not p.requires_grad for p in medusa_model.backbone.parameters())
    assert frozen, "Backbone parameters are not properly frozen!"
    print("Backbone freezing test passed.")
    
    # Verify heads are trainable
    trainable_heads = all(p.requires_grad for p in medusa_model.heads.parameters())
    assert trainable_heads, "Medusa head parameters are incorrectly frozen!"
    
    prompt = "Hello, Medusa!"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        original_logits, head_logits = medusa_model(**inputs)
        
    # Verify shapes
    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]
    vocab_size = backbone.config.vocab_size
    
    expected_shape = (batch_size, seq_len, vocab_size)
    assert original_logits.shape == expected_shape, f"Original logits shape {original_logits.shape} does not match {expected_shape}"
    
    assert len(head_logits) == num_heads, f"Expected {num_heads} head logits, got {len(head_logits)}"
    for i, hl in enumerate(head_logits):
        assert hl.shape == expected_shape, f"Head {i} logits shape {hl.shape} does not match {expected_shape}"
        
    print(f"Shapes test passed: {len(head_logits) + 1} logit tensors of shape {expected_shape}.")
    print("All MedusaModel tests passed!\n")

if __name__ == "__main__":
    test_medusa_head()
    test_medusa_model()
