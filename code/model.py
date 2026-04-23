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
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)  # W1
        self.linear2 = nn.Linear(hidden_size, vocab_size, bias=False)   # W2
        
        # CRITICAL INITIALIZATION (paper §2.1.1):
        nn.init.zeros_(self.linear1.weight)         # W1 -> zero
        self.linear2.weight.data = lm_head_weights.clone()  # W2 -> copy of original LM head
        
    def forward(self, h):
        # residual connection passes h unchanged initially because linear1 is zero
        return self.linear2(F.silu(self.linear1(h)) + h)

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

if __name__ == "__main__":
    test_medusa_head()
