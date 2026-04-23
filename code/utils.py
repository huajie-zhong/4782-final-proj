"""
utils.py

Contains core logic for Medusa tree attention, including tree construction,
mask generation, and position ID generation.
"""

import torch

# Static 64-node tree topology based on typical top-k probabilities (s=[10, 3, 2, 2])
MEDUSA_TREE_NODES = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (3, 0, 0), (3, 0, 1), (4, 0, 0), (5, 0, 0), (6, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 1, 0), (2, 1, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0)]
MEDUSA_TREE_PARENT_INDICES = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 16, 16, 18, 19, 20, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 28, 28, 29, 30, 32, 33, 33, 34, 35, 36, 38, 39, 40]

def build_candidate_tree(top_tokens_per_head):
    """
    Builds candidate tree using the static tree topology.
    Args:
        top_tokens_per_head: List of 4 tensors of shape (batch, s_k). 
                             We assume batch=1 for single sequence inference.
    Returns:
        tree_tokens: tensor of shape (tree_size,) containing token IDs.
        tree_parent_indices: tensor of shape (tree_size,) containing parent pointers.
    """
    # Assuming single sequence generation (batch size 1)
    if top_tokens_per_head[0].dim() > 1:
        top_tokens_per_head = [x[0] for x in top_tokens_per_head]
        
    tree_tokens = []
    for node in MEDUSA_TREE_NODES:
        head_idx = len(node) - 1
        token_rank = node[-1]
        token_id = top_tokens_per_head[head_idx][token_rank].item()
        tree_tokens.append(token_id)
        
    return torch.tensor(tree_tokens), torch.tensor(MEDUSA_TREE_PARENT_INDICES)

def generate_tree_mask(tree_parent_indices, prefix_len):
    """Generates attention mask for tree candidates."""
    tree_size = len(tree_parent_indices)
    mask = torch.zeros((tree_size, prefix_len + tree_size), dtype=torch.bool)
    
    for i in range(tree_size):
        # Every node attends to the prefix
        mask[i, :prefix_len] = True
        
        # Every node attends to itself and its ancestors
        curr = i
        while curr != -1:
            mask[i, prefix_len + curr] = True
            curr = tree_parent_indices[curr].item() if isinstance(tree_parent_indices, torch.Tensor) else tree_parent_indices[curr]
            
    return mask

def generate_position_ids(tree_parent_indices, prefix_len):
    """Generates position IDs for tree candidates."""
    tree_size = len(tree_parent_indices)
    positions = torch.zeros(tree_size, dtype=torch.long)
    
    for i in range(tree_size):
        curr = i
        depth = 0
        while curr != -1:
            depth += 1
            curr = tree_parent_indices[curr].item() if isinstance(tree_parent_indices, torch.Tensor) else tree_parent_indices[curr]
            
        # depth 1 means it's a direct child of the prefix. 
        # The prefix ends at prefix_len - 1, so the next token is at prefix_len.
        positions[i] = prefix_len + depth - 1
        
    return positions

if __name__ == "__main__":
    print("Testing Utils...")
    
    # Test 3-node manual verification
    print("Testing 3-node tree...")
    tree_parent_indices = [-1, 0, 1]
    mask = generate_tree_mask(tree_parent_indices, prefix_len=2)
    positions = generate_position_ids(tree_parent_indices, prefix_len=2)
    
    # Node 0 attends to prefix + itself. Parent is root.
    # Node 1 attends to prefix + node 0 + itself.
    # Node 2 attends to prefix + node 0 + node 1 + itself.
    expected_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ], dtype=torch.bool)
    
    assert torch.all(mask == expected_mask), "3-node mask incorrect."
    assert torch.all(positions == torch.tensor([2, 3, 4])), "3-node positions incorrect."
    print("3-node tree tests passed!")
    
    # Test 64-node tree
    print("Testing 64-node static tree...")
    top_tokens = [
        torch.arange(10, 20).unsqueeze(0), # Head 0 top 10 tokens
        torch.arange(20, 23).unsqueeze(0), # Head 1 top 3 tokens
        torch.arange(30, 32).unsqueeze(0), # Head 2 top 2 tokens
        torch.arange(40, 42).unsqueeze(0)  # Head 3 top 2 tokens
    ]
    tree_tokens, parents = build_candidate_tree(top_tokens)
    assert tree_tokens.shape == (64,), f"Expected 64 tokens, got {tree_tokens.shape}"
    assert parents.shape == (64,), f"Expected 64 parents, got {parents.shape}"
    
    mask = generate_tree_mask(parents, prefix_len=10)
    assert mask.shape == (64, 74), f"Expected shape (64, 74), got {mask.shape}"
    
    # Node 63 is (6, 0, 0, 0)
    # It should attend to prefix, itself, and its 3 ancestors
    # The length of the path is 4, so total trues = 10 + 4 = 14
    assert mask[63].sum().item() == 14, f"Expected 14 true values for node 63, got {mask[63].sum().item()}"
    
    pos = generate_position_ids(parents, prefix_len=10)
    assert pos.shape == (64,), f"Expected 64 positions, got {pos.shape}"
    # Node 63 is depth 4, so position should be 10 + 4 - 1 = 13
    assert pos[63].item() == 13, f"Expected pos 13 for node 63, got {pos[63].item()}"
    
    print("64-node tree tests passed!")
