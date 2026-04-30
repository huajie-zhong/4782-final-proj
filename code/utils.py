"""
utils.py

Contains core logic for Medusa tree attention, including tree construction,
mask generation, position ID generation, and acceptance criteria.
"""

import torch
import torch.nn.functional as F

# Static 64-node tree topology based on typical top-k probabilities (s=[10, 3, 2, 2])
MEDUSA_TREE_NODES = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 5, 3), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 3, 3), (1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 4, 3), (1, 5, 0), (1, 5, 1), (1, 5, 2), (1, 5, 3), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 3, 0), (2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 4, 0), (2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 5, 0), (2, 5, 1), (2, 5, 2), (2, 5, 3), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 0), (3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 4, 0), (3, 4, 1), (3, 4, 2), (3, 4, 3), (3, 5, 0), (3, 5, 1), (3, 5, 2), (3, 5, 3), (4, 0, 0), (4, 0, 1), (4, 0, 2), (4, 0, 3), (4, 1, 0), (4, 1, 1), (4, 1, 2), (4, 1, 3), (4, 2, 0), (4, 2, 1), (4, 2, 2), (4, 2, 3), (4, 3, 0), (4, 3, 1), (4, 3, 2), (4, 3, 3), (4, 4, 0), (4, 4, 1), (4, 4, 2), (4, 4, 3), (4, 5, 0), (4, 5, 1), (4, 5, 2), (4, 5, 3), (5, 0, 0), (5, 0, 1), (5, 0, 2), (5, 0, 3), (5, 1, 0), (5, 1, 1), (5, 1, 2), (5, 1, 3), (5, 2, 0), (5, 2, 1), (5, 2, 2), (5, 2, 3), (5, 3, 0), (5, 3, 1), (5, 3, 2), (5, 3, 3), (5, 4, 0), (5, 4, 1), (5, 4, 2), (5, 4, 3), (5, 5, 0), (5, 5, 1), (5, 5, 2), (5, 5, 3), (6, 0, 0), (6, 0, 1), (6, 0, 2), (6, 0, 3), (6, 1, 0), (6, 1, 1), (6, 1, 2), (6, 1, 3), (6, 2, 0), (6, 2, 1), (6, 2, 2), (6, 2, 3), (6, 3, 0), (6, 3, 1), (6, 3, 2), (6, 3, 3), (6, 4, 0), (6, 4, 1), (6, 4, 2), (6, 4, 3), (6, 5, 0), (6, 5, 1), (6, 5, 2), (6, 5, 3), (7, 0, 0), (7, 0, 1), (7, 0, 2), (7, 0, 3), (7, 1, 0), (7, 1, 1), (7, 1, 2), (7, 1, 3), (7, 2, 0), (7, 2, 1), (7, 2, 2), (7, 2, 3), (7, 3, 0), (7, 3, 1), (7, 3, 2), (7, 3, 3), (7, 4, 0), (7, 4, 1), (7, 4, 2), (7, 4, 3), (7, 5, 0), (7, 5, 1), (7, 5, 2), (7, 5, 3), (8, 0, 0), (8, 0, 1), (8, 0, 2), (8, 0, 3), (8, 1, 0), (8, 1, 1), (8, 1, 2), (8, 1, 3), (8, 2, 0), (8, 2, 1), (8, 2, 2), (8, 2, 3), (8, 3, 0), (8, 3, 1), (8, 3, 2), (8, 3, 3), (8, 4, 0), (8, 4, 1), (8, 4, 2), (8, 4, 3), (8, 5, 0), (8, 5, 1), (8, 5, 2), (8, 5, 3), (9, 0, 0), (9, 0, 1), (9, 0, 2), (9, 0, 3), (9, 1, 0), (9, 1, 1), (9, 1, 2), (9, 1, 3), (9, 2, 0), (9, 2, 1), (9, 2, 2), (9, 2, 3), (9, 3, 0), (9, 3, 1), (9, 3, 2), (9, 3, 3), (9, 4, 0), (9, 4, 1), (9, 4, 2), (9, 4, 3), (9, 5, 0), (9, 5, 1), (9, 5, 2), (9, 5, 3), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2), (0, 2, 3, 0), (0, 2, 3, 1), (0, 2, 3, 2), (0, 3, 0, 0), (0, 3, 0, 1), (0, 3, 0, 2), (0, 3, 1, 0), (0, 3, 1, 1), (0, 3, 1, 2), (0, 3, 2, 0), (0, 3, 2, 1), (0, 3, 2, 2), (0, 3, 3, 0), (0, 3, 3, 1), (0, 3, 3, 2), (0, 4, 0, 0), (0, 4, 0, 1), (0, 4, 0, 2), (0, 4, 1, 0), (0, 4, 1, 1), (0, 4, 1, 2), (0, 4, 2, 0), (0, 4, 2, 1), (0, 4, 2, 2), (0, 4, 3, 0), (0, 4, 3, 1), (0, 4, 3, 2), (0, 5, 0, 0), (0, 5, 0, 1), (0, 5, 0, 2), (0, 5, 1, 0), (0, 5, 1, 1), (0, 5, 1, 2), (0, 5, 2, 0), (0, 5, 2, 1), (0, 5, 2, 2), (0, 5, 3, 0), (0, 5, 3, 1), (0, 5, 3, 2), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 3, 0), (1, 0, 3, 1), (1, 0, 3, 2), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 0), (1, 1, 2, 1), (1, 1, 2, 2), (1, 1, 3, 0), (1, 1, 3, 1), (1, 1, 3, 2), (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2), (1, 2, 1, 0), (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 0), (1, 2, 2, 1), (1, 2, 2, 2), (1, 2, 3, 0), (1, 2, 3, 1), (1, 2, 3, 2), (1, 3, 0, 0), (1, 3, 0, 1), (1, 3, 0, 2), (1, 3, 1, 0), (1, 3, 1, 1), (1, 3, 1, 2), (1, 3, 2, 0), (1, 3, 2, 1), (1, 3, 2, 2), (1, 3, 3, 0), (1, 3, 3, 1), (1, 3, 3, 2), (1, 4, 0, 0), (1, 4, 0, 1), (1, 4, 0, 2), (1, 4, 1, 0), (1, 4, 1, 1), (1, 4, 1, 2), (1, 4, 2, 0), (1, 4, 2, 1), (1, 4, 2, 2), (1, 4, 3, 0), (1, 4, 3, 1), (1, 4, 3, 2), (1, 5, 0, 0), (1, 5, 0, 1), (1, 5, 0, 2), (1, 5, 1, 0), (1, 5, 1, 1), (1, 5, 1, 2), (1, 5, 2, 0), (1, 5, 2, 1), (1, 5, 2, 2), (1, 5, 3, 0), (1, 5, 3, 1), (1, 5, 3, 2), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 0, 2), (2, 0, 1, 0), (2, 0, 1, 1), (2, 0, 1, 2), (2, 0, 2, 0), (2, 0, 2, 1), (2, 0, 2, 2), (2, 0, 3, 0), (2, 0, 3, 1), (2, 0, 3, 2), (2, 1, 0, 0), (2, 1, 0, 1), (2, 1, 0, 2), (2, 1, 1, 0), (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 0), (2, 1, 2, 1), (2, 1, 2, 2), (2, 1, 3, 0), (2, 1, 3, 1), (2, 1, 3, 2), (2, 2, 0, 0), (2, 2, 0, 1), (2, 2, 0, 2), (2, 2, 1, 0), (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 2, 0), (2, 2, 2, 1), (2, 2, 2, 2), (2, 2, 3, 0), (2, 2, 3, 1), (2, 2, 3, 2), (2, 3, 0, 0), (2, 3, 0, 1), (2, 3, 0, 2), (2, 3, 1, 0), (2, 3, 1, 1), (2, 3, 1, 2), (2, 3, 2, 0), (2, 3, 2, 1), (2, 3, 2, 2), (2, 3, 3, 0), (2, 3, 3, 1), (2, 3, 3, 2), (2, 4, 0, 0), (2, 4, 0, 1), (2, 4, 0, 2), (2, 4, 1, 0), (2, 4, 1, 1), (2, 4, 1, 2), (2, 4, 2, 0), (2, 4, 2, 1), (2, 4, 2, 2), (2, 4, 3, 0)]
MEDUSA_TREE_PARENT_INDICES = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 67, 68, 68, 68, 68, 69, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137]

def build_candidate_tree(top_tokens_per_head, tree_budget=64):
    """
    Builds candidate tree using the static tree topology.
    Args:
        top_tokens_per_head: List of 4 tensors of shape (batch, s_k).
                             We assume batch=1 for single sequence inference.
        tree_budget: Number of tree nodes to use (32 or 64). Truncates the
                     static node list; all parents of included nodes remain valid
                     because nodes are ordered BFS (parents always precede children).
    Returns:
        tree_tokens: tensor of shape (tree_budget,) containing token IDs.
        tree_parent_indices: tensor of shape (tree_budget,) containing parent pointers.
    """
    if top_tokens_per_head[0].dim() > 1:
        top_tokens_per_head = [x[0] for x in top_tokens_per_head]

    nodes = MEDUSA_TREE_NODES[:tree_budget]
    parents = MEDUSA_TREE_PARENT_INDICES[:tree_budget]

    tree_tokens = []
    for node in nodes:
        head_idx = len(node) - 1
        token_rank = node[-1]
        if token_rank >= top_tokens_per_head[head_idx].size(0):
            raise IndexError(f"Node rank {token_rank} exceeds top_per_head size {top_tokens_per_head[head_idx].size(0)} for head {head_idx}. Ensure S_K matches the tree topology.")
        token_id = top_tokens_per_head[head_idx][token_rank].item()
        tree_tokens.append(token_id)

    return torch.tensor(tree_tokens), torch.tensor(parents)

def build_linear_tree(top_tokens_per_head):
    """Heads-only ablation (Table 3, Row 1): take each head's top-1 token and
    arrange them as a linear chain of K nodes.

    This reproduces paper Table 3 Row 1 — "MEDUSA head, no tree attention" —
    because the resulting tree degenerates into a single root-to-leaf path:
    each node has exactly one parent (the previous depth) and at most one
    child. Verification still uses the tree mask/position machinery, but with
    only K=4 candidates and no branching, it is equivalent to linear
    speculative decoding.
    """
    if top_tokens_per_head[0].dim() > 1:
        top_tokens_per_head = [x[0] for x in top_tokens_per_head]
    tree_tokens = [top_tokens_per_head[k][0].item() for k in range(len(top_tokens_per_head))]
    parents = [k - 1 for k in range(len(top_tokens_per_head))]  # -1, 0, 1, 2
    return torch.tensor(tree_tokens), torch.tensor(parents)


def build_naive_tree(top_tokens_per_head, s_k=(10, 3, 2, 2)):
    """Full Cartesian-product tree (Table 3, Row 2): every depth-k-1 parent
    gets s_k children, with no pruning. Total size = 10 + 30 + 60 + 120 = 220.
    """
    if top_tokens_per_head[0].dim() > 1:
        top_tokens_per_head = [x[0] for x in top_tokens_per_head]

    nodes = []
    parents = []

    # Depth 1 — roots from head 0
    for i in range(s_k[0]):
        nodes.append((i,))
        parents.append(-1)

    prev_start = 0
    prev_count = s_k[0]
    for depth in range(1, len(s_k)):
        new_start = len(nodes)
        for parent_idx in range(prev_start, prev_start + prev_count):
            parent_path = nodes[parent_idx]
            for j in range(s_k[depth]):
                nodes.append(parent_path + (j,))
                parents.append(parent_idx)
        prev_start = new_start
        prev_count = prev_count * s_k[depth]

    tree_tokens = []
    for node in nodes:
        head_idx = len(node) - 1
        token_rank = node[-1]
        tree_tokens.append(top_tokens_per_head[head_idx][token_rank].item())

    return torch.tensor(tree_tokens), torch.tensor(parents)


def generate_tree_mask(tree_parent_indices, prefix_len):
    """Generates attention mask for tree candidates.

    `prefix_len` is the number of tokens already committed to the sequence,
    including the most recently sampled LM-head token from the prior step
    (standard MEDUSA reference-repo convention).
    """
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
    """Generates position IDs for tree candidates.

    `prefix_len` must equal the number of tokens already committed to the
    sequence, **including** the most recently sampled LM-head token from the
    prior step. Depth-1 tree nodes (Medusa head 1, predicting t+2) then land
    at position `prefix_len`, depth-2 at `prefix_len + 1`, etc.
    """
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

def greedy_accept(prop_logits_1d, verify_logits_2d, tree_tokens, tree_parent_indices):
    """
    Greedy acceptance: accept node i if parent's argmax prediction == tree_tokens[i].

    Args:
        prop_logits_1d: (vocab_size,) logits from the proposal pass at lm_token position.
                        Predicts the token after lm_token; used to check depth-1 (root) nodes.
        verify_logits_2d: (tree_size, vocab_size) logits from the verify pass.
                          verify_logits_2d[i] predicts what comes after tree node i;
                          used to check i's children.
        tree_tokens: (tree_size,) token IDs at each tree node.
        tree_parent_indices: (tree_size,) parent pointer for each node (-1 = root).

    Returns:
        accepted_path: list of node indices (in tree order) along the longest
                       accepted root-to-leaf path.  Empty list if no root accepted.
    """
    parents = tree_parent_indices.tolist() if isinstance(tree_parent_indices, torch.Tensor) else list(tree_parent_indices)
    tokens = tree_tokens.tolist() if isinstance(tree_tokens, torch.Tensor) else list(tree_tokens)

    children = {i: [] for i in range(len(parents))}
    roots = []
    for i, p in enumerate(parents):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)

    pred_after_lm = prop_logits_1d.argmax().item()

    best_path = []
    for root in roots:
        if pred_after_lm != tokens[root]:
            continue
        path = [root]
        curr = root
        while children[curr]:
            pred = verify_logits_2d[curr].argmax().item()
            extended = False
            for child in children[curr]:
                if pred == tokens[child]:
                    path.append(child)
                    curr = child
                    extended = True
                    break
            if not extended:
                break
        if len(path) > len(best_path):
            best_path = path
    return best_path


def typical_accept(prop_logits_1d, verify_logits_2d, tree_tokens, tree_parent_indices,
                   epsilon=0.09, delta=0.09):
    """
    Typical acceptance (paper §2.3.1): accept node i if
        p_model(tree_tokens[i]) > min(ε, δ · exp(-H(p_model)))
    where p_model is the softmax distribution at i's parent position and H is its entropy.

    Args same as greedy_accept, plus:
        epsilon, delta: thresholds from the paper (both default 0.09).

    Returns:
        accepted_path: same format as greedy_accept.
    """
    parents = tree_parent_indices.tolist() if isinstance(tree_parent_indices, torch.Tensor) else list(tree_parent_indices)
    tokens = tree_tokens.tolist() if isinstance(tree_tokens, torch.Tensor) else list(tree_tokens)

    children = {i: [] for i in range(len(parents))}
    roots = []
    for i, p in enumerate(parents):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)

    def _check(logits_1d, token_id):
        p = F.softmax(logits_1d.float(), dim=-1)
        H = -(p * (p + 1e-10).log()).sum()
        tau = min(epsilon, delta * float((-H).exp()))
        return p[token_id].item() > tau

    best_path = []
    for root in roots:
        if not _check(prop_logits_1d, tokens[root]):
            continue
        path = [root]
        curr = root
        while children[curr]:
            extended = False
            for child in children[curr]:
                if _check(verify_logits_2d[curr], tokens[child]):
                    path.append(child)
                    curr = child
                    extended = True
                    break
            if not extended:
                break
        if len(path) > len(best_path):
            best_path = path
    return best_path


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

    # Test linear chain (Table 3 Row 1)
    print("Testing linear chain tree (heads-only ablation)...")
    tree_tokens_lin, parents_lin = build_linear_tree(top_tokens)
    assert tree_tokens_lin.shape == (4,), f"Expected 4 tokens, got {tree_tokens_lin.shape}"
    assert parents_lin.tolist() == [-1, 0, 1, 2], f"Expected linear parents, got {parents_lin.tolist()}"
    assert tree_tokens_lin.tolist() == [10, 20, 30, 40], f"Linear chain should pick top-1 per head"
    print("Linear chain tests passed!")

    # Test naive full Cartesian (Table 3 Row 2)
    print("Testing naive full Cartesian tree...")
    tree_tokens_naive, parents_naive = build_naive_tree(top_tokens)
    assert tree_tokens_naive.shape == (220,), f"Expected 220 tokens, got {tree_tokens_naive.shape}"
    # Depth-1 roots: 10 nodes, all parents = -1
    assert (parents_naive[:10] == -1).all(), "First 10 nodes should be roots"
    # Depth-2: 30 nodes, each parent in [0..9]
    assert parents_naive[10:40].min().item() == 0 and parents_naive[10:40].max().item() == 9
    print("Naive Cartesian tests passed!")
