"""
utils.py

Contains core logic for Medusa tree attention, including tree construction,
mask generation, position ID generation, and acceptance criteria.
"""

import torch
import torch.nn.functional as F

# Static 64-node tree topology based on typical top-k probabilities (s=[10, 3, 2, 2])
MEDUSA_TREE_NODES = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (3, 0, 0), (3, 0, 1), (4, 0, 0), (5, 0, 0), (6, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 1, 0), (2, 1, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0), (6, 0, 0, 0)]
MEDUSA_TREE_PARENT_INDICES = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 16, 16, 18, 19, 20, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 28, 28, 29, 30, 32, 33, 33, 34, 35, 36, 38, 39, 40]

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
