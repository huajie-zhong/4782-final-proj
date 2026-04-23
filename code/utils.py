"""
utils.py

Contains core logic for Medusa tree attention, including tree construction,
mask generation, and position ID generation.
"""

def build_candidate_tree(top_tokens_per_head, top_probs_per_head):
    """Builds candidate tree from top tokens."""
    pass

def generate_tree_mask(tree_parent_indices, prefix_len):
    """Generates attention mask for tree candidates."""
    pass

def generate_position_ids(tree_parent_indices, prefix_len):
    """Generates position IDs for tree candidates."""
    pass
