"""
data_utils.py

Handles downloading, preprocessing, and tokenizing the ShareGPT data.
Contains the MedusaDataset class for data loading.

Each conversation is rendered with the model's chat template (Vicuna for
vicuna-* backbones, otherwise tokenizer.apply_chat_template) so that the
training-time hidden-state distribution matches what the heads see at
inference. A per-token `loss_mask` keeps loss restricted to assistant tokens
(paper §2.2.1).
"""

import torch
from torch.utils.data import Dataset, DataLoader

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _extract_turns(item):
    """Normalize a ShareGPT/lmsys item into a [(role, content), ...] list."""
    turns = []
    if "conversations" in item:
        for t in item["conversations"]:
            sender = t.get("from")
            if sender == "human":
                role = "user"
            elif sender == "gpt":
                role = "assistant"
            else:
                continue
            content = (t.get("value") or "").strip()
            if content:
                turns.append((role, content))
    elif "conversation_a" in item:
        for t in item["conversation_a"]:
            role = t.get("role")
            if role in ("user", "assistant"):
                content = (t.get("content") or "").strip()
                if content:
                    turns.append((role, content))
    return turns


def _build_vicuna_conversation(turns, tokenizer, max_length, model_name=None):
    """Vicuna template via FastChat (matches the official MEDUSA repo's
    `medusa/train/train_legacy.py`, which uses
    `fastchat.model.model_adapter.get_conversation_template`).

    Returns (input_ids, loss_mask) as Python int lists, truncated to max_length.
    loss_mask=1 only on assistant tokens (and trailing separator).

    Falls back to a hand-rolled Vicuna template if fschat isn't importable.
    """
    try:
        # Lightweight FastChat import (pure-stdlib module) — works with
        # `pip install --no-deps fschat`, avoiding fastapi/uvicorn conflicts.
        from fastchat.conversation import get_conv_template
    except ImportError:
        return _build_vicuna_conversation_manual(turns, tokenizer, max_length)

    conv = get_conv_template("vicuna_v1.1")
    user_role, asst_role = conv.roles  # ("USER", "ASSISTANT")

    ids, loss = [], []
    for role, content in turns:
        conv.append_message(user_role if role == "user" else asst_role, content)
        full_text = conv.get_prompt()
        new_ids = tokenizer(full_text, add_special_tokens=True).input_ids
        delta = new_ids[len(ids):]
        mask_value = 1 if role == "assistant" else 0
        ids.extend(delta)
        loss.extend([mask_value] * len(delta))
        if len(ids) >= max_length:
            break

    return ids[:max_length], loss[:max_length]


def _build_vicuna_conversation_manual(turns, tokenizer, max_length):
    """Fallback when fschat isn't installed. Same template structure."""
    eos_id = tokenizer.eos_token_id
    sys_ids = tokenizer(VICUNA_SYSTEM, add_special_tokens=True).input_ids
    ids = list(sys_ids)
    loss = [0] * len(ids)

    for role, content in turns:
        if role == "user":
            seg = tokenizer(f" USER: {content} ASSISTANT:", add_special_tokens=False).input_ids
            ids.extend(seg)
            loss.extend([0] * len(seg))
        else:
            seg = tokenizer(f" {content}", add_special_tokens=False).input_ids
            ids.extend(seg)
            loss.extend([1] * len(seg))
            if eos_id is not None:
                ids.append(eos_id)
                loss.append(1)
        if len(ids) >= max_length:
            break

    return ids[:max_length], loss[:max_length]


def _build_chat_template_conversation(turns, tokenizer, max_length):
    """Generic path for models with a registered tokenizer.chat_template
    (e.g. TinyLlama-Chat). Tokenize prefix-by-prefix so we can mark assistant
    tokens with loss=1 and user/system tokens with loss=0.
    """
    messages = []
    ids, loss = [], []
    prev_len = 0

    for role, content in turns:
        messages.append({"role": role, "content": content})
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            return None, None
        new_ids = tokenizer(text, add_special_tokens=False).input_ids
        delta = new_ids[prev_len:]
        mask_value = 1 if role == "assistant" else 0
        ids.extend(delta)
        loss.extend([mask_value] * len(delta))
        prev_len = len(new_ids)
        if len(ids) >= max_length:
            break

    return ids[:max_length], loss[:max_length]


def _build_conversation(item, tokenizer, model_name, max_length):
    turns = _extract_turns(item)
    if not turns:
        return None, None
    mid = (model_name or "").lower()
    if "vicuna" in mid:
        return _build_vicuna_conversation(turns, tokenizer, max_length, model_name)
    if getattr(tokenizer, "chat_template", None):
        out = _build_chat_template_conversation(turns, tokenizer, max_length)
        if out[0] is not None:
            return out
    # Fallback: assistant turns only, no template (legacy behavior).
    text = "\n\n".join(c for r, c in turns if r == "assistant")
    if not text:
        return None, None
    ids = tokenizer(text, truncation=True, max_length=max_length).input_ids
    return ids, [1] * len(ids)


def download_and_preprocess(tokenizer, max_samples=None, max_length=2048, model_name=None):
    """
    Downloads and preprocesses ShareGPT dataset using the model's chat template.

    Returns a list of dicts with `input_ids`, `attention_mask`, and `loss_mask`
    (all 1D long tensors). `loss_mask` marks assistant-only positions; the
    training loop ANDs it with `attention_mask` when computing CE.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' package to run this function.")

    print("Loading dataset...")
    try:
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    except Exception as e:
        print(f"Failed to load default dataset: {e}. Falling back to lmsys/chatbot_arena_conversations...")
        dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Preprocessing with chat template (model={model_name})...")
    tokenized_data = []
    skipped = 0
    for item in dataset:
        ids, loss = _build_conversation(item, tokenizer, model_name, max_length)
        if ids is None or len(ids) < 4:
            skipped += 1
            continue
        tokenized_data.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "loss_mask": torch.tensor(loss, dtype=torch.long),
        })

    print(f"Kept {len(tokenized_data)} samples (skipped {skipped} empty/invalid).")
    return tokenized_data

class MedusaDataset(Dataset):
    """
    PyTorch Dataset for Medusa training.
    Pads or truncates tokenized data to a fixed max_length.
    Label shifting is handled in the training loop, not here.
    """
    def __init__(self, tokenized_data, max_length=2048, pad_token_id=0):
        self.data = tokenized_data
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        # Older preprocessed shards may not carry loss_mask; default to attention.
        loss_mask = item.get("loss_mask", attention_mask)

        # Tolerate both [seq] and [1, seq] storage shapes.
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            loss_mask = loss_mask.squeeze(0) if loss_mask.dim() == 2 else loss_mask

        seq_len = len(input_ids)

        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            loss_mask = loss_mask[:self.max_length]
        elif seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            pad_loss = torch.zeros((pad_len,), dtype=loss_mask.dtype)

            input_ids = torch.cat([input_ids, pad_ids])
            attention_mask = torch.cat([attention_mask, pad_mask])
            loss_mask = torch.cat([loss_mask, pad_loss])

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "loss_mask": loss_mask.long(),
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("Running data pipeline sanity check...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We test on a very small subset to be fast
    try:
        tokenized_data = download_and_preprocess(tokenizer, max_samples=100)
        dataset = MedusaDataset(tokenized_data, max_length=2048, pad_token_id=tokenizer.pad_token_id or 0)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Decode a sample
        sample_idx = 0
        original_len = tokenized_data[sample_idx]["input_ids"].shape[1]
        print(f"Sample 0 original token length: {original_len}")
        
        decoded_text = tokenizer.decode(tokenized_data[sample_idx]["input_ids"][0][:100])
        print(f"Sample 0 decoded prefix (first 100 tokens):\n{decoded_text}...\n")
        
        # Test DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        
        assert batch["input_ids"].shape == (4, 2048), "Incorrect batch shape!"
        print("DataLoader sanity check passed!")
        
    except ImportError as e:
        print(f"Skipping dataset download test: {e}")
