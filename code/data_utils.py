"""
data_utils.py

Handles downloading, preprocessing, and tokenizing the ShareGPT data.
Contains the MedusaDataset class for data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader

def download_and_preprocess(tokenizer, max_samples=None, max_length=2048):
    """
    Downloads and preprocesses ShareGPT dataset.
    Filters for assistant-turn text only and concatenates multi-turn responses.
    Truncates at tokenization time to max_length so no oversized tensors are stored.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' package to run this function.")

    print("Loading dataset...")
    try:
        # Default ShareGPT dataset
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    except Exception as e:
        print(f"Failed to load default dataset: {e}. Falling back to lmsys/chatbot_arena_conversations...")
        dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print("Preprocessing...")
    processed_texts = []
    
    # Process each conversation
    for item in dataset:
        assistant_turns = []
        
        # ShareGPT format
        if "conversations" in item:
            for turn in item["conversations"]:
                if turn.get("from") == "gpt":
                    assistant_turns.append(turn.get("value", ""))
        # lmsys format fallback
        elif "conversation_a" in item:
            for turn in item["conversation_a"]:
                if turn.get("role") == "assistant":
                    assistant_turns.append(turn.get("content", ""))
                    
        if assistant_turns:
            # Concatenate multi-turn responses with a separator
            concatenated_text = "\n\n".join(assistant_turns)
            processed_texts.append(concatenated_text)

    print(f"Tokenizing {len(processed_texts)} samples...")
    # Batch tokenize: pass all texts at once — HuggingFace fast tokenizers run in Rust
    # and parallelize internally, making this 10-50x faster than a per-sample loop.
    # padding=False so MedusaDataset handles padding/truncation per-item as before.
    batch_encoding = tokenizer(
        processed_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,   # return Python lists; convert to tensors below
    )
    tokenized_data = [
        {
            "input_ids": torch.tensor(batch_encoding["input_ids"][i], dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.tensor(batch_encoding["attention_mask"][i], dtype=torch.long).unsqueeze(0),
        }
        for i in range(len(processed_texts))
    ]
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
        input_ids = item["input_ids"].squeeze(0)
        attention_mask = item["attention_mask"].squeeze(0)
        
        seq_len = len(input_ids)
        
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        elif seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            
            input_ids = torch.cat([input_ids, pad_ids])
            attention_mask = torch.cat([attention_mask, pad_mask])
            
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long()
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
