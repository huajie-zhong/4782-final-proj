"""
benchmark.py

Greedy baseline generation and (later) Medusa inference loop metrics.
"""
import time
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def greedy_baseline(model, tokenizer, prompts, max_new_tokens):
    """
    Measures baseline greedy decoding speed on a set of prompts.
    Returns TPS (Tokens Per Second) and logs the output.
    """
    results = []
    
    # Warmup
    if len(prompts) > 0:
        inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.time()
        
        generated_tokens = outputs.shape[1] - input_len
        total_time = end_time - start_time
        tps = generated_tokens / total_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "tps": tps,
            "total_time": total_time,
            "generated_tokens": generated_tokens
        })
        print(f"Prompt: {prompt[:30]}... | TPS: {tps:.2f} | Time: {total_time:.2f}s")
        
    return results

def main():
    print("Running greedy baseline on TinyLlama-1.1B...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    # Eval prompts
    prompts = [
        "Write a Python script to sort a list.",
        "Explain the theory of relativity in simple terms.",
        "What are the benefits of eating healthy?",
        "Compose a short poem about the moon.",
        "Give me a step-by-step recipe for chocolate chip cookies."
    ]
    
    max_new_tokens = 128
    results = greedy_baseline(model, tokenizer, prompts, max_new_tokens)
    
    # Calculate average TPS
    avg_tps = sum(r["tps"] for r in results) / len(results)
    print(f"Average TPS: {avg_tps:.2f}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "greedy_baseline.json")
    with open(output_file, "w") as f:
        json.dump({"average_tps": avg_tps, "results": results}, f, indent=4)
        
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
