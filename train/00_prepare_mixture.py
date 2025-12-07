import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import os
import re
from tqdm import tqdm

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sample_dataset(ds, max_samples):
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds

def normalize_reasoning_format(text):
    """
    Normalize various CoT/reasoning formats to unified <thinking> tags.
    Handles: HuatuoGPT-o1, medical-o1, MedReason, and other formats.
    """
    if not text or not isinstance(text, str):
        return text
    
    if "<thinking>" in text and "</thinking>" in text:
        return text
    
    if "## Thinking" in text or "## thinking" in text.lower():
        thinking_match = re.search(r'##\s*[Tt]hinking\s*\n(.*?)(?:##\s*[Ff]inal|$)', text, re.DOTALL)
        final_match = re.search(r'##\s*[Ff]inal\s*[Rr]esponse\s*\n(.*?)$', text, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            final = final_match.group(1).strip() if final_match else ""
            
            if final:
                return f"<thinking>\n{thinking}\n</thinking>\n\n{final}"
            else:
                return f"<thinking>\n{thinking}\n</thinking>"
    
    if re.search(r'^(Reasoning|Analysis|Explanation|Solution):', text, re.MULTILINE | re.IGNORECASE):
        return f"<thinking>\n{text}\n</thinking>"
    
    if any(indicator in text.lower() for indicator in ['step 1', 'first,', 'let\'s analyze', 'to solve this']):
        return f"<thinking>\n{text}\n</thinking>"
    
    return f"<thinking>\n{text}\n</thinking>"

def format_example(example, input_field, output_field, normalize=True):
    """
    Format example with unified reasoning structure.
    Normalizes all CoT formats to <thinking> tags.
    """
    q = example.get(input_field, "")
    a = example.get(output_field, "")
    
    if normalize:
        output = normalize_reasoning_format(a)
    else:
        output = a

    return {
        "input": q,
        "output": output
    }

def format_to_chatml(example, eos_token="</s>"):
    """Format example into ChatML style for SmolLM3 training"""
    prompt = example["input"]
    answer = example["output"]  
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{answer}{eos_token}"
    
    return {"text": text}

def main():
    config = load_yaml("data/mixture_config.yaml")
    final_datasets = []
    
    print("=== Loading and Normalizing Datasets ===")
    print("All datasets will be normalized to <thinking> tag format\n")

    for item in config["datasets"]:
        print(f"\nLoading {item['name']}...")
        config_name = item.get("config", None)

        if config_name:
            ds = load_dataset(item["name"], config_name, split=item["load_split"])
        else:
            ds = load_dataset(item["name"], split=item["load_split"])

        if "filter" in item:
            keywords = item["filter"]["include_keywords"]
            print(f"  Filtering for keywords: {keywords}")
            ds = ds.filter(lambda x: any(kw in str(x).lower() for kw in keywords))

        original_size = len(ds)
        ds = sample_dataset(ds, item["max_samples"])
        print(f"  Sampled: {len(ds)} / {original_size} examples")

        input_field = config["formatting"]["fields"]["input_field"]
        output_field = config["formatting"]["fields"]["output_field"]
        
        ds = ds.map(
            lambda x: format_example(
                x,
                input_field,
                output_field,
                normalize=True  
            ),
            remove_columns=ds.column_names,
            num_proc=1,
            desc=f"Formatting {item['name']}"
        )

        final_datasets.append(ds)
        print(f"  ✓ Formatted {len(ds)} examples")

    print("\n=== Concatenating datasets ===")
    merged = concatenate_datasets(final_datasets)
    print(f"Total examples before deduplication: {len(merged)}")

  
    if config["deduplication"]["enabled"]:
        print("\n=== Deduplicating ===")
        seen = set()
        unique = []

        for ex in tqdm(merged, desc="Deduplicating"):
            h = hash(ex["input"])
            if h not in seen:
                seen.add(h)
                unique.append(ex)

        merged = Dataset.from_list(unique)
        print(f"After deduplication: {len(merged)} examples")

    
    target_size = config["target_size"]
    if len(merged) > target_size:
        print(f"\n=== Sampling to target size: {target_size} ===")
        merged = merged.shuffle(seed=42).select(range(target_size))

    
    print("\n=== Formatting to ChatML for training ===")
    merged = merged.map(
        format_to_chatml,
        remove_columns=["input", "output"],
        num_proc=1,
        desc="Converting to ChatML"
    )
    
    
    print("\n=== Example Formatted Training Instance ===")
    print(merged[0]['text'][:800])
    print("...\n")

   
    out_dir = config["output"]["dataset_path"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Saving final dataset to {out_dir} ===")
    merged.save_to_disk(out_dir)

    print("\n✓ Done!")
    print(f"  Final dataset size: {len(merged)}")
    print(f"  Dataset columns: {merged.column_names}")
    print(f"  Format: ChatML with unified <thinking> tags")

if __name__ == "__main__":
    main()