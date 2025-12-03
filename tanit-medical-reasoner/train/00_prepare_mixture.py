import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import os
from tqdm import tqdm

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sample_dataset(ds, max_samples):
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds

def format_example(example, input_field, output_field, add_tags=True):
    q = example.get(input_field, "")
    a = example.get(output_field, "")

    if add_tags:
        a = f"<thinking>\n{a}\n</thinking>"

    return {
        "input": q,
        "output": a
    }

def main():
    config = load_yaml("data/mixture_config.yaml")
    final_datasets = []
    

    print("=== Loading datasets ===")

    for item in config["datasets"]:
        print(f"\nLoading {item['name']}...")
        config_name = item.get("config", None)

        if config_name:
            ds = load_dataset(item["name"], config_name, split=item["load_split"])
        else:
            ds = load_dataset(item["name"], split=item["load_split"])


        # Filter medical subsets if needed
        if "filter" in item:
            keywords = item["filter"]["include_keywords"]
            ds = ds.filter(lambda x: any(kw in str(x).lower() for kw in keywords))

        # Sample dataset
        ds = sample_dataset(ds, item["max_samples"])

        # Format
        ds = ds.map(
            lambda x: format_example(
                x,
                config["formatting"]["fields"]["input_field"],
                config["formatting"]["fields"]["output_field"],
                config["formatting"]["add_thinking_tags"]
            ),
            remove_columns=ds.column_names
        )

        final_datasets.append(ds)

    print("\n=== Concatenating datasets ===")
    merged = concatenate_datasets(final_datasets)

    # Deduplication
    if config["deduplication"]["enabled"]:
        print("Deduplicating...")
        seen = set()
        unique = []

        for ex in merged:
            h = hash(ex["input"])
            if h not in seen:
                seen.add(h)
                unique.append(ex)

        merged = Dataset.from_list(unique)

    # Final sampling to target size
    target_size = config["target_size"]
    if len(merged) > target_size:
        merged = merged.shuffle(seed=42).select(range(target_size))

    # Save
    out_dir = config["output"]["dataset_path"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSaving final dataset to {out_dir}...")
    merged.save_to_disk(out_dir)

    print("\nDone! Final dataset size:", len(merged))

if __name__ == "__main__":
    main()
