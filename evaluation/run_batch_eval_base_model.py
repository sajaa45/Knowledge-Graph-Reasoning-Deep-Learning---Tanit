# evaluation/run_batch_eval_base_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import re
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"
MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 25

# --- Helper Functions ---

def load_model():
    print(f"Loading model {BASE_MODEL} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(question):
    return f"<|user|>\n{question}\n<|assistant|>\n<thinking>"


def run_inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_tensor = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_tensor[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False).strip()


def load_dataset_sample(name, split, num_samples, config=None):
    """Load a percentage of the dataset"""
    if config:
        ds = load_dataset(name, config, split=split)
    else:
        ds = load_dataset(name, split=split)
    sample_size = min(len(ds), num_samples)
    return ds.shuffle().select(range(sample_size))


def extract_predicted_answer(generated_text):
    """Extract the answer from model output"""
    match = re.search(r'<thinking>\s*(.*?)\s*</?s>', generated_text, re.DOTALL)
    if match:
        return match.group(1).split('.')[0].split('\n')[0].strip()
    return generated_text.split('<thinking>')[-1].split('</s>')[0].split('.')[0].strip()


# --- Dataset Evaluation Functions ---

def evaluate_medqa(item, predicted_answer):
    options_dict = item.get("options", {})
    correct_letter = item.get("answer_idx")
    if not correct_letter:
        return False
    correct_idx = ord(correct_letter) - ord('A')
    correct_text = options_dict.get(correct_letter, "")

    # Letter match
    letter_match = re.search(r'\b([A-Da-d])\b', predicted_answer)
    if letter_match:
        return letter_match.group(1).upper() == correct_letter

    # Text match
    if correct_text:
        norm_pred = predicted_answer.lower().replace('-', '').replace(' ', '')
        norm_correct = correct_text.lower().replace('-', '').replace(' ', '')
        return norm_correct in norm_pred or norm_pred == norm_correct
    return False


def evaluate_medmcqa(item, predicted_answer):
    options = [item.get("opa", ""), item.get("opb", ""), item.get("opc", ""), item.get("opd", "")]
    correct_idx = item.get("cop")
    if correct_idx is None:
        return False

    predicted_idx = None
    norm_pred = predicted_answer.lower().replace('-', '').replace(' ', '')

    # Letter match
    letter_match = re.search(r'\b([A-Da-d])\b', predicted_answer)
    if letter_match:
        predicted_idx = ord(letter_match.group(1).upper()) - ord('A')

    # Text match
    if predicted_idx is None:
        for idx, opt in enumerate(options):
            if not opt:
                continue
            norm_opt = opt.lower().replace('-', '').replace(' ', '')
            if norm_opt and (norm_opt in norm_pred or norm_pred in norm_opt):
                predicted_idx = idx
                break

    # Numeric match
    if predicted_idx is None:
        num_match = re.search(r'\b([0-4])\b', predicted_answer)
        if num_match:
            num = int(num_match.group(1))
            predicted_idx = num if 0 <= num <= 3 else num - 1

    return predicted_idx == correct_idx


def evaluate_pubmedqa(item, predicted_answer):
    correct_answer = item.get("final_decision")
    if not correct_answer:
        return False
    norm_pred = predicted_answer.lower().replace('-', '').replace(' ', '')
    norm_correct = correct_answer.lower().replace('-', '').replace(' ', '')
    return norm_correct in norm_pred or norm_pred == norm_correct


def evaluate_mmlu(item, predicted_answer):
    options = item.get("choices", [])
    correct_idx = item.get("answer")
    if correct_idx is None:
        return False

    predicted_idx = None
    num_match = re.search(r'\b([0-3])\b', predicted_answer)
    if num_match:
        predicted_idx = int(num_match.group(1))

    if predicted_idx is None:
        letter_match = re.search(r'\b([A-Da-d])\b', predicted_answer)
        if letter_match:
            predicted_idx = ord(letter_match.group(1).upper()) - ord('A')

    if predicted_idx is None and options:
        norm_pred = predicted_answer.lower().replace('-', '').replace(' ', '')
        for idx, opt in enumerate(options):
            if not opt:
                continue
            norm_opt = opt.lower().replace('-', '').replace(' ', '')
            if norm_opt and (norm_opt in norm_pred or norm_pred in norm_opt):
                predicted_idx = idx
                break

    return predicted_idx == correct_idx


def evaluate_dataset(model, tokenizer, dataset, dataset_name, eval_func):
    correct_count = 0
    total_count = len(dataset)
    print(f"\nEvaluating {dataset_name}... ({total_count} samples)")
    for item in tqdm(dataset, desc=dataset_name):
        question = item.get("question") or item.get("QUESTION") or item.get("query") or item.get("input")
        if not question:
            continue
        prompt = format_prompt(question)
        generated_text = run_inference(model, tokenizer, prompt)
        predicted_answer = extract_predicted_answer(generated_text)
        if eval_func(item, predicted_answer):
            correct_count += 1
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    return correct_count, total_count, accuracy


# --- Main Execution ---

def main():
    model, tokenizer = load_model()

    datasets_config = [
        ("MedQA", "GBaker/MedQA-USMLE-4-options", "test", None, evaluate_medqa),
        ("MedMCQA", "openlifescienceai/medmcqa", "validation", None, evaluate_medmcqa),
        ("PubMedQA", "qiaojin/PubMedQA", "train", "pqa_labeled", evaluate_pubmedqa),
        ("MMLU", "cais/mmlu", "test", "clinical_knowledge", evaluate_mmlu),
    ]

    results = {}
    for name, dataset_name, split, config, eval_func in datasets_config:
        try:
            ds = load_dataset_sample(dataset_name, split, NUM_SAMPLES, config)
            correct, total, accuracy = evaluate_dataset(model, tokenizer, ds, name, eval_func)
            results[name] = {'correct': correct, 'total': total, 'accuracy': accuracy}
            print(f"✓ {name}: {correct}/{total} correct ({accuracy:.2f}%)")
        except Exception as e:
            print(f"✗ Error evaluating {name}: {str(e)}")
            results[name] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{name:<15} | {result['correct']:>4}/{result['total']:<4} | Accuracy: {result['accuracy']:>6.2f}%")
    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print("="*60)
    print(f"{'OVERALL':<15} | {total_correct:>4}/{total_samples:<4} | Accuracy: {overall_accuracy:>6.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
