from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

import re

BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"
LORA_DIR = "models/sft_smollm3/"
MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 5 
stop_tokens = ["</s>", "<|user|>", "<|assistant|>"]



def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True
    )

    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    return model, tokenizer



def format_prompt(question):
    return f"<|user|>\n{question}\n<|assistant|>\n<thinking>"

def run_inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_tensor = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_tensor[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False).strip()

def sample_dataset(name, split, n, config=None):
    if config:
        ds = load_dataset(name, config, split=split)
    else:
        ds = load_dataset(name, split=split)
    return ds.shuffle(seed=42).select(range(min(n, len(ds))))

def clean_text(text):
    return re.sub(r'<.*?>|</s>', '', str(text)).strip()



def extract_answer_and_evaluate(generated_text, dataset_name, ground_truth_item):
    
    match = re.search(r'<thinking>\s*(.*?)\s*</?s>', generated_text, re.DOTALL)
    predicted_answer = match.group(1).split('.')[0].split('\n')[0].strip() if match else generated_text.split('<thinking>')[-1].split('</s>')[0].split('.')[0].strip()

    true_label = None
    correct_idx = None
    
    if dataset_name == "MedMCQA":
        correct_idx = ground_truth_item.get("correct_options")
        options = ground_truth_item.get("options")
        if options and isinstance(options, list) and correct_idx is not None and 0 <= correct_idx < len(options):
            true_label = options[correct_idx]
    
    elif dataset_name == "PubMedQA":
        true_label = ground_truth_item.get("final_decision")
        
    elif dataset_name == "MedQA":
        answer_letter = ground_truth_item.get("answer_idx")
        options = ground_truth_item.get("options")
        if answer_letter and options and isinstance(options, dict):
            true_label = options.get(answer_letter)
            correct_idx = ord(answer_letter) - ord('A')  
    elif dataset_name == "MMLU":
        correct_idx = ground_truth_item.get("answer")
        choices = ground_truth_item.get("choices")
        if choices and isinstance(choices, list) and correct_idx is not None and 0 <= correct_idx < len(choices):
            true_label = choices[correct_idx]
    else:
        true_label = ground_truth_item.get("answer") or ground_truth_item.get("final_decision")

    is_correct = False
    if true_label is not None and isinstance(true_label, str):
        norm_pred = predicted_answer.lower().replace('-', '').replace(' ', '')
        norm_true = true_label.lower().replace('-', '').replace(' ', '')
        
        if norm_true in norm_pred or norm_pred == norm_true:
            is_correct = True
        
    return predicted_answer, is_correct, true_label, correct_idx


def main():
    model, tokenizer = load_model()

    medqa = sample_dataset("GBaker/MedQA-USMLE-4-options", "test", SAMPLE_SIZE)
    medmcqa = sample_dataset("openlifescienceai/medmcqa", "validation", SAMPLE_SIZE)
    pubmedqa = sample_dataset(name="qiaojin/PubMedQA", split="train", n=SAMPLE_SIZE, config="pqa_labeled")
    mmlu = sample_dataset("cais/mmlu", "test", SAMPLE_SIZE, "clinical_knowledge")

    datasets = {
        "MedQA": medqa,
        "MedMCQA": medmcqa,
        "PubMedQA": pubmedqa,
        "MMLU": mmlu
    }

    overall_results = {}

    for name, ds in datasets.items():
        correct_count = 0
        total_samples = len(ds)
        
        print(f"\n======== {name} Results ({total_samples} samples) ========\n")
        
        for i in range(total_samples):
            q = ds[i]
            question = q.get("question") or q.get("QUESTION") or q.get("query") or q.get("input") or q.get("Q") 
            if not question:
                 continue

            prompt = format_prompt(question)
            generated_text_with_tags = run_inference(model, tokenizer, prompt)
            
            predicted_answer, is_correct, true_label_text, correct_answer_idx = extract_answer_and_evaluate(generated_text_with_tags, name, q)

            clean_question = clean_text(question)
            clean_predicted = clean_text(predicted_answer)
            clean_true = clean_text(true_label_text)

            print(f"Question {i+1}: {clean_question}")

            if name == "MedQA":
                options_dict = q.get("options", {})
                correct_letter = q.get("answer")
                correct_idx = ord(correct_letter) - ord('A') if correct_letter else None
                
                options = [options_dict.get(chr(65+i), "") for i in range(len(options_dict))]
                
                predicted_idx = None
                letter_match = re.search(r'\b([A-Da-d])\b', predicted_answer)
                if letter_match:
                    letter = letter_match.group(1).upper()
                    predicted_idx = ord(letter) - ord('A')
                
                if predicted_idx is None:
                    norm_pred = clean_predicted.lower().replace('-', '').replace(' ', '')
                    for idx, opt in enumerate(options):
                        if not opt:
                            continue
                        norm_opt = str(opt).lower().replace('-', '').replace(' ', '')
                        if norm_opt and (norm_opt in norm_pred or norm_pred in norm_opt):
                            predicted_idx = idx
                            break
                
                predicted_idx_display = chr(65 + predicted_idx) if predicted_idx is not None else "No match"
                is_correct = (predicted_idx is not None and predicted_idx == correct_idx)

                print("Options:")
                for idx, opt in enumerate(options):
                    print(f"  {chr(65+idx)}. {opt}")
                print(f"Correct Option: {correct_letter}")
                print(f"Model Answer: {predicted_idx_display}")

            elif name == "MedMCQA":
                options = [
                    q.get("opa", ""),
                    q.get("opb", ""),
                    q.get("opc", ""),
                    q.get("opd", "")
                ]
                correct_idx = q.get("cop", None)
                
                predicted_idx = None
                norm_pred = clean_predicted.lower().replace('-', '').replace(' ', '')
                
                letter_match = re.search(r'\b([A-Da-d])\b', predicted_answer)
                if letter_match:
                    letter = letter_match.group(1).upper()
                    predicted_idx = ord(letter) - ord('A')
                
                if predicted_idx is None:
                    for idx, opt in enumerate(options):
                        if not opt:
                            continue
                        norm_opt = str(opt).lower().replace('-', '').replace(' ', '')
                        if norm_opt and (norm_opt in norm_pred or norm_pred in norm_opt):
                            predicted_idx = idx
                            break
                
                if predicted_idx is None:
                    num_match = re.search(r'\b([0-4])\b', predicted_answer)
                    if num_match:
                        num = int(num_match.group(1))
                        if 0 <= num <= 3:
                            predicted_idx = num
                        elif 1 <= num <= 4:
                            predicted_idx = num - 1
                
                predicted_idx_display = predicted_idx if predicted_idx is not None else "No match"
                is_correct = (predicted_idx is not None and predicted_idx == correct_idx)

                print("Options:")
                for idx, opt in enumerate(options):
                    print(f"  {chr(65+idx)}. {opt}")
                if correct_idx is not None and 0 <= correct_idx < len(options):
                    print(f"Correct Option: {chr(65+correct_idx)}. {options[correct_idx]}")
                else:
                    print(f"Correct Option: {correct_idx} (Unknown)")
                print(f"Model Answer Index: {predicted_idx_display}")
                print(f"Dataset Answer Index: {correct_idx}")

            elif name == "MMLU":
                options = q.get("choices", [])
                correct_idx = q.get("answer")
                
                predicted_idx = None
                num_match = re.search(r'\b([0-3])\b', predicted_answer)
                if num_match:
                    predicted_idx = int(num_match.group(1))
                
                if predicted_idx is None:
                    letter_match = re.search(r'\b([A-Da-d])\b', predicted_answer)
                    if letter_match:
                        letter = letter_match.group(1).upper()
                        predicted_idx = ord(letter) - ord('A')
                
                if predicted_idx is None:
                    norm_pred = clean_predicted.lower().replace('-', '').replace(' ', '')
                    for idx, opt in enumerate(options):
                        if not opt:
                            continue
                        norm_opt = str(opt).lower().replace('-', '').replace(' ', '')
                        if norm_opt and (norm_opt in norm_pred or norm_pred in norm_opt):
                            predicted_idx = idx
                            break
                
                predicted_idx_display = predicted_idx if predicted_idx is not None else "No match"
                is_correct = (predicted_idx is not None and predicted_idx == correct_idx)

                print("Options:")
                for idx, opt in enumerate(options):
                    print(f"  {idx}. {opt}")
                if correct_idx is not None and 0 <= correct_idx < len(options):
                    print(f"Correct Option: {correct_idx}. {options[correct_idx]}")
                else:
                    print(f"Correct Option: {correct_idx} (Unknown)")
                print(f"Model Answer Index: {predicted_idx_display}")
                print(f"Dataset Answer Index: {correct_idx}")
                
            if is_correct:
                correct_count += 1

            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"| Q{i+1} | Status: {status:<9} | Predicted: {predicted_answer[:50]:<50} | True Label: {str(clean_true):<15} |")

        accuracy = (correct_count / total_samples) * 100
        overall_results[name] = accuracy
        print(f"\n--- {name} Accuracy: {accuracy:.2f}% ({correct_count}/{total_samples}) ---\n")

    print("\n\n--- SUMMARY OF MINI-EVALUATION ACCURACIES ---\n")
    for name, accuracy in overall_results.items():
        print(f"| {name:<10} | Accuracy: {accuracy:.2f}% |")
    print("------------------------------------------\n")
    
if __name__ == "__main__":
    main()