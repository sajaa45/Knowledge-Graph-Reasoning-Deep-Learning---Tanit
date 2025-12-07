from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
from threading import Thread, Event
import json
from datetime import datetime
from pathlib import Path
import sys

BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"  
LORA_DIR = "models/sft_smollm3/" 
MAX_NEW_TOKENS = 2048  
TEMPERATURE = 0.7
TOP_P = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FERTILITY_CASES = [
    {
        "id": 1,
        "title": "Recurrent Implantation Failure",
        "case": """34F, 3 failed IVF cycles with 4AA blastocysts. AMH 2.8 ng/mL, FSH 6.2 mIU/mL. Partner SA normal. Hysteroscopy normal. No autoimmune conditions.

Q: Most likely causes and recommended investigations?"""
    },
    {
        "id": 2,
        "title": "Poor Ovarian Response",
        "case": """38F, regular cycles. Day 3 FSH 15.2 mIU/mL, AMH 0.4 ng/mL, AFC 4. TTC 18 months. BMI 22. Partner SA normal.

Q: IVF prognosis, treatment protocol, and alternatives?"""
    },
    {
        "id": 3,
        "title": "Male Factor Infertility",
        "case": """32M, SA: Volume 2.5mL, concentration 8M/mL, motility 25% progressive, morphology 2% normal. Partner 30F, normal reserve. TTC 2 years.

Q: Diagnosis, investigations, and treatment options?"""
    },
    {
        "id": 4,
        "title": "PCOS and Ovulation Induction",
        "case": """29F, PCOS (Rotterdam+), cycles q45-90d. BMI 32, fasting insulin 18 mIU/L, mild hirsutism. TTC 12 months. Partner SA normal.

Q: Step-wise ovulation induction approach, lifestyle/meds, IVF escalation timing?"""
    },
    {
        "id": 5,
        "title": "Recurrent Pregnancy Loss",
        "case": """36F, 3 consecutive 1st trimester losses at 6-8w with prior fetal cardiac activity. No living children. Parental karyotypes normal. Thyroid normal, non-diabetic.

Q: Differential diagnosis, workup, and treatments to improve success?"""
    }
]


class StopOnString(StoppingCriteria):
    """Custom stopping criteria that stops on specific strings"""
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.stop_token_ids = []
        
        for stop_str in stop_strings:
            tokens = tokenizer.encode(stop_str, add_special_tokens=False)
            if tokens:
                self.stop_token_ids.append(tokens)
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_tokens in self.stop_token_ids:
            if len(input_ids[0]) >= len(stop_tokens):
                if input_ids[0][-len(stop_tokens):].tolist() == stop_tokens:
                    return True
        return False


def load_model_and_tokenizer(base_model_path, lora_path=None, load_in_4bit=True):
    """Load model with optional LoRA adapter"""
    print(f"Loading model from {base_model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully on {DEVICE}")
    return model, tokenizer


def format_fertility_prompt(case_text, style="o1"):
    """Format the prompt for fertility reasoning"""
    if style == "o1":
        return f"""<|user|>
You are an expert reproductive endocrinologist. Analyze this fertility case and provide your clinical reasoning.

{case_text}

<|assistant|>
<thinking>"""
    else:
        return f"""<|user|>
{case_text}

<|assistant|>
"""


def stream_generate(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.7, top_p=0.95, stop_at_thinking=True):
    """Generate text with streaming output - FIXED VERSION with proper stopping
    
    Args:
        stop_at_thinking: If True, stops after </thinking></s>. If False, continues generating.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    
    stop_strings = ["</thinking></s>", "</thinking> </s>", "</s>"] if stop_at_thinking else []
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=10.0
    )
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        repetition_penalty=1.1,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.daemon = True
    thread.start()
    
    generated_text = ""
    stop_detected = False
    
    try:
        for new_text in streamer:
            if stop_detected:
                break
                
            generated_text += new_text
            print(new_text, end="", flush=True)
            
            # Check if we've completed the expected output pattern
            if stop_at_thinking and ("</thinking>" in generated_text):
                # Allow a bit more to capture </s>
                if "</s>" in generated_text or len(generated_text.split("</thinking>")[1]) > 10:
                    stop_detected = True
    
    except Exception as e:
        print(f"\n[Warning: Streaming interrupted: {e}]")
    
    # Wait for thread with timeout
    thread.join(timeout=2.0)
    
    # Suppress warning if generation completed successfully
    if not (thread.is_alive() and not stop_detected):
        pass  # Clean termination
    
    print()  # Newline after generation
    
    return generated_text


def extract_thinking_and_answer(full_response):
    """Extract thinking process and final answer from response"""
    import re
    full_response = re.sub(r'(</thinking>\s*</s>\s*){2,}', '</thinking>', full_response)
    full_response = re.sub(r'(</thinking>\s*){2,}', '</thinking>', full_response)
    
    
    if '</thinking>' in full_response:
        parts = full_response.split('</thinking>', 1)
        thinking = parts[0].strip()
        
        thinking = re.sub(r'</?s>|<\|.*?\|>', '', thinking).strip()
        
        if len(parts) > 1:
            answer = parts[1].strip()
            answer = re.sub(r'</?s>|<\|.*?\|>', '', answer).strip()
        else:
            answer = ""
    else:
        # If no closing tag, treat everything as thinking (incomplete response)
        thinking = re.sub(r'</?s>|<\|.*?\|>', '', full_response).strip()
        answer = ""
    
    return thinking, answer


def evaluate_case(model, tokenizer, case_data, output_dir=None, stop_at_thinking=True):
    """Evaluate a single fertility case
    
    Args:
        stop_at_thinking: If True, only generates thinking. If False, continues after thinking.
    """
    
    print("\n" + "="*80)
    print(f"CASE {case_data['id']}: {case_data['title']}")
    print("="*80)
    print(f"\n{case_data['case']}\n")
    print("-"*80)
    print("MODEL RESPONSE (Streaming):")
    print("-"*80)
    
    prompt = format_fertility_prompt(case_data['case'])
    full_response = stream_generate(
        model, 
        tokenizer, 
        prompt, 
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop_at_thinking=stop_at_thinking
    )
    
    thinking, answer = extract_thinking_and_answer(full_response)
    
    result = {
        "case_id": case_data['id'],
        "title": case_data['title'],
        "case": case_data['case'],
        "full_response": full_response,
        "thinking": thinking,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }
    
    if output_dir:
        output_path = Path(output_dir) / f"case_{case_data['id']}_result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Result saved to {output_path}")
    
    return result


def run_all_cases(model, tokenizer, cases, output_dir="inference/results"):
    """Run inference on all fertility cases"""
    
    results = []
    
    for case in cases:
        result = evaluate_case(model, tokenizer, case, output_dir)
        results.append(result)
        print("\n" + "="*80 + "\n")
    
    summary_path = Path(output_dir) / "fertility_cases_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": BASE_MODEL,
            "lora_adapter": LORA_DIR,
            "timestamp": datetime.now().isoformat(),
            "num_cases": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    return results


def generate_markdown_report(results, output_path="inference/results/fertility_report.md"):
    """Generate a markdown report of all cases"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Fertility Use Cases - Model Evaluation Report\n\n")
        f.write(f"**Model:** {BASE_MODEL}\n")
        f.write(f"**LoRA Adapter:** {LORA_DIR}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for result in results:
            f.write(f"## Case {result['case_id']}: {result['title']}\n\n")
            f.write(f"### Clinical Scenario\n\n")
            f.write(f"{result['case']}\n\n")
            
            f.write(f"### Clinical Analysis\n\n")
            if result['thinking']:
                f.write(f"{result['thinking']}\n\n")
            else:
                f.write("*No analysis provided*\n\n")
            
            # Include extracted recommendations if available
            if result.get('answer') and result['answer'].strip():
                f.write(f"### Key Recommendations\n\n")
                f.write(f"{result['answer']}\n\n")
            
            f.write("---\n\n")
    
    print(f"\n✓ Markdown report saved to {output_path}")


def compare_with_base_model(base_model_path, lora_path, case_data):
    """Compare base model vs fine-tuned model on a single case"""
    
    print("\n" + "="*80)
    print("COMPARISON: Base Model vs Fine-tuned Model")
    print("="*80)
    
    print("\n>>> Loading BASE MODEL...")
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_path, lora_path=None)
    
    print("\n>>> BASE MODEL RESPONSE:")
    print("-"*80)
    prompt = format_fertility_prompt(case_data['case'])
    base_response = stream_generate(base_model, base_tokenizer, prompt, max_new_tokens=1024)
    
    del base_model
    torch.cuda.empty_cache()
    
    print("\n>>> Loading FINE-TUNED MODEL...")
    ft_model, ft_tokenizer = load_model_and_tokenizer(base_model_path, lora_path=lora_path)
    
    print("\n>>> FINE-TUNED MODEL RESPONSE:")
    print("-"*80)
    ft_response = stream_generate(ft_model, ft_tokenizer, prompt, max_new_tokens=2048)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


def main():
    import argparse
    
    global BASE_MODEL, LORA_DIR, TEMPERATURE, MAX_NEW_TOKENS
    parser = argparse.ArgumentParser(description="Stream Medical CoT Inference for Fertility Cases")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="Base model path")
    parser.add_argument("--lora", type=str, default=LORA_DIR, help="LoRA adapter path")
    parser.add_argument("--output_dir", type=str, default="inference/results", help="Output directory")
    parser.add_argument("--case_id", type=int, help="Run specific case only (1-5)")
    parser.add_argument("--compare", action="store_true", help="Compare base vs fine-tuned")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens")
    
    args = parser.parse_args()
    
    BASE_MODEL = args.base_model
    LORA_DIR = args.lora
    TEMPERATURE = args.temperature
    MAX_NEW_TOKENS = args.max_tokens
    
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, LORA_DIR)
    
    if args.compare:
        if args.case_id:
            case = FERTILITY_CASES[args.case_id - 1]
        else:
            case = FERTILITY_CASES[0]  
        compare_with_base_model(BASE_MODEL, LORA_DIR, case)
        return
    
    if args.case_id:
        if 1 <= args.case_id <= len(FERTILITY_CASES):
            case = FERTILITY_CASES[args.case_id - 1]
            evaluate_case(model, tokenizer, case, args.output_dir)
        else:
            print(f"Error: case_id must be between 1 and {len(FERTILITY_CASES)}")
        return
    
    print("\n" + "="*80)
    print("FERTILITY USE CASES EVALUATION")
    print(f"Running {len(FERTILITY_CASES)} clinical scenarios")
    print("="*80)
    
    results = run_all_cases(model, tokenizer, FERTILITY_CASES, args.output_dir)
    generate_markdown_report(results, f"{args.output_dir}/fertility_report.md")
    
    print("\n" + "="*80)
    print("✓ ALL CASES COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"- Individual JSON files: case_{{1-5}}_result.json")
    print(f"- Summary: fertility_cases_summary.json")
    print(f"- Report: fertility_report.md")


if __name__ == "__main__":
    main()