from unsloth import FastLanguageModel

import os
import torch
from datasets import load_from_disk, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)

BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"  
DATA_PATH = "data/processed/final_mix/"
OUTPUT_DIR = "models/sft_smollm3/"
MAX_LENGTH = 2048  
BATCH_SIZE = 2  
GRAD_ACCUM = 8  
LR = 2e-4
EPOCHS = 3
SEED = 42
WARMUP_STEPS = 100


torch.manual_seed(SEED)

logging.info("Loading pre-formatted dataset...")
dataset = load_from_disk(DATA_PATH)

if not isinstance(dataset, DatasetDict):
    dataset = DatasetDict({"train": dataset})

logging.info(f"Dataset columns: {dataset['train'].column_names}")
logging.info(f"Number of training examples: {len(dataset['train'])}")
logging.info(f"\nFirst training example:\n{dataset['train'][0]['text'][:500]}...\n")

logging.info(f"Loading base model: {BASE_MODEL}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LENGTH,
    dtype=None,  
    load_in_4bit=True,  
)

logging.info("Configuring LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=64,  
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,  # 
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    
    logging_steps=20,
    logging_first_step=True,
    report_to="none", 
    gradient_checkpointing=True,
    
    seed=SEED,
)

logging.info("Initializing SFT Trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    dataset_text_field="text",  
    max_seq_length=MAX_LENGTH,
    packing=False,  )

logging.info("\n" + "="*50)
logging.info("Starting training...")
logging.info("="*50 + "\n")

trainer.train()

logging.info("\n" + "="*50)
logging.info("Training complete! Saving model...")
logging.info("="*50 + "\n")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logging.info(f" LoRA adapters saved to: {OUTPUT_DIR}")
