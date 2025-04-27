# train_with_accelerate.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

# Config 
MODEL_NAME    = "EleutherAI/gpt-neo-125M"    # or your preferred HF model
JSON_FILE     = "data/processed/nicomachean_ethics_sections.json"
OUTPUT_DIR    = "checkpoints/accel_ethics"
NUM_EPOCHS    = 3
BATCH_SIZE    = 4
GRAD_ACCUM    = 8
LEARNING_RATE = 2e-4
MAX_LENGTH    = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tokenizer & Base Model + LoRA 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

# Ensure there’s a pad token:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# Dataset & Tokenization 
ds = load_dataset("json", data_files=JSON_FILE)["train"]

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

ds = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=ds.column_names
)
ds.set_format("torch", columns=["input_ids", "attention_mask"])

# DataLoader
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)



#  Accelerator, Optimizer, Scheduler 
accelerator = Accelerator(mixed_precision="no")
optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE)

model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

num_update_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_update_steps
)

# Training Loop 
progress_bar = tqdm(range(num_update_steps), desc="Training")

global_step = 0
model.train()
for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"]
            )

            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1

    # Save a checkpoint at end of each epoch
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    unwrapped.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

progress_bar.close()