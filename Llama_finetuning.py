import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Define the base model from Hugging Face Hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Define the dataset for fine-tuning
dataset_name = "mlabonne/guanaco-llama2-1k"

# Define the name for the fine-tuned model
new_model = "Llama-2-7b-chat-finetune"

# QLoRA Parameters
lora_r = 64  # LoRA attention dimension
lora_alpha = 16  # Alpha parameter for LoRA scaling
lora_dropout = 0.1  # Dropout probability for LoRA layers

# bitsandbytes Configuration
use_4bit = True  # Activate 4-bit precision
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit models
bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
use_nested_quant = False  # Activate nested quantization (double quantization)

# Training Arguments
output_dir = "./results"  # Directory for model predictions and checkpoints
num_train_epochs = 1  # Number of training epochs
fp16 = False  # Enable fp16 training
bf16 = False  # Enable bf16 training (for GPUs supporting bfloat16)
per_device_train_batch_size = 4  # Training batch size per GPU
per_device_eval_batch_size = 4  # Evaluation batch size per GPU
gradient_accumulation_steps = 1  # Number of update steps for gradient accumulation
gradient_checkpointing = True  # Enable gradient checkpointing
max_grad_norm = 0.3  # Maximum gradient norm (gradient clipping)
learning_rate = 2e-4  # Initial learning rate (AdamW optimizer)
weight_decay = 0.001  # Weight decay for optimizer
optim = "paged_adamw_32bit"  # Optimizer type
lr_scheduler_type = "cosine"  # Learning rate schedule
group_by_length = True  # Group sequences by length for efficiency
save_steps = 0  # Save checkpoint every X steps
logging_steps = 25  # Log training progress every X steps

# Load dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train")

# Configure QLoRA settings
print("Configuring model for QLoRA...")
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("Your GPU supports bfloat16: Consider setting bf16=True for faster training.")

# Load base model with quantization
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configure LoRA settings
print("Configuring LoRA...")
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
print("Configuring training arguments...")
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=0.03,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)

# Initialize Trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start training
print("Starting training...")
trainer.train()

# Save trained model
print("Saving fine-tuned model...")
trainer.model.save_pretrained(new_model)

# Run text generation pipeline for testing
print("Testing fine-tuned model...")
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print("Generated response:", result[0]['generated_text'])

# Clean up VRAM
del model, pipe, trainer
import gc
gc.collect()

# Reload model in FP16 and merge LoRA weights
print("Merging LoRA weights with base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer for saving
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Push model to Hugging Face Hub
print("Uploading model to Hugging Face Hub...")
model.push_to_hub("entbappy/Llama-2-7b-chat-finetune", check_pr=True)
tokenizer.push_to_hub("entbappy/Llama-2-7b-chat-finetune", check_pr=True)

print("Fine-tuning completed and model uploaded successfully!")
