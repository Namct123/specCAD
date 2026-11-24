"""
Unsloth QLoRA Fine-Tuning Script for Kaggle Free GPU (Tesla T4 16GB)
Model: unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit
Exports directly to 4-bit quantized GGUF format: cad-worker-7b-q4_k_m.gguf (~4.5 GB).
"""

import os
import sys
import torch

# 1. Check GPU environment
print("=" * 60)
print("[+] Checking GPU Environment for QLoRA Training...")
print(f"    - PyTorch Version: {torch.__version__}")
print(f"    - CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    - GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"    - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("    [!] WARNING: No CUDA GPU detected. Unsloth requires NVIDIA GPU (e.g. Tesla T4).")
print("=" * 60)

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
except ImportError:
    print("\n[!] Error: `unsloth` is not installed. In Kaggle, run:")
    print("    pip install --no-deps unsloth \"xformers<0.0.28\" \"trl<0.9.0\" peft accelerate bitsandbytes\n")
    sys.exit(1)

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Hyperparameters
    # --------------------------------------------------------------------------
    MAX_SEQ_LENGTH = 4096
    MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    DATASET_PATH = "training/dataset_sft.jsonl"
    OUTPUT_DIR = "outputs_cad_qlora"
    GGUF_DIR = "cad-worker-7b-q4_k_m"

    # Find dataset file (check local path or Kaggle input directories)
    if not os.path.exists(DATASET_PATH):
        kaggle_candidate = "/kaggle/input/dataset-sft/dataset_sft.jsonl"
        if os.path.exists(kaggle_candidate):
            DATASET_PATH = kaggle_candidate
        else:
            print(f"[!] Dataset not found at {DATASET_PATH} or {kaggle_candidate}.")
            print("    Please upload `dataset_sft.jsonl` to Kaggle input or place in `training/`.")

    print(f"[+] Loading Base Model: {MODEL_NAME} with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detection
        load_in_4bit=True,
    )

    # --------------------------------------------------------------------------
    # 2. Configure PEFT / QLoRA
    # --------------------------------------------------------------------------
    print("[+] Applying QLoRA Target Modules (Rank=16, Alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # --------------------------------------------------------------------------
    # 3. ChatML Template & Dataset Preparation
    # --------------------------------------------------------------------------
    print("[+] Configuring ChatML Prompt Formatter...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    print(f"[+] Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"    - Loaded {len(dataset)} training samples.")

    # --------------------------------------------------------------------------
    # 4. Supervised Fine-Tuning (SFTTrainer)
    # --------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # False for individual CAD script instruction sequences
        args=training_args,
    )

    print("\n[+] Starting Unsloth QLoRA Training on Tesla T4...")
    trainer_stats = trainer.train()
    print("\n[OK] Training completed successfully!")
    print(f"     - Train Runtime: {trainer_stats.metrics['train_runtime']:.1f} seconds")
    print(f"     - Loss: {trainer_stats.metrics['train_loss']:.4f}")

    # --------------------------------------------------------------------------
    # 5. Export Merged Weights directly to GGUF (q4_k_m)
    # --------------------------------------------------------------------------
    print("\n[+] Merging LoRA Weights & Exporting GGUF (Quantization: q4_k_m)...")
    model.save_pretrained_gguf(
        GGUF_DIR,
        tokenizer,
        quantization_method="q4_k_m",
    )

    print(f"\n[OK] GGUF Quantized Model saved to `{GGUF_DIR}/`!")
    print(f"     You can now download the `.gguf` file to your local machine and run:")
    print(f"     ollama create cad-worker-sft -f training/Modelfile")


if __name__ == "__main__":
    main()
