"""
LoRA Adapter Training Script for Mechanistic Interpretability Research
=======================================================================
Trains 4 LoRA adapters on google/gemma-2-9b at ranks [4, 8, 16, 32]
with all hyperparameters fixed except rank, for controlled SAE analysis.
"""



import os
import gc
import json
import time
import logging
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME     = "google/gemma-2-9b"
DATASET_NAME   = "tatsu-lab/alpaca"
NUM_SAMPLES    = 10000
MAX_SEQ_LENGTH = 512
RANKS          = [16, 32]
OUTPUT_ROOT    = "./lora_adapters"
SEED           = 42

# Fixed hyperparameters — only rank varies across runs
LEARNING_RATE  = 2e-4
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
NUM_EPOCHS     = 3
DROPOUT        = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Output dir must exist before FileHandler is created ───────────────────────
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_ROOT, "train.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.bfloat16
    log.info("Device: Apple MPS (M-series). Using bfloat16.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.bfloat16
    log.info(f"Device: CUDA ({torch.cuda.get_device_name(0)}). Using bfloat16.")
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32
    log.warning("Device: CPU. Training will be very slow.")


# ── Dataset preparation ───────────────────────────────────────────────────────
def prepare_dataset(tokenizer):
    log.info(f"Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME, split="train")
    log.info(f"Full dataset size: {len(raw)} samples. Selecting {NUM_SAMPLES}.")
    raw = raw.select(range(NUM_SAMPLES))

    def format_example(example):
        if example["input"].strip():
            text = (
                f"### Instruction:\n{example['instruction']}\n"
                f"### Input:\n{example['input']}\n"
                f"### Response:\n{example['output']}"
            )
        else:
            text = (
                f"### Instruction:\n{example['instruction']}\n"
                f"### Response:\n{example['output']}"
            )
        return {"text": text}

    log.info("Formatting examples...")
    formatted = raw.map(format_example, remove_columns=raw.column_names)
    log.info(f"Sample formatted example:\n{formatted[0]['text'][:300]}\n...")

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    log.info("Tokenizing dataset...")
    tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    log.info(f"Tokenization complete. Dataset size: {len(tokenized)} samples.")
    return tokenized


# ── Single rank training ──────────────────────────────────────────────────────
def train_lora(rank: int, tokenized_dataset, tokenizer, run_log: list):
    log.info("=" * 60)
    log.info(f"Starting LoRA training — rank={rank}, alpha={2 * rank}")
    log.info("=" * 60)

    adapter_dir = os.path.join(OUTPUT_ROOT, f"r{rank}")
    os.makedirs(adapter_dir, exist_ok=True)

    # ── Load base model ───────────────────────────────────────────────────────
    log.info(f"[r={rank}] Loading base model {MODEL_NAME}...")
    log.info(f"[r={rank}] (~18GB download on first run, cached after that)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
        attn_implementation="eager",  # required for Gemma 2 on MPS
    )
    model.config.use_cache = False    # required during training
    log.info(f"[r={rank}] Base model loaded. Total parameters: {model.num_parameters():,}")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    log.info(f"[r={rank}] Applying LoRA (r={rank}, alpha={2 * rank}, dropout={DROPOUT})...")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        target_modules=TARGET_MODULES,
        lora_dropout=DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        f"[r={rank}] Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=(DTYPE == torch.bfloat16),
        fp16=False,
        logging_steps=20,
        save_strategy="no",           # we save manually after training
        seed=SEED,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    log.info(f"[r={rank}] Initialising Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(f"[r={rank}] Training started...")
    t_start = time.time()
    train_result = trainer.train()
    elapsed = round(time.time() - t_start, 1)

    log.info(f"[r={rank}] Training complete in {elapsed:.1f}s")
    log.info(f"[r={rank}] Final loss: {train_result.training_loss:.4f}")

    # ── Save adapter immediately ──────────────────────────────────────────────
    log.info(f"[r={rank}] Saving adapter weights to {adapter_dir}...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Human-readable run config saved alongside adapter weights
    config_summary = {
        "rank": rank,
        "lora_alpha": 2 * rank,
        "target_modules": TARGET_MODULES,
        "lora_dropout": DROPOUT,
        "base_model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "num_samples": NUM_SAMPLES,
        "max_seq_length": MAX_SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": NUM_EPOCHS,
        "seed": SEED,
        "trainable_params": trainable,
        "total_params": total,
        "training_loss": round(train_result.training_loss, 6),
        "training_time_sec": elapsed,
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(adapter_dir, "run_config.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    log.info(f"[r={rank}] ✅ Adapter saved. Files: {os.listdir(adapter_dir)}")
    run_log.append(config_summary)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    log.info(f"[r={rank}] Cleaning up memory...")
    del model, trainer
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
    log.info(f"[r={rank}] Memory cleared. Ready for next run.\n")

    return config_summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║     LoRA Adapter Training — Mech Interp Research         ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Model        : {MODEL_NAME}")
    log.info(f"Dataset      : {DATASET_NAME} ({NUM_SAMPLES} samples)")
    log.info(f"Ranks        : {RANKS}")
    log.info(f"Device       : {DEVICE}")
    log.info(f"Output root  : {OUTPUT_ROOT}")
    log.info(f"Started at   : {datetime.now().isoformat()}")

    # Load tokenizer once — reused across all ranks
    log.info(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Prepare and tokenize dataset once — reused across all ranks
    tokenized_dataset = prepare_dataset(tokenizer)

    run_log = []

    for rank in RANKS:
        try:
            result = train_lora(rank, tokenized_dataset, tokenizer, run_log)
            log.info(
                f"[Summary] r={rank} | loss={result['training_loss']:.4f} "
                f"| time={result['training_time_sec']}s "
                f"| trainable={result['trainable_params']:,}"
            )
        except Exception as e:
            log.error(f"[r={rank}] Training FAILED: {e}", exc_info=True)
            run_log.append({"rank": rank, "status": "FAILED", "error": str(e)})
        finally:
            # Always write global log after each rank — even on failure
            log_path = os.path.join(OUTPUT_ROOT, "training_log.json")
            with open(log_path, "w") as f:
                json.dump(run_log, f, indent=2)
            log.info(f"[Global log] Updated: {log_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("ALL RUNS COMPLETE — Final Summary")
    log.info("=" * 60)
    for entry in run_log:
        if "error" in entry:
            log.info(f"  r={entry['rank']:>2} | FAILED: {entry['error']}")
        else:
            log.info(
                f"  r={entry['rank']:>2} | "
                f"loss={entry['training_loss']:.4f} | "
                f"time={entry['training_time_sec']}s | "
                f"trainable={entry['trainable_params']:,}"
            )

    log.info(f"\nAdapters saved in : {os.path.abspath(OUTPUT_ROOT)}/")
    log.info("Next step         : SAE delta analysis using Gemma Scope.")


if __name__ == "__main__":
    main()
