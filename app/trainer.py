import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTTrainer, SFTConfig

from .config import (
    ADAPTERS_DIR,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_TARGET_MODULES,
    DATALOADER_NUM_WORKERS,
    get_device_map,
)
from .jobs import JobStatus, get_store

log = logging.getLogger("trainer")


class ProgressCallback(TrainerCallback):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._store = get_store()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss", logs.get("train_loss", 0.0))
        line = (f"epoch={state.epoch:.2f} step={state.global_step} "
                f"loss={loss:.4f}")
        self._store.update_progress(
            self.job_id,
            epoch=int(state.epoch),
            step=state.global_step,
            loss=float(loss),
        )
        self._store.append_log(self.job_id, line)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", 0.0)
            line = f"[eval] step={state.global_step} eval_loss={eval_loss:.4f}"
            self._store.append_log(self.job_id, line)


def _should_quantize(model_name_or_path: str) -> bool:
    name_lower = model_name_or_path.lower()
    for small in ["1b", "0.5b", "350m", "125m", "tiny"]:
        if small in name_lower:
            return False
    for large in ["3b", "7b", "8b", "13b", "70b"]:
        if large in name_lower:
            return True
    return True


def _load_model_and_tokenizer(base_model: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=get_device_map(),
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=get_device_map(),
            trust_remote_code=True,
        )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=DEFAULT_TARGET_MODULES,
    )


def _format_chat_sample(sample: dict) -> str:
    messages = sample.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    return "".join(parts)


def run_training_job(
    job_id: str,
    base_model: str,
    data_path: str,
    adapter_name: str,
    training_args_override: dict,
):
    store = get_store()
    store.update_status(job_id, JobStatus.RUNNING)
    store.append_log(job_id, f"Starting training job {job_id}")
    store.append_log(job_id, f"Base model: {base_model}")

    try:
        data_dir = Path(data_path)
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "val.jsonl"
        if not train_file.exists():
            raise FileNotFoundError(f"train.jsonl not found in {data_path}")
        if not val_file.exists():
            raise FileNotFoundError(f"val.jsonl not found in {data_path}")

        store.append_log(job_id, f"Data loaded from {data_path}")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        adapter_output_path = ADAPTERS_DIR / adapter_name / timestamp
        adapter_output_path.mkdir(parents=True, exist_ok=True)

        use_4bit = _should_quantize(base_model)
        store.append_log(job_id, f"Quantization: {'4-bit NF4' if use_4bit else 'float16'}")

        store.append_log(job_id, "Loading model and tokenizer...")
        model, tokenizer = _load_model_and_tokenizer(base_model, use_4bit)
        store.append_log(job_id, "Model loaded")

        lora_config = _build_lora_config()

        dataset = load_dataset(
            "json",
            data_files={"train": str(train_file), "validation": str(val_file)},
        )

        defaults = {
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 1024,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "fp16": True,
            "bf16": False,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "load_best_model_at_end": False,
            "report_to": "none",
        }
        defaults.update(training_args_override or {})

        sft_config = SFTConfig(
            output_dir=str(adapter_output_path),
            **defaults,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=lora_config,
            formatting_func=_format_chat_sample,
            callbacks=[ProgressCallback(job_id)],
        )

        store.append_log(job_id, "Training started")
        train_result = trainer.train()
        store.append_log(job_id, f"Training complete: {train_result.metrics}")

        trainer.model.save_pretrained(str(adapter_output_path))
        tokenizer.save_pretrained(str(adapter_output_path))

        metadata = {
            "base_model": base_model,
            "adapter_name": adapter_name,
            "version": timestamp,
            "job_id": job_id,
            "data_path": data_path,
            "created_at": datetime.utcnow().isoformat(),
            "eval_loss": train_result.metrics.get("eval_loss"),
            "train_loss": train_result.metrics.get("train_loss"),
            "use_4bit": use_4bit,
            "lora_r": DEFAULT_LORA_R,
            "lora_alpha": DEFAULT_LORA_ALPHA,
        }
        (adapter_output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

        store.update_status(job_id, JobStatus.COMPLETED, adapter_path=str(adapter_output_path))
        store.append_log(job_id, f"Adapter saved to {adapter_output_path}")

    except Exception as exc:
        log.exception("Training job %s failed", job_id)
        store.update_status(job_id, JobStatus.FAILED, error=str(exc))
        store.append_log(job_id, f"ERROR: {exc}")

    finally:
        try:
            del model
        except NameError:
            pass
        try:
            del trainer
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        store.append_log(job_id, "GPU memory released")
