import gc
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ADAPTERS_DIR, get_device_map
from .trainer import _should_quantize

log = logging.getLogger("inference")

_cache_lock = threading.Lock()
_cached_model = None
_cached_tokenizer = None
_cached_base_model: Optional[str] = None
_cached_adapter_path: Optional[str] = None


def get_latest_adapter_path(adapter_name: str) -> Optional[Path]:
    adapter_family_dir = ADAPTERS_DIR / adapter_name
    if not adapter_family_dir.exists():
        return None
    versions = sorted(
        [d for d in adapter_family_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return versions[0] if versions else None


def get_base_model_from_adapter(adapter_name: str) -> Optional[str]:
    adapter_path = get_latest_adapter_path(adapter_name)
    if adapter_path is None:
        return None

    metadata_file = adapter_path / "metadata.json"
    if not metadata_file.exists():
        return None

    try:
        metadata = json.loads(metadata_file.read_text())
        return metadata.get("base_model")
    except Exception:
        return None


def get_adapter_version(adapter_path: Path) -> str:
    return adapter_path.name


def unload_model():
    global _cached_model, _cached_tokenizer, _cached_base_model, _cached_adapter_path
    with _cache_lock:
        if _cached_model is not None:
            del _cached_model
            del _cached_tokenizer
            _cached_model = None
            _cached_tokenizer = None
            _cached_base_model = None
            _cached_adapter_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Inference model unloaded")


def _load_for_inference(base_model: str, adapter_path: str):
    global _cached_model, _cached_tokenizer, _cached_base_model, _cached_adapter_path

    if (_cached_base_model == base_model and
            _cached_adapter_path == adapter_path and
            _cached_model is not None):
        return

    if _cached_model is not None:
        del _cached_model
        del _cached_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = _should_quantize(base_model)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=get_device_map(),
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=get_device_map(),
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_base_model = base_model
    _cached_adapter_path = adapter_path
    log.info("Loaded %s + adapter %s", base_model, adapter_path)


def generate(
    prompt: str,
    base_model: str,
    adapter_name: str,
    generation_args: Optional[dict] = None,
) -> tuple[str, str, str]:
    adapter_path = get_latest_adapter_path(adapter_name)
    if adapter_path is None:
        raise ValueError(f"No adapter found with name '{adapter_name}'")

    adapter_version = get_adapter_version(adapter_path)

    with _cache_lock:
        _load_for_inference(base_model, str(adapter_path))

        gen_defaults = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
        }
        if generation_args:
            gen_defaults.update(generation_args)

        chat_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = _cached_tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
        ).to(_cached_model.device)

        with torch.no_grad():
            output_ids = _cached_model.generate(
                **inputs,
                **gen_defaults,
                pad_token_id=_cached_tokenizer.pad_token_id,
                eos_token_id=_cached_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = _cached_tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip(), adapter_version, str(adapter_path)
