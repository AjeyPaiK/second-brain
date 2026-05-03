# Second Brain — LLM Finetuning System

## Hardware
- NVIDIA Jetson Orin Nano Super, 8GB unified RAM
- JetPack 6 / Ubuntu 22.04, ARM64
- NVMe SSD storage

## Overview
CLI-based LLM finetuning infrastructure on Jetson. Use `second-brain` commands to:
- **understand**: finetune open-source models on custom JSONL data with QLoRA
- **elucidate**: query finetuned models interactively or one-shot
- **daemon**: manage background training service
- **status**: inspect job history and progress
- **adapters**: list trained model versions

## Constraints
- All training/inference runs locally on Jetson (no external APIs)
- Python 3.10 required (NVIDIA PyTorch wheel is cp310 only)
- Single-worker training (8GB unified RAM, training + inference can't co-exist)
- Memory budget: ~2.7GB for training, ~1.8GB for inference
- Quantization: 4-bit NF4 for 3B+ models, float16 for 1B models
- No Flash Attention 2 (not supported on Orin GPU)

## Stack
- Python 3.10, PyTorch 2.5 (NVIDIA wheel for JetPack 6)
- transformers, peft (LoRA), trl (SFTTrainer)
- bitsandbytes (4-bit quantization, ARM64 ≥0.46.0)
- FastAPI uvicorn (daemon backend)
- Click + Rich (CLI frontend)
- chromadb removed, ollama removed
