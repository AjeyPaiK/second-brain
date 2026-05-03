# Installation Guide — Second Brain LLM Finetuning

## Requirements
- NVIDIA Jetson Orin Nano Super with JetPack 6 (Ubuntu 22.04, CUDA 12.x)
- 8GB unified RAM (shared GPU/CPU memory)
- Python 3.10 (system-level)

## Step 1: Rebuild venv with Python 3.10

The existing `.venv` runs Python 3.12, but NVIDIA's PyTorch wheel for JetPack 6 only supports Python 3.10. Replace it:

```bash
cd ~/Projects/second-brain

# Remove old venv
rm -rf .venv

# Create new Python 3.10 venv
python3.10 -m venv .venv
```

## Step 2: Install PyTorch for JetPack 6.1

This must be first — other packages pin their versions to PyTorch's CUDA version.

```bash
.venv/bin/pip install --upgrade pip setuptools wheel

# Install NVIDIA PyTorch wheel for Jetson
.venv/bin/pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

**If that URL changes**, check NVIDIA's download page: https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/

## Step 3: Install remaining dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

## Step 4: Install second-brain CLI

```bash
.venv/bin/pip install -e .
```

This registers the `second-brain` command in `.venv/bin/`. Test it:

```bash
second-brain --help
```

## Step 5: Verify CUDA and bitsandbytes

```bash
.venv/bin/python << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import bitsandbytes as bnb
print("bitsandbytes version:", bnb.__version__)
print("bitsandbytes loaded successfully")
EOF
```

**Expected output:**
```
PyTorch version: 2.5.0a0+872d972e41.nv24.08
CUDA available: True
CUDA device: NVIDIA ORIN Nano (nvgpu)
bitsandbytes version: 0.46.1 (or higher)
bitsandbytes loaded successfully
```
# Next Steps
## Start the daemon

The daemon is a background process that handles training and inference. Start it:

```bash
second-brain daemon start
```

Expected output:
```
ℹ Starting daemon...
✓ Daemon started (PID 12345)
```

Check its status:

```bash
second-brain daemon status
```

You'll see GPU memory info and active job count.

Once the daemon is running, you can use the CLI:

1. **Start the daemon** (one-time setup):
   ```bash
   second-brain daemon start
   ```

2. **Submit a training job**:
   ```bash
   second-brain understand /path/to/data \
     --adapter my-adapter \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --epochs 3
   ```
   The CLI will show live progress and update every 2 seconds. Press Ctrl+C to detach (training continues in background).

3. **Check job status**:
   ```bash
   second-brain status              # show all jobs
   second-brain status <job_id>     # show details for one job
   ```

4. **Generate output (one-shot)**:
   ```bash
   second-brain elucidate \
     --adapter my-adapter \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --prompt "Your question here"
   ```

5. **Interactive REPL mode**:
   ```bash
   second-brain elucidate \
     --adapter my-adapter \
     --model meta-llama/Llama-3.2-3B-Instruct
   ```
   You'll get an interactive prompt where you can type multiple questions.

6. **List adapters**:
   ```bash
   second-brain adapters
   ```

7. **Stop the daemon** (when done):
   ```bash
   second-brain daemon stop
   ```

For details, run:
```bash
second-brain --help
second-brain understand --help
second-brain elucidate --help
```
