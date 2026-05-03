# Second Brain CLI Reference

## Quick Start

```bash
# 1. Start the daemon (once, or before training)
second-brain daemon start

# 2. Check daemon is running
second-brain daemon status

# 3. Prepare your data: create /path/to/data with train.jsonl and val.jsonl
# Each file must have format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# 4. Finetune a model
second-brain understand /path/to/data \
  --adapter my-adapter \
  --model meta-llama/Llama-3.2-3B-Instruct

# 5. Check training progress
second-brain status
second-brain status <job_id>

# 6. Once training completes, query the model
second-brain elucidate \
  --adapter my-adapter \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt "What's the main theme?"

# 7. Or use interactive mode (no --prompt)
second-brain elucidate --adapter my-adapter --model meta-llama/Llama-3.2-3B-Instruct

# 8. Stop daemon when done
second-brain daemon stop
```

## Commands

### `second-brain daemon start`
Start the background daemon process. Daemon runs at `http://127.0.0.1:8000`.

### `second-brain daemon stop`
Stop the daemon.

### `second-brain daemon status`
Show daemon health and GPU memory usage.

### `second-brain understand <data_path> [OPTIONS]`
Submit an async QLoRA training job.

**Options:**
- `--adapter TEXT` (required): Name for the adapter family (e.g., "domain-v1")
- `--model TEXT` (required): Base model (e.g., "meta-llama/Llama-3.2-3B-Instruct")
- `--epochs INT`: Number of training epochs (default: 3)
- `--lr FLOAT`: Learning rate (default: 0.0002)
- `--max-seq INT`: Max sequence length (default: 1024)
- `--batch INT`: Batch size per device (default: 1)

**Example:**
```bash
second-brain understand ~/my_data \
  --adapter customer-support \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --epochs 5 \
  --lr 5e-5 \
  --max-seq 512
```

**Output:**
- Shows job ID
- Live-updates training progress every 2 seconds
- Press Ctrl+C to detach (training continues in background)

### `second-brain elucidate [OPTIONS]`
Query a finetuned model.

**Options:**
- `--adapter TEXT` (required): Adapter name
- `--model TEXT` (required): Base model
- `--prompt TEXT`: Single query (optional; if omitted, starts REPL)
- `--max-tokens INT`: Max tokens to generate (default: 512)
- `--temp FLOAT`: Temperature for sampling (default: 0.7)

**Examples:**

One-shot query:
```bash
second-brain elucidate \
  --adapter customer-support \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt "How should we handle refund requests?"
```

Interactive REPL:
```bash
second-brain elucidate \
  --adapter customer-support \
  --model meta-llama/Llama-3.2-3B-Instruct
```
Then type multiple questions with `>` prompt, `exit` or Ctrl+D to quit.

### `second-brain status [JOB_ID]`
Show job status.

- `second-brain status` — Table of all jobs
- `second-brain status abc123...` — Full details for one job (including logs)

**Output:**
- Job ID, Adapter name, Status, Epoch, Loss, Created timestamp
- For specific job: Full progress details + last 10 logs + adapter path

### `second-brain adapters`
List all trained adapters.

**Output:**
- Adapter name, number of versions, latest version, base model, eval loss
- Adapters are stored in `~/Projects/second-brain/adapters/{name}/{YYYYMMDD_HHMMSS}/`

## Data Format

Training data must be in JSONL format (one JSON object per line).

**Required schema:**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Example train.jsonl:**
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language."}]}
{"messages": [{"role": "user", "content": "Who wrote Hamlet?"}, {"role": "assistant", "content": "William Shakespeare."}]}
```

**Rules:**
- `data_path` must contain both `train.jsonl` and `val.jsonl`
- Both files must have the same schema
- Validation set can be same as training set for quick tests
- Each entry is one training example (one Q&A pair)

## Troubleshooting

### Daemon won't start
Check the log:
```bash
tail -f ~/Projects/second-brain/logs/daemon.log
```

Common issues:
- Port 8000 already in use: `lsof -i :8000` to find process
- CUDA not available: Check `nvidia-smi`
- bitsandbytes issue: Run `.venv/bin/python -c "import bitsandbytes; print('OK')"`

### Training is slow
- First training will download the model (~6GB). Subsequent runs reuse the cache.
- Check GPU memory: `second-brain daemon status`
- Reduce `--max-seq` or `--batch` if running out of memory

### Can't query model while training
Intentional design. Daemon returns HTTP 503 while training. Training takes 5-15 minutes depending on dataset size and epochs. Wait for `second-brain status` to show `completed`.

### Out of memory during training
- Use a smaller model (Llama-3.2-1B instead of 3B)
- Reduce `--max-seq`
- Reduce `--batch` (but 1 is already minimum)
- Reduce `--epochs`

### Query seems to take forever
- Model is being loaded for the first time (can take 2-3 minutes for 3B model)
- Check daemon logs: `tail -f ~/Projects/second-brain/logs/daemon.log`
- Timeout is set to 120 seconds for generation

## File Locations

```
~/Projects/second-brain/
├── adapters/              # Trained LoRA weights
│   └── {name}/{timestamp}/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── metadata.json
├── logs/                  # Log files
│   ├── api.log           # API server logs
│   ├── daemon.log        # Daemon startup logs
│   └── *.log
├── jobs.json             # Job state (persistent)
├── .daemon.pid           # Daemon process ID
└── .venv/bin/second-brain  # CLI executable
```

## Environment

If `second-brain` command is not found, activate the venv:

```bash
source ~/Projects/second-brain/.venv/bin/activate
second-brain --help
```

Or use directly:
```bash
~/Projects/second-brain/.venv/bin/second-brain --help
```

## Performance Notes

**Training time estimates** (on Jetson Orin Nano):
- 1B model, 5 samples, 1 epoch: ~2 minutes
- 3B model, 5 samples, 1 epoch: ~5 minutes
- 3B model, 100 samples, 3 epochs: ~30-45 minutes

**Inference time:**
- Loading model + generating 100 tokens: ~30-60 seconds (first time)
- Subsequent queries: ~10-20 seconds (model cached)

**Memory usage:**
- Training: ~2.7GB GPU (3B model, 4-bit)
- Inference: ~1.8GB GPU (3B model)
- 8GB unified RAM is sufficient for one or the other

## Advanced Usage

### Multiple adapters
You can train multiple adapters on different datasets:

```bash
second-brain understand /data/domain-a --adapter domain-a --model llama-3b
second-brain understand /data/domain-b --adapter domain-b --model llama-3b
```

Then query any of them:
```bash
second-brain elucidate --adapter domain-a --model llama-3b --prompt "..."
second-brain elucidate --adapter domain-b --model llama-3b --prompt "..."
```

### Different base models
You can mix base models:

```bash
second-brain understand /data --adapter light --model meta-llama/Llama-3.2-1B-Instruct
second-brain understand /data --adapter heavy --model meta-llama/Llama-3.2-3B-Instruct

# Then use each separately
second-brain elucidate --adapter light --model meta-llama/Llama-3.2-1B-Instruct --prompt "..."
second-brain elucidate --adapter heavy --model meta-llama/Llama-3.2-3B-Instruct --prompt "..."
```

### Hyperparameter tuning
Experiment with different learning rates and epochs:

```bash
second-brain understand /data --adapter v1-lr-1e5 --model llama-3b --lr 1e-5 --epochs 5
second-brain understand /data --adapter v1-lr-5e5 --model llama-3b --lr 5e-5 --epochs 3

# Compare results
second-brain status
second-brain adapters
```

### Re-run training with different data
```bash
second-brain understand /data/v2 --adapter domain --model llama-3b
```
This creates a new timestamped version in `adapters/domain/`. The `--adapter` name groups versions together.

## CLI Help

Get help for any command:

```bash
second-brain --help                  # Show all commands
second-brain understand --help       # Help for understand
second-brain elucidate --help        # Help for elucidate
second-brain daemon --help           # Daemon subcommands
```
