from pathlib import Path
from dataclasses import dataclass
import torch

ROOT = Path.home() / "Projects" / "second-brain"
ADAPTERS_DIR = ROOT / "adapters"
LOG_DIR = ROOT / "logs"
JOBS_FILE = ROOT / "jobs.json"

LARGE_MODEL_PARAM_THRESHOLD = 1_500_000_000
SMALL_MODEL_DTYPE = "float16"

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]

@dataclass
class DefaultTrainingArgs:
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_steps: int = 100
    logging_steps: int = 10
    bf16: bool = False
    fp16: bool = True

MAX_LOG_LINES = 200
DATALOADER_NUM_WORKERS = 2

DAEMON_PID_FILE = ROOT / ".daemon.pid"
DAEMON_LOG_FILE = LOG_DIR / "daemon.log"
DAEMON_URL = "http://127.0.0.1:8000"
DAEMON_PORT = 8000


def get_device_map():
    if torch.cuda.is_available():
        return {"": "cuda:0"}
    return {"": "cpu"}
