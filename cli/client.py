import httpx
from typing import Optional

DAEMON_URL = "http://127.0.0.1:8000"
TIMEOUT = 30.0


def is_daemon_running() -> bool:
    try:
        r = httpx.get(f"{DAEMON_URL}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


class Client:
    def __init__(self):
        self.client = httpx.Client(base_url=DAEMON_URL, timeout=TIMEOUT)

    def health(self) -> dict:
        r = self.client.get("/health")
        r.raise_for_status()
        return r.json()

    def understand(self, data_path: str, adapter_name: str, base_model: str,
                   epochs: int = 3, lr: float = 2e-4, max_seq: int = 1024,
                   batch_size: int = 1) -> dict:
        payload = {
            "data_path": data_path,
            "adapter_name": adapter_name,
            "base_model": base_model,
            "training_args": {
                "num_train_epochs": epochs,
                "learning_rate": lr,
                "max_seq_length": max_seq,
                "per_device_train_batch_size": batch_size,
            }
        }
        r = self.client.post("/understand", json=payload)
        r.raise_for_status()
        return r.json()

    def elucidate(self, adapter_name: str, base_model: str, prompt: str,
                  max_tokens: int = 512, temperature: float = 0.7) -> dict:
        payload = {
            "prompt": prompt,
            "adapter_name": adapter_name,
            "base_model": base_model,
            "generation_args": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
        }
        r = self.client.post("/elucidate", json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()

    def status(self, job_id: str) -> dict:
        r = self.client.get(f"/status/{job_id}")
        r.raise_for_status()
        return r.json()

    def adapters(self) -> dict:
        r = self.client.get("/adapters")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
