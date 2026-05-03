from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ADAPTERS_DIR, LOG_DIR
from .jobs import JobStatus, get_store
from .trainer import run_training_job
from .inference import generate, get_latest_adapter_path, unload_model

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("api")

app = FastAPI(title="Second Brain — LLM Finetuning", version="2.0")

_executor = ThreadPoolExecutor(max_workers=1)
_active_future = None
_active_future_lock = asyncio.Lock()


class TrainingArgsIn(BaseModel):
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024


class UnderstandRequest(BaseModel):
    data_path: str
    adapter_name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=64)
    base_model: str
    training_args: Optional[TrainingArgsIn] = None


class UnderstandResponse(BaseModel):
    job_id: str
    adapter_name: str
    status: str


class JobProgressOut(BaseModel):
    epoch: int
    step: int
    loss: float


class StatusResponse(BaseModel):
    job_id: str
    status: str
    adapter_name: str
    base_model: str
    progress: JobProgressOut
    logs: list[str]
    adapter_path: Optional[str]
    error: Optional[str]


class ElucidateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    adapter_name: str
    base_model: str
    generation_args: Optional[dict] = None


class ElucidateResponse(BaseModel):
    response: str
    adapter_name: str
    adapter_version: str
    base_model: str


class AdapterVersionInfo(BaseModel):
    version: str
    path: str
    metadata: dict


class AdapterInfo(BaseModel):
    adapter_name: str
    versions: list[AdapterVersionInfo]
    latest_version: Optional[str]


@app.post("/understand", response_model=UnderstandResponse)
async def understand(body: UnderstandRequest):
    global _active_future

    async with _active_future_lock:
        if _active_future is not None and not _active_future.done():
            raise HTTPException(
                status_code=409,
                detail="A training job is already running. Wait for it to complete.",
            )

    data_dir = Path(body.data_path)
    if not data_dir.exists():
        raise HTTPException(status_code=400, detail=f"data_path not found: {body.data_path}")
    if not (data_dir / "train.jsonl").exists():
        raise HTTPException(status_code=400, detail="train.jsonl not found in data_path")

    store = get_store()
    training_args_dict = body.training_args.model_dump() if body.training_args else {}
    job = store.create(
        adapter_name=body.adapter_name,
        base_model=body.base_model,
        data_path=body.data_path,
    )

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, unload_model)

    async with _active_future_lock:
        _active_future = _executor.submit(
            run_training_job,
            job_id=job.job_id,
            base_model=body.base_model,
            data_path=body.data_path,
            adapter_name=body.adapter_name,
            training_args_override=training_args_dict,
        )

    log.info("Job %s queued: adapter=%s base=%s", job.job_id, body.adapter_name, body.base_model)
    return UnderstandResponse(
        job_id=job.job_id,
        adapter_name=body.adapter_name,
        status=JobStatus.QUEUED.value,
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def job_status(job_id: str):
    store = get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return StatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        adapter_name=job.adapter_name,
        base_model=job.base_model,
        progress=JobProgressOut(
            epoch=job.progress.epoch,
            step=job.progress.step,
            loss=job.progress.loss,
        ),
        logs=job.logs[-50:],
        adapter_path=job.adapter_path,
        error=job.error,
    )


@app.post("/elucidate", response_model=ElucidateResponse)
async def elucidate(body: ElucidateRequest):
    if _active_future is not None and not _active_future.done():
        raise HTTPException(
            status_code=503,
            detail="Training job is running. Inference is unavailable until training completes.",
        )

    adapter_path = get_latest_adapter_path(body.adapter_name)
    if adapter_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No adapter found with name '{body.adapter_name}'",
        )

    loop = asyncio.get_event_loop()
    try:
        response_text, adapter_version, _ = await loop.run_in_executor(
            None,
            lambda: generate(
                prompt=body.prompt,
                base_model=body.base_model,
                adapter_name=body.adapter_name,
                generation_args=body.generation_args,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=503,
            detail="Out of GPU memory. Try again after training completes.",
        )
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))

    return ElucidateResponse(
        response=response_text,
        adapter_name=body.adapter_name,
        adapter_version=adapter_version,
        base_model=body.base_model,
    )


@app.get("/adapters")
async def list_adapters() -> dict:
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    adapters = []
    for family_dir in sorted(ADAPTERS_DIR.iterdir()):
        if not family_dir.is_dir():
            continue
        versions = []
        for version_dir in sorted(family_dir.iterdir(), key=lambda d: d.name, reverse=True):
            if not version_dir.is_dir():
                continue
            metadata = {}
            meta_file = version_dir / "metadata.json"
            if meta_file.exists():
                try:
                    metadata = json.loads(meta_file.read_text())
                except Exception:
                    pass
            versions.append({
                "version": version_dir.name,
                "path": str(version_dir),
                "metadata": metadata,
            })
        adapters.append({
            "adapter_name": family_dir.name,
            "versions": versions,
            "latest_version": versions[0]["version"] if versions else None,
        })
    return {"adapters": adapters}


@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free = total - reserved
        gpu_info = {
            "total_mb": round(total / 1024 / 1024),
            "allocated_mb": round(allocated / 1024 / 1024),
            "reserved_mb": round(reserved / 1024 / 1024),
            "free_mb": round(free / 1024 / 1024),
        }
    else:
        gpu_info = {"error": "CUDA not available"}

    active_jobs = [
        j for j in get_store().list_all()
        if j.status in (JobStatus.RUNNING, JobStatus.QUEUED)
    ]

    return {
        "status": "ok",
        "gpu": gpu_info,
        "active_jobs": len(active_jobs),
        "training_busy": _active_future is not None and not _active_future.done(),
    }
