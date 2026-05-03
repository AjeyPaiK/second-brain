import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from .config import JOBS_FILE, MAX_LOG_LINES


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobProgress:
    epoch: int = 0
    step: int = 0
    loss: float = 0.0


@dataclass
class Job:
    job_id: str
    adapter_name: str
    base_model: str
    data_path: str
    status: JobStatus = JobStatus.QUEUED
    progress: JobProgress = field(default_factory=JobProgress)
    logs: list[str] = field(default_factory=list)
    adapter_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["progress"] = asdict(self.progress)
        return d


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if JOBS_FILE.exists():
            try:
                data = json.loads(JOBS_FILE.read_text())
                for jd in data.values():
                    status = JobStatus(jd["status"])
                    if status == JobStatus.RUNNING:
                        status = JobStatus.FAILED
                        jd["error"] = "Service restarted while job was running"
                    progress_data = jd.get("progress", {})
                    job = Job(
                        job_id=jd["job_id"],
                        adapter_name=jd["adapter_name"],
                        base_model=jd["base_model"],
                        data_path=jd["data_path"],
                        status=status,
                        progress=JobProgress(
                            epoch=progress_data.get("epoch", 0),
                            step=progress_data.get("step", 0),
                            loss=progress_data.get("loss", 0.0),
                        ),
                        logs=jd.get("logs", []),
                        adapter_path=jd.get("adapter_path"),
                        error=jd.get("error"),
                        created_at=jd.get("created_at", ""),
                        completed_at=jd.get("completed_at"),
                    )
                    self._jobs[job.job_id] = job
            except Exception:
                pass

    def _persist(self):
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {jid: job.to_dict() for jid, job in self._jobs.items()}
        JOBS_FILE.write_text(json.dumps(data, indent=2))

    def create(self, adapter_name: str, base_model: str, data_path: str) -> Job:
        job = Job(
            job_id=str(uuid.uuid4()),
            adapter_name=adapter_name,
            base_model=base_model,
            data_path=data_path,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._persist()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus,
                      error: str = None, adapter_path: str = None):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if error is not None:
                    job.error = error
                if adapter_path is not None:
                    job.adapter_path = adapter_path
                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = datetime.utcnow().isoformat()
                self._persist()

    def update_progress(self, job_id: str, epoch: int, step: int, loss: float):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = JobProgress(epoch=epoch, step=step, loss=loss)

    def append_log(self, job_id: str, line: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.logs.append(line)
                if len(job.logs) > MAX_LOG_LINES:
                    job.logs = job.logs[-MAX_LOG_LINES:]

    def list_all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


_store: Optional[JobStore] = None

def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
