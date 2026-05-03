import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from app.config import ROOT, DAEMON_PID_FILE, DAEMON_LOG_FILE, DAEMON_PORT
from . import client, display


def _is_running() -> bool:
    """Check if daemon process is alive."""
    if not DAEMON_PID_FILE.exists():
        return False

    try:
        pid = int(DAEMON_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # Doesn't send signal, just checks if process exists
        return True
    except (ValueError, ProcessLookupError, OSError):
        return False


def _read_pid() -> int:
    return int(DAEMON_PID_FILE.read_text().strip())


def start():
    """Start the daemon process."""
    if _is_running():
        display.warn("Daemon is already running")
        return

    DAEMON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Start daemon in background
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.api:app",
                "--host", "127.0.0.1", "--port", str(DAEMON_PORT),
                "--workers", "1",
            ],
            cwd=str(ROOT),
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            stdout=open(DAEMON_LOG_FILE, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent terminal
        )

        # Write PID file
        DAEMON_PID_FILE.write_text(str(proc.pid))

        # Wait for daemon to be ready (poll /health)
        display.info("Starting daemon...")
        for i in range(15):
            time.sleep(1)
            if client.is_daemon_running():
                display.success(f"Daemon started (PID {proc.pid})")
                return
            display.info(f"  Waiting... ({i + 1}/15)")

        display.error("Daemon failed to start. Check logs: daemon log at")
        display.info(f"  tail -f {DAEMON_LOG_FILE}")
        DAEMON_PID_FILE.unlink(missing_ok=True)

    except Exception as e:
        display.error(f"Failed to start daemon: {e}")
        DAEMON_PID_FILE.unlink(missing_ok=True)


def stop():
    """Stop the daemon process."""
    if not _is_running():
        display.warn("Daemon is not running")
        return

    try:
        pid = _read_pid()
        os.kill(pid, signal.SIGTERM)
        display.success("Daemon stopped")
        DAEMON_PID_FILE.unlink(missing_ok=True)
    except Exception as e:
        display.error(f"Failed to stop daemon: {e}")


def status():
    """Show daemon status."""
    if not _is_running():
        display.warn("Daemon is not running")
        return

    try:
        with client.Client() as c:
            health = c.health()
        display.print_daemon_status(health)
    except Exception as e:
        display.error(f"Failed to check daemon status: {e}")
