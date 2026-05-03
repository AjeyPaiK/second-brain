import time
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn
from rich.live import Live
from rich.markdown import Markdown

console = Console()


def error(msg: str):
    console.print(f"[red]✗[/red] {msg}")


def success(msg: str):
    console.print(f"[green]✓[/green] {msg}")


def warn(msg: str):
    console.print(f"[yellow]⚠[/yellow] {msg}")


def info(msg: str):
    console.print(f"[blue]ℹ[/blue] {msg}")


def print_job_table(jobs: list[dict]):
    table = Table(title="Jobs")
    table.add_column("Job ID", style="cyan", width=12)
    table.add_column("Adapter", style="magenta", width=15)
    table.add_column("Status", style="yellow", width=10)
    table.add_column("Epoch", width=8)
    table.add_column("Loss", width=8)
    table.add_column("Created", width=20)

    for job in jobs:
        job_id = job["job_id"][:12]
        adapter = job["adapter_name"]
        status = job["status"]
        epoch = f"{job['progress']['epoch']:.1f}"
        loss = f"{job['progress']['loss']:.4f}"
        created = job["created_at"][:19]

        status_color = {
            "queued": "yellow",
            "running": "blue",
            "completed": "green",
            "failed": "red",
        }.get(status, "white")

        table.add_row(job_id, adapter, f"[{status_color}]{status}[/]", epoch, loss, created)

    console.print(table)


def print_adapter_table(adapters: list[dict]):
    table = Table(title="Adapters")
    table.add_column("Name", style="magenta", width=20)
    table.add_column("Versions", width=8)
    table.add_column("Latest Version", style="cyan", width=18)
    table.add_column("Base Model", width=25)
    table.add_column("Eval Loss", width=10)

    for adapter in adapters:
        name = adapter["adapter_name"]
        versions = len(adapter["versions"])
        latest = adapter["latest_version"] or "—"
        base_model = adapter["versions"][0]["metadata"].get("base_model", "?")[:25] if adapter["versions"] else "—"
        eval_loss = "—"
        if adapter["versions"] and adapter["versions"][0]["metadata"]:
            el = adapter["versions"][0]["metadata"].get("eval_loss")
            if el is not None:
                eval_loss = f"{el:.4f}"

        table.add_row(name, str(versions), latest, base_model, eval_loss)

    console.print(table)


def print_job_detail(job: dict):
    progress = job["progress"]
    status = job["status"]

    status_color = {
        "queued": "yellow",
        "running": "blue",
        "completed": "green",
        "failed": "red",
    }.get(status, "white")

    lines = [
        f"[bold]Job ID:[/bold] {job['job_id']}",
        f"[bold]Adapter:[/bold] {job['adapter_name']}",
        f"[bold]Base Model:[/bold] {job['base_model']}",
        f"[bold]Status:[/bold] [{status_color}]{status}[/]",
        f"[bold]Progress:[/bold] Epoch {progress['epoch']} | Step {progress['step']} | Loss {progress['loss']:.4f}",
        f"[bold]Created:[/bold] {job['created_at']}",
    ]

    if job["completed_at"]:
        lines.append(f"[bold]Completed:[/bold] {job['completed_at']}")

    if job["adapter_path"]:
        lines.append(f"[bold]Adapter Path:[/bold] {job['adapter_path']}")

    if job["error"]:
        lines.append(f"[bold]Error:[/bold] [red]{job['error']}[/]")

    lines.append("")
    lines.append("[bold]Recent Logs:[/bold]")
    for log_line in job["logs"][-10:]:
        lines.append(f"  {log_line}")

    content = "\n".join(lines)
    panel = Panel(content, title=f"Job {job['job_id'][:12]}", expand=False)
    console.print(panel)


def live_follow_job(job_id: str, client, poll_interval: float = 2.0, timeout: float = 3600):
    """Live follow job progress with Rich Live display. Blocks until job is done."""
    start_time = time.time()
    last_logs_count = 0

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Waiting...", total=100, visible=False)

            while time.time() - start_time < timeout:
                try:
                    job = client.status(job_id)
                except Exception as e:
                    warn(f"Failed to fetch job status: {e}")
                    time.sleep(poll_interval)
                    continue

                status = job["status"]
                prog = job["progress"]
                epoch = prog["epoch"]
                step = prog["step"]
                loss = prog["loss"]

                # Display new logs
                if len(job["logs"]) > last_logs_count:
                    new_logs = job["logs"][last_logs_count:]
                    for log_line in new_logs:
                        console.print(f"  {log_line}")
                    last_logs_count = len(job["logs"])

                if status == "completed":
                    success(f"Training complete! Adapter saved to: {job['adapter_path']}")
                    return job
                elif status == "failed":
                    error(f"Training failed: {job['error']}")
                    return job
                elif status == "running":
                    progress.update(
                        task,
                        description=f"[cyan]Epoch {epoch:.1f} | Step {step} | Loss {loss:.4f}",
                        completed=min(step, 100),
                        visible=True,
                    )
                else:
                    progress.update(task, description=f"[yellow]{status}...", visible=True)

                time.sleep(poll_interval)

        error("Job timed out")
        return None

    except KeyboardInterrupt:
        console.print("\n[yellow]Detached from job (continues in background)[/]")
        return None


def print_response(text: str, adapter_name: Optional[str] = None):
    """Display model response in a nice panel."""
    if adapter_name:
        title = f"Response from {adapter_name}"
    else:
        title = "Response"

    try:
        md = Markdown(text)
        panel = Panel(md, title=title, expand=False)
        console.print(panel)
    except Exception:
        # Fallback if markdown parsing fails
        panel = Panel(text, title=title, expand=False)
        console.print(panel)


def print_daemon_status(health: dict):
    """Print daemon health status."""
    gpu = health.get("gpu", {})
    active_jobs = health.get("active_jobs", 0)
    training_busy = health.get("training_busy", False)

    status_color = "green" if health["status"] == "ok" else "red"

    lines = [
        f"[bold]Status:[/bold] [{status_color}]{health['status']}[/]",
        f"[bold]Active Jobs:[/bold] {active_jobs}",
        f"[bold]Training Busy:[/bold] {'Yes' if training_busy else 'No'}",
    ]

    if "error" in gpu:
        lines.append(f"[bold]GPU:[/bold] [red]{gpu['error']}[/]")
    else:
        lines.append(f"[bold]GPU Memory:[/bold]")
        lines.append(f"  Total: {gpu.get('total_mb', 0)} MB")
        lines.append(f"  Allocated: {gpu.get('allocated_mb', 0)} MB")
        lines.append(f"  Reserved: {gpu.get('reserved_mb', 0)} MB")
        lines.append(f"  Free: {gpu.get('free_mb', 0)} MB")

    content = "\n".join(lines)
    panel = Panel(content, title="Daemon Status", expand=False)
    console.print(panel)
