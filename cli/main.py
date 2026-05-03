import json
import sys
from pathlib import Path

import click

from app.config import ROOT, JOBS_FILE, ADAPTERS_DIR
from app.jobs import get_store, JobStatus
from . import client, daemon, display


@click.group()
def cli():
    """Second Brain — LLM Finetuning CLI"""
    pass


@cli.group()
def daemon_group():
    """Daemon management"""
    pass


@daemon_group.command("start")
def daemon_start():
    """Start the background daemon."""
    daemon.start()


@daemon_group.command("stop")
def daemon_stop():
    """Stop the background daemon."""
    daemon.stop()


@daemon_group.command("status")
def daemon_status():
    """Check daemon status and GPU memory."""
    daemon.status()


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--adapter", "adapter_name", required=True, help="Adapter family name")
@click.option("--model", "base_model", required=True, help="Base model name")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
@click.option("--max-seq", default=1024, type=int, help="Max sequence length")
@click.option("--batch", default=1, type=int, help="Batch size")
def understand(data_path, adapter_name, base_model, epochs, lr, max_seq, batch):
    """Submit a training job to finetune a model."""
    if not client.is_daemon_running():
        display.error("Daemon is not running. Start it with: second-brain daemon start")
        sys.exit(1)

    data_dir = Path(data_path)
    if not (data_dir / "train.jsonl").exists():
        display.error("train.jsonl not found in data_path")
        sys.exit(1)
    if not (data_dir / "val.jsonl").exists():
        display.error("val.jsonl not found in data_path")
        sys.exit(1)

    try:
        with client.Client() as c:
            display.info(f"Submitting training job to daemon...")
            result = c.understand(
                data_path=str(data_dir),
                adapter_name=adapter_name,
                base_model=base_model,
                epochs=epochs,
                lr=lr,
                max_seq=max_seq,
                batch_size=batch,
            )

        job_id = result["job_id"]
        display.success(f"Job submitted: {job_id}")
        display.info("Watching progress (Ctrl+C to detach)...\n")

        # Live-follow the job
        with client.Client() as c:
            job = display.live_follow_job(job_id, c)

    except Exception as e:
        display.error(f"Failed to submit job: {e}")
        sys.exit(1)


@cli.command()
@click.option("--adapter", "adapter_name", required=True, help="Adapter name")
@click.option("--prompt", default=None, help="Query prompt (if omitted, starts REPL)")
@click.option("--max-tokens", default=512, type=int, help="Max tokens to generate")
@click.option("--temp", default=0.7, type=float, help="Temperature for sampling")
def elucidate(adapter_name, prompt, max_tokens, temp):
    """Query a finetuned model."""
    if not client.is_daemon_running():
        display.error("Daemon is not running. Start it with: second-brain daemon start")
        sys.exit(1)

    from app.inference import get_base_model_from_adapter
    base_model = get_base_model_from_adapter(adapter_name)
    if not base_model:
        display.error(f"Could not determine base model for adapter '{adapter_name}'. "
                     "Ensure the adapter exists and has valid metadata.")
        sys.exit(1)

    try:
        with client.Client() as c:
            if prompt:
                # One-shot query
                display.info("Generating response...")
                result = c.elucidate(
                    adapter_name=adapter_name,
                    base_model=base_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                display.print_response(result["response"], adapter_name)
            else:
                # Interactive REPL
                display.info(f"Second Brain — {adapter_name} ({base_model})")
                display.info("Type 'exit' or Ctrl+D to quit.\n")

                while True:
                    try:
                        prompt = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break

                    if prompt.lower() in ("exit", "quit"):
                        break

                    if not prompt:
                        continue

                    try:
                        result = c.elucidate(
                            adapter_name=adapter_name,
                            base_model=base_model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temp,
                        )
                        print(f"\n{result['response']}\n")
                    except Exception as e:
                        display.error(f"Query failed: {e}")

    except Exception as e:
        display.error(f"Failed to query model: {e}")
        sys.exit(1)


@cli.command()
@click.argument("job_id", required=False)
def status(job_id):
    """Show job status."""
    if not JOBS_FILE.exists():
        display.warn("No jobs found")
        return

    try:
        store = get_store()

        if job_id:
            # Show detail for one job
            job = store.get(job_id)
            if not job:
                display.error(f"Job {job_id} not found")
                sys.exit(1)
            display.print_job_detail(job.to_dict())
        else:
            # Show table of all jobs
            jobs = [j.to_dict() for j in store.list_all()]
            if not jobs:
                display.warn("No jobs found")
            else:
                display.print_job_table(jobs)

    except Exception as e:
        display.error(f"Failed to fetch status: {e}")
        sys.exit(1)


@cli.command()
def adapters():
    """List all trained adapters."""
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    adapters_list = []
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

        adapters_list.append({
            "adapter_name": family_dir.name,
            "versions": versions,
            "latest_version": versions[0]["version"] if versions else None,
        })

    if not adapters_list:
        display.warn("No adapters found")
    else:
        display.print_adapter_table(adapters_list)


# Register daemon commands under "daemon" group
cli.add_command(daemon_group, name="daemon")


if __name__ == "__main__":
    cli()
