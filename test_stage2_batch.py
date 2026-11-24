"""
Phase 2 Batch Verification Script: test_stage2_batch.py
Tests single-call generate_batch_local with 3 medium-complexity mechanical parts:
1. Slotted Base (Đế có rãnh & lỗ bắt vít)
2. Bearing Pillow Block (Gối đỡ có bậc ổ bi)
3. Linear Carriage (Con trượt có 4 lỗ ren & lỗ trục chính)

Validates:
- 100% geometric pass (is_valid, is_watertight, volume > 0)
- Ollama token accounting (prompt_eval_count, eval_count)
- Runtime elapsed time
- SFT dataset pipeline format (workspace/dataset_sft.jsonl in ChatML)
- Batch usage statistics (workspace/ollama_usage.json)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows UTF-8 stdout configuration
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mcp_server.server import CADService, DEFAULT_OLLAMA_MODEL

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    console = Console()
except ImportError:
    console = None
    rprint = print


# 3 Medium-Complexity Mechanical Parts (Compact Spec Format)
MEDIUM_COMPLEXITY_PARTS = [
    {
        "name": "slotted_base",
        "description": "Đế máy có rãnh trượt dọc và 4 lỗ bắt vít chân đế",
        "origin": "center_bottom",
        "base": {
            "shape": "box",
            "dims": [120.0, 80.0, 15.0],
            "fillet_z": 4.0,
            "chamfer": 1.0,
        },
        "features": [
            {"op": "slot", "w": 12.0, "depth": 6.0, "axis": "X"},
            {"op": "hole_rect", "d": 6.5, "dx": 100.0, "dy": 60.0, "thru": True},
            {"op": "hole", "d": 5.0, "at": [0, 25], "thru": True},
            {"op": "hole", "d": 5.0, "at": [0, -25], "thru": True},
        ],
    },
    {
        "name": "bearing_pillow_block",
        "description": "Gối đỡ có bậc chứa ổ lăn và tai bắt bu lông",
        "origin": "center_bottom",
        "base": {
            "shape": "box",
            "dims": [75.0, 35.0, 40.0],
            "fillet_z": 3.0,
        },
        "features": [
            {"op": "hole", "d": 32.0, "depth": 15.0, "at": [0, 0]},
            {"op": "hole", "d": 20.0, "thru": True, "at": [0, 0]},
            {"op": "hole", "d": 7.0, "at": [-28.0, 0], "thru": True},
            {"op": "hole", "d": 7.0, "at": [28.0, 0], "thru": True},
            {"op": "chamfer", "val": 0.8, "sel": ">Z"},
        ],
    },
    {
        "name": "linear_carriage",
        "description": "Con trượt có lỗ trục dẫn hướng và 4 lỗ ren bắt đồ gá",
        "origin": "center_bottom",
        "base": {
            "shape": "box",
            "dims": [50.0, 50.0, 32.0],
            "fillet_z": 2.0,
        },
        "features": [
            {"op": "hole", "d": 20.0, "thru": True, "at": [0, 0]},
            {"op": "hole_rect", "d": 4.2, "dx": 36.0, "dy": 36.0, "depth": 12.0},
            {"op": "chamfer", "val": 1.0, "sel": ">Z"},
        ],
    },
]


async def run_batch_test():
    if console:
        console.rule("[bold cyan]STAGE 2: BATCH VERIFICATION (test_stage2_batch.py)[/bold cyan]")
    else:
        print("\n========================================================")
        print("STAGE 2: BATCH VERIFICATION (test_stage2_batch.py)")
        print("========================================================\n")

    cad_service = CADService(workspace_root=PROJECT_ROOT / "workspace")

    # Clear previous SFT lines or track count
    sft_file = cad_service.workspace_root / "dataset_sft.jsonl"
    initial_sft_lines = 0
    if sft_file.exists():
        with open(sft_file, "r", encoding="utf-8") as f:
            initial_sft_lines = sum(1 for line in f if line.strip())

    rprint(f"[bold blue][+] Calling generate_batch_local with 3 Medium-Complexity Parts:[/bold blue]")
    for part in MEDIUM_COMPLEXITY_PARTS:
        rprint(f"    - [cyan]{part['name']}[/cyan]: {part['description']} ({len(part['features'])} features)")

    start_time = time.perf_counter()
    batch_output = await cad_service.generate_batch_local(
        parts_specs=MEDIUM_COMPLEXITY_PARTS,
        max_retries=2,
        model=DEFAULT_OLLAMA_MODEL,
    )
    total_elapsed = time.perf_counter() - start_time

    summary = batch_output["summary"]
    results = batch_output["results"]

    rprint(f"\n[bold green][OK] Batch Execution Completed![/bold green]")
    rprint(f"   Elapsed Time: [bold yellow]{total_elapsed:.2f}s[/bold yellow] ({summary['total_time_ms']}ms reported by runner)")
    rprint(f"   Success: [bold green]{summary['success']}/{summary['total']}[/bold green] | Escalate: [bold red]{summary['escalate']}[/bold red]")
    rprint(f"   Total Prompt Tokens: [cyan]{summary.get('prompt_eval_count', 0)}[/cyan]")
    rprint(f"   Total Generated Tokens: [cyan]{summary.get('eval_count', 0)}[/cyan]")

    # Verify 100% geometric success
    assert summary["success"] == len(MEDIUM_COMPLEXITY_PARTS), f"Expected all {len(MEDIUM_COMPLEXITY_PARTS)} to succeed!"
    assert summary["escalate"] == 0, "No parts should escalate in this test batch!"

    # Verify each part on disk & origin alignment
    if console:
        table = Table(title="Batch Parts Validation Summary", header_style="bold magenta")
        table.add_column("Part Name", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Attempts", justify="center")
        table.add_column("BBox Dims (mm)", justify="center")
        table.add_column("BBox Min (mm)", justify="center")
        table.add_column("Volume (mm³)", justify="right")
        table.add_column("STEP File Exists", justify="center")

        for r in results:
            step_exists = Path(r["step_path"]).exists() if r.get("step_path") else False
            status_style = "[green]SUCCESS[/green]" if r["status"] == "SUCCESS" else "[red]ESCALATE[/red]"
            step_style = "[green]YES[/green]" if step_exists else "[red]NO[/red]"

            table.add_row(
                r["part_name"],
                status_style,
                str(r["attempts"]),
                str(r.get("bbox_mm", [])),
                str(r.get("bbox_min", [])),
                f"{r.get('volume_mm3', 0.0):.2f}",
                step_style,
            )
        console.print(table)

    # Origin center_bottom verification
    for r in results:
        assert "bbox_min" in r, f"bbox_min must be present in result for {r['part_name']}"
        zmin = r["bbox_min"][2]
        assert abs(zmin) < 0.05, f"Origin check failed for {r['part_name']}: abs(zmin)={abs(zmin)} >= 0.05!"
        rprint(f"   [bold green][OK] Origin center_bottom verified for {r['part_name']}:[/bold green] zmin = {zmin:.3f} mm (< 0.05)")

    # Verify SFT Dataset Pipeline (workspace/dataset_sft.jsonl)
    rprint("\n[bold blue][+] Validating SFT Dataset Pipeline (workspace/dataset_sft.jsonl)...[/bold blue]")
    assert sft_file.exists(), "dataset_sft.jsonl was not created!"
    with open(sft_file, "r", encoding="utf-8") as f:
        sft_lines = [line.strip() for line in f if line.strip()]

    new_lines = len(sft_lines) - initial_sft_lines
    rprint(f"   Total SFT Samples: [bold green]{len(sft_lines)}[/bold green] (Added [bold cyan]+{new_lines}[/bold cyan] from this batch)")
    assert new_lines >= len(MEDIUM_COMPLEXITY_PARTS), "All successful parts must be logged to dataset_sft.jsonl!"

    # Verify ChatML schema format on the last line
    last_sample = json.loads(sft_lines[-1])
    assert "messages" in last_sample, "ChatML format must have 'messages' root key!"
    messages = last_sample["messages"]
    assert len(messages) == 3, "Sample must contain exactly 3 messages (system, user, assistant)!"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert "```python" in messages[2]["content"], "Assistant content must be a Python code block!"
    rprint("   [bold green][OK] SFT ChatML Format Validated:[/bold green] Ready for Kaggle Unsloth QLoRA fine-tuning!")

    # Verify Ollama Token Usage Statistics (workspace/ollama_usage.json)
    rprint("\n[bold blue][+] Validating Ollama Usage Log (workspace/ollama_usage.json)...[/bold blue]")
    usage_file = cad_service.workspace_root / "ollama_usage.json"
    assert usage_file.exists(), "ollama_usage.json was not created!"
    with open(usage_file, "r", encoding="utf-8") as uf:
        usage_data = json.load(uf)

    assert len(usage_data) > 0, "ollama_usage.json must contain at least 1 record!"
    latest_usage = usage_data[-1]
    rprint(f"   Latest Batch ID: [cyan]{latest_usage.get('batch_id')}[/cyan] ({latest_usage.get('timestamp')})")
    rprint(f"   Batch Parts: {latest_usage.get('total_parts')} | Success: {latest_usage.get('success_count')}")
    rprint(f"   Prompt Eval Tokens: [yellow]{latest_usage.get('prompt_eval_count')}[/yellow]")
    rprint(f"   Eval (Generation) Tokens: [yellow]{latest_usage.get('eval_count')}[/yellow]")
    rprint(f"   Total Tokens Consumed: [bold yellow]{latest_usage.get('total_tokens')}[/bold yellow]")
    rprint(f"   Elapsed Time: [cyan]{latest_usage.get('elapsed_sec')}s[/cyan]")
    rprint("   [bold green][OK] Ollama Usage Statistics Logged Successfully![/bold green]")

    rprint("\n[bold green]======================================================================[/bold green]")
    rprint("[bold green][OK] ALL STAGE 2 BATCH TESTS & PIPELINE CHECKS PASSED 100%![/bold green]")
    rprint("[bold green]======================================================================[/bold green]\n")


if __name__ == "__main__":
    asyncio.run(run_batch_test())
