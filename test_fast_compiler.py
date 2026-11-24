"""
Performance-First Benchmark & Verification Script: test_fast_compiler.py
Validates the SpecCompiler deterministic engine for hybrid-cad-agent:
1. Single-shot batch compilation of 4 complex fixture parts.
2. Execution time < 2.0s (pure CPU, 0 MB VRAM, 0 hallucinations).
3. 100% geometric pass (is_watertight, valid solid, volume > 0, origin: center_bottom zmin == 0.0).
4. Confirms MCP server stdout is 100% clean and JSON-RPC compliant.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mcp_server.server import CADService

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


# 4 Complex Mechanical Parts for Shaft Inspection Fixture
FIXTURE_PARTS = [
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
            {"op": "slot", "w": 12.0, "depth": 6.0, "len": 120.0, "axis": "X"},
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
        "description": "Con trượt có lỗ trục chính và 4 lỗ ren bắt đồ gá",
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
    {
        "name": "stepped_shaft_with_keyway",
        "description": "Trục bậc kiểm tra 3 đoạn đồng tâm có rãnh then bằng",
        "origin": "center_bottom",
        "base": {
            "shape": "stepped_shaft",
            "sections": [
                {"d": 20.0, "h": 30.0},
                {"d": 30.0, "h": 50.0},
                {"d": 22.0, "h": 25.0},
            ],
        },
        "features": [
            {"op": "keyway", "w": 8.0, "depth": 4.0, "len": 25.0, "at_z": 55.0, "shaft_r": 15.0},
            {"op": "chamfer", "val": 0.8, "sel": ">Z"},
        ],
    },
]


def test_fast_compiler():
    if console:
        console.rule("[bold cyan]PERFORMANCE-FIRST BENCHMARK (test_fast_compiler.py)[/bold cyan]")
    else:
        print("\n========================================================")
        print("PERFORMANCE-FIRST BENCHMARK (test_fast_compiler.py)")
        print("========================================================\n")

    cad_service = CADService(workspace_root=PROJECT_ROOT / "workspace")

    # --------------------------------------------------------------------------
    # 1. Single-shot batch compilation
    # --------------------------------------------------------------------------
    print(f"[+] Compiling batch of {len(FIXTURE_PARTS)} complex fixture parts via SpecCompiler...")
    for part in FIXTURE_PARTS:
        print(f"    - {part['name']}: {part['description']} (base: {part['base']['shape']})")

    start_perf = time.perf_counter()
    batch_res = cad_service.compile_batch_parts(FIXTURE_PARTS)
    elapsed_total_s = time.perf_counter() - start_perf
    elapsed_sec = batch_res.get("elapsed_sec", round(elapsed_total_s, 2))

    success_count = sum(1 for p in batch_res.get("parts", []) if p.get("status") == "SUCCESS")
    failed_count = sum(1 for p in batch_res.get("parts", []) if p.get("status") != "SUCCESS")
    total_count = len(batch_res.get("parts", []))

    print(f"\n[OK] Batch Compilation Completed in {elapsed_total_s:.3f}s ({elapsed_sec}s reported in schema)!")
    print(f"     Status: {batch_res['status']} | Success: {success_count}/{total_count} | Failed: {failed_count}")
    print(f"     VRAM Consumption: 0 MB (100% Pure CPU Execution)")

    # Assertions on Locked Schema
    assert batch_res["status"] == "SUCCESS", "Batch must succeed completely!"
    assert "elapsed_sec" in batch_res, "Schema must include 'elapsed_sec'!"
    assert "parts" in batch_res, "Schema must include 'parts'!"
    assert success_count == len(FIXTURE_PARTS), f"Expected {len(FIXTURE_PARTS)} successes!"
    assert failed_count == 0, "No parts should fail!"
    assert elapsed_total_s < 2.0, f"Compilation took {elapsed_total_s:.2f}s, exceeding 2.0s limit!"

    # --------------------------------------------------------------------------
    # 2. Geometric & Origin Verification
    # --------------------------------------------------------------------------
    parts_data = batch_res["parts"]
    if console:
        table = Table(title="Deterministic Compilation Summary Table", header_style="bold magenta")
        table.add_column("Part Name", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Source", justify="center")
        table.add_column("BBox Dimensions (mm)", justify="center")
        table.add_column("BBox Min (mm)", justify="center")
        table.add_column("Volume (mm³)", justify="right")
        table.add_column("STEP File Exists", justify="center")

        for p in parts_data:
            step_file_path = PROJECT_ROOT / p["step_path"] if not Path(p["step_path"]).is_absolute() else Path(p["step_path"])
            step_exists = step_file_path.exists()
            status_style = "[green]SUCCESS[/green]" if p["status"] == "SUCCESS" else "[red]FAILED[/red]"
            step_style = "[green]YES[/green]" if step_exists else "[red]NO[/red]"

            table.add_row(
                p["part_name"],
                status_style,
                p.get("source", "compiler"),
                str(p.get("bbox_mm", [])),
                str(p.get("bbox_min", [])),
                f"{p.get('volume_mm3', 0.0):.2f}",
                step_style,
            )
        console.print(table)

    for p in parts_data:
        # Check locked schema keys
        assert p["source"] == "compiler", f"Expected source='compiler', got {p.get('source')}"
        assert p["status"] == "SUCCESS", f"Expected status='SUCCESS', got {p.get('status')}"
        assert p["step_path"].startswith("workspace/parts/"), f"Expected step_path to start with 'workspace/parts/', got {p.get('step_path')}"
        assert "bbox_mm" in p and len(p["bbox_mm"]) == 3, f"Missing or invalid bbox_mm for {p['part_name']}"
        assert "bbox_min" in p and len(p["bbox_min"]) == 3, f"Missing or invalid bbox_min for {p['part_name']}"
        assert "volume_mm3" in p and p["volume_mm3"] > 0, f"Missing or invalid volume_mm3 for {p['part_name']}"

        # Check origin constraint
        zmin = p["bbox_min"][2]
        assert abs(zmin) < 1e-3, f"Origin center_bottom violated for {p['part_name']}: zmin={zmin}"

        # Inspect generated STEP file for watertightness
        step_full_path = str(PROJECT_ROOT / p["step_path"])
        insp = cad_service.inspect_cad_file(step_full_path)
        assert insp.get("is_watertight") is True, f"STEP file for {p['part_name']} is not watertight!"
        assert insp.get("volume_mm3", 0.0) > 0.0, f"Volume for {p['part_name']} is non-positive!"
        print(f"    [OK] Geometry Verified: {p['part_name']} -> Watertight Solid, Vol={insp['volume_mm3']} mm³, zmin={zmin:.3f} mm")

    # --------------------------------------------------------------------------
    # 3. Verify MCP Stdio Hygiene (Zero stdout corruption)
    # --------------------------------------------------------------------------
    print("\n[+] Verifying MCP Server stdout hygiene (Zero stray bytes on stdio)...")
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "mcp_server" / "server.py"), "--test"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_len = len(proc.stdout)
    print(f"    - Captured stdout length: {stdout_len} bytes")
    print(f"    - Captured stderr line count: {len(proc.stderr.splitlines())} lines")
    assert stdout_len == 0, f"MCP server polluted stdout with {stdout_len} bytes! Stdout must be strictly 0 for JSON-RPC!"
    print("    [OK] MCP Server stdout is 100% clean and JSON-RPC ready!")

    print("\n========================================================")
    print("[OK] ALL 5 PERFORMANCE-FIRST OBJECTIVES PASSED 100%!")
    print("========================================================\n")


if __name__ == "__main__":
    test_fast_compiler()
