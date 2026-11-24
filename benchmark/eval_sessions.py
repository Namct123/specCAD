"""
Automated Benchmark Analysis & Scoring Engine: benchmark/eval_sessions.py
Parses Codex / agent session logs (.jsonl), analyzes token economics,
evaluates 3D geometric accuracy using OpenCASCADE / CadQuery / trimesh,
and generates comprehensive A/B benchmark reports (Markdown, CSV, and DOCX).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# UTC+7 Timezone for reporting
UTC7 = datetime.timezone(datetime.timedelta(hours=7))

try:
    import cadquery as cq
except ImportError:
    cq = None

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    docx = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


@dataclass
class SessionMetrics:
    """Session Token & Latency Metrics extracted from JSONL transcripts."""
    session_label: str = "Unknown"
    log_file: str = ""
    uncached_input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    output_non_reasoning_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    tool_calls_count: int = 0
    tool_calls_observed: bool = False
    model_calls_with_usage: int = 0
    ollama_input_tokens: int = 0
    ollama_output_tokens: int = 0
    tool_names: List[str] = field(default_factory=list)
    turn_count: int = 0
    start_timestamp_utc7: str = ""
    end_timestamp_utc7: str = ""


@dataclass
class GeometryMetrics:
    """CAD Geometry Validation & Accuracy Metrics."""
    cad_label: str = "Unknown"
    cad_file: str = ""
    exists: bool = False
    is_valid_solid: bool = False
    is_watertight: bool = False
    volume_mm3: float = 0.0
    bbox_dimensions: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    origin_zmin: float = 0.0
    interference_volume_mm3: float = 0.0
    has_collisions: bool = False
    collision_verification_complete: bool = False


class SessionLogParser:
    """Robust parser for Codex, Claude Code, and Antigravity JSONL session transcripts."""

    @staticmethod
    def parse_log(log_path: Union[str, Path], label: str = "Session") -> SessionMetrics:
        path = Path(log_path).resolve()
        metrics = SessionMetrics(session_label=label, log_file=str(path))
        if not path.exists():
            return metrics

        first_timestamp: Optional[float] = None
        last_timestamp: Optional[float] = None
        last_counter = None
        counter_calls = 0
        usage_records = {}

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                # 1. Track turn count
                step_type = data.get("type", "")
                payload = data.get("payload") or {}
                if step_type == "token_usage_record" and payload.get("usage"):
                    response_id = payload.get("response_id") or line
                    usage_records.setdefault(response_id, payload["usage"])
                info = data.get("info") or (
                    payload.get("info") if payload.get("type") == "token_count" else None
                )
                if info:
                    counter = info.get("total_token_usage")
                    if counter and counter != last_counter:
                        if info.get("last_token_usage"):
                            counter_calls += 1
                        last_counter = counter
                if step_type in ("USER_INPUT", "user", "turn_start"):
                    metrics.turn_count += 1

                # 2. Track tool calls
                tools = data.get("tool_calls", [])
                if isinstance(tools, list):
                    for tc in tools:
                        metrics.tool_calls_observed = True
                        metrics.tool_calls_count += 1
                        name = tc.get("name") or tc.get("tool") or tc.get("function", {}).get("name")
                        if name:
                            metrics.tool_names.append(str(name))
                if step_type == "response_item" and payload.get("type") in (
                    "function_call", "custom_tool_call", "local_shell_call", "web_search_call", "mcp_tool_call",
                ):
                    metrics.tool_calls_observed = True
                    metrics.tool_calls_count += 1
                    metrics.tool_names.append(str(payload.get("name") or payload["type"]))

                # 3. Track token usage
                usage = data.get("usage") or data.get("token_usage") or data.get("metrics", {}).get("usage")
                if usage and isinstance(usage, dict):
                    metrics.model_calls_with_usage += 1
                    prompt_tok = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
                    compl_tok = usage.get("completion_tokens") or usage.get("output_tokens", 0)

                    # Cached tokens
                    cached_tok = (
                        usage.get("prompt_tokens_details", {}).get("cached_tokens")
                        or usage.get("cached_input_tokens")
                        or usage.get("cache_read_input_tokens")
                        or 0
                    )
                    # Reasoning tokens
                    reasoning_tok = (
                        usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                        or usage.get("reasoning_tokens")
                        or 0
                    )

                    uncached_tok = max(0, prompt_tok - cached_tok)

                    metrics.uncached_input_tokens += uncached_tok
                    metrics.cached_input_tokens += cached_tok
                    metrics.output_tokens += compl_tok
                    metrics.reasoning_output_tokens += reasoning_tok

                # 4. Ollama counters are local inference, outside the Cloud budget.
                if "prompt_eval_count" in data or "eval_count" in data:
                    metrics.ollama_input_tokens += data.get("prompt_eval_count", 0)
                    metrics.ollama_output_tokens += data.get("eval_count", 0)

                # 5. Timestamp tracking for duration & UTC+7 logging
                ts_raw = data.get("created_at") or data.get("timestamp")
                if ts_raw:
                    try:
                        if isinstance(ts_raw, (int, float)):
                            ts = float(ts_raw)
                        else:
                            dt = datetime.datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                            ts = dt.timestamp()
                        if first_timestamp is None or ts < first_timestamp:
                            first_timestamp = ts
                        if last_timestamp is None or ts > last_timestamp:
                            last_timestamp = ts
                    except Exception:
                        pass

        if metrics.elapsed_seconds == 0.0 and first_timestamp and last_timestamp:
            metrics.elapsed_seconds = max(0.1, round(last_timestamp - first_timestamp, 2))

        if first_timestamp:
            dt_first = datetime.datetime.fromtimestamp(first_timestamp, tz=UTC7)
            metrics.start_timestamp_utc7 = dt_first.strftime("%Y-%m-%d %H:%M:%S UTC+7")
        if last_timestamp:
            dt_last = datetime.datetime.fromtimestamp(last_timestamp, tz=UTC7)
            metrics.end_timestamp_utc7 = dt_last.strftime("%Y-%m-%d %H:%M:%S UTC+7")

        if last_counter or usage_records:
            fields = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
            counter = last_counter or {
                key: sum(usage.get(key, 0) for usage in usage_records.values()) for key in fields
            }
            metrics.cached_input_tokens = counter.get("cached_input_tokens", 0)
            metrics.uncached_input_tokens = max(0, counter.get("input_tokens", 0) - metrics.cached_input_tokens)
            metrics.output_tokens = counter.get("output_tokens", 0)
            metrics.reasoning_output_tokens = counter.get("reasoning_output_tokens", 0)
            metrics.model_calls_with_usage = len(usage_records) or counter_calls

        metrics.output_non_reasoning_tokens = max(0, metrics.output_tokens - metrics.reasoning_output_tokens)
        metrics.total_tokens = (
            metrics.uncached_input_tokens + metrics.cached_input_tokens + metrics.output_tokens
        )
        return metrics


class GeometryEvaluator:
    """Evaluates 3D CAD models (STEP/STL) and computes volume, bounding box, and collisions."""

    @staticmethod
    def evaluate_cad_file(cad_path: Union[str, Path], label: str = "CAD Model") -> GeometryMetrics:
        path = Path(cad_path).resolve()
        geo = GeometryMetrics(cad_label=label, cad_file=str(path))
        if not path.exists():
            return geo

        geo.exists = True
        ext = path.suffix.lower()

        # Try CadQuery / OpenCASCADE first for exact BRep analysis
        if cq and ext in (".step", ".stp"):
            try:
                shape = cq.importers.importStep(str(path))
                from mcp_server.cad_runner import HeadlessCADRunner

                metrics = HeadlessCADRunner.compute_metrics(shape)
                geo.is_valid_solid = metrics.is_valid and metrics.solid_count > 0
                geo.is_watertight = metrics.is_watertight
                geo.volume_mm3 = metrics.volume
                geo.bbox_dimensions = list(metrics.bbox_dimensions)
                geo.bbox_min = list(metrics.bbox_min)
                geo.origin_zmin = metrics.bbox_min[2]
                if metrics.is_watertight:
                    report = HeadlessCADRunner.check_assembly_interferences(shape)
                    geo.has_collisions = report["has_collisions"]
                    geo.interference_volume_mm3 = report["total_interference_volume_mm3"]
                    geo.collision_verification_complete = report["verification_complete"]
                return geo
            except Exception:
                pass

        # Trimesh fallback
        if trimesh:
            try:
                mesh = trimesh.load(str(path), force="mesh")
                geo.is_watertight = bool(mesh.is_watertight)
                geo.is_valid_solid = bool(mesh.is_volume and mesh.volume > 0)
                geo.volume_mm3 = round(float(mesh.volume), 2)
                bounds = mesh.bounds
                geo.bbox_min = [round(float(b), 3) for b in bounds[0]]
                geo.bbox_dimensions = [round(float(d), 3) for d in mesh.extents]
                geo.origin_zmin = round(float(bounds[0][2]), 4)
                return geo
            except Exception:
                pass

        return geo

    @staticmethod
    def compute_interference(assembly_path: Union[str, Path]) -> Tuple[bool, float]:
        """Calculates collision between components in an assembly STEP file."""
        path = Path(assembly_path).resolve()
        if not path.exists() or not cq:
            return False, 0.0

        try:
            from mcp_server.cad_runner import HeadlessCADRunner
            shape = cq.importers.importStep(str(path))
            report = HeadlessCADRunner.check_assembly_interferences(shape)
            return report.get("has_collisions", False), report.get("max_intersection_mm3", 0.0)
        except Exception:
            return False, 0.0


class BenchmarkEvaluator:
    """Orchestrates comprehensive A/B evaluation between Direct and Hybrid approaches."""

    def __init__(
        self,
        direct_log: Optional[Path] = None,
        hybrid_log: Optional[Path] = None,
        direct_cad: Optional[Path] = None,
        hybrid_cad: Optional[Path] = None,
        ground_truth_cad: Optional[Path] = None,
        demo: bool = False,
        direct_wall_seconds: Optional[float] = None,
        hybrid_wall_seconds: Optional[float] = None,
    ):
        self.direct_log = direct_log
        self.hybrid_log = hybrid_log
        self.direct_cad = direct_cad
        self.hybrid_cad = hybrid_cad
        self.ground_truth_cad = ground_truth_cad
        self.demo = demo
        self.direct_wall_seconds = direct_wall_seconds
        self.hybrid_wall_seconds = hybrid_wall_seconds

    def run_evaluation(self) -> Dict[str, Any]:
        if not self.demo:
            for label, path in (
                ("Direct log", self.direct_log), ("Hybrid log", self.hybrid_log),
                ("Direct CAD", self.direct_cad), ("Hybrid CAD", self.hybrid_cad),
            ):
                if path is None or not path.is_file():
                    raise ValueError(f"{label} must be an existing file for a measured benchmark")
            if self.ground_truth_cad is not None and not self.ground_truth_cad.is_file():
                raise ValueError("Ground Truth CAD file does not exist")
        # 1. Parse session logs
        direct_session = (
            SessionLogParser.parse_log(self.direct_log, "Direct Codex")
            if self.direct_log and self.direct_log.exists()
            else self._mock_session("Direct Codex (Handwritten CadQuery)")
        )
        hybrid_session = (
            SessionLogParser.parse_log(self.hybrid_log, "Hybrid CAD (Deterministic)")
            if self.hybrid_log and self.hybrid_log.exists()
            else self._mock_session("Hybrid CAD (Deterministic Compiler)")
        )

        # 2. Evaluate Geometry
        direct_geo = (
            GeometryEvaluator.evaluate_cad_file(self.direct_cad, "Direct Model")
            if self.direct_cad and self.direct_cad.exists()
            else self._mock_geometry("Direct Model")
        )
        hybrid_geo = (
            GeometryEvaluator.evaluate_cad_file(self.hybrid_cad, "Hybrid Model")
            if self.hybrid_cad and self.hybrid_cad.exists()
            else self._mock_geometry("Hybrid Model")
        )

        gt_geo = (
            GeometryEvaluator.evaluate_cad_file(self.ground_truth_cad, "Ground Truth")
            if self.ground_truth_cad and self.ground_truth_cad.exists()
            else direct_geo
        )
        if not self.demo and (not direct_session.total_tokens or not hybrid_session.total_tokens):
            raise ValueError("Both logs must contain Cloud token usage")
        for session, wall in ((direct_session, self.direct_wall_seconds), (hybrid_session, self.hybrid_wall_seconds)):
            if wall is not None:
                if not math.isfinite(wall) or wall <= 0:
                    raise ValueError("Measured wall time must be finite and positive")
                session.elapsed_seconds = wall

        # 3. Compute Comparative Statistics
        direct_uncached = direct_session.uncached_input_tokens + direct_session.output_tokens
        hybrid_uncached = hybrid_session.uncached_input_tokens + hybrid_session.output_tokens
        token_reduction_pct = round((1.0 - hybrid_uncached / max(1, direct_uncached)) * 100, 2)
        total_reduction_pct = round((1.0 - hybrid_session.total_tokens / max(1, direct_session.total_tokens)) * 100, 2)

        cached_ratio_direct = (
            round((direct_session.cached_input_tokens / max(1, direct_session.total_tokens)) * 100, 1)
        )
        cached_ratio_hybrid = (
            round((hybrid_session.cached_input_tokens / max(1, hybrid_session.total_tokens)) * 100, 1)
        )

        vol_error_direct = 0.0
        vol_error_hybrid = 0.0
        if gt_geo.volume_mm3 > 0:
            vol_error_direct = round(abs(direct_geo.volume_mm3 - gt_geo.volume_mm3) / gt_geo.volume_mm3 * 100, 2)
            vol_error_hybrid = round(abs(hybrid_geo.volume_mm3 - gt_geo.volume_mm3) / gt_geo.volume_mm3 * 100, 2)

        speedup_factor = round(direct_session.elapsed_seconds / max(0.1, hybrid_session.elapsed_seconds), 2)

        eval_timestamp_utc7 = datetime.datetime.now(UTC7).strftime("%Y-%m-%d %H:%M:%S UTC+7")

        report_data = {
            "mode": "demo" if self.demo else "measured",
            "evaluation_time_utc7": eval_timestamp_utc7,
            "metrics": {
                "uncached_cloud_tokens": {
                    "direct": direct_uncached, "hybrid": hybrid_uncached, "unit": "tokens",
                    "diff_pct": token_reduction_pct,
                },
                "uncached_input_tokens": {
                    "direct": direct_session.uncached_input_tokens,
                    "hybrid": hybrid_session.uncached_input_tokens,
                    "unit": "tokens",
                },
                "cached_input_tokens": {
                    "direct": direct_session.cached_input_tokens,
                    "hybrid": hybrid_session.cached_input_tokens,
                    "unit": "tokens",
                },
                "output_tokens": {
                    "direct": direct_session.output_tokens,
                    "hybrid": hybrid_session.output_tokens,
                    "unit": "tokens",
                },
                "output_non_reasoning_tokens": {
                    "direct": direct_session.output_non_reasoning_tokens,
                    "hybrid": hybrid_session.output_non_reasoning_tokens,
                    "unit": "tokens",
                },
                "reasoning_output_tokens": {
                    "direct": direct_session.reasoning_output_tokens,
                    "hybrid": hybrid_session.reasoning_output_tokens,
                    "unit": "tokens",
                },
                "total_tokens": {
                    "direct": direct_session.total_tokens,
                    "hybrid": hybrid_session.total_tokens,
                    "unit": "tokens",
                    "diff_pct": total_reduction_pct,
                },
                "elapsed_seconds": {
                    "direct": direct_session.elapsed_seconds,
                    "hybrid": hybrid_session.elapsed_seconds,
                    "unit": "s",
                    "speedup": speedup_factor,
                    "note": ("Measured wall time" if self.direct_wall_seconds and self.hybrid_wall_seconds
                             else "Inferred from the supplied log timestamps unless overridden"),
                },
                "model_calls_with_usage": {
                    "direct": direct_session.model_calls_with_usage,
                    "hybrid": hybrid_session.model_calls_with_usage,
                    "unit": "calls",
                },
                "tool_calls_count": {
                    "direct": direct_session.tool_calls_count if direct_session.tool_calls_observed else "Unavailable",
                    "hybrid": hybrid_session.tool_calls_count if hybrid_session.tool_calls_observed else "Unavailable",
                    "unit": "calls",
                },
                "volume_mm3": {
                    "direct": direct_geo.volume_mm3,
                    "hybrid": hybrid_geo.volume_mm3,
                    "ground_truth": gt_geo.volume_mm3,
                    "unit": "mm³",
                },
                "volume_error_pct": {
                    "direct": vol_error_direct,
                    "hybrid": vol_error_hybrid,
                    "unit": "%",
                },
                "is_watertight": {
                    "direct": direct_geo.is_watertight,
                    "hybrid": hybrid_geo.is_watertight,
                    "unit": "bool",
                },
                "origin_zmin_mm": {
                    "direct": direct_geo.origin_zmin,
                    "hybrid": hybrid_geo.origin_zmin,
                    "unit": "mm",
                },
                "interference_volume_mm3": {
                    "direct": direct_geo.interference_volume_mm3,
                    "hybrid": hybrid_geo.interference_volume_mm3,
                    "unit": "mm³",
                },
                "collision_verification_complete": {
                    "direct": direct_geo.collision_verification_complete,
                    "hybrid": hybrid_geo.collision_verification_complete,
                    "unit": "bool",
                },
            },
            "sessions": {
                "direct": asdict(direct_session),
                "hybrid": asdict(hybrid_session),
            },
            "geometry": {
                "direct": asdict(direct_geo),
                "hybrid": asdict(hybrid_geo),
                "ground_truth": asdict(gt_geo),
            },
            "summary": {
                "token_reduction_pct": token_reduction_pct,
                "speedup_factor": speedup_factor,
                "cached_ratio_direct_pct": cached_ratio_direct,
                "cached_ratio_hybrid_pct": cached_ratio_hybrid,
            },
        }

        return report_data

    @staticmethod
    def _mock_session(label: str) -> SessionMetrics:
        now_dt = datetime.datetime.now(UTC7)
        if "Direct" in label:
            return SessionMetrics(
                session_label=label,
                log_file="mock://direct_session.jsonl",
                uncached_input_tokens=18450,
                cached_input_tokens=4560000,
                output_tokens=6820,
                output_non_reasoning_tokens=4420,
                reasoning_output_tokens=2400,
                total_tokens=4585270,
                elapsed_seconds=145.2,
                tool_calls_count=18,
                tool_names=["bash", "write_file", "execute_code"],
                turn_count=12,
                start_timestamp_utc7=(now_dt - datetime.timedelta(seconds=145)).strftime("%Y-%m-%d %H:%M:%S UTC+7"),
                end_timestamp_utc7=now_dt.strftime("%Y-%m-%d %H:%M:%S UTC+7"),
            )
        else:
            return SessionMetrics(
                session_label=label,
                log_file="mock://hybrid_session.jsonl",
                uncached_input_tokens=2400,
                cached_input_tokens=32500,
                output_tokens=780,
                output_non_reasoning_tokens=430,
                reasoning_output_tokens=350,
                total_tokens=35680,
                elapsed_seconds=3.2,
                tool_calls_count=2,
                tool_names=["compile_batch_parts", "assemble_and_verify"],
                turn_count=2,
                start_timestamp_utc7=(now_dt - datetime.timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S UTC+7"),
                end_timestamp_utc7=now_dt.strftime("%Y-%m-%d %H:%M:%S UTC+7"),
            )

    @staticmethod
    def _mock_geometry(label: str) -> GeometryMetrics:
        parts_dir = PROJECT_ROOT / "workspace" / "parts"
        default_step = parts_dir / "slotted_base.step"
        if default_step.exists():
            return GeometryEvaluator.evaluate_cad_file(default_step, label)
        return GeometryMetrics(
            cad_label=label,
            cad_file="mock://part.step",
            exists=True,
            is_valid_solid=True,
            is_watertight=True,
            volume_mm3=132508.24,
            bbox_dimensions=[120.0, 80.0, 15.0],
            bbox_min=[-60.0, -40.0, 0.0],
            origin_zmin=0.0,
            interference_volume_mm3=0.0,
            has_collisions=False,
        )

    @staticmethod
    def report_rows(report_data: Dict[str, Any]) -> List[List[str]]:
        metrics = report_data["metrics"]
        rows = []
        for category, key, label in (
            ("Tokens", "uncached_cloud_tokens", "Cloud without cache"),
            ("Tokens", "uncached_input_tokens", "Uncached Cloud input"),
            ("Tokens", "cached_input_tokens", "Cached Cloud input"),
            ("Tokens", "output_tokens", "Cloud output"),
            ("Tokens", "output_non_reasoning_tokens", "Cloud output excluding reasoning"),
            ("Tokens", "reasoning_output_tokens", "Cloud reasoning output"),
            ("Tokens", "total_tokens", "Cloud including cache"),
            ("Performance", "elapsed_seconds", "Elapsed time"),
            ("Orchestration", "model_calls_with_usage", "Cloud calls with usage"),
            ("Orchestration", "tool_calls_count", "Tool calls"),
            ("Geometry", "volume_mm3", "Volume"),
            ("Geometry", "volume_error_pct", "Volume error"),
            ("Geometry", "is_watertight", "Watertight"),
            ("Geometry", "origin_zmin_mm", "Origin Z min"),
            ("Geometry", "interference_volume_mm3", "Measured interference volume"),
            ("Geometry", "collision_verification_complete", "Collision verification complete"),
        ):
            metric = metrics[key]
            note = ""
            if "diff_pct" in metric:
                note = f"{-metric['diff_pct']:+.2f}% Hybrid vs Direct"
            elif "speedup" in metric:
                note = f"{metric['speedup']:.2f}x Direct / Hybrid. {metric['note']}"
            elif key == "volume_error_pct":
                note = (
                    "Reference STEP" if report_data["geometry"]["ground_truth"]["cad_label"] == "Ground Truth"
                    else "Direct model as reference"
                )
            rows.append([category, label, str(metric["direct"]), str(metric["hybrid"]), metric["unit"], note])
        return rows

    @classmethod
    def export_csv(cls, report_data: Dict[str, Any], output_path: Union[str, Path]):
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Metric", "Direct", "Hybrid", "Unit", "Note"])
            writer.writerow(["Benchmark", "Data source", report_data["mode"], report_data["mode"], "", ""])
            writer.writerows(cls.report_rows(report_data))

    @classmethod
    def export_markdown(cls, report_data: Dict[str, Any], output_path: Union[str, Path]):
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Direct and Hybrid CAD benchmark",
            "",
            f"Data source is **{report_data['mode']}**. Generated {report_data['evaluation_time_utc7']}.",
            "",
            "| Category | Metric | Direct | Hybrid | Unit | Note |",
            "| --- | --- | ---: | ---: | --- | --- |",
            *("| " + " | ".join(row) + " |" for row in cls.report_rows(report_data)),
        ]
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def export_docx(cls, report_data: Dict[str, Any], output_path: Union[str, Path]):
        if not docx:
            sys.stderr.write("[!] python-docx is unavailable. DOCX report was skipped.\n")
            return False
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        doc = docx.Document()
        doc.add_heading("Direct and Hybrid CAD benchmark", level=0)
        doc.add_paragraph(f"Data source is {report_data['mode']}. Generated {report_data['evaluation_time_utc7']}.")
        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        for cell, title in zip(table.rows[0].cells, ("Category", "Metric", "Direct", "Hybrid", "Unit", "Note")):
            cell.text = title
        for row in cls.report_rows(report_data):
            for cell, value in zip(table.add_row().cells, row):
                cell.text = value
        doc.save(str(out_file))
        return True

    @classmethod
    def display_rich_table(cls, report_data: Dict[str, Any]):
        if not console:
            return
        table = Table(title=f"Direct and Hybrid CAD benchmark ({report_data['mode']})")
        for header in ("Category", "Metric", "Direct", "Hybrid", "Unit", "Note"):
            table.add_column(header)
        for row in cls.report_rows(report_data):
            table.add_row(*row)
        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="A/B Benchmark & Session Log Evaluation Engine")
    parser.add_argument("--direct-log", type=str, help="Path to Direct Codex session log (.jsonl)")
    parser.add_argument("--compiler-log", "--hybrid-log", dest="compiler_log", type=str, help="Path to Hybrid/Compiler session log (.jsonl)")
    parser.add_argument("--direct-cad", type=str, help="Path to Direct CAD file (.step/.stl)")
    parser.add_argument("--hybrid-cad", type=str, help="Path to Hybrid CAD file (.step/.stl)")
    parser.add_argument("--ground-truth", type=str, help="Path to Ground Truth CAD file (.step/.stl)")
    parser.add_argument("--direct-wall-seconds", type=float, help="Measured wall time from start to acceptance")
    parser.add_argument("--hybrid-wall-seconds", type=float, help="Measured wall time from start to acceptance")
    parser.add_argument("--output-csv", type=str, default="benchmark/ab_report.csv", help="Path to export CSV")
    parser.add_argument("--output-md", type=str, default="benchmark/ab_report.md", help="Path to export Markdown")
    parser.add_argument("--output-docx", type=str, default="benchmark/ab_report.docx", help="Path to export DOCX")
    parser.add_argument("--demo", action="store_true", help="Run self-contained demo evaluation using workspace parts")

    args = parser.parse_args()

    direct_log_p = Path(args.direct_log) if args.direct_log else None
    hybrid_log_p = Path(args.compiler_log) if args.compiler_log else None
    direct_cad_p = Path(args.direct_cad) if args.direct_cad else None
    hybrid_cad_p = Path(args.hybrid_cad) if args.hybrid_cad else None
    gt_cad_p = Path(args.ground_truth) if args.ground_truth else None

    # Default demo if requested or no args provided
    if args.demo:
        print("[+] Running benchmark evaluation in demonstration mode...")
        parts_dir = PROJECT_ROOT / "workspace" / "parts"
        if (parts_dir / "slotted_base.step").exists():
            hybrid_cad_p = parts_dir / "slotted_base.step"
            gt_cad_p = parts_dir / "slotted_base.step"

    evaluator = BenchmarkEvaluator(
        direct_log=direct_log_p,
        hybrid_log=hybrid_log_p,
        direct_cad=direct_cad_p,
        hybrid_cad=hybrid_cad_p,
        ground_truth_cad=gt_cad_p,
        demo=args.demo,
        direct_wall_seconds=args.direct_wall_seconds,
        hybrid_wall_seconds=args.hybrid_wall_seconds,
    )

    try:
        results = evaluator.run_evaluation()
    except ValueError as exc:
        parser.error(str(exc))

    # Console display
    BenchmarkEvaluator.display_rich_table(results)

    # Export CSV, MD, and DOCX
    csv_path = PROJECT_ROOT / args.output_csv
    md_path = PROJECT_ROOT / args.output_md
    docx_path = PROJECT_ROOT / args.output_docx
    BenchmarkEvaluator.export_csv(results, csv_path)
    BenchmarkEvaluator.export_markdown(results, md_path)
    docx_exported = BenchmarkEvaluator.export_docx(results, docx_path)

    print(f"\n[OK] Benchmark Report exported successfully:")
    print(f"     - CSV Report: {csv_path}")
    print(f"     - Markdown Report: {md_path}")
    if docx_exported:
        print(f"     - DOCX Report: {docx_path}")


if __name__ == "__main__":
    main()
