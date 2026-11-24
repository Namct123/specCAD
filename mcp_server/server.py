"""
MCP (Model Context Protocol) Server for Hybrid CAD Automation - Performance-First
Exposes strictly 3 high-performance tools:
1. `compile_batch_parts`: Deterministic SpecCompiler (< 500ms, 0 hallucinations, pure CPU).
2. `assemble_and_verify`: cq.Assembly execution, pair-wise collision detection & STEP+GLB export.
3. `inspect_cad_file`: Fast CAD dimensional metadata inspection.

STRICT STDOUT HYGIENE: All logging is routed to sys.stderr and workspace/mcp_server.log.
sys.stdout is reserved 100% for MCP JSON-RPC protocol.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import logging
import math
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.cad_runner import CADExecutionResult, HeadlessCADRunner
from mcp_server.compact_plan import expand_linear
from mcp_server.spec_compiler import SpecCompiler

try:
    import cadquery as cq
except ImportError:
    cq = None

# Configure logging strictly to file and sys.stderr (NEVER stdout)
LOG_DIR = Path(__file__).resolve().parent.parent / "workspace"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mcp_server.log"

logger = logging.getLogger("hybrid-cad-agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)


DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


class ConeDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    bottom_radius: float = Field(gt=0)
    top_radius: float = Field(ge=0)
    height: float = Field(gt=0)


class RawConeDimensions(ConeDimensions):
    bottom_radius: float | None = Field(gt=0)
    top_radius: float | None = Field(ge=0)
    height: float | None = Field(gt=0)


_BOM_NUMBER = r"[0-9]+(?:[.,][0-9]+)?"
_CONE_BOM_PATTERNS = (
    re.compile(
        rf"nón (?:cụt|tròn nhọn)\s*[,;]?\s*"
        rf"(?P<bottom_kind>đường kính|bán kính) đáy\s+(?P<bottom>{_BOM_NUMBER})\s*mm\s*[,;]\s*"
        rf"(?P<top_kind>đường kính|bán kính) đỉnh\s+(?P<top>{_BOM_NUMBER})\s*mm\s*[,;]\s*"
        rf"cao\s+(?P<height>{_BOM_NUMBER})\s*mm\.?"
    ),
    re.compile(
        rf"circular (?:frustum|cone)\s*[,;]?\s*"
        rf"bottom (?P<bottom_kind>diameter|radius)\s+(?P<bottom>{_BOM_NUMBER})\s*mm\s*[,;]\s*"
        rf"top (?P<top_kind>diameter|radius)\s+(?P<top>{_BOM_NUMBER})\s*mm\s*[,;]\s*"
        rf"height\s+(?P<height>{_BOM_NUMBER})\s*mm\.?"
    ),
)


def parse_cone_bom(text: str) -> dict[str, float] | None:
    """Parse a complete, explicitly labeled millimeter BOM line."""
    normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    for pattern in _CONE_BOM_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match is None:
            continue
        dimensions = {key: float(match[key].replace(",", ".")) for key in ("bottom", "top", "height")}
        for end in ("bottom", "top"):
            if match[f"{end}_kind"] in ("đường kính", "diameter"):
                dimensions[end] /= 2
        return {
            "bottom_radius": dimensions["bottom"],
            "top_radius": dimensions["top"],
            "height": dimensions["height"],
        }
    return None


class CircularPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    part: str
    count: int = Field(ge=1, le=200)
    pcd: float = Field(gt=0)
    center: list[float] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    start_angle: float = 0.0


def expand_placements(plan: dict) -> dict:
    explicit = plan.get("placements", {})
    patterns = plan.get("circular_patterns", {})
    if not isinstance(explicit, dict) or not isinstance(patterns, dict):
        raise ValueError("placements and circular_patterns must be objects")
    placements = explicit.copy()
    for prefix, item in patterns.items():
        if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            raise ValueError("Invalid circular pattern name")
        try:
            pattern = CircularPlacement.model_validate(item)
        except ValueError as exc:
            raise ValueError(f"{prefix}: expected part, count (1–200), positive pcd, optional center and start_angle") from exc
        x, y, z = pattern.center
        start = math.radians(pattern.start_angle % 360)
        radius = pattern.pcd / 2
        for index in range(pattern.count):
            name = f"{prefix}_{index + 1:03d}"
            if name in placements:
                raise ValueError(f"Duplicate placement name: {name}")
            angle = start + math.tau * index / pattern.count
            translation = [x + radius * math.cos(angle), y + radius * math.sin(angle), z]
            if not all(map(math.isfinite, translation)):
                raise ValueError(f"{prefix}: placement exceeds coordinate range")
            placements[name] = {"part": pattern.part, "translation": translation}
    return placements


class CADService:
    """Core CAD Business Logic accessible both via MCP tools and programmatic Python API."""

    def __init__(self, workspace_root: Optional[Union[str, Path]] = None):
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = (Path(__file__).parent.parent / "workspace").resolve()

        self.specs_dir = self.workspace_root / "specs"
        self.parts_dir = self.workspace_root / "parts"
        self.assembly_dir = self.workspace_root / "assembly"

        for p in (self.specs_dir, self.parts_dir, self.assembly_dir):
            p.mkdir(parents=True, exist_ok=True)

        self.runner = HeadlessCADRunner(default_output_dir=self.parts_dir)

    @staticmethod
    def extract_python_code(text: str) -> str:
        """Extracts clean python code from markdown or raw string."""
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()
        return text.strip()

    @staticmethod
    def compile_compact_spec_to_cq(spec: Dict[str, Any]) -> str:
        """Alias for backward compatibility with audit fixtures."""
        return SpecCompiler.compile_spec_to_code(spec)

    @staticmethod
    def validate_part_spec_compliance(spec: Dict[str, Any], result_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates bounding dimensions and origin against the specification."""
        errors: List[str] = []
        metrics = result_dict.get("metrics")
        if not metrics:
            return False, ["Missing metrics in execution result"]

        dims = metrics.get("bbox_dimensions", [0.0, 0.0, 0.0])
        bbox_min = metrics.get("bbox_min", [0.0, 0.0, 0.0])

        if spec.get("origin") == "center_bottom":
            if abs(bbox_min[2]) >= 0.05:
                errors.append(f"Origin Z mismatch: expected center_bottom (zmin ~ 0.0), got {bbox_min[2]:.3f} mm")

        base = spec.get("base", {})
        shape = base.get("shape", base.get("kind"))
        if shape == "box" and "dims" in base:
            exp_dx, exp_dy, exp_dz = base["dims"]
            if abs(dims[0] - exp_dx) > 0.1:
                errors.append(f"Dimension X mismatch: expected {exp_dx} mm, got {dims[0]:.2f} mm")
            if abs(dims[1] - exp_dy) > 0.1:
                errors.append(f"Dimension Y mismatch: expected {exp_dy} mm, got {dims[1]:.2f} mm")
            if abs(dims[2] - exp_dz) > 0.1:
                errors.append(f"Dimension Z mismatch: expected {exp_dz} mm, got {dims[2]:.2f} mm")
        elif shape in ("cylinder", "tube", "pipe") and ("d" in base or "od" in base):
            exp_d = base.get("d") or base.get("od", 0.0)
            exp_h = base.get("h", 0.0)
            if abs(dims[0] - exp_d) > 0.1 or abs(dims[1] - exp_d) > 0.1:
                errors.append(f"Dimension diameter mismatch: expected {exp_d} mm, got {dims[0]:.2f}x{dims[1]:.2f} mm")
            if abs(dims[2] - exp_h) > 0.1:
                errors.append(f"Dimension Z mismatch: expected {exp_h} mm, got {dims[2]:.2f} mm")

        return len(errors) == 0, errors

    def compile_batch_parts(self, parts_specs: List[Dict[str, Any]], output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Compiles batch mechanical parts deterministically in < 500ms using SpecCompiler."""
        default = Path("workspace/parts")
        target = output_dir or (default if default.resolve() == self.parts_dir else self.parts_dir)
        return SpecCompiler.compile_batch(parts_specs=parts_specs, output_dir=target)

    def assemble_and_verify(
        self,
        assembly_script: Optional[str] = None,
        export_name: Optional[str] = None,
        assembly_code: Optional[str] = None,
        assembly_name: Optional[str] = None,
        check_collisions: bool = True,
        output_dir: Optional[str] = None,
        expected_volume: Optional[float] = None,
        volume_tolerance_pct: float = 0.1,
        full_brep: bool = False,
    ) -> Dict[str, Any]:
        """Executes a CadQuery Assembly script, verifies pair-wise collisions,
        validates volume against expected volume with relative 0.1% tolerance (max(1.0 mm³, 0.001*V)),
        and exports STEP, STL, and GLB models.
        """
        raw_code = assembly_script if assembly_script is not None else (assembly_code or "")
        name = export_name if export_name is not None else (assembly_name or "assembly")
        clean_code = self.extract_python_code(raw_code)
        target_path = Path(output_dir) if output_dir else self.assembly_dir
        res: CADExecutionResult = self.runner.execute_assembly_code(
            code=clean_code,
            assembly_name=name,
            output_dir=target_path,
            check_collisions=check_collisions,
            full_brep=full_brep,
            assembly_volume=(expected_volume, volume_tolerance_pct) if expected_volume is not None else None,
        )

        if not res.success:
            result = {
                "status": "FAILED",
                "assembly_name": name,
                "error": res.error_message or "Assembly execution failed",
                "diagnostics": res.diagnostics,
            }
            if "volume_check" in res.artifacts:
                result["volume_check"] = json.loads(res.artifacts["volume_check"])
            return result

        return json.loads(res.artifacts["structured_metadata"])

    def generate_freeform_local(
        self,
        part_spec: Dict[str, Any],
        max_retries: int = 3,
        model_name: str = "cad-worker-sft",
        output_dir: Optional[Path] = None,
        allow_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Tier 2 (Freeform Worker): Generates complex 3D freeform shapes (Loft, Sweep, Shell)
        via Local Fine-tuned LLM on Ollama with self-healing CADDiagnosticAnalyzer loop.
        """
        import urllib.error
        import urllib.request
        start_time = time.perf_counter()
        part_name = part_spec.get("name", "freeform_part")
        prompt_text = part_spec.get("raw_text") or part_spec.get("prompt") or json.dumps(part_spec)
        target_dir = output_dir if output_dir is not None else self.parts_dir
        geometry_contract = {
            key: part_spec[key]
            for key in ("expected_volume", "expected_bbox_mm", "expected_void_boxes")
            if key in part_spec
        }

        system_prompt = (
            "You are an expert CAD automation engineer and precision CadQuery programmer. "
            "Write clean, robust, and watertight CadQuery Python code satisfying the mechanical specification. "
            "Always assign the final valid 3D solid to `result`. "
            "When origin is center_bottom, ensure the base rests at Z=0 (zmin == 0.0)."
        )
        cone_recipe = (
            not allow_fallback and part_spec.get("geometry_kind") == "cone"
            and not part_spec.get("features") and part_spec.get("origin", "center_bottom") == "center_bottom"
        )
        raw_cone = cone_recipe and bool(part_spec.get("raw_text"))
        if cone_recipe:
            system_prompt = (
                "Extract exact cone dimensions in mm. Return only JSON with bottom_radius, top_radius and height. "
                "Halve any stated diameters to get radii. A pointed top has radius zero."
            )
            if raw_cone:
                system_prompt += " Return null for a dimension if it is missing or ambiguous. Do not invent dimensions."
        elif part_spec.get("geometry_kind") == "cone":
            system_prompt += " Use cq.Solid.makeCone(base_radius, top_radius, height) for a straight circular taper."

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        target_model = model_name or "cad-worker-sft"
        if cone_recipe and not part_spec.get("model") and target_model == "cad-worker-sft":
            target_model = os.getenv("CAD_OLLAMA_RECIPE_MODEL") or (
                "qwen2.5:1.5b" if raw_cone else target_model
            )
        last_error = ""
        options = {"temperature": 0.1, "num_ctx": 2048}
        if not allow_fallback:
            options["num_predict"] = 768
        if cone_recipe:
            options.update(temperature=0, num_predict=96, num_ctx=1024, num_batch=64)
        gpu_layers = os.getenv("CAD_OLLAMA_NUM_GPU")
        if gpu_layers is not None:
            options["num_gpu"] = int(gpu_layers)
        recipe_schema = RawConeDimensions if raw_cone else ConeDimensions
        schema = {"format": recipe_schema.model_json_schema()} if cone_recipe else {}

        for attempt in range(1, max_retries + 1):
            py_code = None
            raw_msg = ""
            try:
                req_payload = json.dumps({
                    "model": target_model,
                    "messages": history,
                    "stream": False,
                    "options": options,
                    "keep_alive": "1m" if raw_cone else "5m",
                    **schema,
                }).encode("utf-8")
                req = urllib.request.Request(
                    OLLAMA_CHAT_URL,
                    data=req_payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    raw_msg = res_json.get("message", {}).get("content", "")
                    logger.info(
                        "Ollama %s attempt=%d prompt_tokens=%s output_tokens=%s duration_ns=%s load_ns=%s",
                        part_name, attempt, res_json.get("prompt_eval_count"),
                        res_json.get("eval_count"), res_json.get("total_duration"), res_json.get("load_duration"),
                    )
                    if res_json.get("done_reason") == "length":
                        last_error = "Model reached its token limit"
                        raw_msg = ""
                    elif not cone_recipe:
                        py_code = self.extract_python_code(raw_msg)
            except urllib.error.HTTPError as e:
                detail = e.read(1024).decode("utf-8", errors="replace")
                last_error = f"HTTP {e.code}: {' '.join(detail.split())[:300]}"
            except Exception as e:
                last_error = str(e)

            if cone_recipe and raw_msg:
                try:
                    dimensions = ConeDimensions.model_validate_json(raw_msg)
                    compiled = SpecCompiler.compile_part(
                        {**part_spec, "base": {"shape": "cone", **dimensions.model_dump()}},
                        output_dir=target_dir,
                    )
                    if compiled["status"] == "SUCCESS":
                        return {**compiled, "source": "local_llm", "attempts": attempt,
                                "elapsed_sec": round(time.perf_counter() - start_time, 3)}
                    last_error = compiled["error"]
                except (ValueError, TypeError, IndexError) as exc:
                    last_error = str(exc)
                history.append({"role": "assistant", "content": raw_msg})
                history.append({"role": "user", "content": f"Check the original dimensions. {last_error[:200]}"})
                continue

            if not py_code:
                if allow_fallback and "base" in part_spec:
                    res = SpecCompiler.compile_part(part_spec, output_dir=target_dir)
                    if res.get("status") == "SUCCESS":
                        res["source"] = "compiler_fallback"
                        return res
                return {
                    "part_name": part_name,
                    "status": "FAILED",
                    "source": "local_llm",
                    "error": f"No usable response from Ollama model {target_model}: {last_error}",
                    "attempts": attempt,
                    "failure_code": ("sft_memory_pressure" if "failed to allocate" in last_error.lower()
                                     else "sft_unavailable"),
                }

            # Execute and validate
            exec_res = self.runner.execute_part_code(
                code=py_code,
                part_name=part_name,
                output_dir=target_dir,
                export_formats=("step",),
                geometry_contract=geometry_contract,
                expected_origin=part_spec.get("origin", "center_bottom"),
                expected_z_range=part_spec.get("expected_z_range"),
                z_tolerance=float(part_spec.get("z_tolerance", 0.05)),
            )

            if exec_res.success and exec_res.metrics and exec_res.metrics.is_valid:
                elapsed_sec = round(time.perf_counter() - start_time, 3)
                return {
                    "part_name": part_name,
                    "status": "SUCCESS",
                    "source": "local_llm",
                    "attempts": attempt,
                    "step_path": exec_res.artifacts.get("step", f"workspace/parts/{part_name}.step"),
                    "bbox_mm": list(exec_res.metrics.bbox_dimensions),
                    "volume_mm3": round(exec_res.metrics.volume, 2),
                    "elapsed_sec": elapsed_sec,
                }

            # Self-correction feedback loop
            diag = "\n".join(exec_res.diagnostics) if exec_res.diagnostics else exec_res.error_message
            suggestions = []
            if "shell" in str(exec_res.error_message).lower() or "loft" in str(exec_res.error_message).lower():
                suggestions.append("Rule: Use Core Subtraction instead of .shell() on lofts: loft outer solid, loft inner solid with offset profile, then subtract result = outer.cut(inner).")
            if "origin" in str(exec_res.error_message).lower() or "zmin" in str(exec_res.error_message).lower():
                suggestions.append("Rule: When origin is center_bottom, ensure base rests at Z=0 (zmin == 0.0) or use .box(..., centered=(True, True, False)).")

            suggestion_text = ("\n" + "\n".join(suggestions)) if suggestions else ""
            history.append({"role": "assistant", "content": f"```python\n{py_code}\n```"})
            history.append({
                "role": "user",
                "content": f"Execution failed: {exec_res.error_message}\nDiagnostics: {diag}{suggestion_text}\nFix the CadQuery code.",
            })

        elapsed_sec = round(time.perf_counter() - start_time, 3)
        return {
            "part_name": part_name,
            "status": "FAILED",
            "source": "local_llm",
            "attempts": max_retries,
            "elapsed_sec": elapsed_sec,
            "error": last_error if cone_recipe else (exec_res.error_message if 'exec_res' in locals() else "Retries exhausted"),
            "failure_code": "geometry_rejected" if (
                cone_recipe or ('exec_res' in locals() and exec_res.error_type == "GeometryContractError")
            ) else "sft_failed",
        }

    def run_hybrid_batch(
        self,
        parts_specs: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        max_sft_retries: int = 1,
    ) -> Dict[str, Any]:
        """Unified Performance-First Hybrid CAD Pipeline:
        Executes a pre-classified batch of parts across 3 tiers:
        - compiler_cpu: Deterministic SpecCompiler (< 50ms/part, pure CPU, 0 hallucinations).
        - local_sft: cad-worker-sft (Ollama) with at most 1 retry (total 2 attempts).
        - cloud_direct: Skip-Local route for complex geometries (impellers, freeform), escalates directly.
        Returns locked JSON summary with passed .step paths and concise escalation reasons.
        """
        batch_start = time.perf_counter()
        target_dir = Path(output_dir) if output_dir else self.parts_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        passed_parts: List[Dict[str, Any]] = []
        escalated_parts: List[Dict[str, Any]] = []

        for spec in parts_specs:
            part_name = spec.get("name", "unnamed_part")
            tier = str(spec.get("tier") or spec.get("route") or spec.get("mode") or "").strip().lower()
            base = spec.get("base") or {}
            shape = base.get("shape", base.get("kind"))
            kind = spec.get("geometry_kind")
            complex_geometry = any(
                term in part_name.lower()
                for term in ("impeller", "blade", "runner", "bellmouth", "combustion", "liner", "manifold", "nozzle")
            )
            local_kind = kind in ("cone", "single_chamfer", "shelled_enclosure", "single_rib")
            compiler_tiers = ("compiler_cpu", "compiler", "cpu", "parametric")
            local_tiers = ("local_sft", "sft", "local", "freeform")
            cloud_tiers = ("cloud_direct", "cloud", "cloud_escalate", "skip_local")

            if tier in cloud_tiers:
                tier = "cloud_direct"
            elif (not tier or tier in local_tiers) and shape == "cone" and all(
                key in base for key in ("bottom_radius", "top_radius", "height")
            ):
                tier = "compiler_cpu"
            elif shape in ("tube", "pipe", "hollow_cylinder") and (not complex_geometry or tier in compiler_tiers):
                tier = "compiler_cpu"
            elif tier in compiler_tiers or (not tier and base and not complex_geometry):
                tier = "compiler_cpu"
            elif tier in local_tiers or (not tier and local_kind and not complex_geometry):
                tier = "local_sft"
            elif not tier:
                tier = "cloud_direct"
            else:
                escalated_parts.append({"name": part_name, "tier": "unknown", "reason": "unsupported_tier"})
                continue

            if tier == "cloud_direct":
                escalated_parts.append({"name": part_name, "tier": tier, "reason": "skip_local"})
                continue
            if tier == "local_sft":
                if complex_geometry or not local_kind:
                    escalated_parts.append({"name": part_name, "tier": "cloud_direct", "reason": "outside_sft_scope"})
                    continue
                raw_text = spec.get("raw_text")
                raw_cone = (
                    isinstance(raw_text, str) and bool(raw_text.strip()) and kind == "cone"
                    and not spec.get("features") and spec.get("origin", "center_bottom") == "center_bottom"
                )
                if (not raw_cone and (raw_text is not None or not spec.get("prompt")
                        or spec.get("expected_volume") is None
                        or (kind == "shelled_enclosure" and not spec.get("expected_void_boxes")))):
                    escalated_parts.append({"name": part_name, "tier": tier, "reason": "missing_geometry_contract"})
                    continue
                dimensions = parse_cone_bom(raw_text) if raw_cone and not spec.get("model") else None
                if dimensions is not None:
                    base = {"shape": "cone", **dimensions}
                    spec = {**spec, "base": base}
                    tier = "compiler_cpu"

            if tier == "local_sft":
                res = self.generate_freeform_local(
                    part_spec=spec,
                    max_retries=1 + min(1, max(0, max_sft_retries)),
                    model_name=spec.get("model", "cad-worker-sft"),
                    output_dir=target_dir,
                    allow_fallback=False,
                )
            else:
                if not base:
                    escalated_parts.append({"name": part_name, "tier": tier, "reason": "missing_base"})
                    continue
                res = SpecCompiler.compile_part(spec, output_dir=target_dir, export_stl=False)

            if res.get("status") == "SUCCESS":
                passed_parts.append({
                    "name": part_name,
                    "tier": tier,
                    "step_path": res["step_path"],
                    "volume_mm3": res["volume_mm3"],
                })
            else:
                logger.warning("Hybrid batch %s (%s): %s", part_name, tier, res.get("error"))
                code = "compiler_failed" if tier == "compiler_cpu" else res.get("failure_code", "sft_failed")
                escalated_parts.append({"name": part_name, "tier": tier, "reason": code})

        total_elapsed = round(time.perf_counter() - batch_start, 3)
        total_count = len(parts_specs)
        passed_count = len(passed_parts)
        escalated_count = len(escalated_parts)

        if escalated_count == 0:
            status = "SUCCESS"
        elif passed_count > 0:
            status = "PARTIAL_SUCCESS"
        else:
            status = "FAILED"

        return {
            "status": status,
            "summary": {
                "total_parts": total_count,
                "passed_count": passed_count,
                "escalated_count": escalated_count,
                "elapsed_sec": total_elapsed,
            },
            "passed_parts": passed_parts,
            "escalated_parts": escalated_parts,
        }

    def run_assembly_plan(self, parts_specs, plan, output_dir=None, max_sft_retries=1, full_brep=False):
        """Run one batch and one assembly from a complete plan."""
        started = time.perf_counter()
        target = Path(output_dir or self.parts_dir).resolve()
        try:
            parts_specs, plan = expand_linear(parts_specs, plan)
        except (TypeError, ValueError, OverflowError) as exc:
            return {"status": "FAILED", "phase": "plan", "error": str(exc)}
        names = [spec.get("name") for spec in parts_specs]
        export_name = plan.get("export_name")
        if (not names
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                             for name in [*names, export_name])
                or len(set(names)) != len(names)):
            return {"status": "FAILED", "phase": "plan", "error": "Use unique part names and a valid export_name"}
        try:
            placements = expand_placements(plan)
        except ValueError as exc:
            return {"status": "FAILED", "phase": "plan", "error": str(exc)}
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
               or not isinstance(item, dict) or not isinstance(item.get("part", name), str)
               for name, item in placements.items()):
            return {"status": "FAILED", "phase": "plan", "error": "Each placement needs a valid part reference"}
        if {item.get("part", name) for name, item in placements.items()} != set(names):
            return {"status": "FAILED", "phase": "plan", "error": "Use a placement for every part and only reference defined parts"}

        batch = self.run_hybrid_batch(parts_specs, str(target), max_sft_retries)
        paths = {part["name"]: part["step_path"] for part in batch["passed_parts"]}
        unresolved = [part for part in batch["escalated_parts"]
                      if not plan.get("cloud_parts_script") or (full_brep and part["reason"] != "skip_local")]
        if unresolved:
            return {"status": "ESCALATE", "phase": "parts", **batch["summary"],
                    "escalated_parts": unresolved}

        worker_plan = {key: value for key, value in plan.items() if key not in {"circular_patterns", "linear_series"}}
        worker_plan["placements"] = placements
        root = Path(__file__).resolve().parent.parent
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(root)!r})\n"
            "from mcp_server.assembly_plan import build_planned_assembly\n"
            f"result = build_planned_assembly({worker_plan!r}, {parts_specs!r}, {paths!r}, {str(target)!r})\n"
        )
        assembly = self.assemble_and_verify(
            assembly_script=code, export_name=export_name,
            output_dir=str(target / "assembly"),
            expected_volume=plan.get("expected_volume"),
            full_brep=full_brep,
        )
        result = {key: assembly[key] for key in (
            "status", "error", "outputs", "solid_count", "volume_check", "collision_verification_complete",
        ) if key in assembly}
        if "interferences" in assembly:
            report = assembly["interferences"]
            result["interferences"] = {key: report[key] for key in (
                "pairs_total", "pairs_aabb_pruned", "pairs_boolean_checked",
                "total_interference_volume_mm3", "collision_pairs", "unverified_pairs",
            )}
        result["batch"] = batch["summary"]
        result["instance_count"] = len(placements)
        if assembly["status"] != "SUCCESS":
            result["phase"] = "assembly"
        if "elapsed_sec" in assembly:
            result["assembly_elapsed_sec"] = assembly["elapsed_sec"]
        result["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return result

    def inspect_cad_file(self, file_path: str) -> Dict[str, Any]:
        """Inspects STEP or STL file for dimensions, origin, volume, and watertightness."""
        path = Path(file_path).resolve()
        if not path.exists():
            return {"error": f"File not found: {str(path)}"}

        ext = path.suffix.lower()
        if ext in (".step", ".stp") and cq:
            try:
                imported = cq.importers.importStep(str(path))
                metrics = HeadlessCADRunner.compute_metrics(imported)
                return {
                    "file": path.name,
                    "format": "STEP",
                    "bbox_mm": list(metrics.bbox_dimensions),
                    "bbox_min": list(metrics.bbox_min),
                    "volume_mm3": round(metrics.volume, 2),
                    "is_watertight": metrics.is_watertight,
                }
            except Exception as e:
                return {"error": f"Failed to inspect STEP: {str(e)}"}

        elif ext == ".stl":
            try:
                import trimesh

                mesh = trimesh.load(str(path), file_type="stl")
                return {
                    "file": path.name,
                    "format": "STL",
                    "is_watertight": bool(mesh.is_watertight),
                    "volume_mm3": round(float(mesh.volume), 2) if mesh.is_watertight else 0.0,
                    "bbox_min": mesh.bounds[0].tolist(),
                    "bounds": mesh.bounds.tolist(),
                }
            except Exception as e:
                return {"error": f"Failed to inspect STL: {str(e)}"}

        return {"error": f"Unsupported format or missing engine for {ext}"}

    def list_workspace_artifacts(self) -> Dict[str, Any]:
        """Lists all files in specs, parts, and assembly folders."""
        def scan_folder(folder: Path) -> List[Dict[str, Any]]:
            items = []
            if folder.exists():
                for f in sorted(folder.glob("*")):
                    if f.is_file():
                        items.append({
                            "name": f.name,
                            "path": str(f.resolve()),
                            "size_bytes": f.stat().st_size,
                            "modified": f.stat().st_mtime,
                        })
            return items

        return {
            "workspace_root": str(self.workspace_root),
            "specs": scan_folder(self.specs_dir),
            "parts": scan_folder(self.parts_dir),
            "assembly": scan_folder(self.assembly_dir),
        }


# ==============================================================================
# Model Context Protocol (MCP) Server Setup (Performance-First Streamlined 3 Tools)
# ==============================================================================

cad_service = CADService()

try:
    from mcp.server.mcpserver import MCPServer
    mcp_app = MCPServer("hybrid-cad-agent")

    # TOOL 1: Unified Performance-First Hybrid CAD Pipeline (Compiler CPU + Local SFT + Cloud Direct)
    @mcp_app.tool(structured_output=False)
    def run_hybrid_batch(
        parts_specs: list[dict],
        output_dir: str = "workspace/parts",
        max_sft_retries: int = 1,
        assembly_plan: dict | None = None,
    ) -> str:
        """Process a preclassified batch and return one compact JSON result.

        compiler_cpu takes numeric geometry, including tube (od, id, h) and cone
        (bottom_radius, top_radius, height). For a raw cone description from a user
        or BOM, local_sft accepts raw_text and geometry_kind=cone. Complete labeled
        millimeter BOM lines compile directly; other descriptions go to Ollama.
        Other local_sft requests require prompt, expected_volume (mm³), and geometry_kind: cone,
        single_chamfer, single_rib or shelled_enclosure. Enclosures also require
        expected_void_boxes: lists of [xmin, ymin, zmin, xmax, ymax, zmax].
        cloud_direct escalates immediately. Match escalation names to the input specs.
        With assembly_plan, also place, verify and export the assembly. Provide
        export_name, placements keyed by instance name (translation XYZ, optional
        rotation_axis XYZ, rotation_deg, color and expected_global_z), and
        expected_volume. Use part in a placement to reuse a named definition.
        For repeats on an XY bolt circle, use circular_patterns keyed by prefix:
        part, count (1–200), pcd, optional center XYZ and start_angle (degrees, CCW).
        A cloud_parts_script can produce a cloud_parts dict of
        CadQuery solids for pending_parts inside the assembly worker.
        """
        if assembly_plan is None:
            result = cad_service.run_hybrid_batch(parts_specs, output_dir, max_sft_retries)
        else:
            result = cad_service.run_assembly_plan(parts_specs, assembly_plan, output_dir, max_sft_retries)
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    # TOOL 2: Deterministic Batch Part Compilation (SpecCompiler, < 500ms, pure CPU)
    @mcp_app.tool()
    def compile_batch_parts(parts_specs: list[dict]) -> dict:
        """Compiles and validates a batch of CAD mechanical parts from Compact Spec JSON
        using the deterministic compiler (pure CPU, < 500ms, 0 hallucinations).
        Returns an ultra-compact summary table with dimensions and step file paths.
        """
        return cad_service.compile_batch_parts(parts_specs)

    # TOOL 2: Tier 2 Freeform Local CAD Worker (Ollama cad-worker-sft with retry loop)
    @mcp_app.tool()
    def generate_freeform_local(
        part_spec: dict,
        max_retries: int = 3,
        model_name: str = "cad-worker-sft",
    ) -> dict:
        """Generates complex freeform CAD parts (Loft, Sweep, Shell) using local fine-tuned LLM on Ollama
        with automated CADDiagnosticAnalyzer self-healing feedback loop.
        """
        return cad_service.generate_freeform_local(
            part_spec=part_spec,
            max_retries=max_retries,
            model_name=model_name,
        )

    # TOOL 4: Assembly & Pair-wise Collision Verification (STEP + GLB Export)
    @mcp_app.tool()
    def assemble_and_verify(
        assembly_script: str,
        export_name: str,
        check_collisions: bool = True,
        output_dir: str = "workspace/assembly",
        expected_volume: float = None,
        volume_tolerance_pct: float = 0.1,
    ) -> dict:
        """Export a CadQuery assembly as STEP and GLB with volume and collision results.

        Volume tolerance defaults to max(1 mm³, 0.1% of expected_volume).
        AABB pruning and exact BRep checks report detected collisions and unverified
        pairs. VERIFICATION_INCOMPLETE means part of the collision check was skipped.
        """
        return cad_service.assemble_and_verify(
            assembly_code=assembly_script,
            assembly_name=export_name,
            check_collisions=check_collisions,
            output_dir=output_dir,
            expected_volume=expected_volume,
            volume_tolerance_pct=volume_tolerance_pct,
        )

    # TOOL 4: Fast Dimensional Inspection
    @mcp_app.tool()
    def inspect_cad_file(file_path: str) -> dict:
        """Inspects STEP or STL file to compute volume, bounding box dimensions, and origin zmin."""
        return cad_service.inspect_cad_file(file_path)

except ImportError:
    mcp_app = None


def main():
    """Runs the MCP server over stdio or executes diagnostic check."""
    if "--test" in sys.argv:
        sys.stderr.write("[MCP Server Performance-First] Running in test mode.\n")
        sys.stderr.write(f"CAD Service Workspace: {cad_service.workspace_root}\n")
        sys.stderr.write("Registered MCP Tools: run_hybrid_batch, compile_batch_parts, assemble_and_verify, inspect_cad_file\n")
        sys.stderr.flush()
        return

    if mcp_app:
        mcp_app.run(transport="stdio")
    else:
        sys.stderr.write("[MCP Server] MCPServer class not found. Running in standalone mode.\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
