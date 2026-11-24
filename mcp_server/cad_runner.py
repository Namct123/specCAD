"""
CAD Runner & Headless Sandbox Engine for CadQuery
Provides safe headless execution, topological validation, metrics extraction,
STEP/STL export, and self-healing error diagnostics.
"""

from __future__ import annotations

import io
import json
import math
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cadquery as cq
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
except ImportError:
    cq = None  # Handled gracefully if cadquery is not yet loaded in active env

@dataclass
class CADMetrics:
    """Quantitative geometric metrics of an executed CAD solid."""
    volume: float = 0.0
    surface_area: float = 0.0
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_of_mass: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    face_count: int = 0
    edge_count: int = 0
    vertex_count: int = 0
    solid_count: int = 0
    is_valid: bool = False
    is_watertight: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CADExecutionResult:
    """Standardized output from a headless CadQuery execution."""
    success: bool
    part_name: str
    execution_time_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    diagnostics: List[str] = field(default_factory=list)
    metrics: Optional[CADMetrics] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.metrics:
            data["metrics"] = self.metrics.to_dict()
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CADDiagnosticAnalyzer:
    """Diagnoses CadQuery and OpenCASCADE geometry construction failures and

    recommends concrete fixes for LLM Self-Healing.
    """

    ERROR_PATTERNS = [
        (
            r"(Standard_ConstructionError|radius.*too large|fillet.*failed|BRep_API: command not done)",
            "FILLET_CHAMFER_OVERFLOW",
            "Fillet or chamfer radius exceeds edge boundary length or adjacent face topology.",
            [
                "Reduce the fillet/chamfer radius by 50% or check adjacent geometry length.",
                "Ensure edges being filleted do not intersect tight corners or holes.",
                "Inspect the edge selector (e.g. use .edges('|Z') instead of .edges()).",
            ],
        ),
        (
            r"(IndexError: list index out of range|find.*no faces|empty selection)",
            "SELECTOR_EMPTY_RESULT",
            "CadQuery selector query returned 0 elements, causing an index error.",
            [
                "Check face/edge selector string (e.g. '>Z', '<X', '|Z', '%CIRCLE').",
                "Ensure prior boolean/cut operation did not remove or split the targeted face.",
                "Tag the workplane or feature earlier using .tag('feature_name') and restore via .workplaneFromTagged('feature_name').",
            ],
        ),
        (
            r"(Wires must be closed|BRepLib_MakeWire|Wire is not closed)",
            "UNCLOSED_WIRE_PROFILE",
            "Profile wire is open or discontinuous, preventing extrusion or revolution.",
            [
                "Call .close() at the end of polyline/spline chains.",
                "Verify the start coordinate matches the ending coordinate exactly.",
                "Check for self-intersecting segments in line/spline arrays.",
            ],
        ),
        (
            r"(StdFail_NotDone|Boolean operation failed|BRepAlgoAPI_Cut|BRepAlgoAPI_Fuse)",
            "BOOLEAN_COPLANAR_SINGULARITY",
            "Boolean Cut or Union failed, typically due to exactly co-planar touching surfaces.",
            [
                "Add an infinitesimal overlap (0.01mm - 0.05mm) to the cutting tool depth.",
                "Ensure cutting tool extends completely through the target body (e.g. clean through-holes).",
                "Verify both bodies are valid watertight solids before boolean combination.",
            ],
        ),
        (
            r"(AttributeError:.*Workplane.*object has no attribute '(?P<attr>.*)')",
            "INVALID_CADQUERY_API",
            "Called non-existent method on Workplane.",
            [
                "Check CadQuery API documentation: use .box(x, y, z) for solids, .rect(x, y) on 2D workplanes.",
                "Use .circle(radius).extrude(depth) or .hole(diameter, depth) for cylindrical cuts.",
                "Ensure method is called on Workplane rather than Shape/Solid object.",
            ],
        ),
        (
            r"(TypeError: Workplane\.cylinder\(\) got an unexpected keyword argument|'list' object is not callable)",
            "CYLINDER_SELECTOR_API_MISMATCH",
            "Workplane.cylinder does not take keyword 'd', or string selector was passed incorrectly.",
            [
                "Use cq.Workplane('XY').circle(d / 2.0).extrude(h) for cylinder primitives instead of .cylinder().",
                "Ensure face selectors are string literals like .faces('>Z') rather than passing objects into .faces().",
            ],
        ),
        (
            r"(ORIGIN_ALIGNMENT_MISMATCH|Origin mismatch|abs\(zmin\)|zmin.*deviation)",
            "ORIGIN_ALIGNMENT_MISMATCH",
            "Origin specification requires center_bottom (abs(zmin) < 0.05), but the geometry is centered or shifted along Z.",
            [
                "For box primitives, ALWAYS use: result = cq.Workplane('XY').box(x, y, z, centered=(True, True, False))",
                "Alternatively, translate the final shape: result = result.translate((0, 0, z / 2.0))",
                "Ensure zmin == 0 so that the part bottom rests exactly on the XY plane (Z=0).",
            ],
        ),
        (
            r"(AssertionError.*volume|AssertionError.*Volume|assert.*abs\(.*volume|assert.*abs\(.*Volume)",
            "VOLUME_ASSERTION_TOLERANCE_MISMATCH",
            "Volume assertion failed. Absolute tolerance (e.g. 0.15 mm³) is excessively strict and fails falsely due to STEP polynomial tessellation/rounding.",
            [
                "Use relative tolerance of 0.1% or max(1.0 mm³, 0.001 * expected_volume):",
                "  vol_diff = abs(actual_volume - expected_volume)",
                "  assert vol_diff <= max(1.0, 0.001 * expected_volume), f'Volume deviation {vol_diff:.2f} mm3 exceeds 0.1%'",
            ],
        ),
    ]

    @classmethod
    def analyze(cls, error_msg: str, tb_str: str, code: str) -> List[str]:
        diagnostics = []
        combined_text = f"{error_msg}\n{tb_str}"

        matched = False
        for pattern, code_key, explanation, suggestions in cls.ERROR_PATTERNS:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                matched = True
                diagnostics.append(f"[{code_key}] {explanation}")
                for s in suggestions:
                    diagnostics.append(f"  -> Suggestion: {s}")

        if not matched:
            diagnostics.append(
                "[RUNTIME_GEOMETRY_ERROR] Unclassified CadQuery execution error."
            )
            diagnostics.append(
                "  -> Suggestion: Validate coordinate frames, ensure valid 2D sketches before 3D extrusions/revolutions."
            )

        return diagnostics


def check_volume(actual: float, expected: float, tolerance_pct: float = 0.1) -> Dict[str, Any]:
    """Compare BRep volumes in mm³ using the shared relative tolerance."""
    expected = float(expected)
    tolerance_pct = float(tolerance_pct)
    if not math.isfinite(expected) or expected <= 0:
        raise ValueError("expected_volume must be finite and positive")
    if not math.isfinite(tolerance_pct) or tolerance_pct < 0:
        raise ValueError("volume_tolerance_pct must be finite and non-negative")
    diff = abs(actual - expected)
    allowed = max(1.0, tolerance_pct / 100.0 * expected)
    return {
        "passed": math.isfinite(actual) and actual > 0 and diff <= allowed,
        "expected_volume_mm3": round(expected, 3),
        "actual_volume_mm3": round(actual, 3),
        "diff_mm3": round(diff, 3),
        "error_pct": round(diff / expected * 100, 4),
        "allowed_tolerance_mm3": round(allowed, 3),
    }


def validate_part_geometry(shape: Any, volume: float, contract: Dict[str, Any]) -> None:
    expected = contract.get("expected_volume")
    if expected is not None:
        check = check_volume(volume, expected)
        if not check["passed"]:
            raise ValueError(f"Volume mismatch: expected {expected} mm³, measured {volume} mm³")

    expected_bbox = contract.get("expected_bbox_mm")
    if expected_bbox is not None:
        dimensions = tuple(map(float, expected_bbox))
        if len(dimensions) != 3 or any(not math.isfinite(dim) or dim <= 0 for dim in dimensions):
            raise ValueError("expected_bbox_mm must contain three positive dimensions")
        box = shape.BoundingBox()
        actual = (box.xlen, box.ylen, box.zlen)
        if any(abs(measured - nominal) > 0.1 for measured, nominal in zip(actual, dimensions)):
            raise ValueError(f"Bounding dimensions mismatch: expected {dimensions} mm, measured {actual} mm")

    for bounds in contract.get("expected_void_boxes", []):
        xmin, ymin, zmin, xmax, ymax, zmax = map(float, bounds)
        if not all(map(math.isfinite, (xmin, ymin, zmin, xmax, ymax, zmax))):
            raise ValueError("Void box coordinates must be finite")
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("Void box maxima must exceed minima")
        probe = cq.Solid.makeBox(xmax - xmin, ymax - ymin, zmax - zmin, cq.Vector(xmin, ymin, zmin))
        if shape.intersect(probe).Volume() > 1e-6:
            raise ValueError(f"Expected void box {bounds} contains solid material")


class HeadlessCADRunner:
    """Headless Execution Engine for CadQuery parts and assemblies."""

    def __init__(self, default_output_dir: Union[str, Path] = "workspace/parts"):
        self.default_output_dir = Path(default_output_dir)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_cq_object(local_vars: Dict[str, Any]) -> Optional[Any]:
        """Locates the CadQuery Workplane, Shape, or Assembly object from execution context."""
        # Priority keys
        for key in ("result", "assembly", "part", "model", "assy"):
            if key in local_vars:
                obj = local_vars[key]
                if hasattr(obj, "val") or hasattr(obj, "toCompound") or hasattr(obj, "save"):
                    return obj

        # Search for any Workplane or Assembly instance
        for val in local_vars.values():
            if cq:
                if isinstance(val, (cq.Workplane, cq.Assembly, cq.Shape, cq.Compound)):
                    return val
            elif hasattr(val, "val") or hasattr(val, "toCompound"):
                return val

        return None

    @classmethod
    def compute_metrics(cls, shape_obj: Any, expected_origin: Optional[str] = None) -> CADMetrics:
        """Computes comprehensive topological and physical metrics of a CadQuery shape."""
        metrics = CADMetrics()
        try:
            shape = cls._as_shape(shape_obj)

            if not hasattr(shape, "wrapped"):
                return metrics

            occ_shape = shape.wrapped

            # Basic validity
            analyzer = BRepCheck_Analyzer(occ_shape)
            metrics.is_valid = bool(analyzer.IsValid())

            # Bounding Box
            bbox = shape.BoundingBox()
            metrics.bbox_min = (round(bbox.xmin, 3), round(bbox.ymin, 3), round(bbox.zmin, 3))
            metrics.bbox_max = (round(bbox.xmax, 3), round(bbox.ymax, 3), round(bbox.zmax, 3))
            metrics.bbox_dimensions = (round(bbox.xlen, 3), round(bbox.ylen, 3), round(bbox.zlen, 3))

            # Volume & Surface Properties
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(occ_shape, props)
            metrics.volume = round(props.Mass(), 4)

            # Center of mass
            cm = props.CentreOfMass()
            metrics.center_of_mass = (round(cm.X(), 3), round(cm.Y(), 3), round(cm.Z(), 3))

            # Surface Area
            surf_props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(occ_shape, surf_props)
            metrics.surface_area = round(surf_props.Mass(), 4)

            # Topological element counts
            if hasattr(shape, "Faces"):
                metrics.face_count = len(shape.Faces())
            if hasattr(shape, "Edges"):
                metrics.edge_count = len(shape.Edges())
            if hasattr(shape, "Vertices"):
                metrics.vertex_count = len(shape.Vertices())
            if hasattr(shape, "Solids"):
                solids = shape.Solids()
                metrics.solid_count = len(solids)
                metrics.is_watertight = (
                    metrics.is_valid and bool(solids) and metrics.volume > 0
                    and all(shell.Closed() for solid in solids for shell in solid.Shells())
                )

            # Origin constraint verification
            if expected_origin == "center_bottom" and abs(metrics.bbox_min[2]) > 0.05:
                metrics.is_valid = False
                metrics.is_watertight = False

        except Exception as e:
            metrics.is_valid = False

        return metrics

    @staticmethod
    def _as_shape(obj: Any) -> Any:
        if isinstance(obj, cq.Assembly):
            return obj.toCompound()
        if isinstance(obj, cq.Workplane):
            shapes = [value for value in obj.vals() if isinstance(value, cq.Shape)]
            if len(shapes) == 1:
                return shapes[0]
            if shapes:
                return cq.Compound.makeCompound(shapes)
            return None
        return obj

    def _execute_part_code_in_process(
        self,
        code: str,
        part_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        export_formats: Tuple[str, ...] = ("step", "stl"),
        expected_origin: Optional[str] = None,
        expected_z_range: Optional[Tuple[float, float]] = None,
        geometry_contract: Optional[Dict[str, Any]] = None,
        z_tolerance: float = 0.05,
    ) -> CADExecutionResult:
        """Executes a CadQuery Python script headless inside the current process, validates geometry,
        and exports CAD artifacts (STEP/STL/JSON).
        """
        start_time = time.perf_counter()
        target_dir = Path(output_dir) if output_dir else self.default_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        captured_objects: List[Any] = []

        def mock_show_object(obj: Any, **kwargs: Any) -> None:
            captured_objects.append(obj)

        exec_globals: Dict[str, Any] = {
            "__name__": "__main__",
            "show_object": mock_show_object,
            "show": mock_show_object,
        }

        if cq:
            exec_globals["cq"] = cq
            exec_globals["cadquery"] = cq

        exec_locals: Dict[str, Any] = {}

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exec(code, exec_globals, exec_locals)

        except Exception as exc:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            tb_str = traceback.format_exc()
            err_msg = str(exc)
            diagnostics = CADDiagnosticAnalyzer.analyze(err_msg, tb_str, code)

            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error_type=type(exc).__name__,
                error_message=err_msg,
                error_traceback=tb_str,
                diagnostics=diagnostics,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Retrieve the resulting shape
        target_obj = None
        if captured_objects:
            target_obj = captured_objects[-1]
        else:
            target_obj = self._extract_cq_object(exec_locals) or self._extract_cq_object(exec_globals)

        if target_obj is None:
            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error_type="MissingResultError",
                error_message="The script completed without defining a 'result' Workplane or calling show_object().",
                diagnostics=[
                    "[NO_RETURN_OBJECT] No CadQuery geometry assigned to 'result' or 'part'.",
                    "  -> Suggestion: Assign the final solid to variable 'result = ...' or call show_object(result).",
                ],
            )

        # Compute geometrical metrics
        metrics = self.compute_metrics(target_obj)
        if not metrics.is_watertight:
            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                error_type="InvalidPartGeometry",
                error_message="Expected valid watertight solids with positive volume",
                metrics=metrics,
            )

        # Validate origin constraint if specified
        if expected_origin == "center_bottom" and metrics.is_valid:
            zmin = metrics.bbox_min[2]
            if abs(zmin) >= 0.05:
                err_msg = (
                    f"Origin mismatch: spec requires 'origin': 'center_bottom' (abs(zmin) < 0.05), "
                    f"but got zmin = {zmin:.3f} mm. "
                    f"Use .box(..., centered=(True, True, False)) or .translate((0, 0, z/2))."
                )
                diagnostics = CADDiagnosticAnalyzer.analyze(err_msg, "", code)
                return CADExecutionResult(
                    success=False,
                    part_name=part_name,
                    execution_time_ms=round(elapsed_ms, 2),
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                    error_type="OriginMismatchError",
                    error_message=err_msg,
                    diagnostics=diagnostics,
                    metrics=metrics,
                )

        # Validate Z-envelope if specified (relaxed tolerance of 0.05mm for Sweep/Loft)
        if expected_z_range and metrics.is_valid:
            zmin_exp, zmax_exp = expected_z_range
            actual_zmin = metrics.bbox_min[2]
            actual_zmax = metrics.bbox_max[2]
            zmin_err = abs(actual_zmin - zmin_exp)
            zmax_err = abs(actual_zmax - zmax_exp)
            if zmin_err > z_tolerance or zmax_err > z_tolerance:
                err_msg = (
                    f"Z-envelope tolerance exceeded (tol={z_tolerance:.3f} mm): "
                    f"expected Z in [{zmin_exp:.3f}, {zmax_exp:.3f}], but got [{actual_zmin:.3f}, {actual_zmax:.3f}] "
                    f"(deviations: zmin={zmin_err:.4f} mm, zmax={zmax_err:.4f} mm)."
                )
                return CADExecutionResult(
                    success=False,
                    part_name=part_name,
                    execution_time_ms=round(elapsed_ms, 2),
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                    error_type="ZEnvelopeToleranceError",
                    error_message=err_msg,
                    diagnostics=[
                        f"Align sweep path or loft profiles to Z=[{zmin_exp:.3f}, {zmax_exp:.3f}] (+/- {z_tolerance:.3f} mm).",
                    ],
                    metrics=metrics,
                )

        if geometry_contract is not None:
            try:
                if metrics.solid_count != 1:
                    raise ValueError("Expected exactly one solid")
                if isinstance(target_obj, cq.Workplane) and len(target_obj.vals()) != 1:
                    raise ValueError("Expected exactly one result on the Workplane stack")
                shape = self._as_shape(target_obj)
                validate_part_geometry(shape, metrics.volume, geometry_contract)
            except Exception as exc:
                return CADExecutionResult(
                    success=False,
                    part_name=part_name,
                    execution_time_ms=round(elapsed_ms, 2),
                    metrics=metrics,
                    error_type="GeometryContractError",
                    error_message=str(exc),
                )

        artifacts: Dict[str, str] = {}

        # Save python source file
        code_file = target_dir / f"{part_name}.py"
        code_file.write_text(code, encoding="utf-8")
        artifacts["script"] = str(code_file.resolve())

        # Export STEP and STL
        try:
            if "step" in export_formats:
                step_file = target_dir / f"{part_name}.step"
                if hasattr(target_obj, "toCompound"):
                    # Assembly export
                    target_obj.save(str(step_file))
                else:
                    cq.exporters.export(target_obj, str(step_file), cq.exporters.ExportTypes.STEP)
                artifacts["step"] = str(step_file.resolve())

            if "stl" in export_formats:
                stl_file = target_dir / f"{part_name}.stl"
                if hasattr(target_obj, "toCompound"):
                    compound = target_obj.toCompound()
                    cq.exporters.export(compound, str(stl_file), cq.exporters.ExportTypes.STL)
                else:
                    cq.exporters.export(target_obj, str(stl_file), cq.exporters.ExportTypes.STL)
                artifacts["stl"] = str(stl_file.resolve())

            if "glb" in export_formats and "stl" in artifacts:
                glb_file = target_dir / f"{part_name}.glb"
                try:
                    import trimesh
                    mesh = trimesh.load(artifacts["stl"], file_type="stl")
                    mesh.export(str(glb_file), file_type="glb")
                    artifacts["glb"] = str(glb_file.resolve())
                except Exception:
                    pass

        except Exception as export_exc:
            tb_str = traceback.format_exc()
            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error_type="CADExportError",
                error_message=f"Export failed: {str(export_exc)}",
                error_traceback=tb_str,
                diagnostics=["Verify shape geometry is valid solid before exporting."],
                metrics=metrics,
                artifacts=artifacts,
            )

        # Save metadata summary
        result = CADExecutionResult(
            success=True,
            part_name=part_name,
            execution_time_ms=round(elapsed_ms, 2),
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            metrics=metrics,
            artifacts=artifacts,
        )

        metadata_file = target_dir / f"{part_name}_meta.json"
        metadata_file.write_text(result.to_json(), encoding="utf-8")
        result.artifacts["metadata"] = str(metadata_file.resolve())

        return result

    def _execute_in_subprocess(
        self,
        code: str,
        part_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        export_formats: Tuple[str, ...] = ("step", "stl"),
        expected_origin: Optional[str] = None,
        expected_z_range: Optional[Tuple[float, float]] = None,
        z_tolerance: float = 0.05,
        is_assembly: bool = False,
        check_collisions: bool = True,
        timeout_sec: float = 60.0,
        geometry_contract: Optional[Dict[str, Any]] = None,
        assembly_volume: Optional[Tuple[float, float]] = None,
        full_brep: bool = False,
    ) -> CADExecutionResult:
        """Runs the CAD worker in an isolated subprocess to protect the main MCP server process."""
        import subprocess

        start_time = time.perf_counter()
        target_dir = Path(output_dir) if output_dir else self.default_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "code": code,
            "part_name": part_name,
            "output_dir": str(target_dir.resolve()),
            "export_formats": list(export_formats),
            "expected_origin": expected_origin,
            "expected_z_range": list(expected_z_range) if expected_z_range else None,
            "z_tolerance": z_tolerance,
            "is_assembly": is_assembly,
            "check_collisions": check_collisions,
            "full_brep": full_brep,
            "geometry_contract": geometry_contract,
            "assembly_volume": assembly_volume,
        }

        runner_script = Path(__file__).resolve()
        cmd = [sys.executable, str(runner_script), "--worker"]

        # Prevent OpenBLAS TLS allocation failures on Windows
        worker_env = os.environ.copy()
        worker_env["OPENBLAS_NUM_THREADS"] = "1"
        worker_env["OMP_NUM_THREADS"] = "1"
        worker_env["MKL_NUM_THREADS"] = "1"
        worker_env["NUMEXPR_NUM_THREADS"] = "1"

        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                encoding="utf-8",
                errors="replace",
                env=worker_env,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            stdout_text = proc.stdout.strip()
            parsed_data = None
            if "{" in stdout_text and "}" in stdout_text:
                json_start = stdout_text.find("{")
                json_end = stdout_text.rfind("}") + 1
                try:
                    parsed_data = json.loads(stdout_text[json_start:json_end])
                except Exception:
                    pass

            if parsed_data:
                metrics_data = parsed_data.get("metrics")
                metrics = CADMetrics(**metrics_data) if metrics_data else None
                return CADExecutionResult(
                    success=parsed_data.get("success", False),
                    part_name=parsed_data.get("part_name", part_name),
                    execution_time_ms=parsed_data.get("execution_time_ms", round(elapsed_ms, 2)),
                    stdout=parsed_data.get("stdout", ""),
                    stderr=parsed_data.get("stderr", "") + (f"\n{proc.stderr}" if proc.stderr else ""),
                    error_type=parsed_data.get("error_type"),
                    error_message=parsed_data.get("error_message"),
                    error_traceback=parsed_data.get("error_traceback"),
                    diagnostics=parsed_data.get("diagnostics", []),
                    metrics=metrics,
                    artifacts=parsed_data.get("artifacts", {}),
                )

            err_msg = f"Worker subprocess exited with code {proc.returncode}."
            if proc.stderr:
                err_msg += f" Stderr: {proc.stderr[:400]}"
            diagnostics = CADDiagnosticAnalyzer.analyze(err_msg, proc.stderr, code)
            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                stdout=proc.stdout,
                stderr=proc.stderr,
                error_type="WorkerSubprocessError",
                error_message=err_msg,
                error_traceback=proc.stderr,
                diagnostics=diagnostics,
            )

        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                error_type="TimeoutExpired",
                error_message=f"CadQuery execution timed out after {timeout_sec}s.",
                diagnostics=["Execution timed out. Check for infinite loops or non-terminating geometry operations."],
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return CADExecutionResult(
                success=False,
                part_name=part_name,
                execution_time_ms=round(elapsed_ms, 2),
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                diagnostics=[f"Subprocess launcher error: {str(e)}"],
            )

    def execute_part_code(
        self,
        code: str,
        part_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        export_formats: Tuple[str, ...] = ("step", "stl"),
        expected_origin: Optional[str] = None,
        expected_z_range: Optional[Tuple[float, float]] = None,
        z_tolerance: float = 0.05,
        use_subprocess: bool = True,
        timeout_sec: float = 60.0,
        geometry_contract: Optional[Dict[str, Any]] = None,
    ) -> CADExecutionResult:
        """Executes a CadQuery Python script headless, validates geometry, and exports artifacts."""
        if not use_subprocess:
            return self._execute_part_code_in_process(
                code=code,
                part_name=part_name,
                output_dir=output_dir,
                export_formats=export_formats,
                expected_origin=expected_origin,
                expected_z_range=expected_z_range,
                z_tolerance=z_tolerance,
                geometry_contract=geometry_contract,
            )
        return self._execute_in_subprocess(
            code=code,
            part_name=part_name,
            output_dir=output_dir,
            export_formats=export_formats,
            expected_origin=expected_origin,
            expected_z_range=expected_z_range,
            z_tolerance=z_tolerance,
            is_assembly=False,
            timeout_sec=timeout_sec,
            geometry_contract=geometry_contract,
        )

    @classmethod
    def check_assembly_interferences(cls, target_obj: Any, full_brep: bool = False) -> Dict[str, Any]:
        """Calculates pair-wise collisions/interference between components in a cq.Assembly
        using a high-performance 2-Phase Algorithm:
        1. Broad-Phase AABB filtering (tol = 1e-4 mm): prunes ~90%+ disjoint pairs with 0 Boolean calls.
        2. Narrow-Phase B-Rep Boolean intersection: only runs OpenCASCADE Boolean on candidates with overlapping AABBs.
        """
        def collect(node: Any, prefix: str, parent_loc: Any) -> List[Tuple[str, Any]]:
            world_loc = parent_loc * node.loc
            components = []
            shapes = node.shapes
            if shapes:
                components.append((prefix, cq.Compound.makeCompound(shapes).locate(world_loc)))
            for child in node.children:
                components.extend(collect(child, f"{prefix}/{child.name}" if prefix else child.name, world_loc))
            return components

        if isinstance(target_obj, cq.Assembly):
            components = collect(target_obj, "", cq.Location())
        else:
            shape = cls._as_shape(target_obj)
            components = [(f"solid_{i}", solid) for i, solid in enumerate(shape.Solids(), 1)]

        # Compute per-component properties once, even when testing many pairs.
        solids = [solid for _, shape in components for solid in shape.Solids()]
        valid_count = sum(
            BRepCheck_Analyzer(solid.wrapped).IsValid() and all(shell.Closed() for shell in solid.Shells())
            for solid in solids
        )
        boxes = [shape.BoundingBox() for _, shape in components]
        face_counts: Dict[int, int] = {}
        n = len(components)
        report: Dict[str, Any] = {
            "has_collisions": False,
            "verification_complete": True,
            "max_intersection_mm3": 0.0,
            "total_interference_volume_mm3": 0.0,
            "collision_pairs": [],
            "unverified_pairs": [],
            "warnings": [],
            "pairs_total": n * (n - 1) // 2,
            "pairs_aabb_pruned": 0,
            "pairs_boolean_checked": 0,
            "pairs_complex_face_skipped": 0,
            "pairs_boolean_failed": 0,
            "solid_count": len(solids),
            "valid_solids": valid_count,
        }

        tol = 1e-4
        for i, (name_a, shape_a) in enumerate(components):
            bb_a = boxes[i]
            for j in range(i + 1, n):
                bb_b = boxes[j]
                if (min(bb_a.xmax, bb_b.xmax) - max(bb_a.xmin, bb_b.xmin) <= tol
                        or min(bb_a.ymax, bb_b.ymax) - max(bb_a.ymin, bb_b.ymin) <= tol
                        or min(bb_a.zmax, bb_b.zmax) - max(bb_a.zmin, bb_b.zmin) <= tol):
                    report["pairs_aabb_pruned"] += 1
                    continue

                name_b, shape_b = components[j]
                pair = {"part_a": name_a or "root", "part_b": name_b or "root"}
                if i not in face_counts:
                    face_counts[i] = len(shape_a.Faces())
                if j not in face_counts:
                    face_counts[j] = len(shape_b.Faces())
                if not full_brep and face_counts[i] * face_counts[j] > 1500:
                    report["pairs_complex_face_skipped"] += 1
                    report["unverified_pairs"].append({**pair, "reason": "face_complexity"})
                    continue

                try:
                    vol = round(shape_a.intersect(shape_b).Volume(), 4)
                    if not math.isfinite(vol) or vol < 0:
                        raise ValueError("Invalid intersection volume")
                except Exception as exc:
                    report["pairs_boolean_failed"] += 1
                    report["unverified_pairs"].append({**pair, "reason": type(exc).__name__})
                    continue

                report["pairs_boolean_checked"] += 1
                if vol > 0.01:
                    report["has_collisions"] = True
                    report["total_interference_volume_mm3"] += vol
                    report["max_intersection_mm3"] = max(report["max_intersection_mm3"], vol)
                    def extents(box):
                        return [round(value, 4) for value in (box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax)]
                    report["collision_pairs"].append({
                        **pair, "overlap_volume_mm3": vol, "bbox_a": extents(bb_a), "bbox_b": extents(bb_b),
                    })

        report["total_interference_volume_mm3"] = round(report["total_interference_volume_mm3"], 4)
        if report["unverified_pairs"]:
            report["verification_complete"] = False
            report["warnings"].append(f"{len(report['unverified_pairs'])} overlapping AABB pairs could not be verified")
        return report

    def _execute_assembly_code_in_process(
        self,
        code: str,
        assembly_name: str,
        output_dir: Optional[Union[str, Path]] = "workspace/assembly",
        check_collisions: bool = True,
        assembly_volume: Optional[Tuple[float, float]] = None,
        full_brep: bool = False,
    ) -> CADExecutionResult:
        """Executes a CadQuery Assembly script in a SINGLE PASS:
        captures stdout/stderr cleanly, extracts Assembly in-memory, computes metrics & 2-phase collisions,
        and exports STEP and GLB.
        """
        import contextlib
        import io

        t_start = time.perf_counter()
        target_dir = Path(output_dir) if output_dir else Path("workspace/assembly")
        target_dir.mkdir(parents=True, exist_ok=True)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        exec_globals: Dict[str, Any] = {
            "__name__": "__main__",
            "cq": cq,
            "cadquery": cq,
        }
        # Use one namespace so functions and comprehensions see script imports.
        # 1. Execute script ONCE with complete stdout/stderr redirection
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, exec_globals)
        except Exception as exc:
            tb_str = traceback.format_exc()
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return CADExecutionResult(
                success=False,
                part_name=assembly_name,
                execution_time_ms=round(elapsed_ms, 2),
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_traceback=tb_str,
                diagnostics=CADDiagnosticAnalyzer.analyze(str(exc), tb_str, code),
            )

        t_script = time.perf_counter() - t_start

        # 2. Extract Assembly object
        target_obj = self._extract_cq_object(exec_globals)
        if target_obj is None:
            return CADExecutionResult(
                success=False,
                part_name=assembly_name,
                execution_time_ms=round(t_script * 1000, 2),
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error_type="MissingResultError",
                error_message="The script completed without defining a 'result' or 'assy' Assembly object.",
                diagnostics=["Assign the final assembly to variable 'result = assy'."],
            )

        metrics = self.compute_metrics(target_obj)
        if not metrics.is_watertight:
            return CADExecutionResult(
                success=False, part_name=assembly_name,
                execution_time_ms=round(t_script * 1000, 2),
                error_message="Assembly must contain valid watertight solids with positive volume",
            )
        volume_check = None
        if assembly_volume is not None:
            try:
                volume_check = check_volume(metrics.volume, *assembly_volume)
            except ValueError as exc:
                return CADExecutionResult(success=False, part_name=assembly_name,
                                          execution_time_ms=round(t_script * 1000, 2),
                                          error_message=str(exc))
            if not volume_check["passed"]:
                return CADExecutionResult(
                    success=False, part_name=assembly_name,
                    execution_time_ms=round(t_script * 1000, 2),
                    error_message="Assembly volume exceeds tolerance",
                    artifacts={"volume_check": json.dumps(volume_check)},
                )

        # 3. Two-phase Collision Verification (Broad-phase AABB + Narrow-phase B-Rep)
        t_col_start = time.perf_counter()
        inter_report: Dict[str, Any] = {
            "has_collisions": False,
            "verification_complete": False,
            "max_intersection_mm3": 0.0,
            "total_interference_volume_mm3": 0.0,
            "collision_pairs": [],
            "warnings": [],
            "pairs_total": 0,
            "pairs_aabb_pruned": 0,
            "pairs_boolean_checked": 0,
            "solid_count": 0,
            "valid_solids": 0,
        }
        if check_collisions:
            inter_report = self.check_assembly_interferences(target_obj, full_brep=full_brep)
        t_collision = time.perf_counter() - t_col_start

        # 4. Metrics & In-Memory Export
        t_exp_start = time.perf_counter()
        artifacts: Dict[str, str] = {}
        try:
            with TemporaryDirectory(dir=target_dir) as staging_dir:
                stage = Path(staging_dir)
                step_file = stage / f"{assembly_name}.step"
                stl_file = stage / f"{assembly_name}.stl"
                glb_file = stage / f"{assembly_name}.glb"
                if isinstance(target_obj, cq.Assembly):
                    target_obj.export(str(step_file))
                    compound = target_obj.toCompound()
                else:
                    compound = target_obj
                    cq.exporters.export(compound, str(step_file), cq.exporters.ExportTypes.STEP)

                cq.exporters.export(compound, str(stl_file), cq.exporters.ExportTypes.STL)
                if isinstance(target_obj, cq.Assembly):
                    target_obj.export(str(glb_file))
                else:
                    import trimesh

                    mesh = trimesh.load(str(stl_file), file_type="stl")
                    mesh.export(str(glb_file), file_type="glb")

                for file in (step_file, stl_file, glb_file):
                    if file.stat().st_size == 0:
                        raise RuntimeError(f"Empty export {file.name}")
                for file in (step_file, stl_file, glb_file):
                    destination = target_dir / file.name
                    file.replace(destination)
                    artifacts[file.suffix[1:]] = str(destination.resolve())
        except Exception as exc:
            return CADExecutionResult(
                success=False, part_name=assembly_name,
                execution_time_ms=round((time.perf_counter() - t_start) * 1000, 2),
                error_message=f"Assembly export failed: {exc}", artifacts=artifacts,
            )
        t_export = time.perf_counter() - t_exp_start

        t_total = time.perf_counter() - t_start

        elapsed_sec_dict = {
            "script": round(t_script, 3),
            "collision": round(t_collision, 3),
            "export": round(t_export, 3),
            "total": round(t_total, 3),
        }

        status = "SUCCESS"
        if inter_report["has_collisions"]:
            status = "COLLISION_DETECTED"
        elif check_collisions and not inter_report["verification_complete"]:
            status = "VERIFICATION_INCOMPLETE"

        meta_data = {
            "status": status,
            "collision_verification_complete": inter_report["verification_complete"],
            "solid_count": metrics.solid_count,
            "valid_solids": metrics.solid_count,
            "bbox_mm": list(metrics.bbox_dimensions) if metrics else [],
            "bbox_min": list(metrics.bbox_min) if metrics else [],
            "total_volume_mm3": round(metrics.volume, 3) if metrics else 0.0,
            "pairs_total": inter_report.get("pairs_total", 0),
            "pairs_aabb_pruned": inter_report.get("pairs_aabb_pruned", 0),
            "pairs_boolean_checked": inter_report.get("pairs_boolean_checked", 0),
            "max_intersection_mm3": inter_report.get("max_intersection_mm3", 0.0),
            "elapsed_sec": elapsed_sec_dict,
            "outputs": {
                "step": artifacts["step"],
                "glb": artifacts["glb"],
            },
            "artifacts": artifacts,
            "interferences": inter_report,
        }

        if volume_check is not None:
            meta_data["volume_check"] = volume_check
        meta_path = target_dir / f"{assembly_name}_meta.json"
        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
        artifacts["metadata"] = str(meta_path.resolve())
        artifacts["interferences"] = json.dumps(inter_report)
        artifacts["structured_metadata"] = json.dumps(meta_data)

        return CADExecutionResult(
            success=True,
            part_name=assembly_name,
            execution_time_ms=round(t_total * 1000, 2),
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            metrics=metrics,
            artifacts=artifacts,
        )

    def execute_assembly_code(
        self,
        code: str,
        assembly_name: str,
        output_dir: Optional[Union[str, Path]] = "workspace/assembly",
        check_collisions: bool = True,
        use_subprocess: bool = True,
        timeout_sec: float = 60.0,
        assembly_volume: Optional[Tuple[float, float]] = None,
        full_brep: bool = False,
    ) -> CADExecutionResult:
        """Executes a CadQuery Assembly script with pair-wise collision detection and STEP+GLB export."""
        if not use_subprocess:
            return self._execute_assembly_code_in_process(
                code=code,
                assembly_name=assembly_name,
                output_dir=output_dir,
                check_collisions=check_collisions,
                assembly_volume=assembly_volume,
                full_brep=full_brep,
            )
        return self._execute_in_subprocess(
            code=code,
            part_name=assembly_name,
            output_dir=output_dir,
            export_formats=("step", "stl", "glb"),
            is_assembly=True,
            check_collisions=check_collisions,
            timeout_sec=timeout_sec,
            assembly_volume=assembly_volume,
            full_brep=full_brep,
        )


def _run_worker_cli():
    """CLI worker entrypoint for isolated subprocess execution."""
    import sys
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input)
        runner = HeadlessCADRunner(default_output_dir=payload.get("output_dir", "workspace/parts"))

        is_assembly = payload.get("is_assembly", False)
        check_collisions = payload.get("check_collisions", True)

        if is_assembly:
            result = runner._execute_assembly_code_in_process(
                code=payload["code"],
                assembly_name=payload["part_name"],
                output_dir=payload.get("output_dir"),
                check_collisions=check_collisions,
                assembly_volume=payload.get("assembly_volume"),
                full_brep=payload.get("full_brep", False),
            )
        else:
            exp_z = payload.get("expected_z_range")
            result = runner._execute_part_code_in_process(
                code=payload["code"],
                part_name=payload["part_name"],
                output_dir=payload.get("output_dir"),
                export_formats=tuple(payload.get("export_formats", ("step", "stl"))),
                expected_origin=payload.get("expected_origin"),
                expected_z_range=tuple(exp_z) if exp_z else None,
                z_tolerance=float(payload.get("z_tolerance", 0.05)),
                geometry_contract=payload.get("geometry_contract"),
            )
        sys.stdout.write(result.to_json())
        sys.stdout.flush()
        sys.exit(0 if result.success else 1)
    except Exception as exc:
        err_dict = {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(),
            "diagnostics": [f"Worker CLI exception: {str(exc)}"],
        }
        sys.stdout.write(json.dumps(err_dict))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _run_worker_cli()
