"""
Synthetic CAD Training Data Generator with OpenCASCADE Validation Filter
Generates 2,000+ verified, watertight, single-solid CadQuery samples in ChatML format.
Strict 2-Group Distribution:
  - Group A: Freeform / Non-standard (60%, 1,200 samples) [shell, loft, sweep, gusset/rib, asymmetric flange]
  - Group B: Complex Parametric (40%, 800 samples) [stepped shaft dual keyway, double-stepped bearing housing, multi-pattern fixture plate]
Optimized for Unsloth QLoRA fine-tuning on Kaggle Tesla T4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cadquery as cq
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
except ImportError as e:
    sys.stderr.write(f"CadQuery/OCP is required: {e}\n")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    console = Console()
except ImportError:
    console = None


SYSTEM_PROMPT = (
    "You are an expert CAD automation engineer and precision CadQuery programmer. "
    "Write clean, robust, and watertight CadQuery Python code satisfying the mechanical specification. "
    "Always assign the final valid 3D solid to `result`. "
    "When origin is center_bottom, ensure the base rests at Z=0 (zmin == 0.0)."
)


# ------------------------------------------------------------------------------
# 1. GROUP A: FREEFORM & NON-STANDARD CAD GENERATORS (60% Target)
# ------------------------------------------------------------------------------

def gen_shelled_enclosure(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group A (Freeform/Non-standard): Thin-walled hollow casing using shell."""
    L = round(rng.uniform(75.0, 130.0), 1)
    W = round(rng.uniform(50.0, 85.0), 1)
    H = round(rng.uniform(25.0, 45.0), 1)
    wall = round(rng.uniform(2.5, 3.5), 1)
    hole_d = 4.0
    hx = round((L / 2) - wall * 2.5, 1)
    hy = round((W / 2) - wall * 2.5, 1)

    prompt = (
        f"Design a thin-walled electronic housing enclosure: outer length {L}mm, width {W}mm, height {H}mm. "
        f"Hollow out the interior cavity with uniform wall thickness {wall}mm using an open top face. "
        f"Include 4 internal corner mounting through-holes (diameter {hole_d}mm) positioned at (+/-{hx}mm, +/-{hy}mm). "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

base = (
    cq.Workplane("XY")
    .box({L}, {W}, {H}, centered=(True, True, False))
    .faces(">Z")
    .shell(-{wall})
)

result = (
    base
    .faces("<Z")
    .workplane()
    .pushPoints([(-{hx}, -{hy}), ({hx}, -{hy}), (-{hx}, {hy}), ({hx}, {hy})])
    .hole({hole_d})
)
"""
    meta = {"type": "shelled_enclosure", "group": "freeform_non_standard_group_a", "bbox": [L, W, H]}
    return prompt, code, meta


def gen_loft_adapter(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group A (Freeform/Non-standard): Smooth loft transition from rectangular base to circular flange."""
    rect_l = round(rng.uniform(50.0, 90.0), 1)
    rect_w = round(rng.uniform(35.0, 65.0), 1)
    top_r = round(rng.uniform(15.0, 25.0), 1)
    height = round(rng.uniform(40.0, 80.0), 1)
    bore_d = round(top_r * rng.uniform(0.7, 0.9), 1)

    prompt = (
        f"Design a smooth loft transition adapter: bottom base rectangle {rect_l}mm x {rect_w}mm at Z=0, "
        f"lofting smoothly to a circular top cross-section (radius {top_r}mm) at height {height}mm. "
        f"Cut a central internal through passage (diameter {bore_d}mm). "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

result = (
    cq.Workplane("XY")
    .rect({rect_l}, {rect_w})
    .workplane(offset={height})
    .circle({top_r})
    .loft(combine=True)
    .faces("<Z")
    .workplane()
    .hole({bore_d})
)
"""
    meta = {"type": "loft_adapter", "group": "freeform_non_standard_group_a", "bbox": [rect_l, rect_w, height]}
    return prompt, code, meta


def gen_curved_sweep_pipe(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group A (Freeform/Non-standard): 90-degree curved sweep duct along 3D arc path with base flange."""
    outer_r = round(rng.uniform(10.0, 16.0), 1)
    inner_r = round(outer_r - rng.uniform(2.5, 4.0), 1)
    bend_r = round(rng.uniform(25.0, 45.0), 1)
    flange_r = round(outer_r + rng.uniform(10.0, 18.0), 1)
    flange_h = round(rng.uniform(6.0, 12.0), 1)

    prompt = (
        f"Design a 90-degree curved sweep conduit pipe with integral base flange: "
        f"mounting base flange thickness {flange_h}mm and radius {flange_r}mm at Z=0. "
        f"Sweep a circular pipe section (outer radius {outer_r}mm, inner radius {inner_r}mm) "
        f"along a 90-degree smooth arc path with bend radius {bend_r}mm. "
        f"Provide a continuous central opening of diameter {inner_r * 2}mm. Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

bend_r = {bend_r}
path = (
    cq.Workplane("XZ")
    .moveTo(0, 0)
    .lineTo(0, bend_r)
    .threePointArc((bend_r * 0.293, bend_r + bend_r * 0.707), (bend_r, bend_r * 2))
    .lineTo(bend_r + 20.0, bend_r * 2)
)

pipe = cq.Workplane("XY").circle({outer_r}).circle({inner_r}).sweep(path)
flange = (
    cq.Workplane("XY")
    .cylinder({flange_h}, {flange_r}, centered=(True, True, False))
    .faces(">Z")
    .workplane()
    .hole({inner_r * 2})
)

result = flange.union(pipe.translate((0, 0, {flange_h})))
"""
    meta = {"type": "curved_sweep_pipe", "group": "freeform_non_standard_group_a", "bbox": [flange_r * 2, flange_r * 2, bend_r * 2 + flange_h]}
    return prompt, code, meta


def gen_gusset_bracket(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group A (Freeform/Non-standard): L-bracket with inclined triangular stiffening rib/gusset."""
    leg1 = round(rng.uniform(50.0, 90.0), 1)
    leg2 = round(rng.uniform(50.0, 90.0), 1)
    w = round(rng.uniform(35.0, 60.0), 1)
    t = round(rng.uniform(6.0, 12.0), 1)
    rib_t = round(rng.uniform(4.0, 8.0), 1)
    hole_d = round(rng.choice([5.5, 6.6, 8.5]), 1)
    hx = round(w / 4, 1)
    hy1 = round(leg1 * 0.65, 1)
    hy2 = round(leg2 * 0.65, 1)
    rib_offset_y = round(leg1 - 12.0, 1)
    rib_offset_z = round(leg2 - 12.0, 1)

    prompt = (
        f"Design a heavy-duty angle bracket with an inclined triangular stiffening gusset: "
        f"horizontal foot length {leg1}mm, vertical leg height {leg2}mm, extrusion width {w}mm, "
        f"wall thickness {t}mm. Add a central triangular gusset rib of thickness {rib_t}mm connecting "
        f"the inner faces from Y={rib_offset_y}mm to Z={rib_offset_z}mm. "
        f"Include 2 mounting holes (diameter {hole_d}mm) on the horizontal foot and 2 on the vertical leg. "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

pts = [(0, 0), ({leg1}, 0), ({leg1}, {t}), ({t}, {t}), ({t}, {leg2}), (0, {leg2})]
bracket = cq.Workplane("YZ").polyline(pts).close().extrude({w}).translate((- {w / 2}, 0, 0))

rib_pts = [({t}, {t}), ({rib_offset_y}, {t}), ({t}, {rib_offset_z})]
rib = cq.Workplane("YZ").polyline(rib_pts).close().extrude({rib_t}).translate((- {rib_t / 2}, 0, 0))

combined = bracket.union(rib)

result = (
    combined
    .faces("<Z")
    .workplane()
    .pushPoints([(- {hx}, {hy1}), ({hx}, {hy1})])
    .hole({hole_d})
    .faces("<Y")
    .workplane()
    .pushPoints([(- {hx}, {hy2}), ({hx}, {hy2})])
    .hole({hole_d})
)
"""
    meta = {"type": "gusset_bracket", "group": "freeform_non_standard_group_a", "bbox": [w, leg1, leg2]}
    return prompt, code, meta


def gen_asymmetric_flange(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group A (Freeform/Non-standard): Asymmetric mounting flange with offset lobes and unequal hole pattern."""
    base_r = round(rng.uniform(32.0, 48.0), 1)
    flange_h = round(rng.uniform(10.0, 16.0), 1)
    ear_x = round(base_r * rng.uniform(0.70, 0.85), 1)
    ear_y = round(rng.uniform(-6.0, 6.0), 1)
    ear_r = round(rng.uniform(16.0, 22.0), 1)
    bore_d = round(base_r * 0.5, 1)
    ear_hole_d = 6.5
    side_hole_d = 5.5

    prompt = (
        f"Design an asymmetric mounting flange: circular base body radius {base_r}mm, thickness {flange_h}mm. "
        f"Merge an asymmetric mounting ear lobe (radius {ear_r}mm) positioned at offset X={ear_x}mm, Y={ear_y}mm. "
        f"Cut a central bore (diameter {bore_d}mm). "
        f"Drill a clamping through hole (diameter {ear_hole_d}mm) on the offset ear lobe, "
        f"and 2 mounting holes (diameter {side_hole_d}mm) on the circular base at (-{round(base_r*0.6, 1)}mm, +/-{round(base_r*0.35, 1)}mm). "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

base = cq.Workplane("XY").cylinder({flange_h}, {base_r}, centered=(True, True, False))
ear = cq.Workplane("XY").moveTo({ear_x}, {ear_y}).circle({ear_r}).extrude({flange_h})
flange = base.union(ear)

result = (
    flange
    .faces(">Z")
    .workplane()
    .hole({bore_d})
    .faces(">Z")
    .workplane()
    .pushPoints([({ear_x}, {ear_y})])
    .hole({ear_hole_d})
    .faces(">Z")
    .workplane()
    .pushPoints([(-{round(base_r*0.6, 1)}, -{round(base_r*0.35, 1)}), (-{round(base_r*0.6, 1)}, {round(base_r*0.35, 1)})])
    .hole({side_hole_d})
)
"""
    meta = {"type": "asymmetric_flange", "group": "freeform_non_standard_group_a", "bbox": [ear_x + ear_r + base_r, base_r * 2, flange_h]}
    return prompt, code, meta


# ------------------------------------------------------------------------------
# 2. GROUP B: COMPLEX PARAMETRIC CAD GENERATORS (40% Target)
# ------------------------------------------------------------------------------

def gen_stepped_shaft_dual_keyway(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group B (Complex Parametric): Multi-stage stepped shaft with dual keyways offset by 90 degrees."""
    d1 = round(rng.uniform(16.0, 24.0), 1)
    h1 = round(rng.uniform(25.0, 45.0), 1)
    d2 = round(d1 + rng.uniform(10.0, 18.0), 1)
    h2 = round(rng.uniform(40.0, 70.0), 1)
    d3 = round(rng.uniform(d1 - 2.0, d1 + 4.0), 1)
    h3 = round(rng.uniform(25.0, 40.0), 1)

    kw1_w = round(rng.choice([4.0, 5.0, 6.0]), 1)
    kw1_len = round(h1 * 0.55, 1)
    kw1_z = round(h1 * 0.2, 1)

    kw2_w = round(rng.choice([5.0, 6.0, 8.0]), 1)
    kw2_len = round(h3 * 0.55, 1)
    kw2_z = round(h1 + h2 + h3 * 0.2, 1)

    total_h = h1 + h2 + h3

    prompt = (
        f"Design a 3-stage stepped transmission shaft with dual keyways: "
        f"Stage 1 diameter {d1}mm (length {h1}mm), Stage 2 diameter {d2}mm (length {h2}mm), "
        f"Stage 3 diameter {d3}mm (length {h3}mm). "
        f"Mill a primary keyway on Stage 1 (width {kw1_w}mm, length {kw1_len}mm, Z start {kw1_z}mm). "
        f"Mill a secondary keyway on Stage 3 rotated 90 degrees (width {kw2_w}mm, length {kw2_len}mm, Z start {kw2_z}mm). "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

s1 = cq.Workplane("XY").cylinder({h1}, {d1 / 2}, centered=(True, True, False))
s2 = cq.Workplane("XY").workplane(offset={h1}).cylinder({h2}, {d2 / 2}, centered=(True, True, False))
s3 = cq.Workplane("XY").workplane(offset={h1 + h2}).cylinder({h3}, {d3 / 2}, centered=(True, True, False))
shaft = s1.union(s2).union(s3)

kw1 = (
    cq.Workplane("XY")
    .workplane(offset={kw1_z})
    .box({kw1_w}, {d1}, {kw1_len}, centered=(True, False, False))
)
kw2 = (
    cq.Workplane("XY")
    .workplane(offset={kw2_z})
    .box({d3}, {kw2_w}, {kw2_len}, centered=(False, True, False))
)

result = shaft.cut(kw1).cut(kw2)
"""
    meta = {"type": "stepped_shaft_dual_keyway", "group": "complex_parametric_group_b", "bbox": [d2, d2, total_h]}
    return prompt, code, meta


def gen_double_stepped_bearing_housing(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group B (Complex Parametric): Pillow block with stepped bearing bores from both top and bottom."""
    L = round(rng.uniform(75.0, 120.0), 1)
    W = round(rng.uniform(35.0, 55.0), 1)
    H = round(rng.uniform(35.0, 60.0), 1)
    fillet_r = round(rng.uniform(2.5, 4.5), 1)

    bore1_d = round(rng.choice([32.0, 35.0, 42.0]), 1)
    depth1 = round(rng.uniform(10.0, 16.0), 1)
    bore2_d = round(bore1_d + rng.uniform(4.0, 8.0), 1)
    depth2 = round(rng.uniform(10.0, 16.0), 1)
    thru_d = round(bore1_d - rng.uniform(8.0, 14.0), 1)

    ear_x = round((L / 2) - rng.uniform(9.0, 15.0), 1)
    mount_d = round(rng.choice([5.5, 6.6, 8.5]), 1)

    prompt = (
        f"Design a double-stepped bearing housing block: length {L}mm, width {W}mm, height {H}mm. "
        f"Apply vertical edge fillets of radius {fillet_r}mm. "
        f"Bore a central through hole (diameter {thru_d}mm). "
        f"From the top face, counterbore bearing seat 1 (diameter {bore1_d}mm, depth {depth1}mm). "
        f"From the bottom face, counterbore bearing seat 2 (diameter {bore2_d}mm, depth {depth2}mm). "
        f"Include 2 side mounting holes (diameter {mount_d}mm) positioned at X = +/-{ear_x}mm. "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

result = (
    cq.Workplane("XY")
    .box({L}, {W}, {H}, centered=(True, True, False))
    .edges("|Z")
    .fillet({fillet_r})
    .faces(">Z")
    .workplane()
    .hole({thru_d})
    .faces(">Z")
    .workplane()
    .hole({bore1_d}, {depth1})
    .faces("<Z")
    .workplane()
    .hole({bore2_d}, {depth2})
    .faces(">Z")
    .workplane()
    .pushPoints([(-{ear_x}, 0), ({ear_x}, 0)])
    .hole({mount_d})
)
"""
    meta = {"type": "double_stepped_bearing_housing", "group": "complex_parametric_group_b", "bbox": [L, W, H]}
    return prompt, code, meta


def gen_multi_pattern_fixture_plate(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    """Group B (Complex Parametric): Tooling fixture plate with slot, matrix bolt array, and dowel locator holes."""
    L = round(rng.uniform(110.0, 180.0), 1)
    W = round(rng.uniform(70.0, 110.0), 1)
    H = round(rng.uniform(15.0, 26.0), 1)
    slot_w = round(rng.uniform(10.0, 16.0), 1)
    slot_d = round(rng.uniform(4.0, H * 0.45), 1)
    slot_len = round(L * rng.uniform(0.7, 0.9), 1)

    mat_x = round(rng.choice([30.0, 35.0, 40.0]), 1)
    mat_y = round(rng.choice([25.0, 30.0, 35.0]), 1)
    nx = 3
    ny = 2
    bolt_d = round(rng.choice([5.5, 6.6, 9.0]), 1)

    dowel_x = round((L / 2) - 12.0, 1)
    dowel_d = round(rng.choice([4.0, 5.0, 6.0]), 1)

    prompt = (
        f"Design a precision CNC multi-pattern tooling fixture plate: length {L}mm, width {W}mm, thickness {H}mm. "
        f"Mill a central alignment slot (length {slot_len}mm, width {slot_w}mm, depth {slot_d}mm) along X. "
        f"Drill a 3x2 matrix array of mounting holes (diameter {bolt_d}mm) spaced {mat_x}mm in X and {mat_y}mm in Y. "
        f"Add 2 precision dowel locator holes (diameter {dowel_d}mm) at (+/-{dowel_x}mm, 0). "
        f"Origin must be center_bottom."
    )

    code = f"""import cadquery as cq

result = (
    cq.Workplane("XY")
    .box({L}, {W}, {H}, centered=(True, True, False))
    .faces(">Z")
    .workplane()
    .slot2D({slot_len}, {slot_w}, 0)
    .cutBlind(-{slot_d})
    .faces(">Z")
    .workplane()
    .rarray({mat_x}, {mat_y}, {nx}, {ny})
    .hole({bolt_d})
    .faces(">Z")
    .workplane()
    .pushPoints([(-{dowel_x}, 0), ({dowel_x}, 0)])
    .hole({dowel_d})
)
"""
    meta = {"type": "multi_pattern_fixture_plate", "group": "complex_parametric_group_b", "bbox": [L, W, H]}
    return prompt, code, meta


# ------------------------------------------------------------------------------
# 3. STRICT QUALITY FILTER (OpenCASCADE + Topology Analyzer)
# ------------------------------------------------------------------------------

class QualityVerifier:
    """Executes generated CadQuery code and enforces strict 3D topological validity."""

    @staticmethod
    def verify_code(code_str: str) -> Tuple[bool, str, Dict[str, Any]]:
        exec_globals = {"cq": cq, "cadquery": cq, "math": math}
        exec_locals: Dict[str, Any] = {}

        try:
            exec(code_str, exec_globals, exec_locals)
        except Exception as e:
            return False, f"Code Execution Failed: {str(e)}", {}

        result_obj = exec_locals.get("result") or exec_globals.get("result")
        if result_obj is None:
            return False, "Variable `result` was not defined or assigned.", {}

        # Extract underlying OpenCASCADE shape
        if isinstance(result_obj, cq.Workplane) and len(result_obj.vals()) != 1:
            return False, "Expected exactly one result on the Workplane stack.", {}
        shape = result_obj.val() if hasattr(result_obj, "val") else result_obj
        wrapped = getattr(shape, "wrapped", None)
        if wrapped is None:
            return False, "Extracted object has no valid OCC wrapped shape.", {}

        # 1. Topological validity check
        try:
            analyzer = BRepCheck_Analyzer(wrapped)
            if not analyzer.IsValid():
                return False, "Topological Analyzer failed: shape is invalid or degenerate.", {}
        except Exception as e:
            return False, f"Topology analyzer error: {str(e)}", {}

        # 2. Volume & Watertight check
        try:
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(wrapped, props)
            volume = float(props.Mass())
            if volume <= 0.001:
                return False, f"Non-positive volume: {volume:.4f} mm³.", {}
        except Exception as e:
            return False, f"Volume computation error: {str(e)}", {}

        # 3. Single coherent solid check
        try:
            solids = shape.Solids()
            if len(solids) != 1:
                return False, f"Solid count violation: expected exactly 1 solid, got {len(solids)}.", {}
            if not all(shell.Closed() for shell in solids[0].Shells()):
                return False, "Solid contains an open shell.", {}
        except Exception as e:
            return False, f"Solid count check error: {str(e)}", {}

        # 4. Origin locking check (center_bottom -> abs(zmin) < 1e-3)
        try:
            bb = shape.BoundingBox()
            if abs(bb.zmin) > 1e-3:
                return False, f"Origin center_bottom violated: zmin = {bb.zmin:.5f} (must be 0.0).", {}
            bbox_dims = [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)]
        except Exception as e:
            return False, f"Bounding box extraction error: {str(e)}", {}

        metrics = {
            "volume_mm3": round(volume, 2),
            "bbox_dimensions": bbox_dims,
            "zmin": round(bb.zmin, 4),
            "is_watertight": True,
            "solid_count": 1,
        }
        return True, "OK", metrics


# ------------------------------------------------------------------------------
# 4. PIPELINE ORCHESTRATION & EXPORT
# ------------------------------------------------------------------------------

GENERATORS_GROUP_A = [
    gen_shelled_enclosure,
    gen_loft_adapter,
    gen_curved_sweep_pipe,
    gen_gusset_bracket,
    gen_asymmetric_flange,
]

GENERATORS_GROUP_B = [
    gen_stepped_shaft_dual_keyway,
    gen_double_stepped_bearing_housing,
    gen_multi_pattern_fixture_plate,
]


def generate_synthetic_dataset(
    count: int = 2000,
    output_path: Union[str, Path] = "training/dataset_sft.jsonl",
    stats_path: Union[str, Path] = "training/dataset_stats.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates synthetic dataset with strict 60/40 group distribution and OCC quality verification."""
    rng = random.Random(seed)
    if count <= 0:
        raise ValueError("count must be positive")
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file = Path(stats_path).resolve()
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    target_group_a = int(count * 0.60)
    target_group_b = count - target_group_a

    start_time = time.perf_counter()
    accepted_count = 0
    type_counts: Dict[str, int] = {}
    group_counts: Dict[str, int] = {"freeform_non_standard_group_a": 0, "complex_parametric_group_b": 0}
    rejections: Dict[str, int] = {}
    total_attempts = 0

    print(f"\n[+] Starting Synthetic CAD Dataset Generation (Target: {count} samples)...")
    print(f"    - Target Group A (Freeform/Non-standard, 60%): {target_group_a} samples")
    print(f"    - Target Group B (Complex Parametric, 40%):    {target_group_b} samples")
    print(f"    - Output JSONL: {out_file}")
    print(f"    - Validation: OpenCASCADE BRepCheck + Watertight + Single Solid + Origin Lock")

    with TemporaryDirectory(dir=out_file.parent) as stage:
        staged_output = Path(stage) / out_file.name
        with staged_output.open("w", encoding="utf-8") as jsonl_f:
            while accepted_count < count and total_attempts < max(100, 30 * count):
                total_attempts += 1

                # Decide which group needs more samples
                count_a = group_counts["freeform_non_standard_group_a"]
                count_b = group_counts["complex_parametric_group_b"]

                if count_a < target_group_a and count_b < target_group_b:
                    # Weighted pick 60/40
                    pick_group_a = rng.random() < 0.60
                elif count_a < target_group_a:
                    pick_group_a = True
                else:
                    pick_group_a = False

                if pick_group_a:
                    gen_func = rng.choice(GENERATORS_GROUP_A)
                else:
                    gen_func = rng.choice(GENERATORS_GROUP_B)

                prompt, code, meta = gen_func(rng)

                # Strict quality gate
                is_valid, reason, metrics = QualityVerifier.verify_code(code)
                if not is_valid:
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue

                # Format in standard ChatML
                chatml_sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
                    ],
                    "metadata": {
                        "sample_id": f"cad_{accepted_count + 1:05d}",
                        "part_type": meta["type"],
                        "group": meta["group"],
                        "metrics": metrics,
                    },
                }

                jsonl_f.write(json.dumps(chatml_sample) + "\n")
                accepted_count += 1
                t_name = meta["type"]
                g_name = meta["group"]
                type_counts[t_name] = type_counts.get(t_name, 0) + 1
                group_counts[g_name] = group_counts.get(g_name, 0) + 1

                if accepted_count % 10 == 0 or accepted_count == count:
                    elapsed = time.perf_counter() - start_time
                    rate = accepted_count / max(0.1, elapsed)
                    sys.stdout.write(
                        f"\r    [Progress] {accepted_count}/{count} generated "
                        f"({rate:.1f} samples/s, pass rate: {accepted_count/total_attempts*100:.1f}%)"
                    )
                    sys.stdout.flush()
        if accepted_count != count:
            raise RuntimeError(f"Accepted {accepted_count}/{count} in {total_attempts} attempts: {rejections}")
        staged_output.replace(out_file)

    total_time = round(time.perf_counter() - start_time, 2)
    pass_rate = round((accepted_count / total_attempts) * 100, 2)
    stats = {
        "target_count": count,
        "accepted_count": accepted_count,
        "total_attempts": total_attempts,
        "pass_rate_pct": pass_rate,
        "elapsed_seconds": total_time,
        "samples_per_sec": round(count / max(0.1, total_time), 2),
        "group_distribution": group_counts,
        "type_distribution": type_counts,
        "rejection_reasons": rejections,
    }
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n[OK] Generated {accepted_count} samples in {total_time}s. Stats: {stats_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Synthetic CAD Dataset Generator with OCC Quality Filter")
    parser.add_argument("--count", type=int, default=2000, help="Target number of validated CAD samples (default: 2000)")
    parser.add_argument("--output", type=str, default="training/dataset_sft.jsonl", help="Output JSONL dataset path")
    parser.add_argument("--stats", type=str, default="training/dataset_stats.json", help="Output stats JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")

    args = parser.parse_args()
    generate_synthetic_dataset(
        count=args.count,
        output_path=args.output,
        stats_path=args.stats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
