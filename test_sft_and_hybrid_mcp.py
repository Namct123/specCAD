"""
Smoke test suite for Hybrid CAD Pipeline & SFT Model Fixes:
1. Opt-in live Ollama check for the nose cone.
2. Tests SpecCompiler sequential batch execution (0 race conditions).
3. Tests run_hybrid_batch MCP method across all 3 tiers (compiler_cpu, local_sft, cloud_direct).
4. Tests assemble_and_verify on the B2 liner STEP with relative volume tolerance.
"""

import json
import os
from io import BytesIO
from math import pi
import time
import urllib.request
from pathlib import Path

import pytest

import mcp_server.server as server
from mcp_server.server import CADService, cad_service
from mcp_server.spec_compiler import SpecCompiler
from mcp_server.cad_runner import HeadlessCADRunner

@pytest.mark.skipif(os.getenv("CAD_RUN_OLLAMA_TESTS") != "1", reason="Requires a local Ollama model")
def test_ollama_sft_generation(tmp_path):
    res = CADService(tmp_path).run_hybrid_batch(
        [{
            "name": "01_intake_nose_cone",
            "tier": "local_sft",
            "geometry_kind": "cone",
            "expected_volume": 16210.618,
            "expected_bbox_mm": [36, 36, 40],
            "expected_end_radii_mm": [18, 3],
            "expected_z_range": [0, 40],
            "prompt": "Solid circular taper centered on Z, base radius 18 mm, top radius 3 mm, height 40 mm.",
        }],
        max_sft_retries=0,
    )
    assert res["status"] == "SUCCESS", res
    assert Path(res["passed_parts"][0]["step_path"]).is_file()


def test_compiler_sequential_batch():
    print("\n" + "=" * 60)
    print("TEST 2: SpecCompiler Sequential Batch (Box, Shaft, Tube)")
    print("=" * 60)

    specs = [
        {
            "name": "test_plate",
            "origin": "center_bottom",
            "base": {"shape": "box", "dims": [60.0, 40.0, 10.0], "fillet_z": 2.0},
            "features": [{"op": "hole", "d": 6.0, "pt": [0.0, 0.0]}],
        },
        {
            "name": "test_shaft",
            "origin": "center_bottom",
            "base": {
                "shape": "stepped_shaft",
                "sections": [
                    {"d": 30.0, "h": 20.0},
                    {"d": 20.0, "h": 40.0},
                ],
            },
            "features": [{"op": "keyway", "w": 4.0, "l": 15.0, "d": 2.0, "z": 5.0}],
        },
        {
            "name": "06_outer_casing",
            "origin": "center_bottom",
            "base": {"shape": "tube", "od": 116.0, "id": 110.0, "h": 160.0},
        },
    ]

    t0 = time.perf_counter()
    batch_res = SpecCompiler.compile_batch(specs, output_dir="workspace/test_parts", export_stl=False)
    elapsed = round(time.perf_counter() - t0, 3)

    print(f"Batch compiled 3 parts in {elapsed}s (status={batch_res['status']})")
    assert batch_res["status"] == "SUCCESS", f"Batch compilation failed: {batch_res}"
    assert len(batch_res["parts"]) == 3
    casing = next(p for p in batch_res["parts"] if p["part_name"] == "06_outer_casing")
    print(f"06_outer_casing: bbox={casing['bbox_mm']}, volume={casing['volume_mm3']} mm³")
    assert abs(casing["volume_mm3"] - 170400.0) < 1.0, f"Expected 170400 mm³, got {casing['volume_mm3']}"
    print("TEST 2 PASSED: 06_outer_casing tube compiled via CPU in < 0.05s with 100% exact volume!")

def test_run_hybrid_batch_3_tiers(tmp_path, monkeypatch):
    print("\n" + "=" * 60)
    print("TEST 3: run_hybrid_batch (3 Tiers with 06_outer_casing in CPU)")
    print("=" * 60)

    parts_batch = [
        # Tier 1: Compiler CPU (Shaft + 06_outer_casing tube)
        {
            "name": "03_front_bearing_housing",
            "tier": "compiler_cpu",
            "origin": "center_bottom",
            "base": {
                "shape": "stepped_shaft",
                "sections": [
                    {"d": 50.0, "h": 10.0},
                    {"d": 40.0, "h": 15.0},
                ],
            },
            "features": [{"op": "hole", "d": 16.0, "pt": [0.0, 0.0]}],
        },
        {
            "name": "06_outer_casing",
            "tier": "local_sft",
            "origin": "center_bottom",
            "base": {"shape": "tube", "od": 116.0, "id": 110.0, "h": 160.0},
        },
        # Tier 2: Local SFT (Single solid / one primary feature: nose cone)
        {
            "name": "01_intake_nose_cone",
            "tier": "local_sft",
            "geometry_kind": "cone",
            "expected_volume": pi * 40 / 3 * (18**2 + 18 * 3 + 3**2),
            "expected_bbox_mm": [36, 36, 40],
            "origin": "center_bottom",
            "prompt": "Design a smooth loft transition adapter: bottom circular cross-section (radius 18.0mm) at Z=0, lofting smoothly to a circular top cross-section (radius 3.0mm) at height 40.0mm. Solid cone. Origin must be center_bottom.",
        },
        # Tier 3: Cloud Direct (Skip-Local for complex loft assembly & BSpline impeller)
        {
            "name": "02_intake_bellmouth",
            "tier": "local_sft",
            "prompt": "A large prompt " * 500,
        },
        {
            "name": "04_compressor_impeller",
            "tier": "cloud_direct",
            "reason": "Complex 8-blade curved BSpline geometry (Skip-Local)",
        },
    ]

    requests = []

    def reply(request, *, timeout):
        requests.append(json.loads(request.data))
        recipe = json.dumps({"bottom_radius": 18, "top_radius": 3, "height": 40})
        return BytesIO(json.dumps({"message": {"content": recipe}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", reply)
    monkeypatch.setattr(server, "cad_service", CADService(tmp_path))
    t0 = time.perf_counter()
    output_dir = tmp_path / "custom_parts"
    body = server.run_hybrid_batch(parts_batch, output_dir=str(output_dir))
    assert "\n" not in body
    res = json.loads(body)
    elapsed = round(time.perf_counter() - t0, 2)

    print(f"run_hybrid_batch result: status={res['status']}, elapsed={elapsed}s")
    print(f"Summary: {json.dumps(res['summary'], indent=2)}")
    print(f"Passed parts ({len(res['passed_parts'])}): {[p['name'] for p in res['passed_parts']]}")
    print(f"Escalated parts ({len(res['escalated_parts'])}): {[e['name'] for e in res['escalated_parts']]}")

    assert res["summary"]["total_parts"] == 5
    assert res["summary"]["passed_count"] == 3
    assert res["summary"]["escalated_count"] == 2
    # Verify 06_outer_casing compiled via CPU with exact volume
    casing = next(p for p in res["passed_parts"] if p["name"] == "06_outer_casing")
    assert casing["tier"] == "compiler_cpu"
    assert len(requests) == 1 and requests[0]["model"] == "cad-worker-sft"
    assert requests[0]["options"]["num_predict"] == 96
    assert set(requests[0]["format"]["required"]) == {"bottom_radius", "top_radius", "height"}
    assert all(Path(part["step_path"]).parent == output_dir for part in res["passed_parts"])
    assert all(Path(part["step_path"]).is_file() for part in res["passed_parts"])
    assert res["escalated_parts"][0]["reason"] == "outside_sft_scope"
    assert "A large prompt" not in json.dumps(res)
    assert abs(casing["volume_mm3"] - 170400.0) < 1.0, f"Expected 170400 mm³, got {casing['volume_mm3']}"
    print("TEST 3 PASSED: run_hybrid_batch executed seamlessly across 3 tiers (06_outer_casing in CPU < 0.05s)!")

def test_sft_rejects_material_in_expected_void(tmp_path, monkeypatch):
    service = CADService(tmp_path)
    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10, centered=(True, True, False))"
    requests = []

    def reply(request, *, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"message": {"content": code}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", reply)
    spec = {
        "name": "enclosure",
        "tier": "local_sft",
        "geometry_kind": "shelled_enclosure",
        "prompt": "Build a hollow enclosure.",
        "expected_volume": 1000,
        "expected_void_boxes": [[-2, -2, 2, 2, 2, 8]],
    }
    result = service.run_hybrid_batch([spec], max_sft_retries=8)
    assert result["escalated_parts"] == [
        {"name": "enclosure", "tier": "local_sft", "reason": "geometry_rejected"}
    ]
    assert len(requests) == 2
    assert "contains solid material" in requests[-1]["messages"][-1]["content"]
    assert not (service.parts_dir / "enclosure.step").exists()

    spec["expected_volume"] = 500
    result = service.run_hybrid_batch([spec], max_sft_retries=0)
    assert result["escalated_parts"][0]["reason"] == "geometry_rejected"
    assert len(requests) == 3

    valid_code = code + "\nresult = result.cut(cq.Workplane('XY').box(6, 6, 8).translate((0, 0, 5)))"
    accepted = service.runner.execute_part_code(
        valid_code, "hollow", use_subprocess=False, export_formats=("step",),
        geometry_contract={"expected_volume": 712, "expected_void_boxes": spec["expected_void_boxes"]},
    )
    assert accepted.success, accepted.error_message


def test_assembly_relative_volume_tolerance(tmp_path):
    print("\n" + "=" * 60)
    print("TEST 4: Assembly Collision Check & 0.1% Relative Volume Tolerance")
    print("=" * 60)

    assy_code = (
        "import cadquery as cq\n"
        "result = cq.Assembly()\n"
        "result.add(cq.Workplane('XY').box(100, 100, 6.80596), name='coupon')"
    )
    expected_vol = 68059.010  # Independent reference with a sub-mm³ deviation.

    service = CADService(tmp_path)
    res = service.assemble_and_verify(
        assembly_code=assy_code,
        assembly_name="test_assembly",
        check_collisions=True,
        expected_volume=expected_vol,
        volume_tolerance_pct=0.1,
    )
    print(f"assemble_and_verify result: status={res.get('status')}")
    if "volume_check" in res:
        print(f"Volume check: {json.dumps(res['volume_check'], indent=2)}")
    assert res.get("status") in ("SUCCESS", "COLLISION_FREE", "OK")
    assert res.get("volume_check", {}).get("passed") is True, f"Volume check failed with relative 0.1% tolerance: {res}"
    assert res["volume_check"]["diff_mm3"] > 0.15
    assert all(Path(res["outputs"][ext]).is_file() for ext in ("step", "glb"))
    failed = service.assemble_and_verify(assy_code, "wrong_volume", expected_volume=expected_vol + 100)
    assert failed["status"] == "FAILED"
    assert failed["volume_check"]["passed"] is False
    assert not (service.assembly_dir / "wrong_volume.step").exists()


def test_assembly_script_comprehension_sees_top_level_imports(tmp_path):
    result = CADService(tmp_path).assemble_and_verify(
        assembly_script=(
            "import math\n"
            "heights = [math.sqrt(n) for n in (4,)]\n"
            "result = cq.Assembly(name='assembly')\n"
            "result.add(cq.Workplane('XY').box(2, 3, heights[0]), name='box')\n"
        ),
        export_name="assembly", expected_volume=12,
    )
    assert result["status"] == "SUCCESS", result


def test_assembly_plan_rejects_wrong_global_z(tmp_path):
    from mcp_server.assembly_plan import build_planned_assembly

    plan = {
        "export_name": "mount",
        "placements": {
            "foot": {"translation": [0, 0, 25], "rotation_axis": [1, 0, 0],
                     "rotation_deg": 90, "expected_global_z": [25, 65]},
        },
        "cloud_parts_script": 'cloud_parts = {"foot": cq.Solid.makeBox(180, 40, 25, cq.Vector(-90, -20, 0))}',
    }
    with pytest.raises(ValueError, match=r"foot: placed Z \[5.000, 45.000\]"):
        build_planned_assembly(plan, [{"name": "foot"}], {}, tmp_path)
    assert not (tmp_path / "foot.step").exists()


def test_assembly_plan_requests_missing_cloud_geometry(tmp_path):
    service = CADService(tmp_path)
    result = service.run_assembly_plan(
        [{"name": "liner", "tier": "cloud_direct"}],
        {"export_name": "pump", "placements": {"liner": {"translation": [0, 0, 0]}}},
    )
    assert result["status"] == "ESCALATE"
    assert result["escalated_parts"] == [{"name": "liner", "tier": "cloud_direct", "reason": "skip_local"}]


def test_cone_recipe_rejects_reversed_ends(tmp_path, monkeypatch):
    service = CADService(tmp_path)
    recipe = json.dumps({"bottom_radius": 3, "top_radius": 18, "height": 40})
    answer = json.dumps({"message": {"content": recipe}}).encode()
    monkeypatch.setenv("CAD_OLLAMA_RECIPE_MODEL", "recipe-model")

    def reply(request, *, timeout):
        assert json.loads(request.data)["model"] == "recipe-model"
        return BytesIO(answer)

    monkeypatch.setattr(urllib.request, "urlopen", reply)
    result = service.run_hybrid_batch([{
        "name": "nose", "tier": "local_sft", "geometry_kind": "cone",
        "prompt": "Bottom radius 18 mm, top radius 3 mm, height 40 mm.",
        "expected_end_radii_mm": [18, 3], "expected_volume": 16210.618,
    }], max_sft_retries=0)
    assert result["escalated_parts"][0]["reason"] == "geometry_rejected"
    assert not (service.parts_dir / "nose.step").exists()


def test_assembly_plan_reuses_one_geometry_for_multiple_instances(tmp_path):
    service = CADService(tmp_path)
    result = service.run_assembly_plan(
        [{"name": "foot", "base": {"shape": "box", "dims": [10, 10, 10]}}],
        {
            "export_name": "mount",
            "placements": {
                "left": {"part": "foot", "translation": [0, 0, 0]},
                "right": {"part": "foot", "translation": [20, 0, 0]},
            },
            "expected_volume": 2000,
        },
    )
    assert result["status"] == "SUCCESS", result
    assert result["batch"]["passed_count"] == 1
    assert result["instance_count"] == result["solid_count"] == 2
    assert result["interferences"]["pairs_aabb_pruned"] == 1
    print("TEST 4 PASSED: Assembly verified with 0.0 mm³ collision and relative 0.1% volume tolerance passed!")

def test_numeric_cone_bypasses_ollama(tmp_path, monkeypatch):
    service = CADService(tmp_path)

    def unexpected_ollama(*args, **kwargs):
        pytest.fail("Numeric cone called Ollama")

    monkeypatch.setattr(service, "generate_freeform_local", unexpected_ollama)
    result = service.run_hybrid_batch([{
        "name": "nose", "tier": "local_sft",
        "base": {"shape": "cone", "bottom_radius": 18, "top_radius": 3, "height": 40},
        "expected_volume": 16210.618,
    }])
    assert result["status"] == "SUCCESS", result
    assert result["passed_parts"][0]["tier"] == "compiler_cpu"
    spec = {
        "name": "wrong_ends", "tier": "compiler_cpu",
        "base": {"shape": "cone", "bottom_radius": 3, "top_radius": 18, "height": 40},
        "expected_end_radii_mm": [18, 3],
    }
    rejected = service.run_hybrid_batch([spec])
    assert rejected["escalated_parts"][0]["reason"] == "compiler_failed"


def test_base_kind_preserves_hollow_geometry(tmp_path):
    service = CADService(tmp_path)
    spec = {"name": "collar", "base": {"kind": "tube", "od": 18, "id": 10, "h": 12}}
    result = service.run_hybrid_batch([spec])
    assert result["status"] == "SUCCESS", result
    assert result["passed_parts"][0]["volume_mm3"] == pytest.approx(pi * (9**2 - 5**2) * 12, abs=0.01)

    spec["base"]["shape"] = "box"
    rejected = service.run_hybrid_batch([spec], output_dir=str(tmp_path / "conflict"))
    assert rejected["escalated_parts"][0]["reason"] == "compiler_failed"
    assert not (tmp_path / "conflict" / "collar.step").exists()


def test_raw_cone_uses_small_json_model(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD_OLLAMA_RECIPE_MODEL", raising=False)

    def reply(request, *, timeout):
        payload = json.loads(request.data)
        assert request.full_url == "http://127.0.0.1:11434/api/chat"
        assert payload["model"] == "qwen2.5:1.5b"
        assert payload["messages"][-1]["content"] == "Nón Ø36/Ø6, cao 40 mm."
        assert payload["format"]["required"] == ["bottom_radius", "top_radius", "height"]
        return BytesIO(json.dumps({"message": {"content": json.dumps({
            "bottom_radius": 18, "top_radius": 3, "height": 40,
        })}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", reply)
    result = CADService(tmp_path).run_hybrid_batch([{
        "name": "nose", "tier": "local_sft", "geometry_kind": "cone",
        "raw_text": "Nón Ø36/Ø6, cao 40 mm.",
    }], max_sft_retries=0)
    assert result["status"] == "SUCCESS", result
    assert result["passed_parts"][0]["volume_mm3"] == pytest.approx(16210.618, abs=0.01)


@pytest.mark.parametrize(("raw_text", "radii", "height"), [
    ("Nón cụt, đường kính đáy 24 mm, đường kính đỉnh 12 mm, cao 18 mm.", [12, 6], 18),
    ("Circular frustum, bottom diameter 22 mm, top diameter 10 mm, height 20 mm.", [11, 5], 20),
    ("Nón tròn nhọn, bán kính đáy 10 mm, bán kính đỉnh 0 mm, cao 22 mm.", [10, 0], 22),
    ("Nón cụt; đường kính đáy 25 mm, bán kính đỉnh 3,5 mm, cao 12,5 mm.", [12.5, 3.5], 12.5),
])
def test_labeled_cone_bom_compiles_without_ollama(tmp_path, monkeypatch, raw_text, radii, height):
    def unexpected_ollama(*args, **kwargs):
        pytest.fail("Labeled BOM called Ollama")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_ollama)
    bottom, top = radii
    result = CADService(tmp_path).run_hybrid_batch([{
        "name": "locator", "tier": "local_sft", "geometry_kind": "cone",
        "raw_text": raw_text, "expected_end_radii_mm": radii, "expected_z_range": [0, height],
        "expected_volume": pi * height * (bottom**2 + bottom*top + top**2) / 3,
    }], max_sft_retries=0)
    assert result["status"] == "SUCCESS", result
    assert result["passed_parts"][0]["tier"] == "compiler_cpu"


def test_cone_bom_leaves_unrecognized_text_to_ollama():
    assert server.parse_cone_bom("Nón cụt, đường kính đáy 24 mm, đường kính đỉnh 12 mm.") is None
    assert server.parse_cone_bom("Circular frustum, bottom diameter 22 cm, top diameter 10 mm, height 20 mm.") is None
    assert server.parse_cone_bom("Circular cone, bottom radius 10 mm, top radius 0 mm, height 22 mm; add a hole") is None


def test_raw_cone_with_missing_height_is_rejected(tmp_path, monkeypatch):
    answer = {"message": {"content": '{"bottom_radius":18,"top_radius":3,"height":null}'}}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: BytesIO(json.dumps(answer).encode()))
    result = CADService(tmp_path).run_hybrid_batch([{
        "name": "nose", "tier": "local_sft", "geometry_kind": "cone",
        "raw_text": "A frustum with bottom radius 18 mm and top radius 3 mm.",
    }], max_sft_retries=0)
    assert result["escalated_parts"][0]["reason"] == "geometry_rejected"


def test_circular_placements_reject_duplicate_and_nonfinite():
    from mcp_server.server import expand_placements

    patterns = {"pin": {"part": "pin", "pcd": 100, "count": 4,
                         "center": [10, 20, -4], "start_angle": 90}}
    placements = expand_placements({"circular_patterns": patterns})
    assert placements["pin_001"]["translation"] == pytest.approx([10, 70, -4])
    assert placements["pin_002"]["translation"] == pytest.approx([-40, 20, -4])
    with pytest.raises(ValueError, match="Duplicate placement name"):
        expand_placements({"placements": {"pin_001": {}}, "circular_patterns": patterns})
    patterns["pin"]["pcd"] = float("inf")
    with pytest.raises(ValueError, match="positive pcd"):
        expand_placements({"circular_patterns": patterns})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
