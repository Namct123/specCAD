"""
Test Audit Fixes for Hybrid CAD Agent
Verifies 4 findings:
1. P1: Chi tiết sai spec không được báo thành công (Spec compliance verification)
2. P1: Kiểm va chạm không kết luận sạch sai (Nested assembly & robust collision detection)
3. P1: Mã worker chạy trong tiến trình con cô lập (Subprocess isolation protects MCP server)
4. P2: Fallback không biến lỗ mù thành lỗ xuyên (Blind hole preserved with depth)
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mcp_server.cad_runner import HeadlessCADRunner
from mcp_server.server import CADService
import cadquery as cq


def test_p2_blind_hole_fallback():
    print("\n--- Test P2: Fallback Blind Hole vs Through Hole ---")
    spec_blind = {
        "name": "test_blind_part",
        "origin": "center_bottom",
        "base": {"shape": "box", "dims": [60, 60, 30]},
        "features": [
            {"op": "hole", "d": 20.0, "depth": 12.0, "at": [0, 0]},
            {"op": "hole_rect", "d": 4.0, "dx": 40.0, "dy": 40.0, "depth": 8.0},
            {"op": "hole", "d": 10.0, "thru": True, "at": [15, 15]}
        ]
    }
    code = CADService.compile_compact_spec_to_cq(spec_blind)
    print("Generated Fallback Code:\n", code)

    # Assertions
    assert "hole(20.0, 12.0)" in code, "Blind center hole must include depth 12.0!"
    assert "hole(4.0, 8.0)" in code, "Blind rectangular hole array must include depth 8.0!"
    assert "hole(10.0)" in code, "Explicit thru hole must remain thru hole!"
    print("[OK] P2 Fixed: Blind holes retain their depth and are not converted to through-holes.")


def test_p1_worker_subprocess_isolation():
    print("\n--- Test P1: Worker Subprocess Isolation ---")
    runner = HeadlessCADRunner()

    # 1. Test that sys.exit() in script does NOT kill the parent process
    suicide_code = """
import sys
sys.exit(42)
"""
    res = runner.execute_part_code(suicide_code, "test_suicide", use_subprocess=True)
    print("Suicide code result: success =", res.success, "| error_type =", res.error_type)
    assert res.success is False, "sys.exit(42) must result in execution failure!"
    assert res.error_type in ("WorkerSubprocessError", "SubprocessCrashError"), f"Unexpected error type: {res.error_type}"

    # 2. Test valid execution in subprocess
    valid_code = """
import cadquery as cq
result = cq.Workplane('XY').box(40, 40, 20, centered=(True, True, False))
"""
    res_valid = runner.execute_part_code(valid_code, "test_valid_subproc", expected_origin="center_bottom", use_subprocess=True)
    print("Valid subproc result: success =", res_valid.success, "| volume =", res_valid.metrics.volume if res_valid.metrics else None)
    assert res_valid.success is True, "Valid code must succeed in subprocess!"
    assert abs(res_valid.metrics.bbox_min[2]) < 0.05, "Origin must be preserved in subprocess result!"
    print("[OK] P1 Fixed: Worker code executes in isolated subprocess; MCP server process is protected.")


def test_p1_spec_compliance_validation():
    print("\n--- Test P1: Spec Compliance Validation ---")
    spec = {
        "name": "gear_bracket",
        "origin": "center_bottom",
        "base": {"shape": "box", "dims": [100.0, 50.0, 20.0]},
        "features": []
    }

    # Case A: Code generates wrong dimensions (50x50x10 instead of 100x50x20)
    wrong_dims_code = """
import cadquery as cq
result = cq.Workplane('XY').box(50.0, 50.0, 10.0, centered=(True, True, False))
"""
    runner = HeadlessCADRunner()
    exec_res = runner.execute_part_code(wrong_dims_code, "test_wrong_dims", expected_origin="center_bottom", use_subprocess=True)
    is_valid, errors = CADService.validate_part_spec_compliance(spec, exec_res.to_dict())
    print("Wrong dimensions check: is_valid =", is_valid, "| errors =", errors)
    assert is_valid is False, "Wrong bounding dimensions must fail spec compliance!"
    assert any("Dimension X mismatch" in e for e in errors), "Must flag Dimension X mismatch!"
    assert any("Dimension Z mismatch" in e for e in errors), "Must flag Dimension Z mismatch!"

    # Case B: Code generates correct dimensions
    correct_code = """
import cadquery as cq
result = cq.Workplane('XY').box(100.0, 50.0, 20.0, centered=(True, True, False))
"""
    exec_res_ok = runner.execute_part_code(correct_code, "test_correct_dims", expected_origin="center_bottom", use_subprocess=True)
    is_valid_ok, errors_ok = CADService.validate_part_spec_compliance(spec, exec_res_ok.to_dict())
    print("Correct dimensions check: is_valid =", is_valid_ok, "| errors =", errors_ok)
    assert is_valid_ok is True, "Correct dimensions must pass spec compliance!"
    assert len(errors_ok) == 0
    print("[OK] P1 Fixed: Parts deviating from spec dimensions are strictly rejected.")


def test_p1_assembly_collision_detection():
    print("\n--- Test P1: Assembly Collision Detection ---")
    # 1. Flat assembly with actual collision (overlapping boxes)
    assy_collision = cq.Assembly()
    b1 = cq.Workplane("XY").box(20, 20, 20)
    b2 = cq.Workplane("XY").box(20, 20, 20)
    assy_collision.add(b1, name="part_1")
    assy_collision.add(b2, name="part_2", loc=cq.Location(cq.Vector(10, 0, 0))) # Overlaps 10x20x20 = 4000 mm3

    report_col = HeadlessCADRunner.check_assembly_interferences(assy_collision)
    print("Flat collision report:", report_col)
    assert report_col["has_collisions"] is True, "Must detect collision between overlapping boxes!"
    assert report_col["total_interference_volume_mm3"] > 3000, f"Expected ~4000 mm3 overlap, got {report_col['total_interference_volume_mm3']}"

    # 2. Nested sub-assembly collision
    assy_nested = cq.Assembly()
    sub_assy = cq.Assembly(name="sub_assembly")
    b3 = cq.Workplane("XY").box(20, 20, 20)
    sub_assy.add(b3, name="nested_child", loc=cq.Location(cq.Vector(5, 0, 0)))
    assy_nested.add(b1, name="root_box")
    assy_nested.add(sub_assy, name="sub_assembly", loc=cq.Location(cq.Vector(0, 0, 5)))

    report_nested = HeadlessCADRunner.check_assembly_interferences(assy_nested)
    print("Nested collision report:", report_nested)
    assert report_nested["has_collisions"] is True, "Must detect collision in nested sub-assembly hierarchy!"
    assert len(report_nested["collision_pairs"]) >= 1

    body_with_child = cq.Assembly(b1, name="body")
    body_with_child.add(b2, name="child", loc=cq.Location(cq.Vector(10, 0, 0)))
    parent = cq.Assembly()
    parent.add(body_with_child)
    assert HeadlessCADRunner.check_assembly_interferences(parent)["has_collisions"]

    compound = cq.Compound.makeCompound([b1.val(), b2.val().translate((10, 0, 0))])
    assert HeadlessCADRunner.check_assembly_interferences(compound)["has_collisions"]
    multi_root = cq.Workplane("XY").newObject(compound.Solids())
    assert HeadlessCADRunner.compute_metrics(multi_root).solid_count == 2
    assert HeadlessCADRunner.check_assembly_interferences(multi_root)["has_collisions"]

    # 3. Clean separated assembly
    assy_clean = cq.Assembly()
    assy_clean.add(b1, name="box_a")
    assy_clean.add(b2, name="box_b", loc=cq.Location(cq.Vector(100, 0, 0)))
    report_clean = HeadlessCADRunner.check_assembly_interferences(assy_clean)
    print("Clean separation report:", report_clean)
    assert report_clean["has_collisions"] is False, "Separated components must be clean!"
    assert report_clean["verification_complete"] is True
    assert report_clean["total_interference_volume_mm3"] == 0.0

    print("[OK] P1 Fixed: Assembly collision checker reliably handles flat & nested assemblies without false negatives.")


def test_complex_collision_pair_remains_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr(cq.Compound, "Faces", lambda self: [None] * 39)
    code = """
body = cq.Workplane("XY").box(20, 20, 20)
result = cq.Assembly()
result.add(body, name="a")
result.add(body, name="b", loc=cq.Location(cq.Vector(10, 0, 0)))
"""
    result = HeadlessCADRunner().execute_assembly_code(
        code, "guarded", output_dir=tmp_path, use_subprocess=False,
    )
    assert result.success, result.error_message
    meta = json.loads(result.artifacts["structured_metadata"])
    assert meta["status"] == "VERIFICATION_INCOMPLETE"
    assert meta["interferences"]["pairs_boolean_checked"] == 0
    assert meta["interferences"]["unverified_pairs"] == [
        {"part_a": "a", "part_b": "b", "reason": "face_complexity"}
    ]


def test_export_failures_do_not_publish_partial_assembly(tmp_path, monkeypatch):
    import trimesh

    def fail_glb(*args, **kwargs):
        raise RuntimeError("mesh failure")

    monkeypatch.setattr(trimesh, "load", fail_glb)
    runner = HeadlessCADRunner(tmp_path)
    code = 'result = cq.Workplane("XY").box(10, 10, 10)'
    result = runner.execute_assembly_code(code, "case", output_dir=tmp_path, use_subprocess=False)
    assert not result.success
    assert "mesh failure" in result.error_message
    assert not list(tmp_path.iterdir())

    result = runner.execute_part_code(code, "wrong_origin", expected_origin="center_bottom", use_subprocess=False)
    assert not result.success
    assert result.error_type == "OriginMismatchError"
    assert not (tmp_path / "wrong_origin.step").exists()


if __name__ == "__main__":
    test_p2_blind_hole_fallback()
    test_p1_worker_subprocess_isolation()
    test_p1_spec_compliance_validation()
    test_p1_assembly_collision_detection()
    print("\n========================================================")
    print("ALL 4 AUDIT FIXES VERIFIED SUCCESSFULLY!")
    print("========================================================")
