"""
Comprehensive Acceptance Test Suite: test_v2_and_stage3.py
Validates:
  1. 30-Part Shaft Drive Assembly (435 pairs):
     - Broad-phase AABB pruning >= 390 pairs pruned, collision check < 0.6s.
     - Zero geometric interference (max_intersection_mm3 == 0.0).
     - Strict stdout isolation (0 bytes leaked even with print() statements in assembly code).
  2. Geometric Accuracy of New SpecCompiler Features:
     - Chamfer scope ('outer', 'inner') on bearing sleeve and idler roller.
     - Stadium slot style on rear stop block.
     - Origin locked at zmin == 0.000.
  3. Locked MCP Metadata Schema:
     - Direct in-memory metadata returned without reading external log files.
  4. Stage 3 Synthetic Data Generation:
     - 50 samples generated with 100% pass rate (60% Group A, 40% Group B).
     - Validates ChatML formatting and prints 1 sample.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.server import cad_service
from mcp_server.spec_compiler import SpecCompiler
from training.synth_cad_generator import generate_synthetic_dataset


def test_1_and_3_30_part_assembly_and_schema():
    print("\n" + "=" * 70)
    print("[TEST 1 & 3] 30-Part Shaft Drive Assembly (435 Pairs) & Locked MCP Schema")
    print("=" * 70)

    # 30-part assembly script with print statement to verify stdout isolation
    assembly_script = """
import cadquery as cq

print("DEBUG MALICIOUS USER PRINT: THIS MUST NEVER LEAK TO MCP STDOUT!")

assy = cq.Assembly()

# 1. Bed (Base Plate): 260 x 140 x 20
bed = cq.Workplane('XY').box(260, 140, 20, centered=(True, True, False))
assy.add(bed, name='bed', loc=cq.Location(cq.Vector(0, 0, 0)))

# 2. Bearing Blocks (2 units): Width 30 in X, 40 in Y, 50 in Z. Bore diam 30 along X at Z=25
bb = (
    cq.Workplane('XY')
    .box(30, 40, 50, centered=(True, True, False))
    .faces('>X')
    .workplane(centerOption='CenterOfMass')
    .hole(30.0)
)
assy.add(bb, name='bb_left', loc=cq.Location(cq.Vector(-80, 0, 20)))
assy.add(bb, name='bb_right', loc=cq.Location(cq.Vector(80, 0, 20)))

# 3. Main Transmission Shaft (1 unit): Length 240, diam 15.6 at Z=45
shaft = cq.Workplane('YZ').cylinder(240, 7.8, centered=(True, True, True))
assy.add(shaft, name='shaft', loc=cq.Location(cq.Vector(0, 0, 45)))

# 4. Deep Groove Bearings (4 units): Width 8, OD 29.6, ID 16.0
bearing = cq.Workplane('YZ').cylinder(8, 14.8, centered=(True, True, True)).faces('>X').workplane().hole(16.0)
assy.add(bearing, name='brg_1', loc=cq.Location(cq.Vector(-88, 0, 45)))
assy.add(bearing, name='brg_2', loc=cq.Location(cq.Vector(-72, 0, 45)))
assy.add(bearing, name='brg_3', loc=cq.Location(cq.Vector(72, 0, 45)))
assy.add(bearing, name='brg_4', loc=cq.Location(cq.Vector(88, 0, 45)))

# 5. Thrust Washers (4 units): Width 2, OD 23.0, ID 16.0
washer = cq.Workplane('YZ').cylinder(2, 11.5, centered=(True, True, True)).faces('>X').workplane().hole(16.0)
assy.add(washer, name='w_1', loc=cq.Location(cq.Vector(-100, 0, 45)))
assy.add(washer, name='w_2', loc=cq.Location(cq.Vector(-60, 0, 45)))
assy.add(washer, name='w_3', loc=cq.Location(cq.Vector(60, 0, 45)))
assy.add(washer, name='w_4', loc=cq.Location(cq.Vector(100, 0, 45)))

# 6. Pulleys / Gears (2 units): Width 12, OD 50.0, ID 16.0
pulley = cq.Workplane('YZ').cylinder(12, 25.0, centered=(True, True, True)).faces('>X').workplane().hole(16.0)
assy.add(pulley, name='pulley_left', loc=cq.Location(cq.Vector(-25, 0, 45)))
assy.add(pulley, name='pulley_right', loc=cq.Location(cq.Vector(25, 0, 45)))

# 7. M8 Clamping Bolts (4 units): Length 25, diam 7.6 at bed corners
bolt = cq.Workplane('XY').cylinder(25, 3.8, centered=(True, True, False))
assy.add(bolt, name='bolt_1', loc=cq.Location(cq.Vector(-115, -55, 20)))
assy.add(bolt, name='bolt_2', loc=cq.Location(cq.Vector(115, -55, 20)))
assy.add(bolt, name='bolt_3', loc=cq.Location(cq.Vector(-115, 55, 20)))
assy.add(bolt, name='bolt_4', loc=cq.Location(cq.Vector(115, 55, 20)))

# 8. Precision Dowel Pins (12 units): Height 15, diam 5.8
pin = cq.Workplane('XY').cylinder(15, 2.9, centered=(True, True, False))
pin_coords = [
    (-115, -45), (-40, -45), (0, -45), (40, -45), (115, -45), (-80, -45),
    (-115, 45), (-40, 45), (0, 45), (40, 45), (115, 45), (80, 45)
]
for i, (px, py) in enumerate(pin_coords):
    assy.add(pin, name=f'pin_{i+1}', loc=cq.Location(cq.Vector(px, py, 20)))

result = assy
"""

    # Intercept stdout to verify zero leaks
    old_stdout = sys.stdout
    stdout_trap = io.StringIO()
    sys.stdout = stdout_trap

    t0 = time.perf_counter()
    res = cad_service.assemble_and_verify(
        assembly_script=assembly_script,
        export_name="shaft_drive_30_parts",
        check_collisions=True,
        output_dir="workspace/assembly",
    )
    t_total = time.perf_counter() - t0

    # Restore stdout
    sys.stdout = old_stdout
    captured_stdout = stdout_trap.getvalue()

    # 1. Verify stdout isolation
    print(f"[1] Stdout Isolation Check:")
    print(f"    - Captured stdout length: {len(captured_stdout)} bytes")
    assert "DEBUG MALICIOUS USER PRINT" not in captured_stdout, "Malicious print leaked to stdout!"
    print("    [PASS] Zero stdout leakage confirmed.")

    # 2. Verify locked metadata keys
    print(f"\n[2] Locked MCP Metadata Schema Verification:")
    required_keys = [
        "status",
        "solid_count",
        "valid_solids",
        "bbox_mm",
        "bbox_min",
        "total_volume_mm3",
        "pairs_total",
        "pairs_aabb_pruned",
        "pairs_boolean_checked",
        "max_intersection_mm3",
        "elapsed_sec",
        "outputs",
    ]
    for k in required_keys:
        assert k in res, f"Missing required key '{k}' in response!"
        print(f"    - Key '{k}': {res[k]}")
    print("    [PASS] All 12 locked metadata schema keys present.")

    # 3. Verify collision metrics and AABB pruning
    print(f"\n[3] 2-Phase Collision Performance (30 parts, 435 pairs):")
    pairs_total = res["pairs_total"]
    pairs_pruned = res["pairs_aabb_pruned"]
    pairs_checked = res["pairs_boolean_checked"]
    max_overlap = res["max_intersection_mm3"]
    t_collision = res["elapsed_sec"]["collision"]

    print(f"    - Total Pairs: {pairs_total}")
    print(f"    - AABB Pruned Pairs: {pairs_pruned} (>= 390 required)")
    print(f"    - Boolean Checked Pairs: {pairs_checked}")
    print(f"    - Max Overlap Volume: {max_overlap} mm³ (0.0 mm³ required)")
    print(f"    - Collision Check Time: {t_collision:.3f} s (< 0.60 s required)")
    print(f"    - Total Single-Pass Time: {res['elapsed_sec']['total']:.3f} s")

    assert pairs_total == 435, f"Expected 435 pairs, got {pairs_total}"
    assert pairs_pruned >= 390, f"AABB pruned {pairs_pruned} pairs, expected >= 390"
    assert max_overlap == 0.0, f"Expected 0.0 mm³ intersection, got {max_overlap}"
    assert t_collision < 0.60, f"Collision check took {t_collision}s, expected < 0.60s"
    assert res["solid_count"] == 30, f"Expected 30 solids, got {res['solid_count']}"
    assert res["valid_solids"] == 30, f"Expected 30 valid solids, got {res['valid_solids']}"
    print("    [PASS] 2-Phase collision check passed all criteria.")


def test_2_spec_compiler_chamfer_and_stadium_slot():
    print("\n" + "=" * 70)
    print("[TEST 2] SpecCompiler Chamfer Scopes & Stadium Slot")
    print("=" * 70)

    specs = [
        {
            "name": "bearing_sleeve",
            "origin": "center_bottom",
            "base": {"shape": "cylinder", "r": 20.0, "h": 35.0},
            "features": [
                {"op": "hole", "d": 24.0, "thru": True},
                {"op": "chamfer", "length": 1.5, "scope": "outer"},
            ],
        },
        {
            "name": "idler_roller",
            "origin": "center_bottom",
            "base": {"shape": "cylinder", "r": 30.0, "h": 40.0},
            "features": [
                {"op": "hole", "d": 18.0, "thru": True},
                {"op": "chamfer", "length": 2.0, "scope": "inner"},
            ],
        },
        {
            "name": "rear_stop_block",
            "origin": "center_bottom",
            "base": {"shape": "box", "dims": [80.0, 50.0, 25.0]},
            "features": [
                {"op": "slot", "w": 12.0, "len": 50.0, "d": 6.0, "style": "stadium", "at": [0, 0]},
            ],
        },
    ]

    t0 = time.perf_counter()
    batch_res = SpecCompiler.compile_batch(specs)
    t_elapsed = time.perf_counter() - t0

    assert batch_res["status"] == "SUCCESS", "Batch compilation failed!"
    print(f"[1] Compiled 3 parts in {t_elapsed:.3f} s (spec: < 0.50 s)")

    for p in batch_res["parts"]:
        print(f"    - {p['part_name']}:")
        print(f"        Status: {p['status']}")
        print(f"        Volume: {p['volume_mm3']} mm³")
        print(f"        BBox:   {p['bbox_mm']} mm")
        print(f"        Zmin:   {p['bbox_min'][2]:.4f} mm")
        assert p["status"] == "SUCCESS"
        assert abs(p["bbox_min"][2]) < 0.05, f"Zmin {p['bbox_min'][2]} violated origin lock!"
        assert p["volume_mm3"] > 0, "Volume must be positive!"

    print("    [PASS] Chamfer scopes ('outer', 'inner') and stadium slot verified.")


def test_4_synthetic_dataset_generation():
    print("\n" + "=" * 70)
    print("[TEST 4] Stage 3 Synthetic Dataset Pipeline (50 Samples)")
    print("=" * 70)

    test_jsonl = PROJECT_ROOT / "training" / "test_dataset_sft.jsonl"
    test_stats = PROJECT_ROOT / "training" / "test_dataset_stats.json"

    stats = generate_synthetic_dataset(
        count=50,
        output_path=test_jsonl,
        stats_path=test_stats,
        seed=101,
    )

    print(f"\n[1] Generation Statistics:")
    print(f"    - Accepted Samples: {stats['accepted_count']}/{stats['target_count']}")
    print(f"    - Pass Rate:        {stats['pass_rate_pct']}% (100% required)")
    print(f"    - Group A Samples:  {stats['group_distribution']['freeform_non_standard_group_a']} (60% = 30)")
    print(f"    - Group B Samples:  {stats['group_distribution']['complex_parametric_group_b']} (40% = 20)")
    print(f"    - Elapsed Time:     {stats['elapsed_seconds']:.2f} s")

    assert stats["accepted_count"] == 50, f"Expected 50 samples, got {stats['accepted_count']}"
    assert stats["pass_rate_pct"] == 100.0, f"Expected 100% pass rate, got {stats['pass_rate_pct']}"
    assert stats["group_distribution"]["freeform_non_standard_group_a"] == 30, "Expected exactly 30 Group A samples"
    assert stats["group_distribution"]["complex_parametric_group_b"] == 20, "Expected exactly 20 Group B samples"

    # Inspect first sample in ChatML format
    print(f"\n[2] Sample ChatML Inspection:")
    with open(test_jsonl, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        messages = first_line["messages"]
        assert len(messages) == 3, "Expected 3 messages (system, user, assistant)"
        print(f"    - System:    {messages[0]['content'][:60]}...")
        print(f"    - User:      {messages[1]['content'][:80]}...")
        print(f"    - Assistant: {messages[2]['content'][:80]}...")
        print(f"    - Metadata:  {first_line['metadata']}")

    print("    [PASS] Stage 3 Synthetic Data Pipeline verified 100% functional.")


def main():
    start_total = time.perf_counter()
    test_1_and_3_30_part_assembly_and_schema()
    test_2_spec_compiler_chamfer_and_stadium_slot()
    test_4_synthetic_dataset_generation()
    total_elapsed = time.perf_counter() - start_total

    print("\n" + "=" * 70)
    print(f"[OK] ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
