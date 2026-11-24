"""Provider, schema and geometry invariants for the v4 entry point."""

import json
from math import pi

import pytest

from orchestrator import hybrid_orchestrator as v4
from orchestrator.report_generator import write_report
from mcp_server.spec_compiler import SpecCompiler


@pytest.mark.parametrize(("provider", "suffix", "limit_field"), [
    ("openai", "/chat/completions", "max_completion_tokens"),
    ("openrouter", "/chat/completions", "max_tokens"),
])
def test_compatible_provider_usage_and_request(monkeypatch, provider, suffix, limit_field):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unit-test.example/v1")

    def post(url, payload, headers):
        assert url.endswith(suffix) and headers["Authorization"] == "Bearer mock"
        assert payload[limit_field] == 3072 and not payload.get("tools")
        return {"usage": {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 350},
                          "completion_tokens": 120},
                "choices": [{"finish_reason": "stop", "message": {"content": '{"manifest":"test"}'}}]}

    monkeypatch.setattr(v4, "post_json", post)
    client = v4.CloudPlanner("mock", api_key="mock", provider=provider)
    assert client.ask("system", "brief", "plan") == {"manifest": "test"}
    report = client.ledger[0]
    assert report["uncached_total"] == 770
    assert report["cached_tokens"] == 350 and report["total_tokens"] == 1120


def test_yaml_envelope_and_flat_input():
    text = """
parts_specs:
  - {name: pin, tier: compiler_cpu, base: {shape: cylinder, d: 5, h: 10}}
export_name: pins
linear_series:
  pin: {part: pin, n: 3, xyz: [0,0,0], step: [20,0,0]}
"""
    direct = v4.load_manifest(text)
    assert direct == v4.load_manifest({"assembly_plan_yaml": text, "cloud_direct_parts": {}})
    parts, plan = v4.materialize(direct, False)
    assert parts[0]["base"]["h"] == 10 and plan["linear_series"]["pin"]["n"] == 3
    with pytest.raises(ValueError, match="aliases"):
        v4.read_yaml("a: &a [1]\nb: *a")
    with pytest.raises(ValueError, match="Duplicate"):
        v4.read_yaml("x: 1\nx: 2")


@pytest.mark.parametrize(("base","dims","volume"), [
    ({"shape":"hollow_frustum_loft","r0":8,"r1":6,"h":20,"wall":2,"axis":"+X"},
     (20,16,16), pi * 20 / 3 * (64+48+36-36-24-16)),
    ({"shape":"hollow_frustum_loft","r0":8,"r1":6,"h":20,"wall":2,"axis":"+Y"},
     (16,20,16), pi * 20 / 3 * (64+48+36-36-24-16)),
    ({"shape":"hollow_frustum_loft","r0":8,"r1":6,"h":20,"wall":2,"axis":"+Z"},
     (16,16,20), pi * 20 / 3 * (64+48+36-36-24-16)),
    ({"shape":"swept_tube_path","od":6,"id":0,"points":[[0,0,0],[15,0,0]]},
     (15,6,6), pi * 9 * 15),
    ({"shape":"swept_tube_path","od":8,"id":4,"points":[[0,0,0],[30,0,0],[30,20,0]],"bend_radius":10},
     (34,24,8), pi * 12 * (30+5*pi)),
    ({"shape":"bracket_with_cutout","dims":[10,10,8],"bolt_holes":[{"d":2,"xy":[-2.5,0]}],
      "cuts":[{"shape":"plane","xyz":[0,0,0],"axis":[1,0,0],"keep":"negative"}]},
     (5,10,8), 400-8*pi),
])
def test_v4_macros_compile_watertight(tmp_path, base, dims, volume):
    result = SpecCompiler.compile_part({"name":"test","base":base,"expected_volume":volume},
                                       output_dir=tmp_path)
    assert result["status"] == "SUCCESS", result
    assert result["bbox_mm"] == pytest.approx(dims, abs=0.05)
    assert result["volume_mm3"] == pytest.approx(volume, abs=0.05)
    assert result["bbox_min"][2] == pytest.approx(0, abs=0.05)


def test_repair_candidates_limit_xyz_and_report(tmp_path):
    plan = {"assembly_plan":{"placements":{"p":{"translation":[0,0,0]}}}}
    result = {"status":"COLLISION_DETECTED","collision_verification_complete":True,
              "interferences":{"collision_pairs":[{"part_a":"p","part_b":"other","overlap_volume_mm3":10,
                                                   "bbox_a":[0,0,0,1,1,1],"bbox_b":[0,0,0,2,2,2]}]}}
    choices = v4.repair_candidates(plan, result, {"p":[[2,0,0],[0,0,0]],"external":[[2,0,0]]})
    assert choices == [{"instance":"p","xyz":[2,0,0]}]
    assert v4.with_repair(plan, choices[0])["assembly_plan"]["placements"]["p"]["translation"] == [2,0,0]
    assert plan["assembly_plan"]["placements"]["p"]["translation"] == [0,0,0]
    record = {"passed":True,"cloud":[],"uncached_cloud_tokens":0,"elapsed_sec":1.25,
              "attempts":[{"stage":"initial","elapsed_sec":1,"result":{"status":"SUCCESS","batch":{"elapsed_sec":0.2}}}]}
    write_report(tmp_path,record)
    info=json.loads((tmp_path/"result.json").read_text(encoding="utf-8"))
    assert info["api_calls"] == 0 and info["usage"]["total_tokens"] == 0
    assert info["elapsed_by_tier"]["compile_batch"] == 0.2
    assert (tmp_path/"report.md").is_file()


def test_full_brep_does_not_skip_high_face_counts(monkeypatch):
    import cadquery as cq
    from mcp_server.cad_runner import HeadlessCADRunner

    body = cq.Workplane("XY").box(10, 10, 10)
    assembly = cq.Assembly()
    assembly.add(body, name="a")
    assembly.add(body, name="b", loc=cq.Location(cq.Vector(5, 0, 0)))
    monkeypatch.setattr(cq.Compound, "Faces", lambda self: [None] * 60)
    result = HeadlessCADRunner.check_assembly_interferences(assembly, full_brep=True)
    assert result["verification_complete"] and not result["unverified_pairs"]
    assert result["pairs_boolean_checked"] == 1
    pair = result["collision_pairs"][0]
    assert pair["overlap_volume_mm3"] == pytest.approx(500)
    assert pair["bbox_a"] == [-5,-5,-5,5,5,5]
