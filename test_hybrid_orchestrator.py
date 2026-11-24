"""Regression coverage for compact plans and the direct planner boundary."""

import json

import pytest

import hybrid_orchestrator as orchestrator
from mcp_server.compact_plan import expand_linear


def manifest():
    return {
        "parts_specs": [{"name":"stand","tier":"compiler_cpu","base":{"shape":"box","dims":[4,4,4]}}],
        "assembly_plan": {
            "export_name":"demo",
            "placements":{"ground":{"part":"stand","translation":[0,0,0]},
                          "top":{"part":"stand","translation":[0,0,2]}},
            "expected_volume":128,
        },
    }


def test_linear_vary_applies_to_numeric_copy():
    item={"n":3,"xyz":[2,0,0],"step":[10,0,0],
          "spec":{"base":{"shape":"stepped_shaft","sections":[{"d":4,"h":5}]}},
          "vary":{"base.sections.0.d":0.5}}
    parts, plan=expand_linear([], {"linear_series":{"shaft":item}})
    assert [part["base"]["sections"][0]["d"] for part in parts]==[4,4.5,5]
    assert plan["placements"]["shaft_003"]["translation"]==[22,0,0]
    assert item["spec"]["base"]["sections"][0]["d"]==4
    item["vary"]={"base.shape":2}
    with pytest.raises(ValueError,match="numeric"):
        expand_linear([],{"linear_series":{"shaft":item}})


def test_compiler_lanes_reject_python_fragment():
    obj=manifest()
    obj["parts_specs"][0]["features"]=[{"op":"hole","d":2,"sel":">Z'); print('bad"}]
    with pytest.raises(ValueError,match="Unsupported string"):
        orchestrator.materialize(obj,False)


def test_cloud_budget_usage_and_minimal_context(monkeypatch):
    sent=[]
    def reply(url,payload,headers,timeout=90):
        sent.append(payload)
        return {"usage":{"input_tokens":200,"output_tokens":40,"cache_creation_input_tokens":20,
                         "cache_read_input_tokens":15},
                "stop_reason":"end_turn","content":[{"type":"text","text":'{"index":0}'}]}
    monkeypatch.setattr(orchestrator,"post_json",reply)
    client=orchestrator.CloudPlanner("test",api_key="configured-for-test")
    assert client.ask("schema","brief","plan")=={"index":0}
    assert client.ask("fix","one collision","repair")=={"index":0}
    with pytest.raises(ValueError,match="budget"):
        client.ask("fix","third","repair")
    assert client.ledger[0]["uncached_total"]==260
    assert client.ledger[0]["cache_read_tokens"]==15
    assert len(sent)==2 and all(len(call["messages"])==1 and "tools" not in call for call in sent)


def test_local_repair_obeys_candidates_and_rechecks_geometry(tmp_path):
    class Local:
        def ask(self, system, prompt, phase):
            options=json.loads(prompt)["candidates"]
            assert options==[{"instance":"top","z":2.5},{"instance":"top","z":4}]
            assert phase=="repair"
            return {"index":1}
    result=orchestrator.execute(manifest(),tmp_path,repair_policy={"top":[2.5,4]},local=Local())
    assert result["passed"],result
    assert [attempt["result"]["status"] for attempt in result["attempts"]]==["COLLISION_DETECTED","SUCCESS"]


def test_untrusted_cloud_code_requires_explicit_opt_in():
    obj=manifest()
    obj["parts_specs"][0]={"name":"stand","tier":"cloud_direct"}
    obj["cloud_parts"]={"stand":"result=cq.Solid.makeBox(4,4,4)"}
    with pytest.raises(ValueError,match="explicit"):
        orchestrator.materialize(obj,False)
