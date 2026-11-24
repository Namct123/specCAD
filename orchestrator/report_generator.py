"""Write provider usage and verified CAD metadata without a model call."""

from pathlib import Path
import json


def write_report(output, record):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calls = record["cloud"]
    record["api_calls"] = len(calls)
    record["usage"] = {
        key: sum(call[key] for call in calls) if all(key in call for call in calls) else None
        for key in ("prompt_tokens", "cached_tokens", "completion_tokens", "total_tokens")
    }
    reg = record.get("attempts", [])
    record["elapsed_by_tier"] = {
        "cloud": sum(call.get("elapsed_sec", 0) for call in calls),
        "ollama": sum(row.get("elapsed_sec", 0) for row in record.get("local", [])),
        "cad": sum(row.get("elapsed_sec", 0) for row in reg),
        "compile_batch": sum(row["result"].get("batch", {}).get("elapsed_sec", 0) for row in reg),
        **{f"assembly_{stage}": sum((row["result"].get("assembly_elapsed_sec") or {}).get(stage, 0)
                                      for row in reg)
           for stage in ("script", "collision", "export")},
    }
    (output / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    vs = record["usage"]
    lines = ["# Kết quả Hybrid CAD", "", f"- Nghiệm thu — {'Đạt' if record['passed'] else 'Chưa đạt'}",
             f"- Lượt Cloud — {len(calls)}",
             f"- Token Cloud chưa cache + output — {record['uncached_cloud_tokens']}",
             f"- Token input / cache / output / tổng — {vs['prompt_tokens']} / {vs['cached_tokens']} / {vs['completion_tokens']} / {vs['total_tokens']}",
             f"- Thời gian — {record['elapsed_sec']:.3f} s", ""]
    for tier, elapsed in record["elapsed_by_tier"].items():
        lines.append(f"- {tier} — {elapsed:.3f} s")
    for attempt in reg:
        result = attempt["result"]
        lines.append(f"- {attempt['stage']} — {result.get('status', 'FAILED')}")
    if "error" in record:
        lines.append(f"- Lỗi — {record['error']}")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
