"""Single-shot CAD planner, bounded repair, and local execution/reporting."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.report_generator import write_report


SCHEMA_GUIDE = """Work in millimeters. Name parts and instances with letters, digits, underscores or hyphens.
Define each reusable part once under parts_specs, tier compiler_cpu, then reference it by part name in placements. A placement is {part: name, translation: [x,y,z]} with optional rotation_axis XYZ and rotation_deg in degrees. Rotate about the local origin first, then translate to world XYZ. Local center_bottom places Z minimum at zero; centered uses the geometric Z midpoint. Translation is the placement origin, not the world bounding-box minimum. Do not infer contact from overlap of bounding boxes. Keep enough clearance where the brief requires it.
Base shapes and parameters:
- box(dims:[x,y,z]), cylinder(d,h), tube(od,id,h), cone(bottom_radius,top_radius,height).
- stepped_shaft(sections: ordered [{d,h}]); each section height is a length, not a cumulative Z.
- hollow_frustum_loft(r0,r1,h,wall,axis:+X/+Y/+Z), axis defaults to +Z. r0/r1 are the outer radii, wall is radial thickness.
- swept_tube_path(od,id,points:[[x,y,z],...],bend_radius). id=0 makes a solid rod. Interior polyline turns receive tangent fillets. The bend centerline radius must exceed OD/2 and fit both adjoining spans.
- bracket_with_cutout(dims,bolt_holes:[{d,xy:[x,y]}],cuts:[...]). Cylinder cutter {shape:cylinder,r,h,xyz:[x,y,z],axis:[x,y,z]}; box cutter {shape:box,dims:[x,y,z],center:[x,y,z]}; plane cut {shape:plane,xyz:[x,y,z],axis:[x,y,z],keep:positive or negative}. These coordinates are local.
- Legacy loft_cone_hollow(r0,r1,h,wall), mounting_foot_cutout(dims,cuts), sweep_pipe_arc(od,id,plane,start,path,cuts) also work. The legacy sweep uses planar XY or YZ and explicit tangent line/arc entries {to:[u,v]} or {via:[u,v],to:[u,v]}.
Feature entries use op, e.g. hole({d,at:[x,y],depth}) with omitted depth through, hole_rect({d,dx,dy}), hole_pcd({d,pcd,count,start_angle}), counterbore({d,cb_d,cb_depth,at}). PCD is diameter, not radius. Do not add a volume target unless the user supplies an independent value. expected_global_z is [min,max]; expected_global_bbox is [xmin,ymin,zmin,xmax,ymax,zmax].
linear_series maps a prefix to {n,xyz:[x,y,z],step:[dx,dy,dz],part}. For varying geometry replace part with spec (a compiler_cpu prototype without name), and use vary to map an existing numeric base path to the increment per zero-based index. Circular instancing uses circular_patterns mapping prefix to {part,count,pcd,center:[x,y,z],start_angle}; it lies in world XY. Use a series when dimensions or placement follow a formula; all generated definitions and placements are expanded locally. Do not copy a part definition for identical instances. Single named instances use placements directly.
Example 1: two identical pins at 15 mm pitch.
parts_specs: [{name: pin, tier: compiler_cpu, base: {shape: cylinder, d: 6, h: 10}}]
export_name: pins
linear_series: {pin: {part: pin, n: 2, xyz: [0,0,0], step: [15,0,0]}}
Example 2: three diameter variants spaced 40 mm, axis local +Z rotated onto world +Y.
parts_specs: []
export_name: rollers
linear_series: {roller: {n: 3, xyz: [0,-20,0], step: [40,0,0], rotation_axis: [1,0,0], rotation_deg: -90, spec: {base: {shape: cylinder, d: 18, h: 40}}, vary: {base.d: 1}}}
All referenced parts must exist and every definition must be placed. Preserve axes, material boundaries and dimensions exactly. If the stated geometry is ambiguous, return an error object instead of inventing dimensions.
"""


LOCAL_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "parts_specs": {"type": "array", "items": {"type": "object",
            "properties": {"name": {"type": "string"}, "tier": {"const": "compiler_cpu"},
                           "base": {"type": "object", "additionalProperties": True},
                           "features": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                           "origin": {"type": "string"}},
            "required": ["name", "tier", "base"], "additionalProperties": False}},
        "assembly_plan": {"type": "object", "properties": {
            "export_name": {"type": "string"},
            "placements": {"type": "object", "additionalProperties": True},
            "linear_series": {"type": "object", "additionalProperties": True},
            "circular_patterns": {"type": "object", "additionalProperties": True},
            "expected_volume": {"type": "number"},
        }, "required": ["export_name"], "additionalProperties": False},
    },
    "required": ["parts_specs", "assembly_plan"],
    "additionalProperties": False,
}
LOCAL_JSON_EXAMPLE = """Return JSON using this exact object structure:
{"parts_specs":[{"name":"stud","tier":"compiler_cpu","base":{"shape":"cylinder","d":8,"h":12}}],
"assembly_plan":{"export_name":"fixture","placements":{"stud":{"part":"stud","translation":[0,0,0]}}}}
Use dimensions, part names and the assembly name from the actual user brief. Do not copy the example values. Only emit compiler_cpu parts.
"""


def post_json(url, payload, headers, timeout=90):
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Planner endpoint returned HTTP {exc.code}") from exc


class CloudPlanner:
    def __init__(self, model, api_key=None, provider=None):
        self.model = model
        self.provider = provider or os.getenv("HYBRID_CLOUD_PROVIDER")
        if not self.provider:
            if not api_key and (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL")):
                self.provider = "openai"
            elif not api_key and os.getenv("OPENROUTER_API_KEY"):
                self.provider = "openrouter"
            else:
                self.provider = "anthropic"
        defaults = {
            "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
            "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
            "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
        }
        if self.provider not in defaults:
            raise ValueError("HYBRID_CLOUD_PROVIDER must be openai, openrouter or anthropic")
        key_name, default_url = defaults[self.provider]
        self.api_key = api_key or os.getenv(key_name)
        if self.provider == "openrouter":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_env = "ANTHROPIC_BASE_URL" if self.provider == "anthropic" else "OPENAI_BASE_URL"
        self.base_url = (os.getenv(base_env) or default_url).rstrip("/")
        if not self.base_url.startswith("https://") and not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost)(?::\d+)?(?:/[^ ]*)?", self.base_url):
            raise ValueError("Provider endpoint must use HTTPS or local loopback")
        if not self.api_key or not self.model:
            raise ValueError(f"Configure {key_name} and HYBRID_CLOUD_MODEL, or pass --model")
        self.ledger = []

    def ask(self, system, prompt, phase):
        if len(self.ledger) >= 2:
            raise ValueError("Cloud call budget exhausted")
        entry = {"phase": phase, "provider": self.provider, "model": self.model}
        self.ledger.append(entry)
        started = perf_counter()
        limit = 3072 if phase == "plan" else 768
        try:
            if self.provider == "anthropic":
                response = post_json(self.base_url + "/messages", {
                    "model": self.model, "system": system, "max_tokens": limit,
                    "messages": [{"role": "user", "content": prompt}],
                }, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
                usage = response["usage"]
                uncached = usage["input_tokens"]
                write = usage.get("cache_creation_input_tokens", 0)
                cached = usage.get("cache_read_input_tokens", 0)
                completion = usage["output_tokens"]
                stop = response["stop_reason"] == "end_turn"
                text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
            else:
                limit_field = "max_tokens" if self.provider == "openrouter" else "max_completion_tokens"
                response = post_json(self.base_url + "/chat/completions", {
                    "model": self.model, limit_field: limit, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                }, {"Authorization": f"Bearer {self.api_key}"})
                usage = response["usage"]
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                uncached = usage["prompt_tokens"] - cached
                write = 0
                completion = usage["completion_tokens"]
                choice = response["choices"][0]
                stop = choice["finish_reason"] == "stop"
                text = choice["message"]["content"]
            prompt_tokens = uncached + write + cached
            entry.update(input_tokens=uncached, output_tokens=completion, cache_write_tokens=write,
                         cache_read_tokens=cached, prompt_tokens=prompt_tokens, cached_tokens=cached,
                         completion_tokens=completion, total_tokens=prompt_tokens + completion,
                         uncached_total=uncached + write + completion)
            if not stop or not isinstance(text, str):
                raise ValueError("Cloud completion stopped before a complete plan")
            return read_object(text)
        finally:
            entry["elapsed_sec"] = round(perf_counter() - started, 3)


class LocalPlanner:
    def __init__(self, model):
        self.model = model
        self.ledger = []

    def ask(self, system, prompt, phase):
        started = perf_counter()
        try:
            return self._ask(system, prompt, phase)
        finally:
            self.ledger.append({"phase": phase, "elapsed_sec": round(perf_counter() - started, 3)})

    def _ask(self, system, prompt, phase):
        field = "index" if phase == "repair" else "code"
        schema = {"type": "object", "properties": {field: {"type": "integer" if field == "index" else "string"}},
                  "required": [field], "additionalProperties": False}
        reply = post_json("http://127.0.0.1:11434/api/chat", {
            "model": self.model, "stream": False, "format": LOCAL_PLAN_SCHEMA if phase == "plan" else schema, "keep_alive": "1m",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 2048 if phase == "plan" else 768 if phase == "syntax" else 96},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }, {}, timeout=90)
        if reply.get("done_reason") == "length":
            raise ValueError("Ollama reached its output limit")
        return read_object(reply["message"]["content"])


def read_object(text):
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence) and text.endswith(fence):
        text = text.split("\n", 1)[1].rsplit(fence, 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class ManifestLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are unsupported in CAD manifests")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        by_key = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in by_key:
                raise ValueError("Duplicate YAML key")
            by_key[key] = self.construct_object(value_node, deep=deep)
        return by_key


def read_yaml(text):
    if len(text) > 65536:
        raise ValueError("Manifest exceeds 64 KiB")
    value = yaml.load(text, Loader=ManifestLoader)
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML mapping")
    return value


def load_manifest(value):
    if isinstance(value, str):
        value = read_yaml(value)
    if "assembly_plan_yaml" in value:
        if set(value) - {"assembly_plan_yaml", "cloud_direct_parts"}:
            raise ValueError("Unexpected planner envelope fields")
        plan = read_yaml(value["assembly_plan_yaml"])
        parts = plan.pop("parts_specs")
        return {"parts_specs": parts, "assembly_plan": plan,
                "cloud_parts": value.get("cloud_direct_parts", {})}
    if "cloud_direct_parts" in value:
        if "cloud_parts" in value:
            raise ValueError("Duplicate cloud part maps")
        value = {key: item for key, item in value.items() if key != "cloud_direct_parts"} | {
            "cloud_parts": value["cloud_direct_parts"]
        }
    if "assembly_plan" not in value and "parts_specs" in value and "export_name" in value:
        parts = value["parts_specs"]
        snippets = value.get("cloud_parts", {})
        plan = {key: item for key, item in value.items() if key not in {"parts_specs", "cloud_parts"}}
        return {"parts_specs": parts, "assembly_plan": plan, "cloud_parts": snippets}
    return value


def validate_compiler_data(value, key=None):
    if isinstance(value, dict):
        for field, child in value.items():
            validate_compiler_data(child, field)
    elif isinstance(value, list):
        for child in value:
            validate_compiler_data(child, key)
    elif isinstance(value, str):
        labels = {"shape", "kind", "op", "sel", "at_face", "plane", "axis", "style", "slot_style", "scope", "keep"}
        if key not in labels or not re.fullmatch(r"[A-Za-z0-9_<>|+-]+", value):
            raise ValueError("Unsupported string in compiler geometry")
    elif type(value) in (int, float) and not math.isfinite(value):
        raise ValueError("Nonfinite compiler geometry")


def materialize(manifest, allow_code):
    from mcp_server.compact_plan import expand_linear

    parts, plan = manifest.get("parts_specs"), manifest.get("assembly_plan")
    if set(manifest) - {"parts_specs", "assembly_plan", "cloud_parts"} or not isinstance(parts, list) or not isinstance(plan, dict):
        raise ValueError("Expected parts_specs and assembly_plan")
    if set(plan) - {"export_name", "placements", "linear_series", "circular_patterns", "expected_volume"}:
        raise ValueError("Unknown assembly_plan field")
    if len(parts) > 200 or any(
        not isinstance(part, dict) or not isinstance(part.get("name"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", part["name"])
        or part.get("tier") not in {"compiler_cpu", "local_sft", "cloud_direct"}
        for part in parts
    ):
        raise ValueError("Invalid part definitions")
    if any(part["tier"] == "local_sft" and (
        part.get("geometry_kind") != "cone" or not isinstance(part.get("raw_text"), str)
        or part.get("features") or part.get("origin", "center_bottom") != "center_bottom"
    ) for part in parts):
        raise ValueError("Local generation is restricted to structured cone recipes")
    # Bound variants and reject malformed numeric series before opening a worker.
    expanded, expanded_plan = expand_linear(parts, plan)
    position_count = len(expanded_plan.get("placements", {}))
    circular = expanded_plan.get("circular_patterns", {})
    if not isinstance(circular, dict):
        raise ValueError("circular_patterns must be an object")
    for pattern in circular.values():
        if not isinstance(pattern, dict) or type(pattern.get("count")) is not int or pattern["count"] < 1:
            raise ValueError("Invalid circular pattern count")
        position_count += pattern["count"]
    if len(expanded) > 200 or position_count > 200:
        raise ValueError("Manifest exceeds the local part or placement limit")
    for part in expanded:
        if part.get("tier") == "compiler_cpu":
            validate_compiler_data(part.get("base", {}))
            validate_compiler_data(part.get("features", []))
    cloud_names = {part["name"] for part in parts if part["tier"] == "cloud_direct"}
    snippets = manifest.get("cloud_parts", {})
    if not isinstance(snippets, dict) or snippets.keys() != cloud_names:
        raise ValueError("Provide one cloud_parts snippet per cloud_direct part")
    if not snippets:
        return parts, plan
    if not allow_code:
        raise ValueError("Generated Python requires review and explicit --allow-generated-code")
    script = ["cloud_parts = {}"]
    for name, snippet in snippets.items():
        if not isinstance(snippet, str) or len(snippet) > 8000:
            raise ValueError(f"Invalid cloud snippet: {name}")
        # A blank namespace per part prevents accidental reuse of another part's result.
        script.extend([f"if {name!r} in pending_parts:",
                       "    scope = {'cq': cq}",
                       "    try:",
                       f"        exec({snippet!r}, scope)",
                       f"        cloud_parts[{name!r}] = scope['result']",
                       "    except Exception as exc:",
                       f"        raise ValueError('Cloud part {name}: ' + type(exc).__name__ + ': ' + str(exc)[:170]) from exc"])
    return parts, {**plan, "cloud_parts_script": "\n".join(script)}


def repair_candidates(manifest, result, policy):
    report = result.get("interferences", {})
    if result.get("status") != "COLLISION_DETECTED" or not result.get("collision_verification_complete"):
        return []
    involved = {name for pair in report.get("collision_pairs", [])
                for name in (pair.get("part_a"), pair.get("part_b"))}
    candidates = []
    for name, values in policy.items():
        placement = manifest["assembly_plan"].get("placements", {}).get(name)
        if name not in involved or not placement or not isinstance(values, list):
            continue
        original = placement["translation"]
        for value in values:
            if type(value) in (int, float) and math.isfinite(value) and value != original[2]:
                candidates.append({"instance": name, "z": value})
            if (isinstance(value, list) and len(value) == 3 and value != original
                    and all(type(v) in (int, float) and math.isfinite(v) for v in value)):
                candidates.append({"instance": name, "xyz": value})
    return candidates[:8]


def select_candidate(planner, candidates, result, phase):
    summary = {"collisions": result.get("interferences", {}).get("collision_pairs", [])[:4],
               "candidates": candidates}
    choice = planner.ask(
        'Choose only an authorized absolute position candidate. Return {"index":integer}, zero based.',
        json.dumps(summary, separators=(",", ":")), phase,
    )
    index = choice.get("index")
    if set(choice) != {"index"} or type(index) is not int or not 0 <= index < len(candidates):
        raise ValueError("Planner selected an unauthorized repair")
    return candidates[index]


def with_repair(manifest, chosen):
    repaired = deepcopy(manifest)
    placement = repaired["assembly_plan"]["placements"][chosen["instance"]]
    if "xyz" in chosen:
        placement["translation"] = chosen["xyz"]
    else:
        placement["translation"][2] = chosen["z"]
    return repaired


def accepted(result):
    return (result.get("status") == "SUCCESS" and result.get("collision_verification_complete")
            and result.get("solid_count") == result.get("instance_count")
            and (result.get("volume_check") or {}).get("passed", True)
            and all(Path(result.get("outputs", {}).get(kind, "")).is_file() for kind in ("step", "glb")))


def execute(manifest, output, allow_code=False, repair_policy=None, local=None, cloud=None, cloud_repair=False, local_repair_enabled=True):
    from mcp_server.server import CADService

    attempts = []
    service = CADService(output)
    def run(candidate, label):
        started = perf_counter()
        try:
            parts, plan = materialize(candidate, allow_code)
            result = service.run_assembly_plan(parts, plan, str(output / label), max_sft_retries=0, full_brep=True)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            result = {"status": "FAILED", "error": str(exc)}
        attempts.append({"stage": label, "passed": bool(accepted(result)), "result": result,
                         "elapsed_sec": round(perf_counter() - started, 3)})
        return result

    result = run(manifest, "initial")
    candidates = repair_candidates(manifest, result, repair_policy or {})
    if not accepted(result) and candidates:
        if not local_repair_enabled:
            chosen = None
        elif len(candidates) == 1:
            chosen = candidates[0]
        elif local:
            try:
                chosen = select_candidate(local, candidates, result, "repair")
            except (ValueError, RuntimeError, OSError) as exc:
                attempts.append({"stage": "local_repair", "passed": False, "result": {"error": str(exc)}})
                chosen = None
        else:
            chosen = None
        if chosen:
            result = run(with_repair(manifest, chosen), "local_repair")
        remaining = [candidate for candidate in candidates if candidate != chosen]
        if not accepted(result) and remaining and cloud and cloud_repair:
            try:
                selected = select_candidate(cloud, remaining, result, "repair")
                result = run(with_repair(manifest, selected), "cloud_repair")
            except (OSError, RuntimeError, ValueError) as exc:
                attempts.append({"stage": "cloud_repair", "passed": False, "result": {"error": str(exc)}})
    elif not accepted(result) and cloud and cloud_repair and result.get("phase") == "parts":
        failures = result.get("escalated_parts", [])
        failed_name = failures[0]["name"] if len(failures) == 1 and failures[0]["reason"] == "compiler_failed" else None
        original = next((item for item in manifest["parts_specs"]
                         if item["name"] == failed_name and item.get("tier") == "compiler_cpu"), None)
        if original is not None:
            try:
                patch = cloud.ask(
                    'Correct this CAD definition only. Return JSON {"base":{...},"features":[...]}. Preserve the contract.',
                    json.dumps({"part": original, "failure": failures[0]["reason"]}, separators=(",", ":")),
                    "repair",
                )
                if set(patch) - {"base", "features"} or not isinstance(patch.get("base"), dict):
                    raise ValueError("Expected only base and optional features")
                updated = deepcopy(manifest)
                next(item for item in updated["parts_specs"] if item["name"] == failed_name).update(patch)
                result = run(updated, "cloud_repair")
            except (OSError, RuntimeError, ValueError) as exc:
                attempts.append({"stage": "cloud_repair", "passed": False, "result": {"error": str(exc)}})
    elif not accepted(result) and allow_code:
        match = re.match(r"Cloud part ([A-Za-z0-9_-]+): (.*)", result.get("error", ""))
        if match and match[1] in manifest.get("cloud_parts", {}):
            name = match[1]
            current = manifest["cloud_parts"][name]
            error = match[2][:200]
            guidance = 'Replace only the failing CadQuery snippet. Return {"code":"..."} defining result.'
            if local_repair_enabled and local and error.startswith(("SyntaxError", "IndentationError")):
                try:
                    updated = local.ask(guidance, json.dumps({"part": name, "error": error, "code": current}), "syntax")
                    if set(updated) != {"code"} or not isinstance(updated["code"], str):
                        raise ValueError("Expected only replacement code")
                    compile(updated["code"], name, "exec")
                    patched = deepcopy(manifest)
                    patched["cloud_parts"][name] = updated["code"]
                    result = run(patched, "local_repair")
                except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
                    attempts.append({"stage": "local_repair", "passed": False, "result": {"error": str(exc)}})
            if not accepted(result) and cloud and cloud_repair:
                try:
                    updated = cloud.ask(guidance, json.dumps({"part": name, "error": error, "code": current}), "repair")
                    if set(updated) != {"code"} or not isinstance(updated["code"], str):
                        raise ValueError("Expected only replacement code")
                    patched = deepcopy(manifest)
                    patched["cloud_parts"][name] = updated["code"]
                    result = run(patched, "cloud_repair")
                except (OSError, RuntimeError, ValueError) as exc:
                    attempts.append({"stage": "cloud_repair", "passed": False, "result": {"error": str(exc)}})
    return {"passed": bool(accepted(result)), "attempts": attempts,
            "outputs": result["outputs"] if accepted(result) else {}}





def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("input", type=Path, nargs="?", help="UTF-8 brief or saved manifest")
    input_group.add_argument("--prompt", help="Design brief as command-line text")
    parser.add_argument("--mode", choices=("manifest","local","cloud"), default="manifest")
    parser.add_argument("--local-only", action="store_true", help="Plan with Ollama and never call Cloud")
    parser.add_argument("--provider", choices=("openai", "openrouter", "anthropic"))
    parser.add_argument("--output", type=Path, required=True, help="New directory to keep attempts separate")
    parser.add_argument("--model")
    parser.add_argument("--export-name", help="Set the exact assembly name locally")
    parser.add_argument("--local-model", default=os.getenv("CAD_OLLAMA_RECIPE_MODEL") or "qwen2.5:1.5b")
    parser.add_argument("--repair-policy", type=Path, help="Trusted YAML/JSON mapping explicit instances to allowed absolute Z or XYZ values")
    parser.add_argument("--cloud-repair", action="store_true")
    parser.add_argument("--allow-generated-code", action="store_true", help="Only for reviewed/trusted Python")
    args = parser.parse_args()
    if args.local_only:
        if args.mode == "cloud":
            parser.error("--local-only and --mode cloud cannot be combined")
        args.mode = "local"
    if args.prompt is not None and args.mode == "manifest":
        parser.error("--prompt requires --local-only, --mode local or --mode cloud")
    if args.cloud_repair and args.mode != "cloud":
        parser.error("--cloud-repair requires --mode cloud")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output must identify a new directory")
    started = perf_counter()
    cloud = None
    local = LocalPlanner(args.local_model)
    record = {"passed": False, "attempts": []}
    try:
        text = args.prompt if args.prompt is not None else args.input.read_text(encoding="utf-8-sig")
        policy = read_yaml(args.repair_policy.read_text(encoding="utf-8-sig")) if args.repair_policy else {}
        if args.mode == "cloud":
            cloud = CloudPlanner(args.model or os.getenv("HYBRID_CLOUD_MODEL") or os.getenv("CAD_CLOUD_MODEL"), provider=args.provider)
            guidance = SCHEMA_GUIDE + "Return one JSON object with assembly_plan_yaml (a flat YAML string including parts_specs) and cloud_direct_parts (a map)."
            guidance += (" Inside YAML give unsupported parts tier cloud_direct, and place only their independent Python snippets in cloud_direct_parts. Each snippet defines result, with cq prebound." if args.allow_generated_code else
                         " Only emit compiler_cpu shapes and an empty cloud_direct_parts map.")
            manifest = load_manifest(cloud.ask(guidance, text, "plan"))
        elif args.mode == "local":
            manifest = load_manifest(local.ask(SCHEMA_GUIDE.split("Example 1:", 1)[0] + LOCAL_JSON_EXAMPLE, text, "plan"))
        else:
            manifest = load_manifest(text)
        if args.mode != "manifest" and any(
            part.get("tier") == "local_sft" for part in manifest.get("parts_specs", [])
            if isinstance(part, dict)
        ):
            raise ValueError("Planner must emit numeric CPU specs for supported parts")
        if args.export_name:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", args.export_name):
                raise ValueError("Invalid --export-name")
            manifest["assembly_plan"]["export_name"] = args.export_name
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
        repair_enabled = os.getenv("ENABLE_LOCAL_REPAIR", "1").lower() not in {"0", "false", "off"}
        record.update(execute(manifest, output, args.allow_generated_code, policy, local, cloud,
                              args.cloud_repair, local_repair_enabled=repair_enabled))
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, yaml.YAMLError, RecursionError) as exc:
        record["error"] = str(exc)
    record["cloud"] = cloud.ledger if cloud else []
    record["local"] = local.ledger
    measured = [entry.get("uncached_total") for entry in record["cloud"]]
    record["uncached_cloud_tokens"] = None if None in measured else sum(measured)
    record["elapsed_sec"] = round(perf_counter() - started, 3)
    write_report(output, record)
    print(output / "result.json")
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
