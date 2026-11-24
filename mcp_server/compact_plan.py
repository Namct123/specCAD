"""Expand bounded, declarative linear series into ordinary CAD specs and placements."""

from copy import deepcopy
import math
import re


def vector(value):
    if (not isinstance(value, list) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise ValueError("Expected a finite XYZ vector")
    return value


def vary_number(spec, path, step, index):
    keys = path.split(".")
    if len(keys) < 2 or keys[0] != "base" or not isinstance(step, (int, float)) or isinstance(step, bool):
        raise ValueError("vary needs numeric steps at existing base fields")
    node = spec
    try:
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        key = int(keys[-1]) if isinstance(node, list) else keys[-1]
        original = node[key]
        if isinstance(original, bool) or not isinstance(original, (int, float)):
            raise ValueError("vary target must be numeric")
        value = original + index * step
        if not math.isfinite(value):
            raise ValueError("Nonfinite series value")
        node[key] = value
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unknown series field {path}") from exc


def expand_linear(parts, plan):
    series = plan.get("linear_series", {})
    if not isinstance(series, dict):
        raise ValueError("linear_series must be an object")
    if not series:
        return parts, plan
    positions = plan.get("placements", {})
    if not isinstance(positions, dict):
        raise ValueError("placements must be an object")
    positions = positions.copy()
    parts = list(parts)
    total = 0
    for prefix, item in series.items():
        if (not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix)
                or not isinstance(item, dict)
                or set(item) - {"n", "xyz", "step", "part", "spec", "vary", "rotation_axis", "rotation_deg"}):
            raise ValueError("Invalid linear series")
        count = item.get("n")
        if type(count) is not int or count < 1 or total + count > 200:
            raise ValueError("linear_series supports at most 200 generated placements")
        total += count
        origin, step = vector(item.get("xyz")), vector(item.get("step"))
        template, part = item.get("spec"), item.get("part")
        vary = item.get("vary", {})
        if (not isinstance(vary, dict) or (template is None) == (part is None)
                or (template is not None and (not isinstance(template, dict) or template.get("tier", "compiler_cpu") != "compiler_cpu"))
                or (vary and template is None)):
            raise ValueError("Give an existing part or a CPU spec template; vary requires a spec")
        for index in range(count):
            name = f"{prefix}_{index + 1:03d}"
            if name in positions:
                raise ValueError(f"Duplicate placement name: {name}")
            if template is not None:
                spec = deepcopy(template)
                spec.update(name=name, tier="compiler_cpu")
                for path, delta in vary.items():
                    vary_number(spec, path, delta, index)
                parts.append(spec)
            position = vector([start + index * increment for start, increment in zip(origin, step)])
            positions[name] = {"part": name if template is not None else part, "translation": position}
            for field in ("rotation_axis", "rotation_deg"):
                if field in item:
                    positions[name][field] = item[field]
    return parts, {**plan, "placements": positions}
