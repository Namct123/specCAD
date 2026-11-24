"""Execute placement and recovered geometry inside the assembly worker."""

from math import isfinite
from pathlib import Path

import cadquery as cq

from mcp_server.cad_runner import HeadlessCADRunner, validate_part_geometry
from mcp_server.compact_plan import vector


def build_planned_assembly(plan, parts_specs, step_paths, output_dir):
    placements = plan["placements"]
    pending = [spec["name"] for spec in parts_specs if spec["name"] not in step_paths]
    namespace = {"cq": cq, "pending_parts": tuple(pending)}
    if pending:
        exec(plan["cloud_parts_script"], namespace)
    cloud_parts = namespace.get("cloud_parts", {})
    assembly = cq.Assembly(name=plan["export_name"])
    qualified = {}

    for spec in parts_specs:
        name = spec["name"]
        try:
            if name in step_paths:
                shape = HeadlessCADRunner._as_shape(cq.importers.importStep(step_paths[name]))
            else:
                shape = HeadlessCADRunner._as_shape(cloud_parts.get(name))

            metrics = HeadlessCADRunner.compute_metrics(shape)
            if not metrics.is_watertight or metrics.solid_count != 1:
                raise ValueError(f"{name}: expected one valid watertight solid")
            if spec.get("origin", "center_bottom") == "center_bottom" and abs(metrics.bbox_min[2]) >= 0.05:
                raise ValueError(f"{name}: local Z minimum must be within 0.05 mm of zero")
            validate_part_geometry(shape, metrics.volume, spec)
        except (ValueError, TypeError, AttributeError) as exc:
            if name not in step_paths:
                raise ValueError(f"Cloud part {name}: {type(exc).__name__}: {str(exc)[:170]}") from exc
            raise
        qualified[name] = shape

    for name, placement in placements.items():
        shape = qualified[placement.get("part", name)]
        translation = cq.Vector(*vector(placement["translation"]))
        axis = cq.Vector(*vector(placement.get("rotation_axis", [0, 0, 1])))
        angle = placement.get("rotation_deg", 0)
        if axis.Length == 0 or type(angle) not in (int, float) or not isfinite(angle):
            raise ValueError(f"{name}: invalid axis or angle")
        location = cq.Location(translation, axis, angle)
        if "expected_global_z" in placement or "expected_global_bbox" in placement:
            box = shape.moved(location).BoundingBox()
            actual = (box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax)
            for field, bounds in (("expected_global_z", (box.zmin, box.zmax)), ("expected_global_bbox", actual)):
                if field not in placement:
                    continue
                expected = placement[field]
                if len(expected) != len(bounds) or not all(type(v) in (int, float) and isfinite(v) for v in expected):
                    raise ValueError(f"{name}: invalid {field}")
                if any(abs(left - right) >= 0.05 for left, right in zip(bounds, expected)):
                    if field == "expected_global_z":
                        low, high = expected
                        raise ValueError(f"{name}: placed Z [{box.zmin:.3f}, {box.zmax:.3f}] differs from [{low}, {high}]")
                    raise ValueError(f"{name}: world bounding box {actual} differs from {expected}")
        color = cq.Color(placement["color"]) if placement.get("color") else None
        assembly.add(shape, name=name, loc=location, color=color)

    for name in pending:
        cq.exporters.export(qualified[name], str(Path(output_dir) / f"{name}.step"))

    return assembly
