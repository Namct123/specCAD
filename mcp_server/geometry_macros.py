"""Validated geometry recipes shared by the CAD compiler and its exported Python."""

from math import acos, isfinite, sin, tan

import cadquery as cq


def positive(*values):
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and isfinite(value) and value > 0 for value in values):
        raise ValueError("Macro dimensions must be finite and positive")


def loft_cone_hollow(base):
    r0, r1, height, wall = (base[key] for key in ("r0", "r1", "h", "wall"))
    positive(r0, r1, height, wall)
    if wall >= min(r0, r1):
        raise ValueError("Wall must be smaller than both end radii")

    def loft(bottom, top):
        return cq.Workplane("XY").circle(bottom).workplane(offset=height).circle(top).loft(ruled=True)

    return loft(r0, r1).cut(loft(r0 - wall, r1 - wall)).val()


def cut_solid(item):
    if item["shape"] == "box":
        dims = item["dims"]
        positive(*dims)
        cutter = cq.Workplane("XY").box(*dims).translate(tuple(item["center"])).val()
    elif item["shape"] == "cylinder":
        radius, height = item["r"], item["h"]
        positive(radius, height)
        cutter = cq.Solid.makeCylinder(radius, height, cq.Vector(*item["xyz"]),
                                       cq.Vector(*item.get("axis", [0, 0, 1])))
    else:
        raise ValueError("Cutter must be a box or cylinder")
    if "exclude_box" in item:
        cutter = cutter.cut(cut_solid({**item["exclude_box"], "shape": "box"}))
    return cutter


def mounting_foot_cutout(base):
    dims = base["dims"]
    positive(*dims)
    result = cq.Workplane("XY").box(*dims, centered=(True, True, False)).val()
    for item in base["cuts"]:
        result = result.cut(cut_solid(item))
    return result


def sweep_pipe_arc(base):
    outer, inner = base["od"], base["id"]
    positive(outer, inner)
    if inner >= outer:
        raise ValueError("Pipe ID must be smaller than OD")
    axes = {"XY": ((0, 0, 1), (1, 0, 0)), "YZ": ((1, 0, 0), (0, 1, 0))}
    normal, xdir = axes[base.get("plane", "XY")]
    plane = cq.Plane(origin=tuple(base.get("plane_origin", [0, 0, 0])), normal=normal, xDir=xdir)
    point = plane.toWorldCoords(tuple(base["start"]))
    edges = []
    for segment in base["path"]:
        end = plane.toWorldCoords(tuple(segment["to"]))
        if set(segment) == {"to", "via"}:
            edge = cq.Edge.makeThreePointArc(point, plane.toWorldCoords(tuple(segment["via"])), end)
            if edge.radius() <= outer / 2:
                raise ValueError("Bend centerline radius must exceed pipe outer radius")
        elif set(segment) == {"to"}:
            edge = cq.Edge.makeLine(point, end)
        else:
            raise ValueError("Path items need to and optional via")
        edges.append(edge)
        point = end
    if not edges:
        raise ValueError("Pipe path is empty")
    for left, right in zip(edges, edges[1:]):
        if left.tangentAt(1).dot(right.tangentAt(0)) < 0.999:
            raise ValueError("Pipe line and arc segments must meet tangentially")
    section = cq.Plane(origin=edges[0].startPoint(), normal=edges[0].tangentAt(0), xDir=cq.Vector(*normal))
    wire = cq.Wire.assembleEdges(edges)
    result = cq.Workplane(section).circle(outer / 2).circle(inner / 2).sweep(wire).val()
    for cut in base.get("cuts", []):
        result = result.cut(cut_solid(cut))
    return result.translate(tuple(base.get("translate", [0, 0, 0])))



def hollow_frustum_loft(base):
    result = loft_cone_hollow(base)
    axis = base.get("axis", "+Z")
    rotations = {"+X": ((0, 1, 0), 90), "+Y": ((1, 0, 0), -90)}
    if axis == "+Z":
        return result
    if axis not in rotations:
        raise ValueError("Frustum axis must be +X, +Y or +Z")
    direction, angle = rotations[axis]
    return result.rotate((0, 0, 0), direction, angle)


def swept_tube_path(base):
    outer, inner = base["od"], base.get("id", 0)
    positive(outer)
    if inner != 0:
        positive(inner)
    if inner >= outer:
        raise ValueError("Tube ID must be smaller than OD")
    points = [cq.Vector(*point) for point in base["points"]]
    if not 2 <= len(points) <= 64 or any((b - a).Length <= 1e-7 for a, b in zip(points, points[1:])):
        raise ValueError("Tube needs 2–64 distinct consecutive points")
    radius = base.get("bend_radius")
    if len(points) > 2:
        positive(radius)
        if radius <= outer / 2:
            raise ValueError("Bend radius must exceed the tube outer radius")
    edges = []
    previous = points[0]
    for a, vertex, b in zip(points, points[1:], points[2:]):
        toward_a, toward_b = (a - vertex).normalized(), (b - vertex).normalized()
        dot = min(1, max(-1, toward_a.dot(toward_b)))
        if dot < -1 + 1e-9:
            continue
        if dot > 1 - 1e-9:
            raise ValueError("Tube path reverses direction")
        angle = acos(dot) / 2
        trim = radius / tan(angle)
        start, stop = vertex + toward_a * trim, vertex + toward_b * trim
        if (start - previous).dot(toward_a) >= -1e-7 or trim >= (b - vertex).Length:
            raise ValueError("Bend radius does not fit adjoining segments")
        bisector = (toward_a + toward_b).normalized()
        midpoint = vertex + bisector * (radius / sin(angle) - radius)
        edges.extend((cq.Edge.makeLine(previous, start), cq.Edge.makeThreePointArc(start, midpoint, stop)))
        previous = stop
    if (points[-1] - previous).Length <= 1e-7:
        raise ValueError("Tube end has no straight section")
    edges.append(cq.Edge.makeLine(previous, points[-1]))
    section = cq.Plane(origin=points[0], normal=edges[0].tangentAt(0))
    profile = cq.Workplane(section).circle(outer / 2)
    if inner:
        profile = profile.circle(inner / 2)
    return profile.sweep(cq.Wire.assembleEdges(edges)).val()


def bracket_with_cutout(base):
    dims = base["dims"]
    positive(*dims)
    result = cq.Workplane("XY").box(*dims, centered=(True, True, False)).val()
    for hole in base.get("bolt_holes", []):
        diameter = hole["d"]
        positive(diameter)
        x, y = hole["xy"]
        result = result.cut(cq.Solid.makeCylinder(diameter / 2, dims[2] + 2, cq.Vector(x, y, -1)))
    for cut in base.get("cuts", []):
        if cut["shape"] == "plane":
            plane = cq.Plane(origin=tuple(cut["xyz"]), normal=tuple(cut["axis"]))
            keep = cut.get("keep", "negative")
            if keep not in {"positive", "negative"}:
                raise ValueError("Plane keep must be positive or negative")
            box = result.BoundingBox()
            reach = box.DiagonalLength + (plane.origin - box.center).Length + 1
            half = cq.Workplane(plane).rect(2 * reach, 2 * reach).extrude(
                reach if keep == "positive" else -reach,
            ).val()
            result = result.intersect(half)
        else:
            result = result.cut(cut_solid(cut))
    return result


MACROS = {
    "hollow_frustum_loft": hollow_frustum_loft,
    "swept_tube_path": swept_tube_path,
    "bracket_with_cutout": bracket_with_cutout,
    "loft_cone_hollow": loft_cone_hollow,
    "sweep_pipe_arc": sweep_pipe_arc,
    "mounting_foot_cutout": mounting_foot_cutout,
}


def build_macro(shape, base, origin):
    result = MACROS[shape](base)
    bounds = result.BoundingBox()
    floor = bounds.zmin if origin == "center_bottom" else (bounds.zmin + bounds.zmax) / 2
    return result.translate((0, 0, -floor))
