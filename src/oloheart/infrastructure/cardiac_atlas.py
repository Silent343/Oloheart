"""Parametric cardiac atlas with explicit ownership and anatomical connections.

Anterior is +Z and anatomical left is +X. Every detachable named structure has
its own element key. Tissue detail shares the chamber's deformation origin, while
valves and subvalvular structures retain their own picking/drag identities.
"""

from __future__ import annotations

import math
from dataclasses import replace

from oloheart.domain.model import HeartPartId as I, build_anatomy_catalog
from oloheart.infrastructure.geometry import AnatomicalMesh, Vec3, _tube, _torus, _ellipsoid
from oloheart.infrastructure.sculpted_geometry import (
    CHAMBERS, VALVES, _contour, _point, _shell, _hollow_vessel, _leaflets, av_leaflet_point, av_leaflet_open_point,
)

NAMES = {part.identifier: part.display_name for part in build_anatomy_catalog()}
EXPLOSION = {
    I.RIGHT_ATRIUM: Vec3(-0.65, 0.40, -0.05),
    I.LEFT_ATRIUM: Vec3(0.65, 0.42, -0.32),
    I.RIGHT_VENTRICLE: Vec3(-0.60, -0.26, 0.20),
    I.LEFT_VENTRICLE: Vec3(0.60, -0.28, -0.05),
    I.TRICUSPID_VALVE: Vec3(-0.60, 0.12, 0.20),
    I.MITRAL_VALVE: Vec3(0.60, 0.12, -0.05),
    I.AORTIC_VALVE: Vec3(0.24, 0.52, -0.14),
    I.PULMONARY_VALVE: Vec3(-0.24, 0.55, 0.40),
}
ZERO = Vec3(0, 0, 0)
VENTRICULAR_ORIGIN = Vec3(.04, .46, -.02)
MUSCLE = ("#249ba5", "#9e2a35")
LINING = ("#56aeb3", "#b84c4a")
TRABECULAR_MUSCLE = ("#3c9da4", "#a4323c")
PAPILLARY_COLOR = ("#429ea6", "#9a2834")
ARTERIAL = ("#2ab1bd", "#be524e")
VENOUS = ("#337fbe", "#587696")
VALVULAR = ("#65c8c9", "#edd8c4")

# The vessel centerlines are also consumed by the circulation visualization.
# Endpoints terminate in the relevant atrium or ventricular outflow tract.
VESSELS = (
    (I.AORTA, "aorta", "Aorta ascendente, arco y descendente", 0.174, ARTERIAL,
     Vec3(0.20, 0.72, -0.22),
     ((0.075,0.59,0.005),(0.08,0.97,-0.015),(-0.01,1.36,-0.05),
      (0.12,1.68,-0.09),(0.40,1.73,-0.20),(0.68,1.48,-0.39),
      (0.71,0.92,-0.53),(0.65,0.10,-0.65),(0.61,-0.69,-0.61))),
    (I.AORTA, "brachiocephalic", "Tronco braquiocefálico", 0.080, ARTERIAL,
     Vec3(-0.15,0.92,-0.25), ((0.04,1.55,-0.05),(-0.06,1.91,-0.03),(-0.20,2.07,-0.02))),
    (I.AORTA, "left_carotid", "Arteria carótida común izquierda", 0.061, ARTERIAL,
     Vec3(0.10,1.04,-0.26), ((0.29,1.69,-0.13),(0.29,1.96,-0.15),(0.34,2.10,-0.15))),
    (I.AORTA, "left_subclavian", "Arteria subclavia izquierda", 0.068, ARTERIAL,
     Vec3(0.44,1.0,-0.30), ((0.53,1.63,-0.28),(0.65,1.88,-0.35),(0.80,1.98,-0.40))),
    (I.PULMONARY_ARTERY, "pulmonary_trunk", "Tronco pulmonar", 0.178, VENOUS,
     Vec3(-0.20,0.62,0.45), ((-0.19,0.63,0.31),(-0.14,0.94,0.34),
                            (-0.02,1.23,0.27),(0.17,1.37,0.02))),
    (I.PULMONARY_ARTERY, "left_pulmonary_artery", "Arteria pulmonar izquierda", 0.122, VENOUS,
     Vec3(0.66,0.63,0.08), ((0.13,1.35,0.01),(0.44,1.39,-0.07),(0.87,1.35,-0.18))),
    (I.PULMONARY_ARTERY, "right_pulmonary_artery", "Arteria pulmonar derecha", 0.120, VENOUS,
     Vec3(-0.76,0.60,-0.15), ((0.13,1.35,-0.05),(-0.13,1.25,-0.30),
                             (-0.50,1.20,-0.40),(-0.96,1.18,-0.35))),
    (I.VENA_CAVA, "superior_vena_cava", "Vena cava superior", 0.154, VENOUS,
     Vec3(-0.70,0.42,-0.24), ((-0.67,1.72,-0.22),(-0.68,1.35,-0.18),(-0.56,0.99,-0.17))),
    (I.VENA_CAVA, "inferior_vena_cava", "Vena cava inferior", 0.160, VENOUS,
     Vec3(-0.77,-0.27,-0.40), ((-0.63,-0.71,-0.42),(-0.68,-0.11,-0.38),
                             (-0.64,0.37,-0.31),(-0.52,0.59,-0.18))),
    (I.PULMONARY_VEINS, "right_superior_pulmonary_vein", "Vena pulmonar superior derecha", 0.091, ARTERIAL,
     Vec3(-0.60,0.26,-0.63), ((-0.90,0.94,-0.46),(-0.42,0.95,-0.54),(0.11,0.94,-0.50))),
    (I.PULMONARY_VEINS, "right_inferior_pulmonary_vein", "Vena pulmonar inferior derecha", 0.087, ARTERIAL,
     Vec3(-0.61,-0.04,-0.65), ((-0.86,0.66,-0.46),(-0.41,0.68,-0.53),(0.14,0.68,-0.47))),
    (I.PULMONARY_VEINS, "left_superior_pulmonary_vein", "Vena pulmonar superior izquierda", 0.089, ARTERIAL,
     Vec3(0.85,0.27,-0.60), ((0.98,1.02,-0.40),(0.77,1.00,-0.46),(0.61,0.97,-0.47))),
    (I.PULMONARY_VEINS, "left_inferior_pulmonary_vein", "Vena pulmonar inferior izquierda", 0.086, ARTERIAL,
     Vec3(0.86,-0.02,-0.63), ((1.01,0.72,-0.36),(0.79,0.70,-0.42),(0.61,0.72,-0.44))),
)


def _owned(mesh: AnatomicalMesh, key: str, name: str = "", center: Vec3 | None = None,
           **changes) -> AnatomicalMesh:
    return replace(mesh, element_key=key, element_name=name or NAMES[mesh.part_id],
                   deformation_center=center, **changes)


def _envelope_point(t: float, angle: float) -> Vec3:
    """Continuous epicardial silhouette with anatomical sulci, RV conus and LV apex."""
    width = _contour(t, "lv")
    s_pi = math.sin(math.pi * t)
    ant_sulcus = math.exp(-((math.cos(angle) + 0.04) * 5.2) ** 2) * max(0.0, math.sin(angle))
    post_sulcus = math.exp(-((math.cos(angle) + 0.10) * 4.8) ** 2) * max(0.0, -math.sin(angle))
    sulcus_indent = 1.0 - (0.048 * ant_sulcus + 0.036 * post_sulcus) * s_pi

    conus_bulge = 0.038 * math.exp(-((t - 0.16) / 0.18) ** 2) * max(0.0, math.sin(angle - 0.55)) * max(0.0, -math.cos(angle - 0.3))
    lv_margin = 0.028 * s_pi * max(0.0, math.cos(angle - 0.25))
    organic = (
        sulcus_indent
        + conus_bulge
        + lv_margin
        + 0.018 * math.cos(angle * 2.0 + t * 3.8) * s_pi
        + 0.009 * math.sin(angle * 3.0 - t * 5.2) * s_pi
    )
    return Vec3(
        -0.02 + 0.55 * t + 0.92 * width * math.cos(angle) * organic,
        0.46 - 1.70 * t,
        -0.025 + 0.61 * width * math.sin(angle) * organic + 0.025 * s_pi,
    )


def _ventricular_envelope() -> list[AnatomicalMesh]:
    """Partition one common surface into adjacent, independently selectable walls."""
    rows, columns = 64, 96
    vertices = [_envelope_point(r / rows, c * math.tau / columns)
                for r in range(rows + 1) for c in range(columns)]
    groups = {I.LEFT_VENTRICLE: [], I.RIGHT_VENTRICLE: []}
    for r in range(rows):
        for c in range(columns):
            a, b = r * columns + c, r * columns + (c + 1) % columns
            d, e = a + columns, b + columns
            angle = (c + 0.5) * math.tau / columns
            right = math.cos(angle) < 0.20 * math.sin(angle) - 0.04 and r / rows < 0.88
            part = I.RIGHT_VENTRICLE if right else I.LEFT_VENTRICLE
            groups[part].extend(((a, d, b), (b, d, e)))
    previous = list(range(columns))
    roof_center = Vec3(0.025, 0.71, -0.065)
    for row in range(1, 13):
        weight = row / 13
        current = []
        for c in range(columns):
            rim = vertices[c]
            p = rim * (1 - weight) + roof_center * weight
            current.append(len(vertices))
            vertices.append(Vec3(p.x, 0.46 + 0.25 * math.sin(weight * math.pi / 2), p.z))
        for c in range(columns):
            n = (c + 1) % columns
            angle = (c + 0.5) * math.tau / columns
            part = I.RIGHT_VENTRICLE if math.cos(angle) < 0.20 * math.sin(angle) - 0.04 else I.LEFT_VENTRICLE
            groups[part].extend(((previous[c], previous[n], current[c]), (previous[n], current[n], current[c])))
        previous = current
    pole = len(vertices)
    vertices.append(roof_center)
    for c in range(columns):
        angle = (c + 0.5) * math.tau / columns
        part = I.RIGHT_VENTRICLE if math.cos(angle) < 0.20 * math.sin(angle) - 0.04 else I.LEFT_VENTRICLE
        groups[part].append((previous[c], previous[(c + 1) % columns], pole))
    apex = len(vertices)
    vertices.append(Vec3(0.53, -1.245, -0.025))
    for c in range(columns):
        groups[I.LEFT_VENTRICLE].append((rows * columns + c, apex, rows * columns + (c + 1) % columns))
    meshes = []
    for part, faces in groups.items():
        used = sorted({v for face in faces for v in face})
        mapping = {v: n for n, v in enumerate(used)}
        mesh = AnatomicalMesh(part, tuple(vertices[v] for v in used),
                              tuple(tuple(mapping[v] for v in face) for face in faces),
                              *MUSCLE, EXPLOSION[part], True, True, "exterior")
        meshes.append(_owned(mesh, part.value, center=VENTRICULAR_ORIGIN))
    return meshes


def _windowed_vessel(mesh: AnatomicalMesh) -> AnatomicalMesh:
    """Clip a smooth root window, sharing edge intersections and sealing thickness."""
    from collections import Counter
    sides = mesh.faces[0][1]
    vertices = list(mesh.vertices)
    centers = []
    for first in range(0, len(vertices), sides):
        ring = vertices[first:first + sides]
        centers.append(Vec3(sum(v.x for v in ring) / sides, sum(v.y for v in ring) / sides,
                            sum(v.z for v in ring) / sides))
    vertex_centers = [centers[n // sides] for n in range(len(vertices))]
    distances = [max(v.y - 1.08, vertex_centers[n].z - v.z) for n, v in enumerate(vertices)]
    intersections = {}
    faces = []
    for face in mesh.faces:
        polygon = []
        for a, b in zip(face, (*face[1:], face[0])):
            da, db = distances[a], distances[b]
            if da >= 0:
                polygon.append(a)
            if (da >= 0) != (db >= 0):
                edge = tuple(sorted((a, b)))
                if edge not in intersections:
                    weight = da / (da - db)
                    intersections[edge] = len(vertices)
                    vertices.append(vertices[a] * (1 - weight) + vertices[b] * weight)
                    vertex_centers.append(vertex_centers[a] * (1 - weight) + vertex_centers[b] * weight)
                polygon.append(intersections[edge])
        faces.extend((polygon[0], polygon[n], polygon[n + 1]) for n in range(1, len(polygon) - 1))
    count = len(vertices)
    vertices += [c + (v - c) * 0.82 for v, c in zip(vertices, vertex_centers)]
    edges = Counter(tuple(sorted(edge)) for a, b, c in faces for edge in ((a, b), (b, c), (c, a)))
    outer = tuple(faces)
    faces.extend((a + count, c + count, b + count) for a, b, c in outer)
    for (a, b), occurrences in edges.items():
        if occurrences == 1:
            faces.extend(((a, b, a + count), (b, b + count, a + count)))
    return replace(mesh, vertices=tuple(vertices), faces=tuple(faces), view="interior")


def _tilt_valve(point: Vec3, center: Vec3, angle: float = 0.48) -> Vec3:
    """Oblique AV plane; the same local frame is used for leaflet animation."""
    p = point - center
    c, s = math.cos(angle), math.sin(angle)
    return center + Vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c)


def _wall_details(part: I) -> list[AnatomicalMesh]:
    """Embed fleshy muscular trabeculae carneae and the RV moderator band in the endocardium."""
    top, height, radii, tilt, thickness = CHAMBERS[part]
    kind = "rv" if part == I.RIGHT_VENTRICLE else "lv"
    meshes = []
    center = VENTRICULAR_ORIGIN
    # Coarse muscular trabeculae in RV; fine, dense meshwork in LV
    for n in range(21):
        angle = math.pi + (n + 0.5) * math.pi / 21
        path = []
        for k in range(16):
            t = 0.16 + (0.72 + 0.03 * math.sin(n)) * k / 15
            a = angle + 0.065 * math.sin(t * 14 + n * 1.3) + 0.035 * math.cos(t * 21 - n)
            path.append(_point(top, height, radii, tilt, t, a, kind, thickness - 0.003))
        ridge_r = 0.021 if kind == "rv" else 0.015
        ridge = _tube(part, path, ridge_r, TRABECULAR_MUSCLE,
                      EXPLOSION[part], 10, True, False, curve_steps=2, taper=0.42)
        meshes.append(_owned(ridge, part.value, center=center, view="interior"))
    for n in range(40):
        t = 0.20 + (n % 8) * 0.082
        angle = math.pi + 0.14 + (n // 8) * 0.50
        path = [_point(top, height, radii, tilt, t + 0.028 * math.sin(k * 0.55 + n),
                       angle + k * 0.068, kind, thickness - 0.006) for k in range(8)]
        ridge = _tube(part, path, 0.012 if kind == "rv" else 0.0095, TRABECULAR_MUSCLE,
                      EXPLOSION[part], 8, True, False, taper=0.52)
        meshes.append(_owned(ridge, part.value, center=center, view="interior"))
    if kind == "rv":
        # Septomarginal trabecula (Moderator band of the right ventricle)
        mod_band = [
            Vec3(0.06, -0.36, -0.14),
            Vec3(-0.08, -0.44, -0.10),
            Vec3(-0.26, -0.48, -0.05),
            _point(top, height, radii, tilt, 0.65, math.pi * 1.08, "rv", thickness * 0.15),
        ]
        mb_mesh = _tube(part, mod_band, 0.026, TRABECULAR_MUSCLE, EXPLOSION[part], 12, True, False, curve_steps=4, taper=0.25)
        meshes.append(_owned(mb_mesh, part.value, center=center, view="interior"))
    return meshes


def _support_tube(part, points, radius, colors, explosion, sides=14, taper=0.5):
    """Rounded tissue/fibrous cable with closed ends, never a line primitive."""
    mesh = _tube(part, points, radius, colors, explosion, sides, part == I.PAPILLARY_MUSCLES,
                 False, taper=taper)
    vertices, faces = list(mesh.vertices), list(mesh.faces)
    for first, center in ((0, points[0]), (len(vertices) - sides, points[-1])):
        pole = len(vertices)
        vertices.append(center)
        faces.extend((first + c, first + (c + 1) % sides, pole) for c in range(sides))
    return replace(mesh, vertices=tuple(vertices), faces=tuple(faces))


def _subvalvular() -> list[AnatomicalMesh]:
    """Branching volumetric chords share exact membrane and muscle anchors."""
    meshes = []
    for chamber, valve, muscle_names in (
        (I.LEFT_VENTRICLE, I.MITRAL_VALVE, ("anterolateral", "posteromedial")),
        (I.RIGHT_VENTRICLE, I.TRICUSPID_VALVE, ("anterior", "posterior", "septal")),
    ):
        top, height, radii, tilt, thickness = CHAMBERS[chamber]
        kind = "lv" if chamber == I.LEFT_VENTRICLE else "rv"
        side = "izquierdo" if kind == "lv" else "derecho"
        valve_center, radius, _, _ = VALVES[valve]
        for n, name in enumerate(muscle_names):
            angle = math.pi * ((1.64, 1.90)[n] if kind == "lv" else (1.08, 1.30, 1.51)[n])
            base = _point(top, height, radii, tilt, 0.65, angle, kind, thickness * 0.15)
            tip = Vec3(valve_center.x + 0.15 * math.cos(angle),
                       -0.10 - 0.055 * (n % 2), valve_center.z - 0.11 + 0.08 * n)
            # Physiological papillary shortening along its axis during systole
            shortening = (base - tip) * 0.125
            points = [base * (1 - t) + tip * t + Vec3(0.012 * math.sin(math.pi * t), 0, -0.012 * math.sin(math.pi * t))
                      for t in (k / 28 for k in range(29))]
            points[0], points[-1] = base, tip
            muscle = _support_tube(I.PAPILLARY_MUSCLES, points, 0.098 if kind == "lv" else 0.080,
                                   PAPILLARY_COLOR, EXPLOSION[chamber], 24, 0.68)
            # Fusiform muscular belly profile: rooted in trabeculae, bulging at mid-belly, tapering at fibrous apex
            vertices = list(muscle.vertices)
            for k in range(29 * 24):
                row = k // 24
                t_row = row / 28
                root_blend = min(1.0, t_row / 0.20)
                root_factor = 0.08 + 0.92 * root_blend * root_blend * (3 - 2 * root_blend)
                belly_bulge = 1.0 + 0.22 * math.sin(math.pi * t_row)
                factor = root_factor * belly_bulge
                vertices[k] = points[row] + (vertices[k] - points[row]) * factor
            muscle = replace(muscle, vertices=tuple(vertices))
            data = []
            for k in range(len(muscle.vertices)):
                t = min(k // 24, 28) / 28 if k < 29 * 24 else float(k == len(muscle.vertices) - 1)
                delta = shortening * (t * t * (3 - 2 * t))
                data.append((0, 0, 0, 0, delta.x, delta.y, delta.z))
            muscle = replace(muscle, support_motion=tuple(data), motion="papillary")
            key = f"papillary_{chamber.value}_{n}"
            meshes.append(_owned(muscle, key, f"Músculo papilar {name} · ventrículo {side}", VENTRICULAR_ORIGIN))
            leaflet_count = VALVES[valve][2]
            for fan in range(3):
                leaflet = (n + fan % 2) % leaflet_count
                parameters = [(0.16 + 0.38 * n + 0.08 * twig + 0.014 * (fan - 1) if kind == "lv"
                               else 0.22 + 0.15 * twig + 0.024 * (fan - 1)) for twig in range(4)]
                edges = [_tilt_valve(av_leaflet_point(valve_center, radius, leaflet_count,
                          leaflet, u, 1), valve_center) for u in parameters]
                target = sum(edges[1:], edges[0]) * 0.25
                deltas = [_tilt_valve(av_leaflet_open_point(valve_center, radius, leaflet_count,
                          leaflet, u, 1), valve_center) - edge for u, edge in zip(parameters, edges)]
                fork_delta = sum(deltas[1:], deltas[0]) * (0.25 * 0.56)
                fork = tip * 0.44 + target * 0.56
                cord_key = f"chordae_{chamber.value}_{n}_{fan}"
                cord_name = f"Cuerdas tendinosas · {NAMES[valve].lower()} · fascículo {n * 3 + fan + 1}"
                segments = [(tip, fork, ZERO, fork_delta, 0, 0.56, 0.0085)]
                segments += [(fork, edge, fork_delta, delta, 0.56, 1, 0.0052) for edge, delta in zip(edges, deltas)]
                for a, b, delta_a, delta_b, w0, w1, cord_radius in segments:
                    points, data = [], []
                    for k in range(19):
                        t = k / 18
                        bend = math.sin(math.pi * t) if 0 < k < 18 else 0.0
                        slack = Vec3(0.008 * bend, -0.011 * bend, 0.005 * bend)
                        points.append(a * (1 - t) + b * t + slack)
                        opening = delta_a * (1 - t) + delta_b * t
                        weight = w0 * (1 - t) + w1 * t
                        delta = shortening * (1 - weight) - slack
                        data.extend([(opening.x, opening.y, opening.z, weight, delta.x, delta.y, delta.z)] * 14)
                    cord = _support_tube(I.CHORDAE_TENDINEAE, points, cord_radius, VALVULAR,
                                         EXPLOSION[chamber], 14, 0.37)
                    if w1 == 1:
                        vertices = list(cord.vertices)
                        for k in range(19 * 14):
                            t = (k // 14) / 18
                            vertices[k] = points[k // 14] + (vertices[k] - points[k // 14]) * (1 + 1.5 * t ** 10)
                        cord = replace(cord, vertices=tuple(vertices))
                    data.extend((data[0], data[-1]))
                    meshes.append(_owned(replace(cord, support_motion=tuple(data)), cord_key, cord_name, VENTRICULAR_ORIGIN, motion="chordal",
                                         valve_center=valve_center, valve_radius=radius))
    return meshes


def _surface(x: float, y: float, back: bool = False) -> Vec3:
    """Project coronary centerlines onto the assembled epicardial contour."""
    t = max(0, min(0.965, (0.46 - y) / 1.70))
    u = (x + 0.02 - 0.55 * t) / max(0.015, 0.92 * _contour(t, "lv"))
    angle = math.acos(max(-0.965, min(0.965, u)))
    if back:
        angle = math.tau - angle
    p = _envelope_point(t, angle)
    return Vec3(p.x, p.y, p.z + (-0.006 if back else 0.006))


def _coronaries() -> list[AnatomicalMesh]:
    """Build tortuous epicardial coronary arteries and cardiac veins embedded in sulci."""
    meshes = []
    for part, key, name, side, back, color in (
        (I.CORONARY_ARTERIES, "lad", "Arteria interventricular anterior", 1, False, ("#4fc4c2", "#d13b35")),
        (I.CORONARY_ARTERIES, "right_coronary", "Coronaria derecha y rama marginal", -1, False, ("#4fc4c2", "#d13b35")),
        (I.CORONARY_ARTERIES, "circumflex", "Arteria circunfleja", 1, True, ("#4fc4c2", "#d13b35")),
        (I.CORONARY_VEINS, "great_cardiac_vein", "Vena cardíaca magna", 1, False, ("#2f91b8", "#487da8")),
        (I.CORONARY_VEINS, "middle_cardiac_vein", "Vena cardíaca media", -1, True, ("#2f91b8", "#487da8")),
    ):
        path = []
        vein = part == I.CORONARY_VEINS
        for k in range(29):
            t = k / 28
            meander = 0.018 * math.sin(t * 17.0 + side * 1.4) + 0.009 * math.cos(t * 29.0)
            x = (0.015 + 0.48 * t + meander) if side == 1 else (-0.67 + 0.94 * t + meander)
            if vein:
                x += 0.032 * side
            if key == "circumflex":
                p = _envelope_point(0.025 + 0.035 * math.sin(math.pi * t), 0.20 - 2.75 * t)
                path.append(Vec3(p.x, p.y + 0.010 * math.sin(t * 19.0), p.z - 0.008))
            else:
                path.append(_surface(x, 0.41 - 1.50 * t, back))
        if not vein:
            root = Vec3(0.04 if side == 1 else -0.06, 0.67, -0.13 if back else 0.12)
            path = [root, *path]
        tree = [(path, 0.022 if not vein else 0.018)]
        for n in range(7):
            start = 3 + n * 3
            if start >= len(path):
                break
            a = path[start]
            branch = [a]
            direction = -1 if (n + side) % 2 else 1
            for j in range(1, 11):
                t = j / 10
                descent = 0.63 if key == "circumflex" else 0.26
                tort = 0.014 * math.sin(t * 11.0 + n * 2.1)
                bx = a.x + direction * 0.25 * t + tort
                by = a.y - descent * t + 0.022 * math.sin(t * 6.0 + n)
                branch.append(_surface(bx, by, back))
            tree.append((branch, 0.0105))
            if n % 2 == 0 and len(branch) > 5:
                sub_a = branch[4]
                sub_branch = [
                    _surface(
                        sub_a.x + direction * 0.11 * (m / 5),
                        sub_a.y - 0.14 * (m / 5) + 0.008 * math.cos(m),
                        back,
                    )
                    for m in range(6)
                ]
                tree.append((sub_branch, 0.0065))
        for points, radius in tree:
            mesh = _tube(part, points, radius, color, ZERO, 12, False, True, curve_steps=3, taper=0.72)
            meshes.append(_owned(mesh, key, name, VENTRICULAR_ORIGIN))
    sinus = _tube(I.CORONARY_SINUS,
                  [Vec3(0.64, 0.38, -0.48), Vec3(0.28, 0.37, -0.61), Vec3(-0.20, 0.40, -0.57),
                   Vec3(-0.47, 0.54, -0.24)], 0.040, VENOUS, Vec3(0, -0.10, -0.36), 18, False, False, curve_steps=9)
    meshes.append(_owned(sinus, "coronary_sinus"))
    return meshes


def _conduction() -> list[AnatomicalMesh]:
    """Show SA, AV, His, bundle branches and Purkinje network hugging the endocardial walls."""
    color = ("#52c7c6", "#f2cf66")
    ra_top, ra_h, ra_r, ra_tilt, ra_thick = CHAMBERS[I.RIGHT_ATRIUM]
    # SA node at superior-lateral crista terminalis (SVC-RA junction); AV node at Koch's triangle
    sa = _point(ra_top, ra_h, ra_r, ra_tilt, 0.20, math.pi * 1.18, "atrium", ra_thick + 0.005)
    av = Vec3(-0.045, 0.47, -0.19)
    meshes = []
    for key, name, center in (("sa_node", "Nodo sinoauricular", sa), ("av_node", "Nodo auriculoventricular", av)):
        meshes.append(_owned(_ellipsoid(I.CARDIAC_CONDUCTION, center, Vec3(0.024, 0.036, 0.019),
                                        color, ZERO, 16, 24, False, False), key, name))

    # Three classical internodal pathways hugging the right atrial posterior/septal endocardium
    internodal_wenckebach = [
        sa,
        _point(ra_top, ra_h, ra_r, ra_tilt, 0.42, math.pi * 1.42, "atrium", ra_thick + 0.006),
        _point(ra_top, ra_h, ra_r, ra_tilt, 0.68, math.pi * 1.68, "atrium", ra_thick + 0.006),
        Vec3(-0.18, 0.54, -0.27),
        av,
    ]
    internodal_thorel = [
        sa,
        _point(ra_top, ra_h, ra_r, ra_tilt, 0.52, math.pi * 1.24, "atrium", ra_thick + 0.006),
        _point(ra_top, ra_h, ra_r, ra_tilt, 0.82, math.pi * 1.50, "atrium", ra_thick + 0.006),
        Vec3(-0.24, 0.49, -0.24),
        av,
    ]
    # Bachmann's interatrial bundle curving along the posterior atrial roof
    bachmann_path = [
        sa,
        _point(ra_top, ra_h, ra_r, ra_tilt, 0.16, math.pi * 1.58, "atrium", ra_thick + 0.004),
        Vec3(-0.12, 1.00, -0.34),
        Vec3(0.16, 1.02, -0.40),
        Vec3(0.42, 0.96, -0.42),
    ]

    paths = [
        ("atrial_conduction", "Vías internodales auriculares (Wenckebach y Thorel)", internodal_wenckebach),
        ("atrial_conduction", "Vía internodal posterior de Thorel", internodal_thorel),
        ("bachmann", "Haz interauricular de Bachmann", bachmann_path),
        ("his_bundle", "Haz de His (fascículo penetrante)", [av, Vec3(0.02, 0.34, -0.17), Vec3(0.06, 0.14, -0.17)]),
        ("right_bundle", "Rama derecha del haz de His", [
            Vec3(0.06, 0.14, -0.17),
            Vec3(0.01, -0.18, -0.16),
            Vec3(-0.08, -0.44, -0.11),
            Vec3(-0.22, -0.52, -0.05),
            Vec3(-0.12, -0.78, -0.02),
        ]),
        ("left_bundle", "Rama izquierda del haz de His", [
            Vec3(0.06, 0.14, -0.17),
            Vec3(0.16, -0.16, -0.19),
            Vec3(0.28, -0.48, -0.22),
            Vec3(0.42, -0.86, -0.16),
            Vec3(0.48, -1.04, -0.08),
        ]),
    ]
    for part in (I.RIGHT_VENTRICLE, I.LEFT_VENTRICLE):
        top, h, radii, tilt, thickness = CHAMBERS[part]
        kind = "rv" if part == I.RIGHT_VENTRICLE else "lv"
        for n in range(7):
            a = math.pi + (n + 0.5) * math.pi / 7
            path = [_point(top, h, radii, tilt, 0.90 - k * 0.06, a + 0.07 * math.sin(k + n), kind, thickness + 0.014)
                    for k in range(11)]
            paths.append((f"purkinje_{kind}", f"Red subendocárdica de Purkinje · {NAMES[part].lower()}", path))
    for key, name, path in paths:
        center = VENTRICULAR_ORIGIN if key in {"right_bundle", "left_bundle", "purkinje_lv", "purkinje_rv"} else ra_top
        meshes.append(_owned(_tube(I.CARDIAC_CONDUCTION, path, 0.0056, color, ZERO, 8,
                                    False, False, curve_steps=5), key, name, center))
    return meshes


def build_cardiac_atlas() -> tuple[AnatomicalMesh, ...]:
    """Build paired exterior/interior anatomy with smooth vascular surfaces."""
    meshes = []
    envelope = _ventricular_envelope()
    meshes.extend(envelope)
    for part, (top, height, radii, tilt, thickness) in CHAMBERS.items():
        atrium = part in {I.RIGHT_ATRIUM, I.LEFT_ATRIUM}
        kind = "atrium" if atrium else ("rv" if part == I.RIGHT_VENTRICLE else "lv")
        for section in (False, True):
            wall = _shell(part, top, height, radii, tilt, thickness, MUSCLE, EXPLOSION[part], kind, section)
            deform = top if atrium else VENTRICULAR_ORIGIN
            if atrium or section:
                meshes.append(_owned(wall, part.value, center=deform))
            if not atrium:
                muscle_wall = wall if section else next(m for m in envelope if m.part_id == part)
                meshes.append(_owned(replace(muscle_wall, part_id=I.MYOCARDIUM, default_visible=False),
                                     f"myocardium_{part.value}", f"Miocardio · {NAMES[part].lower()}", deform))
            lining = _shell(I.ENDOCARDIUM, top, height,
                            (radii[0] * (1 - thickness - 0.007), radii[1] * (1 - thickness - 0.007)),
                            tilt, 0.010, LINING, EXPLOSION[part], kind, section, False)
            meshes.append(_owned(lining, f"endocardium_{part.value}", f"Endocardio · {NAMES[part].lower()}", deform))
            epicardium = _shell(I.EPICARDIUM, top, height, (radii[0] * 1.007, radii[1] * 1.007),
                               tilt, 0.006, ("#82c7ba", "#e8bbaa"), EXPLOSION[part], kind, section, False, 0.20)
            if not atrium and not section:
                surface = next(m for m in envelope if m.part_id == part)
                epicardium = replace(surface, part_id=I.EPICARDIUM,
                    vertices=tuple(VENTRICULAR_ORIGIN + (v - VENTRICULAR_ORIGIN) * 1.007 for v in surface.vertices),
                    analysis_color="#82c7ba", realistic_color="#e8bbaa", opacity=0.20, default_visible=False)
            meshes.append(_owned(epicardium, f"epicardium_{part.value}", f"Epicardio · {NAMES[part].lower()}", deform))
        if not atrium:
            meshes.extend(_wall_details(part))
        if atrium:
            left = part == I.LEFT_ATRIUM
            center = Vec3(0.64 if left else -0.59, 0.87, 0.10 if left else 0.11)
            auricle = _ellipsoid(part, center, Vec3(0.23, 0.16, 0.12), MUSCLE, EXPLOSION[part], 26, 40)
            meshes.append(_owned(auricle, f"auricle_{part.value}", "Orejuela izquierda" if left else "Orejuela derecha", top, view="exterior"))
            if not left:
                # Fleshy pectinate muscles (músculos pectíneos) & crista terminalis in right atrium
                for n in range(10):
                    path = [_point(top, height, radii, tilt, 0.22 + k * 0.045, math.pi + 0.12 + n * 0.088,
                                   kind, thickness - 0.008) for k in range(13)]
                    mesh = _tube(part, path, 0.0115, TRABECULAR_MUSCLE, EXPLOSION[part], 8, True, False)
                    meshes.append(_owned(mesh, part.value, center=top, view="interior"))
    for part, top, h, radii, tilt, kind in (
        (I.INTERVENTRICULAR_SEPTUM, Vec3(0.015, 0.47, -0.055), 1.63, (0.080, 0.39), 0.42, "septum"),
        (I.INTERATRIAL_SEPTUM, Vec3(-0.01, 1.07, -0.24), 0.61, (0.037, 0.26), 0.02, "atrium"),
    ):
        for section in (False, True):
            mesh = _shell(part, top, h, radii, tilt, 0.92, LINING, Vec3(0, -0.12, -0.28), kind, section, False)
            meshes.append(_owned(mesh, part.value, center=VENTRICULAR_ORIGIN if part == I.INTERVENTRICULAR_SEPTUM else top))
    fossa = _ellipsoid(I.INTERATRIAL_SEPTUM, Vec3(-0.052, 0.77, -0.17), Vec3(0.014, 0.085, 0.07),
                       ("#52adb5", "#c87d72"), Vec3(0, -0.12, -0.28), 18, 28, True, False)
    meshes.append(_owned(fossa, "fossa_ovalis", "Fosa oval", view="interior"))
    for section in (False, True):
        sac = _shell(I.PERICARDIUM, Vec3(0.02, 1.16, -0.08), 2.43, (1.00, 1.00), 0.24, 0.016,
                     ("#82c5cb", "#e6d2c4"), ZERO, "sac", section, True, 0.48 if section else 0.22)
        meshes.append(_owned(sac, "pericardium"))
    for part, key, name, radius, color, explosion, path in VESSELS:
        tube = _tube(part, [Vec3(*p) for p in path], radius, color, explosion, 36, False, True, curve_steps=12)
        intact = _hollow_vessel(tube)
        if key in {"aorta", "pulmonary_trunk"}:
            intact = replace(intact, view="exterior")
            meshes.append(_owned(_windowed_vessel(tube), key, name))
        meshes.append(_owned(intact, key, name))
    for part, (center, radius, count, semilunar) in VALVES.items():
        ring = _torus(part, center, radius, 0.012, VALVULAR, EXPLOSION[part], tilt=0 if semilunar else 0.48, default_visible=False)
        meshes.append(_owned(ring, part.value))
        leaf = _leaflets(part, center, radius, count, semilunar, explosion=EXPLOSION[part])
        if not semilunar:
            motion = []
            for data in leaf.support_motion:
                delta = _tilt_valve(Vec3(*data[:3]), ZERO)
                motion.append((delta.x, delta.y, delta.z, *data[3:]))
            leaf = replace(leaf, vertices=tuple(_tilt_valve(v, center) for v in leaf.vertices), support_motion=tuple(motion))
        meshes.append(_owned(replace(leaf, realistic_color=VALVULAR[1]), part.value,
                             valve_center=center, valve_radius=radius))
    meshes.extend(_subvalvular())
    meshes.extend(_coronaries())
    meshes.extend(_conduction())
    return tuple(meshes)