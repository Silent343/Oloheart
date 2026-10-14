"""Anatomical surface construction and paired, closed-edge dissection meshes.

Coordinates use anterior +Z and anatomical left +X. These procedural surfaces
are educational reconstructions, not patient-derived segmentations. Section
surfaces share the same anatomical coordinates as intact surfaces.
"""

from __future__ import annotations

import math
from dataclasses import replace

from oloheart.domain.model import HeartPartId as I
from oloheart.infrastructure.geometry import AnatomicalMesh, Vec3, _tube, _torus, _ellipsoid


# Chamber centers, heights and contours intentionally differ across chambers.
# The RV is anterior and crescent-like; the LV forms the inferior-left apex.
CHAMBERS = {
    I.RIGHT_VENTRICLE: (Vec3(-0.26, 0.43, 0.12), 1.42, (0.68, 0.48), 0.52, 0.12),
    I.LEFT_VENTRICLE: (Vec3(0.23, 0.46, -0.06), 1.68, (0.67, 0.57), 0.30, 0.28),
    I.RIGHT_ATRIUM: (Vec3(-0.48, 1.10, -0.18), 0.67, (0.44, 0.40), 0.08, 0.065),
    I.LEFT_ATRIUM: (Vec3(0.38, 1.16, -0.33), 0.72, (0.41, 0.36), -0.015, 0.065),
}
VALVES = {
    I.TRICUSPID_VALVE: (Vec3(-0.40, 0.43, -0.01), 0.245, 3, False),
    I.MITRAL_VALVE: (Vec3(0.365, 0.44, -0.09), 0.240, 2, False),
    I.AORTIC_VALVE: (Vec3(0.075, 0.59, 0.005), 0.146, 3, True),
    I.PULMONARY_VALVE: (Vec3(-0.19, 0.63, 0.31), 0.151, 3, True),
}


def _contour(t: float, kind: str) -> float:
    if kind == "atrium":
        # A broad basal opening joins the AV junction; the atrial roof narrows.
        return 0.20 + 0.44 * t + 0.62 * math.sin(math.pi * t) ** 0.70
    if kind == "sac":
        return (0.36 + 0.68 * math.sin(math.pi * t) ** 0.78) * (1.0 - 0.34 * t)
    return max(0.025, (1.0 - t) ** 0.53 * (0.70 + 0.48 * math.sin(math.pi * t)))


def _point(top: Vec3, height: float, radii: tuple[float, float], tilt: float,
           t: float, angle: float, kind: str, inset: float = 0.0) -> Vec3:
    if kind == "sac":
        profile = (
            (0.00, -0.02, -0.13, 0.44, 0.31),
            (0.11, -0.03, -0.10, 0.74, 0.44),
            (0.26, -0.01, -0.03, 0.82, 0.53),
            (0.46,  0.11, -0.01, 0.76, 0.50),
            (0.67,  0.28, -0.01, 0.57, 0.39),
            (0.86,  0.43, -0.02, 0.34, 0.24),
            (1.00,  0.52, -0.025, 0.045, 0.040),
        )
        index = next((k for k in range(len(profile) - 1) if t <= profile[k + 1][0]), len(profile) - 2)
        a, b = profile[index], profile[index + 1]
        weight = (t - a[0]) / max(1e-6, (b[0] - a[0]))
        previous = profile[max(0, index - 1)]
        following = profile[min(len(profile) - 1, index + 2)]
        duration = b[0] - a[0]
        values = []
        for k in (1, 2, 3, 4):
            slope_a = (b[k] - previous[k]) / max(1e-6, (b[0] - previous[0]))
            slope_b = (following[k] - a[k]) / max(1e-6, (following[0] - a[0]))
            w2, w3 = weight * weight, weight * weight * weight
            values.append(
                (2 * w3 - 3 * w2 + 1) * a[k]
                + (w3 - 2 * w2 + weight) * duration * slope_a
                + (-2 * w3 + 3 * w2) * b[k]
                + (w3 - w2) * duration * slope_b
            )
        cx, cz, rx, rz = values
        y = top.y - height * t
        dx = rx * radii[0] * math.cos(angle)
        dz = rz * radii[1] * math.sin(angle)
        radius = max(math.hypot(dx, dz), 0.001)
        nx, nz = dx / radius, dz / radius

        vent_t = (0.46 - y) / 1.70
        if 0.0 <= vent_t <= 1.0:
            vw = _contour(vent_t, "lv")
            vcx = -0.02 + 0.55 * vent_t
            vcz = -0.025 + 0.025 * math.sin(math.pi * vent_t)
            vrx = max(0.03, 0.92 * vw)
            vrz = max(0.03, 0.61 * vw)
            ox, oz = cx - vcx, cz - vcz
            qa = (nx / vrx) ** 2 + (nz / vrz) ** 2
            qb = 2.0 * (ox * nx / vrx ** 2 + oz * nz / vrz ** 2)
            qc = (ox / vrx) ** 2 + (oz / vrz) ** 2 - 1.0
            disc = qb * qb - 4.0 * qa * qc
            if disc > 0.0:
                v_rad = (-qb + math.sqrt(disc)) / (2.0 * qa)
                radius = max(radius, v_rad + 0.030)

        for chamber_top, chamber_height, chamber_radii, chamber_tilt, _ in CHAMBERS.values():
            depth = (chamber_top.y - y) / chamber_height
            if not 0.0 <= depth <= 1.0:
                continue
            shape = "atrium" if chamber_height < 1.0 else "lv"
            w = _contour(depth, shape)
            crx = max(0.025, chamber_radii[0] * w)
            crz = max(0.025, chamber_radii[1] * w)
            ox = cx - (chamber_top.x + chamber_tilt * depth)
            oz = cz - (chamber_top.z + 0.025 * math.sin(math.pi * depth))
            qa = (nx / crx) ** 2 + (nz / crz) ** 2
            qb = 2.0 * (ox * nx / crx ** 2 + oz * nz / crz ** 2)
            qc = (ox / crx) ** 2 + (oz / crz) ** 2 - 1.0
            discriminant = qb * qb - 4.0 * qa * qc
            if discriminant > 0.0:
                c_rad = (-qb + math.sqrt(discriminant)) / (2.0 * qa)
                radius = max(radius, c_rad + 0.026)

        fold = 1.0 + 0.008 * math.sin(angle * 3.0 + t * 5.0) * math.sin(math.pi * t)
        r_final = max(0.012, radius * fold * (1.0 - inset))
        return Vec3(cx + nx * r_final, y, cz + nz * r_final)

    width = _contour(t, kind)
    s_t = math.sin(math.pi * t)
    organic = (
        1.0
        + 0.022 * math.sin(angle * 2.0 + t * 4.5) * s_t
        + 0.011 * math.cos(angle * 3.0 - t * 7.0) * s_t
        + 0.005 * math.sin(angle * 5.0 + t * 11.0) * s_t
    )
    x = math.cos(angle) * radii[0] * width * organic * (1.0 - inset)
    z = math.sin(angle) * radii[1] * width * organic * (1.0 - inset)
    world_x = top.x + tilt * t + x
    y = top.y - height * t
    if kind in {"lv", "rv"}:
        septum = 0.015 + 0.42 * max(0.0, (0.46 - y) / 1.68) ** 1.30
        boundary = septum + (0.06 * inset if kind == "lv" else -0.06 * inset)
        world_x = max(boundary, world_x) if kind == "lv" else min(boundary, world_x)
    atrial_outlet = (0.17 if top.x < 0 else 0.24) * t if kind == "atrium" else 0.0
    return Vec3(world_x, y, top.z + z + 0.025 * s_t + atrial_outlet)


def _shell(part: I, top: Vec3, height: float, radii: tuple[float, float],
           tilt: float, thickness: float, colors: tuple[str, str], explosion: Vec3,
           kind: str, section: bool, default: bool = True,
           opacity: float = 1.0) -> AnatomicalMesh:
    """Build outer and endocardial surfaces plus physical rims along the opening."""
    rows, columns = 48, 80 if not section else 48
    start, span = (0.0, math.tau) if not section else (math.pi, math.pi)
    vertices: list[Vec3] = []
    stride = columns + 1
    for inset in (0.0, thickness):
        for row in range(rows + 1):
            t = row / rows
            for column in range(stride):
                angle = start + span * column / columns
                vertices.append(_point(top, height, radii, tilt, t, angle, kind, inset))
    offset = (rows + 1) * stride
    faces: list[tuple[int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            a = row * stride + column
            b, c, d = a + 1, a + stride, a + stride + 1
            faces.extend(((a, c, b), (b, c, d)))
            faces.extend(((a + offset, b + offset, c + offset),
                          (b + offset, d + offset, c + offset)))
    edges = [(c, c + 1) for c in range(columns)]
    edges += [(rows * stride + c + 1, rows * stride + c) for c in range(columns)]
    if section:
        edges += [((r + 1) * stride, r * stride) for r in range(rows)]
        edges += [(r * stride + columns, (r + 1) * stride + columns) for r in range(rows)]
    for a, b in edges:
        faces.extend(((a, b, a + offset), (b, b + offset, a + offset)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), *colors, explosion,
                          True, default, "interior" if section else "exterior",
                          "none", opacity)


def av_leaflet_point(center: Vec3, radius: float, count: int, leaflet: int,
                     u: float, v: float) -> Vec3:
    """A scalloped parachute-like AV leaflet with saddle annulus and coaptation zone.

    u follows the annular arc; v runs from the fixed hinge (v=0) to the free
    scalloped coapting margin (v=1).
    """
    start = math.pi / 2 + math.tau * leaflet / count
    end = start + math.tau / count
    angle = start + (end - start) * u
    # Saddle-shaped annular height modulation (physiological mitral/tricuspid saddle)
    annular_saddle = 0.014 * math.cos(2.0 * angle) * (1.0 - v)
    outer = Vec3(radius * math.cos(angle), annular_saddle, radius * math.sin(angle))
    a = Vec3(radius * math.cos(start), 0.0, radius * math.sin(start))
    b = Vec3(radius * math.cos(end), 0.0, radius * math.sin(end))
    if count == 2:
        edge = a * (1.0 - u) + b * u
    elif u <= 0.5:
        edge = a * (1.0 - 2.0 * u)
    else:
        edge = b * (2.0 * u - 1.0)
    p = outer * (1.0 - v) + edge * v
    # Scalloped radial fullness (rough zone & clear zone of the leaflet belly)
    scallop = 1.0 + 0.06 * math.sin(math.pi * u) * math.sin(math.pi * v)
    p = Vec3(p.x * scallop, p.y, p.z * scallop) + outer.normalized() * (0.002 * math.sin(math.pi * u) * v)
    # Parachute coaptation belly & downward free-marginrough zone lip
    belly = -0.038 * math.sin(math.pi * (v ** 0.85)) * (0.35 + 0.65 * math.sin(math.pi * u))
    margin = -0.022 * (v ** 1.6) * (0.45 + 0.55 * math.sin(math.pi * u))
    coapt_lip = -0.010 * math.sin(math.pi * u * 3.0) * (v ** 2)
    return center + Vec3(p.x, p.y + belly + margin + coapt_lip, p.z)


def av_leaflet_open_point(center: Vec3, radius: float, count: int, leaflet: int,
                          u: float, v: float) -> Vec3:
    """Funnel-like diastolic opening into the ventricular inflow tract."""
    closed = av_leaflet_point(center, radius, count, leaflet, u, v)
    hinge = av_leaflet_point(center, radius, count, leaflet, u, 0.0)
    # In diastole, the leaflet swings outward toward the ventricular wall (90% of radius)
    # and forms a deep vertical inflow sleeve/funnel toward the apex
    radial_open = closed * (1.0 - 0.90 * v) + hinge * (0.90 * v)
    funnel_drop = -0.195 * (v ** 0.92) * (0.38 + 0.62 * math.sin(math.pi * u))
    return Vec3(radial_open.x, closed.y + funnel_drop, radial_open.z)


def _leaflets(part: I, center: Vec3, radius: float, count: int,
              semilunar: bool, explosion: Vec3) -> AnatomicalMesh:
    """Create two mitral or three tricuspid/semilunar cusps with true 3D pocket morphology."""
    vertices: list[Vec3] = []
    faces: list[tuple[int, int, int]] = []
    motion = []
    radial_steps, angular_steps = 16, 32
    for leaflet in range(count):
        base = len(vertices)
        start_angle = math.tau * leaflet / count
        span_angle = math.tau / count
        mid_angle = start_angle + 0.5 * span_angle
        for row in range(radial_steps + 1):
            v = row / radial_steps  # 0 at outer wall hinge, 1 at central coaptation/free edge
            for column in range(angular_steps + 1):
                u = column / angular_steps  # 0 to 1 across cusp commissures
                if not semilunar:
                    point = av_leaflet_point(center, radius, count, leaflet, u, v)
                    delta = av_leaflet_open_point(center, radius, count, leaflet, u, v) - point
                    vertices.append(point)
                    motion.append((delta.x, delta.y, delta.z, 0, 0, 0, 0))
                    continue

                # Semilunar pocket cusp ("nido de golondrina" / Sinus of Valsalva cusp):
                # Outer U-shaped fibrous attachment crown + deep diastolic belly + Nodule of Arantius
                theta = start_angle + u * span_angle
                sin_u = math.sin(math.pi * u)
                # Outer ring at v=0, converging to the tri-leaflet coaptation lines & central Arantius nodule at v=1
                r_outer = radius * (1.0 + 0.04 * sin_u)
                # At v=1, the free edge runs from commissure (u=0) through the center (u=0.5) to commissure (u=1)
                c_start = Vec3(radius * math.cos(start_angle), 0.052, radius * math.sin(start_angle))
                c_end = Vec3(radius * math.cos(start_angle + span_angle), 0.052, radius * math.sin(start_angle + span_angle))
                nodule = Vec3(0.008 * math.cos(mid_angle), 0.018, 0.008 * math.sin(mid_angle))
                if u <= 0.5:
                    w = u * 2.0
                    free_edge = c_start * (1.0 - w) + nodule * w
                else:
                    w = (u - 0.5) * 2.0
                    free_edge = nodule * (1.0 - w) + c_end * w

                # U-shaped semilunar hinge crown (high at commissures u=0,1; deep at sinus base u=0.5)
                crown_y = 0.052 * (1.0 - 0.85 * sin_u) - 0.032 * sin_u
                hinge_pt = Vec3(r_outer * math.cos(theta), crown_y, r_outer * math.sin(theta))

                # Interpolate from sinus hinge to free lunula edge with deep pocket belly
                pocket_sag = -0.058 * math.sin(math.pi * v) * sin_u
                radial_belly = 0.022 * math.sin(math.pi * v) * sin_u
                p_local = hinge_pt * (1.0 - v) + free_edge * v
                p_local = Vec3(
                    p_local.x + radial_belly * math.cos(theta),
                    p_local.y + pocket_sag,
                    p_local.z + radial_belly * math.sin(theta),
                )
                vertices.append(center + p_local)
        for row in range(radial_steps):
            for column in range(angular_steps):
                a = base + row * (angular_steps + 1) + column
                b, c, d = a + 1, a + angular_steps + 1, a + angular_steps + 2
                faces.extend(((a, c, b), (b, c, d)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), "#41bac2", "#e8c4b0",
                          explosion, False, False, "all",
                          "semilunar" if semilunar else "atrioventricular", support_motion=tuple(motion))


def _hollow_vessel(mesh: AnatomicalMesh) -> AnatomicalMesh:
    """Add a luminal wall and annular end faces to an existing tube surface."""
    sides = mesh.faces[0][1]
    if sides < 5 or len(mesh.vertices) % sides:
        return mesh
    vertices = list(mesh.vertices)
    for start in range(0, len(mesh.vertices), sides):
        ring = mesh.vertices[start:start + sides]
        center = Vec3(sum(v.x for v in ring) / sides, sum(v.y for v in ring) / sides,
                      sum(v.z for v in ring) / sides)
        vertices.extend(center + (v - center) * 0.82 for v in ring)
    offset = len(mesh.vertices)
    faces = list(mesh.faces)
    faces.extend((a + offset, c + offset, b + offset) for a, b, c in mesh.faces)
    for start in (0, offset - sides):
        for side in range(sides):
            a, b = start + side, start + (side + 1) % sides
            faces.extend(((a, b, a + offset), (b, b + offset, a + offset)))
    return replace(mesh, vertices=tuple(vertices), faces=tuple(faces))


def refine_anatomy(original: tuple[AnatomicalMesh, ...]) -> tuple[AnatomicalMesh, ...]:
    """Replace generic walls and rings while retaining vascular/conduction routes."""
    replacement_parts = set(CHAMBERS) | set(VALVES) | {
        I.PERICARDIUM, I.EPICARDIUM, I.MYOCARDIUM, I.ENDOCARDIUM,
        I.PAPILLARY_MUSCLES, I.CHORDAE_TENDINEAE,
        I.CARDIAC_CONDUCTION,
        I.INTERVENTRICULAR_SEPTUM, I.INTERATRIAL_SEPTUM,
    }
    explosion = {m.part_id: m.explosion_vector for m in original}
    meshes = [m for m in original if m.part_id not in replacement_parts]
    meshes = [_hollow_vessel(m) if m.part_id in {
        I.AORTA, I.PULMONARY_ARTERY, I.VENA_CAVA, I.PULMONARY_VEINS,
    } else m for m in meshes]
    for part, top, height, radii, tilt, kind in (
        (I.INTERVENTRICULAR_SEPTUM, Vec3(0.065, 0.45, -0.12), 1.57, (0.080, 0.34), 0.24, "lv"),
        (I.INTERATRIAL_SEPTUM, Vec3(0.0, 1.01, -0.19), 0.61, (0.045, 0.23), 0.0, "atrium"),
    ):
        for section in (False, True):
            meshes.append(_shell(part, top, height, radii, tilt, 0.94, ("#438e98", "#b56865"),
                                 explosion[part], kind, section, False))
    for part, (top, height, radii, tilt, thickness) in CHAMBERS.items():
        atrium = part in {I.RIGHT_ATRIUM, I.LEFT_ATRIUM}
        kind = "atrium" if atrium else ("rv" if part == I.RIGHT_VENTRICLE else "lv")
        colors = ("#228693", "#963d48" if atrium else "#ae4b50")
        for section in (False, True):
            meshes.append(_shell(part, top, height, radii, tilt, thickness,
                                 colors, explosion[part], kind, section))
            meshes.append(_shell(I.EPICARDIUM, top, height,
                                 (radii[0] * 1.008, radii[1] * 1.008), tilt, 0.009,
                                 ("#87c4bc", "#ddb296"), explosion[part], kind,
                                 section, True, 0.16))
            if not atrium:
                wall = next(m for m in reversed(meshes) if m.part_id == part)
                meshes.append(replace(wall, part_id=I.MYOCARDIUM,
                                      default_visible=False, explosion_vector=Vec3(0, 0, 0)))
                lining = _shell(I.ENDOCARDIUM, top, height,
                                 (radii[0] * (1 - thickness - 0.008), radii[1] * (1 - thickness - 0.008)),
                                 tilt, 0.016, ("#6ac4c3", "#d7857b"), explosion[part],
                                 kind, section, False)
                meshes.append(lining)
        ridge_count = 13 if not atrium else (8 if part == I.RIGHT_ATRIUM else 4)
        for n in range(ridge_count):
            angle = math.pi + (n + 0.5) * math.pi / ridge_count
            path = []
            for step in range(14):
                t = 0.18 + step / 13 * (0.70 if not atrium else 0.54)
                bend = angle + 0.13 * math.sin(t * 10 + n)
                path.append(_point(top, height, radii, tilt, t, bend, kind,
                                   thickness + 0.035))
            meshes.append(_tube(part, path, 0.018 if atrium else 0.024,
                                ("#4fadae", "#a83842"), explosion[part],
                                8, True, False, curve_steps=2, taper=0.4))
        if not atrium:
            for n in range(16):
                t = 0.27 + (n % 5) * 0.12
                a = math.pi + 0.16 + (n // 5) * 0.78
                path = [_point(top, height, radii, tilt, t + 0.04 * math.sin(k),
                               a + k * 0.11, kind, thickness + 0.035) for k in range(6)]
                meshes.append(_tube(part, path, 0.015, ("#48999f", "#b2424b"),
                                    explosion[part], 7, True, False))

    for part, radii, thickness, color, opacity in (
        (I.PERICARDIUM, (1.00, 1.00), 0.018, "#e2cfc2", 0.22),
    ):
        for section in (False, True):
            meshes.append(_shell(part, Vec3(0.02, 1.16, -0.08), 2.43,
                                 radii, 0.22, thickness, ("#83c3ce", color),
                                 Vec3(0, 0, 0), "sac", section, True,
                                 0.54 if section else opacity))

    for part, (center, radius, count, semilunar) in VALVES.items():
        meshes.append(_torus(part, center, radius, 0.018,
                             ("#44b2b9", "#d5a994"), explosion[part], default_visible=False))
        meshes.append(_leaflets(part, center, radius, count, semilunar, explosion[part]))

    for chamber, valve, count in ((I.LEFT_VENTRICLE, I.MITRAL_VALVE, 2),
                                 (I.RIGHT_VENTRICLE, I.TRICUSPID_VALVE, 3)):
        center, radius, _, _ = VALVES[valve]
        for muscle in range(count):
            a = math.pi * (1.12 + 0.76 * muscle / max(1, count - 1))
            tip = Vec3(center.x + 0.14 * math.cos(a), -0.24, center.z - 0.10)
            top, height, radii, tilt, thickness = CHAMBERS[chamber]
            base = _point(top, height, radii, tilt, 0.66, a,
                          "rv" if chamber == I.RIGHT_VENTRICLE else "lv", thickness + 0.025)
            meshes.append(_tube(I.PAPILLARY_MUSCLES, [base, tip], 0.066,
                                ("#39959f", "#a3333d"), explosion[chamber], 14,
                                True, False, taper=0.50))
            for cord in range(7):
                theta = math.tau * (muscle + (cord + 0.5) / 7) / count
                endpoint = Vec3(center.x + radius * 0.65 * math.cos(theta),
                                center.y - 0.025, center.z + radius * 0.65 * math.sin(theta))
                meshes.append(_tube(I.CHORDAE_TENDINEAE, [tip, endpoint], 0.007,
                                    ("#90d2cf", "#ead0ba"), explosion[chamber],
                                    6, False, False))
    conduction_color = ("#35b9c0", "#d6b456")
    sa, av = Vec3(-0.52, 0.91, -0.23), Vec3(-0.08, 0.45, -0.16)
    for center in (sa, av):
        meshes.append(_ellipsoid(I.CARDIAC_CONDUCTION, center, Vec3(0.034, 0.045, 0.025),
                                 conduction_color, Vec3(0, 0, 0), 10, 16, False, False))
    paths = ([sa, Vec3(-0.35, 0.72, -0.28), av],
             [av, Vec3(0.0, 0.10, -0.21), Vec3(0.12, -0.54, -0.25)],
             [Vec3(0.12, -0.54, -0.25), Vec3(-0.10, -0.76, -0.14), Vec3(-0.21, -0.40, -0.24)],
             [Vec3(0.12, -0.54, -0.25), Vec3(0.38, -1.06, -0.14), Vec3(0.58, -0.55, -0.25)])
    for path in paths:
        meshes.append(_tube(I.CARDIAC_CONDUCTION, list(path), 0.009,
                            conduction_color, Vec3(0, 0, 0), 8, False, False, curve_steps=5))
    return tuple(meshes)