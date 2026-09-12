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
        return (0.40 + 0.77 * math.sin(math.pi * t) ** 0.75) * (1.0 - 0.36 * t)
    return max(0.025, (1.0 - t) ** 0.53 * (0.70 + 0.48 * math.sin(math.pi * t)))


def _point(top: Vec3, height: float, radii: tuple[float, float], tilt: float,
           t: float, angle: float, kind: str, inset: float = 0.0) -> Vec3:
    if kind == "sac":
        # Cross-sectional envelope follows the atria, shoulders and LV apex.
        profile = ((0.0, -0.02, 0.36, 0.34), (0.12, -0.04, 0.73, 0.49),
                   (0.29, -0.01, 0.91, 0.60), (0.48, 0.04, 0.83, 0.57),
                   (0.68, 0.12, 0.66, 0.48), (0.85, 0.27, 0.42, 0.33),
                   (1.0, 0.44, 0.012, 0.012))
        index = next((k for k in range(len(profile)-1) if t <= profile[k+1][0]), len(profile)-2)
        a, b = profile[index], profile[index+1]
        weight = (t-a[0]) / (b[0]-a[0])
        previous, following = profile[max(0,index-1)], profile[min(len(profile)-1,index+2)]
        duration = b[0]-a[0]
        values = []
        for k in (1,2,3):
            slope_a = (b[k]-previous[k]) / (b[0]-previous[0])
            slope_b = (following[k]-a[k]) / (following[0]-a[0])
            w2, w3 = weight*weight, weight*weight*weight
            values.append((2*w3-3*w2+1)*a[k] + (w3-2*w2+weight)*duration*slope_a
                          + (-2*w3+3*w2)*b[k] + (w3-w2)*duration*slope_b)
        cx, rx, rz = values
        y = top.y-height*t
        dx, dz = rx * 1.06 * math.cos(angle), rz * 1.10 * math.sin(angle)
        radius = max(math.hypot(dx,dz),0.001)
        nx, nz = dx/radius, dz/radius
        # Fit locally around chamber contours rather than inflating the entire
        # sac to accommodate one shoulder. A small clearance avoids intersections.
        for chamber_top, chamber_height, chamber_radii, chamber_tilt, _ in CHAMBERS.values():
            depth = (chamber_top.y-y)/chamber_height
            if not 0 <= depth <= 1:
                continue
            shape = "atrium" if chamber_height < 1 else "lv"
            w = _contour(depth,shape)*1.02
            crx, crz = chamber_radii[0]*w, chamber_radii[1]*w
            ox = cx-chamber_top.x-chamber_tilt*depth
            oz = 0.01-chamber_top.z-0.025*math.sin(math.pi*depth)
            a = (nx/crx)**2+(nz/crz)**2
            b = 2*(ox*nx/crx**2+oz*nz/crz**2)
            c = (ox/crx)**2+(oz/crz)**2-1
            discriminant = b*b-4*a*c
            if discriminant > 0:
                radius = max(radius,(-b+math.sqrt(discriminant))/(2*a)+0.025)
        return Vec3(cx + nx*radius*(1-inset), y, 0.01+nz*radius*(1-inset))
    width = _contour(t, kind)
    organic = 1.0 + 0.026 * math.sin(angle * 3 + t * 6) * math.sin(math.pi * t)
    x = math.cos(angle) * radii[0] * width * organic * (1.0 - inset)
    z = math.sin(angle) * radii[1] * width * organic * (1.0 - inset)
    # The RV wraps around the septal side rather than forming a second LV cone.
    world_x = top.x + tilt * t + x
    y = top.y - height * t
    if kind in {"lv", "rv"}:
        septum = 0.015 + 0.42 * max(0.0, (0.46-y)/1.68) ** 1.30
        # Adjacent D-shaped cavities meet at a common septal wall. The RV
        # terminates on the LV surface instead of creating a second free apex.
        boundary = septum + (0.06 * inset if kind == "lv" else -0.06 * inset)
        world_x = max(boundary, world_x) if kind == "lv" else min(boundary, world_x)
    atrial_outlet = (0.17 if top.x < 0 else 0.24) * t if kind == "atrium" else 0.0
    return Vec3(world_x, y, top.z + z + 0.025 * math.sin(math.pi * t) + atrial_outlet)


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
    # Bridge wall thickness at superior/inferior rims and both cut edges.
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
    """A scalloped leaflet spans its annular arc and a coapting free edge.

    u follows the annular arc; v runs from the fixed hinge to the free margin.
    Both the membrane and its chordal insertions consume this same surface.
    """
    start = math.pi / 2 + math.tau * leaflet / count
    end = start + math.tau / count
    angle = start + (end-start)*u
    outer = Vec3(radius*math.cos(angle),0,radius*math.sin(angle))
    a,b = Vec3(radius*math.cos(start),0,radius*math.sin(start)), Vec3(radius*math.cos(end),0,radius*math.sin(end))
    if count == 2:
        edge = a*(1-u)+b*u
    elif u <= .5:
        edge = a*(1-2*u)
    else:
        edge = b*(2*u-1)
    p = outer*(1-v)+edge*v
    p = p+outer.normalized()*(.0015*math.sin(math.pi*u)*v)
    belly = -.026*math.sin(math.pi*v)*math.sin(math.pi*u)
    margin = -.012*math.sin(math.pi*u)*v
    return center+Vec3(p.x,belly+margin,p.z)


def av_leaflet_open_point(center: Vec3, radius: float, count: int, leaflet: int,
                          u: float, v: float) -> Vec3:
    """Fold toward the leaflet's annular hinge without crossing the orifice."""
    closed = av_leaflet_point(center,radius,count,leaflet,u,v)
    hinge = av_leaflet_point(center,radius,count,leaflet,u,0)
    opened = closed*(1-.90*v)+hinge*(.90*v)
    return Vec3(opened.x,closed.y-.16*v*math.sin(math.pi*u),opened.z)


def _leaflets(part: I, center: Vec3, radius: float, count: int,
              semilunar: bool, explosion: Vec3) -> AnatomicalMesh:
    """Create two mitral or three other cusps, with curved coapting surfaces."""
    vertices: list[Vec3] = []
    faces: list[tuple[int, int, int]] = []
    motion = []
    radial_steps, angular_steps = 16, 32
    for leaflet in range(count):
        base = len(vertices)
        for row in range(radial_steps + 1):
            r = 0.008 + (radius - 0.008) * row / radial_steps
            for column in range(angular_steps + 1):
                if not semilunar:
                    point = av_leaflet_point(center,radius,count,leaflet,column/angular_steps,row/radial_steps)
                    delta = av_leaflet_open_point(center,radius,count,leaflet,column/angular_steps,row/radial_steps)-point
                    vertices.append(point)
                    motion.append((delta.x,delta.y,delta.z,0,0,0,0))
                    continue
                theta = math.tau * (leaflet + column / angular_steps) / count
                belly = math.sin(math.pi * r / radius) * 0.045
                vertices.append(Vec3(center.x + r * math.cos(theta),
                                     center.y - belly,
                                     center.z + r * math.sin(theta)))
        for row in range(radial_steps):
            for column in range(angular_steps):
                a = base + row * (angular_steps + 1) + column
                b, c, d = a + 1, a + angular_steps + 1, a + angular_steps + 2
                faces.extend(((a, c, b), (b, c, d)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), "#41bac2", "#e3b5a4",
                          explosion, False, False, "all",
                          "semilunar" if semilunar else "atrioventricular",support_motion=tuple(motion))


def _hollow_vessel(mesh: AnatomicalMesh) -> AnatomicalMesh:
    """Add a luminal wall and annular end faces to an existing tube surface."""
    sides = mesh.faces[0][1]
    if sides < 5 or len(mesh.vertices) % sides:
        return mesh
    vertices = list(mesh.vertices)
    for start in range(0, len(mesh.vertices), sides):
        ring = mesh.vertices[start:start+sides]
        center = Vec3(sum(v.x for v in ring)/sides, sum(v.y for v in ring)/sides,
                      sum(v.z for v in ring)/sides)
        vertices.extend(center + (v-center)*0.82 for v in ring)
    offset = len(mesh.vertices)
    faces = list(mesh.faces)
    faces.extend((a+offset,c+offset,b+offset) for a,b,c in mesh.faces)
    for start in (0, offset-sides):
        for side in range(sides):
            a, b = start+side, start+(side+1)%sides
            faces.extend(((a,b,a+offset),(b,b+offset,a+offset)))
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
    # Septa retain a thick posterior wall while the anterior cut exposes its edge.
    for part, top, height, radii, tilt, kind in (
        (I.INTERVENTRICULAR_SEPTUM,Vec3(0.065,0.45,-0.12),1.57,(0.080,0.34),0.24,"lv"),
        (I.INTERATRIAL_SEPTUM,Vec3(0.0,1.01,-0.19),0.61,(0.045,0.23),0.0,"atrium"),
    ):
        for section in (False,True):
            meshes.append(_shell(part,top,height,radii,tilt,0.94,("#438e98","#b56865"),
                                 explosion[part],kind,section,False))
    for part, (top, height, radii, tilt, thickness) in CHAMBERS.items():
        atrium = part in {I.RIGHT_ATRIUM, I.LEFT_ATRIUM}
        kind = "atrium" if atrium else ("rv" if part == I.RIGHT_VENTRICLE else "lv")
        colors = ("#228693", "#963d48" if atrium else "#ae4b50")
        for section in (False, True):
            meshes.append(_shell(part, top, height, radii, tilt, thickness,
                                 colors, explosion[part], kind, section))
            meshes.append(_shell(I.EPICARDIUM, top, height,
                                 (radii[0]*1.008, radii[1]*1.008), tilt, 0.009,
                                 ("#87c4bc", "#ddb296"), explosion[part], kind,
                                 section, True, 0.16))
            # Myocardium is the actual chamber wall, not detached helical wires.
            if not atrium:
                wall = next(m for m in reversed(meshes) if m.part_id == part)
                meshes.append(replace(wall, part_id=I.MYOCARDIUM,
                                      default_visible=False, explosion_vector=Vec3(0, 0, 0)))
                lining = _shell(I.ENDOCARDIUM, top, height,
                                 (radii[0] * (1-thickness-0.008), radii[1] * (1-thickness-0.008)),
                                 tilt, 0.016, ("#6ac4c3", "#d7857b"), explosion[part],
                                 kind, section, False)
                meshes.append(lining)
        # Attached ridges follow the inner surface. RV trabeculae are coarser.
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
                                ("#4fadae", "#bb615e"), explosion[part],
                                8, True, False, curve_steps=2, taper=0.4))
        # Fine transverse branches add a trabecular network rather than loose rods.
        if not atrium:
            for n in range(16):
                t = 0.27 + (n % 5) * 0.12
                a = math.pi + 0.16 + (n // 5) * 0.78
                path = [_point(top, height, radii, tilt, t + 0.04 * math.sin(k),
                               a + k * 0.11, kind, thickness + 0.035) for k in range(6)]
                meshes.append(_tube(part, path, 0.015, ("#48999f", "#ce776d"),
                                    explosion[part], 7, True, False))

    # A pericardial sac envelops the entire heart; its thin section rim exposes
    # the parietal wall and potential cavity rather than two separated disks.
    for part, radii, thickness, color, opacity in (
        (I.PERICARDIUM, (1.10, 0.84), 0.026, "#dac4b5", 0.26),
    ):
        for section in (False, True):
            meshes.append(_shell(part, Vec3(0.0, 1.22, -0.08), 2.63,
                                 radii, 0.22, thickness, ("#83c3ce", color),
                                 Vec3(0, 0, 0), "sac", section, True,
                                 0.72 if section else opacity))

    for part, (center, radius, count, semilunar) in VALVES.items():
        meshes.append(_torus(part, center, radius, 0.018,
                             ("#44b2b9", "#d5a994"), explosion[part], default_visible=False))
        meshes.append(_leaflets(part, center, radius, count, semilunar, explosion[part]))

    # Two LV and three RV papillary groups support AV leaflets only.
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
                                ("#39959f", "#bf665d"), explosion[chamber], 14,
                                True, False, taper=0.50))
            for cord in range(7):
                theta = math.tau * (muscle + (cord + 0.5) / 7) / count
                endpoint = Vec3(center.x + radius * 0.65 * math.cos(theta),
                                center.y - 0.025, center.z + radius * 0.65 * math.sin(theta))
                meshes.append(_tube(I.CHORDAE_TENDINEAE, [tip, endpoint], 0.007,
                                    ("#90d2cf", "#ead0ba"), explosion[chamber],
                                    6, False, False))
    # Educational conduction overlay follows septal/endocardial surfaces.
    conduction_color = ("#35b9c0", "#d6b456")
    sa, av = Vec3(-0.52, 0.91, -0.23), Vec3(-0.08, 0.45, -0.16)
    for center in (sa, av):
        meshes.append(_ellipsoid(I.CARDIAC_CONDUCTION, center, Vec3(0.034,0.045,0.025),
                                 conduction_color, Vec3(0,0,0), 10, 16, False, False))
    paths = ([sa, Vec3(-0.35,0.72,-0.28), av],
             [av, Vec3(0.0,0.10,-0.21), Vec3(0.12,-0.54,-0.25)],
             [Vec3(0.12,-0.54,-0.25), Vec3(-0.10,-0.76,-0.14), Vec3(-0.21,-0.40,-0.24)],
             [Vec3(0.12,-0.54,-0.25), Vec3(0.38,-1.06,-0.14), Vec3(0.58,-0.55,-0.25)])
    for path in paths:
        meshes.append(_tube(I.CARDIAC_CONDUCTION, list(path), 0.009,
                            conduction_color, Vec3(0,0,0), 8, False, False, curve_steps=5))
    return tuple(meshes)
