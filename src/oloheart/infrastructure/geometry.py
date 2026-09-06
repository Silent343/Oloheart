"""Procedural anatomical geometry independent from the desktop renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from oloheart.domain.model import HeartPartId


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        length = max(self.length(), 1e-9)
        return self * (1.0 / length)

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)


@dataclass(frozen=True, slots=True)
class AnatomicalMesh:
    part_id: HeartPartId
    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, int, int], ...]
    analysis_color: str
    realistic_color: str
    explosion_vector: Vec3
    tissue: bool = False
    default_visible: bool = True
    view: str = "all"
    motion: str = "none"
    opacity: float = 1.0


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def shade(color: str, intensity: float) -> str:
    """Shade a material color while preserving low-light color information."""

    red, green, blue = _hex(color)
    gain = max(0.16, min(1.35, intensity))
    ambient = 5
    return f"#{min(255, int(red * gain) + ambient):02x}{min(255, int(green * gain) + ambient):02x}{min(255, int(blue * gain) + ambient):02x}"


def _ellipsoid(part: HeartPartId, center: Vec3, radii: Vec3, colors: tuple[str, str],
               explosion: Vec3, rows: int = 10, columns: int = 18, tissue: bool = True,
               default_visible: bool = True) -> AnatomicalMesh:
    vertices: list[Vec3] = []
    for row in range(rows + 1):
        latitude = math.pi * row / rows
        sine = math.sin(latitude)
        cosine = math.cos(latitude)
        for column in range(columns):
            longitude = math.tau * column / columns
            grain = 1.0
            if tissue:
                # Surface variation must read as tissue, not as polygonal damage.
                grain += 0.012 * math.sin(longitude * 5.0 + latitude * 3.0) * sine * sine
            vertices.append(Vec3(center.x + radii.x * grain * sine * math.cos(longitude),
                                 center.y + radii.y * grain * cosine,
                                 center.z + radii.z * grain * sine * math.sin(longitude)))
    faces: list[tuple[int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            next_column = (column + 1) % columns
            a = row * columns + column
            b = row * columns + next_column
            c = (row + 1) * columns + column
            d = (row + 1) * columns + next_column
            faces.extend(((a, c, b), (b, c, d)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), *colors, explosion, tissue,
                          default_visible)


def _ventricle(part: HeartPartId, top: Vec3, height: float, radii: tuple[float, float],
               tilt: float, colors: tuple[str, str], explosion: Vec3,
               default_visible: bool = True, rows: int = 14,
               columns: int = 24, hollow: bool = False) -> AnatomicalMesh:
    vertices: list[Vec3] = []
    # Two recessed shoulder rings blend the ventricular base into the atria.
    # They replace the visibly flat fan that made the intact heart look cut open.
    for shoulder_width, y_offset, z_offset in (
        (0.28, 0.18, -0.11),
        (0.56, 0.10, -0.055),
    ):
        for column in range(columns):
            angle = math.tau * column / columns
            front_bias = 1.0 + 0.08 * math.sin(angle)
            vertices.append(Vec3(
                top.x + math.cos(angle) * radii[0] * shoulder_width,
                top.y + y_offset,
                top.z + z_offset + math.sin(angle) * radii[1] * shoulder_width * front_bias,
            ))
    for row in range(rows + 1):
        t = row / rows
        # A broad ventricular base narrows continuously into a rounded apex.
        width = max(0.04, (1.0 - t) ** 0.48 * (0.76 + 0.34 * math.sin(math.pi * t)))
        y = top.y - height * t
        for column in range(columns):
            angle = math.tau * column / columns
            fiber = 1.0 + 0.012 * math.sin(angle * 5.0 + t * math.tau * 2.0)
            front_bias = 1.0 + 0.10 * math.sin(angle)
            vertices.append(Vec3(top.x + tilt * t + math.cos(angle) * radii[0] * width * fiber,
                                 y,
                                 top.z + math.sin(angle) * radii[1] * width * front_bias * fiber))
    top_center: int | None = None
    if not hollow:
        top_center = len(vertices)
        vertices.append(Vec3(top.x, top.y + 0.215, top.z - 0.12))
    apex = len(vertices)
    vertices.append(Vec3(top.x + tilt, top.y - height - 0.04, top.z))
    faces: list[tuple[int, int, int]] = []
    if top_center is not None:
        for column in range(columns):
            next_column = (column + 1) % columns
            faces.append((top_center, next_column, column))
    ring_count = rows + 3
    for row in range(ring_count - 1):
        for column in range(columns):
            next_column = (column + 1) % columns
            a, b = row * columns + column, row * columns + next_column
            c, d = (row + 1) * columns + column, (row + 1) * columns + next_column
            faces.extend(((a, c, b), (b, c, d)))
    last = (ring_count - 1) * columns
    for column in range(columns):
        faces.append((last + column, apex, last + (column + 1) % columns))

    if hollow:
        # The chamber has a genuine inward-facing endocardial wall. A cutaway in
        # the focused GPU view can therefore reveal wall thickness and internal
        # structures instead of exposing an empty solid primitive.
        inner_start = len(vertices)
        for ring in range(ring_count):
            depth = ring / max(1, ring_count - 1)
            radial_scale = 0.69 + depth * 0.055
            lift = 0.026 + depth * 0.105
            for column in range(columns):
                outer = vertices[ring * columns + column]
                vertices.append(Vec3(
                    top.x + (outer.x - top.x) * radial_scale,
                    outer.y + lift,
                    top.z + (outer.z - top.z) * radial_scale,
                ))
        inner_apex = len(vertices)
        vertices.append(Vec3(top.x + tilt * 0.82, top.y - height + 0.17, top.z))
        for ring in range(ring_count - 1):
            for column in range(columns):
                next_column = (column + 1) % columns
                a = inner_start + ring * columns + column
                b = inner_start + ring * columns + next_column
                c = inner_start + (ring + 1) * columns + column
                d = inner_start + (ring + 1) * columns + next_column
                faces.extend(((a, b, c), (b, d, c)))
        inner_last = inner_start + (ring_count - 1) * columns
        for column in range(columns):
            next_column = (column + 1) % columns
            faces.append((inner_last + column, inner_last + next_column, inner_apex))
        for column in range(columns):
            next_column = (column + 1) % columns
            outer_a, outer_b = column, next_column
            inner_a = inner_start + column
            inner_b = inner_start + next_column
            faces.extend(((outer_a, outer_b, inner_a), (outer_b, inner_b, inner_a)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), *colors, explosion, True,
                          default_visible)


def _curve_points(points: list[Vec3], steps: int) -> list[Vec3]:
    """Interpolate a centripetal-looking path without adding a runtime dependency."""

    if len(points) < 3 or steps <= 1:
        return points
    output: list[Vec3] = []
    for index in range(len(points) - 1):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        for sample in range(steps):
            t = sample / steps
            t2, t3 = t * t, t * t * t
            output.append(Vec3(
                0.5 * ((2.0 * p1.x) + (-p0.x + p2.x) * t
                       + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                       + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3),
                0.5 * ((2.0 * p1.y) + (-p0.y + p2.y) * t
                       + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                       + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3),
                0.5 * ((2.0 * p1.z) + (-p0.z + p2.z) * t
                       + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2
                       + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3),
            ))
    output.append(points[-1])
    return output


def _tube(part: HeartPartId, points: list[Vec3], radius: float, colors: tuple[str, str],
          explosion: Vec3, sides: int = 7, tissue: bool = False,
          default_visible: bool = True, curve_steps: int = 1,
          taper: float = 0.0) -> AnatomicalMesh:
    sides = max(5, min(sides, 16))
    points = _curve_points(points, curve_steps)
    vertices: list[Vec3] = []
    for index, point in enumerate(points):
        previous = points[max(0, index - 1)]
        following = points[min(len(points) - 1, index + 1)]
        tangent = (following - previous).normalized()
        helper = Vec3(0.0, 1.0, 0.0) if abs(tangent.y) < 0.84 else Vec3(1.0, 0.0, 0.0)
        normal = tangent.cross(helper).normalized()
        binormal = tangent.cross(normal).normalized()
        local_radius = radius * (1.0 - max(0.0, min(0.75, taper))
                                 * index / max(1, len(points) - 1))
        for side in range(sides):
            angle = math.tau * side / sides
            vertices.append(point + normal * (math.cos(angle) * local_radius)
                            + binormal * (math.sin(angle) * local_radius))
    faces: list[tuple[int, int, int]] = []
    for segment in range(len(points) - 1):
        for side in range(sides):
            next_side = (side + 1) % sides
            a, b = segment * sides + side, segment * sides + next_side
            c, d = (segment + 1) * sides + side, (segment + 1) * sides + next_side
            faces.extend(((a, c, b), (b, c, d)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), *colors, explosion, tissue,
                          default_visible)


def _torus(part: HeartPartId, center: Vec3, major: float, minor: float,
           colors: tuple[str, str], explosion: Vec3, tilt: float = 0.0,
           default_visible: bool = True) -> AnatomicalMesh:
    major_steps, minor_steps = 40, 10
    vertices: list[Vec3] = []
    cosine_tilt, sine_tilt = math.cos(tilt), math.sin(tilt)
    for major_index in range(major_steps):
        major_angle = math.tau * major_index / major_steps
        for minor_index in range(minor_steps):
            minor_angle = math.tau * minor_index / minor_steps
            radial = major + minor * math.cos(minor_angle)
            x = radial * math.cos(major_angle)
            z = radial * math.sin(major_angle)
            y = minor * math.sin(minor_angle)
            vertices.append(Vec3(center.x + x,
                                 center.y + y * cosine_tilt - z * sine_tilt,
                                 center.z + y * sine_tilt + z * cosine_tilt))
    faces: list[tuple[int, int, int]] = []
    for major_index in range(major_steps):
        for minor_index in range(minor_steps):
            next_major = (major_index + 1) % major_steps
            next_minor = (minor_index + 1) % minor_steps
            a = major_index * minor_steps + minor_index
            b = next_major * minor_steps + minor_index
            c = major_index * minor_steps + next_minor
            d = next_major * minor_steps + next_minor
            faces.extend(((a, b, c), (c, b, d)))
    return AnatomicalMesh(part, tuple(vertices), tuple(faces), *colors, explosion, False,
                          default_visible)


def build_heart_geometry() -> tuple[AnatomicalMesh, ...]:
    """Build independently selectable meshes for all catalog structures."""

    i = HeartPartId
    cyan = ("#0b8ea4", "#731923")
    atrial = ("#08778d", "#67212f")
    vessel_red = ("#087e93", "#922722")
    vessel_blue = ("#075f79", "#263958")
    valve = ("#14a4ad", "#d5b78d")
    internal = ("#126f7d", "#7e4148")
    coronary = ("#20c6d6", "#b93a2e")
    inner_muscle = ("#096a79", "#51121e")
    chordae = ("#1ec3ce", "#d8cbb7")
    pericardial = ("#176f7c", "#d8ae98")
    epicardial = ("#0c8292", "#a84c51")
    endocardial = ("#1593a0", "#d98682")
    conduction = ("#36d7df", "#f5d64f")
    coronary_vein = ("#176e98", "#3c6eaa")
    meshes: list[AnatomicalMesh] = [
        _ventricle(i.RIGHT_VENTRICLE, Vec3(-0.28, 0.40, 0.10), 1.30, (0.62, 0.46), 0.18, cyan, Vec3(-0.80, -0.20, 0.42), rows=28, columns=56, hollow=True),
        _ventricle(i.LEFT_VENTRICLE, Vec3(0.30, 0.42, -0.02), 1.70, (0.57, 0.52), -0.06, ("#0a99ae", "#7d1824"), Vec3(0.82, -0.25, -0.18), rows=28, columns=56, hollow=True),
        _ellipsoid(i.RIGHT_ATRIUM, Vec3(-0.39, 0.73, -0.13), Vec3(0.40, 0.31, 0.33), atrial, Vec3(-0.78, 0.68, 0.16), rows=22, columns=44),
        _ellipsoid(i.LEFT_ATRIUM, Vec3(0.35, 0.76, -0.23), Vec3(0.35, 0.29, 0.30), atrial, Vec3(0.76, 0.72, -0.30), rows=22, columns=44),
        _ellipsoid(i.RIGHT_ATRIUM, Vec3(-0.61, 0.64, 0.14), Vec3(0.24, 0.17, 0.18), atrial, Vec3(-0.78, 0.68, 0.16), rows=16, columns=32),
        _ellipsoid(i.LEFT_ATRIUM, Vec3(0.58, 0.66, 0.08), Vec3(0.22, 0.15, 0.17), atrial, Vec3(0.76, 0.72, -0.30), rows=16, columns=32),
        _tube(i.AORTA, [Vec3(0.24, 0.48, -0.20), Vec3(0.18, 0.86, -0.18), Vec3(0.14, 1.22, -0.12), Vec3(0.30, 1.48, -0.10), Vec3(0.62, 1.57, -0.13), Vec3(0.88, 1.39, -0.18)], 0.17, vessel_red, Vec3(0.40, 1.00, -0.38), 16, curve_steps=4),
        _tube(i.AORTA, [Vec3(0.35, 1.48, -0.10), Vec3(0.32, 1.82, -0.10)], 0.075, vessel_red, Vec3(0.40, 1.00, -0.38), 14),
        _tube(i.AORTA, [Vec3(0.55, 1.54, -0.12), Vec3(0.60, 1.86, -0.12)], 0.075, vessel_red, Vec3(0.40, 1.00, -0.38), 14),
        _tube(i.AORTA, [Vec3(0.73, 1.51, -0.15), Vec3(0.86, 1.78, -0.16)], 0.075, vessel_red, Vec3(0.40, 1.00, -0.38), 14),
        _tube(i.PULMONARY_ARTERY, [Vec3(-0.17, 0.42, 0.26), Vec3(-0.12, 0.82, 0.32), Vec3(-0.10, 1.10, 0.28), Vec3(-0.45, 1.28, 0.20)], 0.15, vessel_blue, Vec3(-0.45, 0.88, 0.70), 16, curve_steps=4),
        _tube(i.PULMONARY_ARTERY, [Vec3(-0.10, 1.10, 0.28), Vec3(0.30, 1.27, 0.12), Vec3(0.60, 1.25, 0.02)], 0.13, vessel_blue, Vec3(-0.45, 0.88, 0.70), 14, curve_steps=4),
        _tube(i.VENA_CAVA, [Vec3(-0.54, 1.70, -0.14), Vec3(-0.51, 1.21, -0.10), Vec3(-0.44, 0.78, -0.04)], 0.16, vessel_blue, Vec3(-1.00, 0.62, -0.20), 12),
        _tube(i.VENA_CAVA, [Vec3(-0.48, 0.58, -0.30), Vec3(-0.58, 0.18, -0.48), Vec3(-0.55, -0.32, -0.53)], 0.14, vessel_blue, Vec3(-1.00, 0.62, -0.20), 16, curve_steps=5),
    ]
    for y, side in ((0.88, -1.0), (0.64, -1.0), (0.88, 1.0), (0.64, 1.0)):
        start = Vec3(0.38, y, -0.25)
        meshes.append(_tube(i.PULMONARY_VEINS, [start, Vec3(0.38 + 0.58 * side, y + 0.04, -0.34)], 0.095,
                            vessel_red, Vec3(0.85 * side, 0.45, -0.82), 9))
    meshes.extend((
        _torus(i.TRICUSPID_VALVE, Vec3(-0.26, 0.42, 0.03), 0.22, 0.045, valve, Vec3(-0.75, 0.18, 0.62), 0.08, False),
        _torus(i.MITRAL_VALVE, Vec3(0.29, 0.45, -0.06), 0.20, 0.045, valve, Vec3(0.76, 0.20, 0.42), -0.08, False),
        _torus(i.AORTIC_VALVE, Vec3(0.22, 0.68, -0.18), 0.14, 0.038, valve, Vec3(0.38, 0.78, -0.40), 0.02, False),
        _torus(i.PULMONARY_VALVE, Vec3(-0.15, 0.68, 0.24), 0.14, 0.038, valve, Vec3(-0.38, 0.70, 0.65), 0.02, False),
        _ellipsoid(i.TRICUSPID_VALVE, Vec3(-0.36, 0.40, 0.08), Vec3(0.14, 0.026, 0.10),
                   valve, Vec3(-0.75, 0.18, 0.62), 4, 10, False, False),
        _ellipsoid(i.TRICUSPID_VALVE, Vec3(-0.16, 0.40, 0.02), Vec3(0.13, 0.026, 0.095),
                   valve, Vec3(-0.75, 0.18, 0.62), 4, 10, False, False),
        _ellipsoid(i.MITRAL_VALVE, Vec3(0.22, 0.43, -0.02), Vec3(0.15, 0.026, 0.10),
                   valve, Vec3(0.76, 0.20, 0.42), 4, 10, False, False),
        _ellipsoid(i.MITRAL_VALVE, Vec3(0.39, 0.43, -0.08), Vec3(0.13, 0.026, 0.09),
                   valve, Vec3(0.76, 0.20, 0.42), 4, 10, False, False),
        _ellipsoid(i.INTERVENTRICULAR_SEPTUM, Vec3(0.02, -0.26, 0.0), Vec3(0.055, 0.82, 0.36), internal,
                   Vec3(0.0, -0.70, 0.78), 24, 40, True, False),
        _ellipsoid(i.INTERATRIAL_SEPTUM, Vec3(0.0, 0.72, -0.11), Vec3(0.045, 0.28, 0.25), internal,
                   Vec3(0.0, 0.68, -0.82), 20, 36, True, False),
    ))
    # Internal myocardial structures prevent the exploded heart from reading as
    # empty shells and provide useful landmarks in isolated chamber views.
    papillary_specs = (
        (i.PAPILLARY_MUSCLES, [Vec3(-0.49, -0.72, 0.12), Vec3(-0.43, -0.30, 0.10)], Vec3(-0.80, -0.20, 0.42)),
        (i.PAPILLARY_MUSCLES, [Vec3(-0.08, -0.76, 0.09), Vec3(-0.16, -0.27, 0.05)], Vec3(-0.80, -0.20, 0.42)),
        (i.PAPILLARY_MUSCLES, [Vec3(0.13, -0.92, -0.06), Vec3(0.18, -0.29, -0.05)], Vec3(0.82, -0.25, -0.18)),
        (i.PAPILLARY_MUSCLES, [Vec3(0.51, -0.84, -0.10), Vec3(0.40, -0.26, -0.08)], Vec3(0.82, -0.25, -0.18)),
    )
    for part, path, explosion in papillary_specs:
        meshes.append(_tube(part, path, 0.085, inner_muscle, explosion, 8, True, False))

    chordae_specs = (
        (i.CHORDAE_TENDINEAE, [Vec3(-0.43, -0.30, 0.10), Vec3(-0.36, 0.20, 0.08)], Vec3(-0.75, 0.18, 0.62)),
        (i.CHORDAE_TENDINEAE, [Vec3(-0.16, -0.27, 0.05), Vec3(-0.18, 0.23, 0.03)], Vec3(-0.75, 0.18, 0.62)),
        (i.CHORDAE_TENDINEAE, [Vec3(0.18, -0.29, -0.05), Vec3(0.23, 0.25, -0.03)], Vec3(0.76, 0.20, 0.42)),
        (i.CHORDAE_TENDINEAE, [Vec3(0.40, -0.26, -0.08), Vec3(0.37, 0.25, -0.07)], Vec3(0.76, 0.20, 0.42)),
    )
    for part, path, explosion in chordae_specs:
        meshes.append(_tube(part, path, 0.018, chordae, explosion, 5, False, False))

    trabecular_specs = (
        (i.RIGHT_VENTRICLE, [Vec3(-0.64, -0.10, 0.18), Vec3(-0.31, -0.26, 0.31)], Vec3(-0.80, -0.20, 0.42)),
        (i.RIGHT_VENTRICLE, [Vec3(-0.59, -0.45, 0.15), Vec3(-0.26, -0.58, 0.28)], Vec3(-0.80, -0.20, 0.42)),
        (i.RIGHT_VENTRICLE, [Vec3(-0.48, -0.78, 0.08), Vec3(-0.17, -0.91, 0.16)], Vec3(-0.80, -0.20, 0.42)),
        (i.LEFT_VENTRICLE, [Vec3(0.05, -0.12, 0.12), Vec3(0.40, -0.29, 0.22)], Vec3(0.82, -0.25, -0.18)),
        (i.LEFT_VENTRICLE, [Vec3(0.06, -0.50, 0.08), Vec3(0.45, -0.62, 0.17)], Vec3(0.82, -0.25, -0.18)),
        (i.LEFT_VENTRICLE, [Vec3(0.12, -0.86, 0.02), Vec3(0.39, -1.01, 0.09)], Vec3(0.82, -0.25, -0.18)),
    )
    for part, path, explosion in trabecular_specs:
        meshes.append(_tube(part, path, 0.035, inner_muscle, explosion, 6, True, False))

    # Educational tissue layers are hidden in the intact overview and become
    # visible when layers are exploded or selected directly.
    meshes.extend((
        _ventricle(i.PERICARDIUM, Vec3(0.02, 0.68, -0.02), 2.12, (0.92, 0.70), 0.04,
                   pericardial, Vec3(-2.00, 0.00, 0.15), False),
        _ventricle(i.EPICARDIUM, Vec3(0.02, 0.62, -0.01), 2.02, (0.85, 0.64), 0.035,
                   epicardial, Vec3(2.00, 0.00, 0.12), False),
        _ventricle(i.ENDOCARDIUM, Vec3(-0.25, 0.31, 0.08), 1.32, (0.39, 0.29), 0.08,
                   endocardial, Vec3(-1.12, -1.42, 0.30), False),
        _ventricle(i.ENDOCARDIUM, Vec3(0.29, 0.34, -0.01), 1.52, (0.35, 0.34), -0.04,
                   endocardial, Vec3(1.12, -1.42, 0.26), False),
    ))

    for fiber_index in range(7):
        phase = math.tau * fiber_index / 7.0
        direction = -1.0 if fiber_index % 2 else 1.0
        path: list[Vec3] = []
        for step in range(20):
            t = step / 19.0
            envelope = 0.22 + 0.68 * math.sin(math.pi * t) ** 0.72
            angle = phase + direction * (2.15 * math.pi * t)
            path.append(Vec3(0.02 + math.cos(angle) * envelope,
                             -1.33 + t * 1.98,
                             0.02 + math.sin(angle) * envelope * 0.67))
        meshes.append(_tube(i.MYOCARDIUM, path, 0.016, inner_muscle,
                            Vec3(0.0, 1.70, 0.36), 7, True, False))

    conduction_paths = (
        [Vec3(-0.45, 1.00, 0.28), Vec3(-0.38, 0.78, 0.33), Vec3(-0.14, 0.48, 0.39)],
        [Vec3(-0.14, 0.48, 0.39), Vec3(-0.03, 0.14, 0.43), Vec3(0.00, -0.32, 0.46)],
        [Vec3(0.00, -0.32, 0.46), Vec3(-0.22, -0.70, 0.38), Vec3(-0.39, -1.04, 0.20)],
        [Vec3(0.00, -0.32, 0.46), Vec3(0.28, -0.68, 0.37), Vec3(0.36, -1.18, 0.12)],
        [Vec3(-0.39, -1.04, 0.20), Vec3(-0.58, -0.58, 0.24), Vec3(-0.55, -0.14, 0.34)],
        [Vec3(0.36, -1.18, 0.12), Vec3(0.58, -0.65, 0.24), Vec3(0.55, -0.10, 0.33)],
    )
    meshes.extend((
        _ellipsoid(i.CARDIAC_CONDUCTION, Vec3(-0.46, 1.02, 0.28), Vec3(0.065, 0.09, 0.035),
                   conduction, Vec3(0.0, 0.10, 1.48), 5, 10, False, False),
        _ellipsoid(i.CARDIAC_CONDUCTION, Vec3(-0.14, 0.48, 0.39), Vec3(0.055, 0.07, 0.03),
                   conduction, Vec3(0.0, 0.10, 1.48), 5, 10, False, False),
    ))
    for path in conduction_paths:
        meshes.append(_tube(i.CARDIAC_CONDUCTION, path, 0.018, conduction,
                            Vec3(0.0, 0.10, 1.48), 6, False, False))

    cardiac_vein_paths = (
        [Vec3(0.08, -1.05, 0.20), Vec3(-0.01, -0.56, 0.48),
         Vec3(-0.03, -0.08, 0.61), Vec3(-0.02, 0.42, 0.57)],
        [Vec3(-0.55, -0.50, 0.22), Vec3(-0.66, -0.14, 0.34),
         Vec3(-0.60, 0.25, 0.40), Vec3(-0.43, 0.43, 0.34)],
        [Vec3(0.05, -1.02, -0.22), Vec3(0.02, -0.52, -0.48),
         Vec3(-0.06, -0.04, -0.57), Vec3(-0.20, 0.30, -0.53)],
    )
    for path in cardiac_vein_paths:
        meshes.append(_tube(i.CORONARY_VEINS, path, 0.021, coronary_vein,
                            Vec3(-0.10, 0.04, 1.08), 10, False, True,
                            curve_steps=5, taper=0.22))

    meshes.append(_tube(i.CORONARY_SINUS,
                        [Vec3(-0.72, 0.38, -0.28), Vec3(-0.30, 0.31, -0.48),
                         Vec3(0.18, 0.30, -0.54), Vec3(0.48, 0.43, -0.39)],
                        0.055, coronary_vein, Vec3(0.0, 0.12, -0.98), 12, False, True,
                        curve_steps=5, taper=0.18))
    coronary_paths = [
        [Vec3(0.10, 0.62, 0.43), Vec3(0.03, 0.30, 0.57), Vec3(0.01, -0.10, 0.59), Vec3(0.02, -0.58, 0.46), Vec3(0.10, -1.05, 0.18)],
        [Vec3(0.15, 0.58, 0.40), Vec3(0.44, 0.38, 0.42), Vec3(0.58, 0.05, 0.36), Vec3(0.60, -0.36, 0.24)],
        [Vec3(-0.14, 0.55, 0.43), Vec3(-0.46, 0.36, 0.46), Vec3(-0.58, 0.02, 0.36), Vec3(-0.50, -0.38, 0.24)],
        [Vec3(0.04, 0.12, 0.58), Vec3(0.30, 0.00, 0.52), Vec3(0.50, -0.22, 0.36)],
        [Vec3(0.03, -0.08, 0.58), Vec3(-0.22, -0.28, 0.50), Vec3(-0.40, -0.52, 0.34)],
        [Vec3(0.45, 0.34, 0.41), Vec3(0.32, 0.08, 0.48), Vec3(0.28, -0.30, 0.42)],
    ]
    for path in coronary_paths:
        meshes.append(_tube(i.CORONARY_ARTERIES, path, 0.025, coronary,
                            Vec3(0.0, 0.0, 0.98), 12, False, True,
                            curve_steps=5, taper=0.38))
    from oloheart.infrastructure.sculpted_geometry import refine_anatomy

    return refine_anatomy(tuple(meshes))
