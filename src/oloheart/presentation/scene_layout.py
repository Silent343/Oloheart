"""Stable model-space placement shared by rendering, picking, and tests."""

from __future__ import annotations

import math
import numpy as np

from oloheart.domain.model import AnatomySystem, HeartPartId, HeartSnapshot, anatomy_system_for


def visible_mesh(mesh, snapshot: HeartSnapshot) -> bool:
    """Choose a paired surface without changing its anatomical coordinates."""
    if anatomy_system_for(mesh.part_id) not in snapshot.visible_systems:
        return False
    if mesh.view == "interior" and not snapshot.interior_view:
        return False
    if mesh.view == "exterior" and snapshot.interior_view:
        return False
    if mesh.part_id == HeartPartId.EPICARDIUM:
        return snapshot.selected_part == HeartPartId.EPICARDIUM
    if mesh.part_id == HeartPartId.MYOCARDIUM:
        return (snapshot.selected_part == HeartPartId.MYOCARDIUM
                or AnatomySystem.CHAMBERS_AND_SEPTA not in snapshot.visible_systems)
    if (snapshot.selected_part == HeartPartId.MYOCARDIUM
            and mesh.part_id in {HeartPartId.LEFT_VENTRICLE, HeartPartId.RIGHT_VENTRICLE}
            and mesh.view != "all"):
        return False
    return (mesh.default_visible or snapshot.interior_view or snapshot.explosion > 0.035
            or snapshot.selected_part == mesh.part_id)


def scene_rotation(snapshot: HeartSnapshot) -> np.ndarray:
    """Return the same rotation for every structure, including the selected one."""
    x, y, z = snapshot.rotation_x, snapshot.rotation_y, snapshot.rotation_z
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array(((1,0,0),(0,cx,-sx),(0,sx,cx)), dtype=np.float32)
    ry = np.array(((cy,0,sy),(0,1,0),(-sy,0,cy)), dtype=np.float32)
    rz = np.array(((cz,-sz,0),(sz,cz,0),(0,0,1)), dtype=np.float32)
    result = np.eye(4, dtype=np.float32)
    result[:3,:3] = rz @ rx @ ry
    return result


def placement_matrix(snapshot: HeartSnapshot, explosion, part: HeartPartId,
                     scale: float) -> np.ndarray:
    """Selection never changes position, scale, rotation, or explosion offsets."""
    shift = np.asarray(explosion, dtype=np.float32) * snapshot.explosion * 0.78
    for offset in snapshot.fragment_offsets:
        if offset.part_id == part:
            shift += np.asarray((offset.horizontal * 4, -offset.vertical * 4,
                                 offset.depth * 3), dtype=np.float32)
            break
    shift[1] -= 0.14
    matrix = scene_rotation(snapshot)
    matrix[:3,:3] *= scale
    matrix[:3,3] = matrix[:3,:3] @ shift
    return matrix


def object_drag(snapshot: HeartSnapshot, dx: float, dy: float, depth: float) -> tuple[float,float,float]:
    """Convert camera-space hand motion into persistent rotating model offsets."""
    delta = scene_rotation(snapshot)[:3,:3].T @ np.asarray((dx * 4, -dy * 4, depth * 3))
    return float(delta[0] / 4), float(-delta[1] / 4), float(delta[2] / 3)
