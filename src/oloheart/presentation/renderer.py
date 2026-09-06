"""Native Tk canvas renderer for the procedural cardiac meshes."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageTk

from oloheart.domain.model import HeartPartId, HeartSnapshot
from oloheart.infrastructure.geometry import AnatomicalMesh, Vec3, shade


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


@dataclass(slots=True)
class _Triangle:
    depth: float
    points: tuple[float, float, float, float, float, float]
    fill: str
    outline: str
    width: int
    part_id: HeartPartId
    pickable: bool


@dataclass(frozen=True, slots=True)
class _MeshData:
    part_id: HeartPartId
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    smooth_normals: np.ndarray
    surface_grain: np.ndarray
    analysis_color: str
    realistic_color: str
    explosion: np.ndarray
    tissue: bool
    default_visible: bool


class HeartRenderer:
    """Project shaded 3D triangles directly into a native desktop canvas."""

    def __init__(self, canvas: tk.Canvas, meshes: tuple[AnatomicalMesh, ...]) -> None:
        self._canvas = canvas
        prepared_meshes: list[_MeshData] = []
        for mesh in meshes:
            vertices = np.asarray(
                [(vertex.x, vertex.y, vertex.z) for vertex in mesh.vertices],
                dtype=np.float32,
            )
            faces = np.asarray(mesh.faces, dtype=np.int32)[:, (0, 2, 1)]
            face_normals, smooth_normals, surface_grain = self._calculate_mesh_surface(
                vertices, faces
            )
            prepared_meshes.append(_MeshData(
                mesh.part_id,
                vertices,
                faces,
                face_normals,
                smooth_normals,
                surface_grain,
                mesh.analysis_color,
                mesh.realistic_color,
                np.asarray((mesh.explosion_vector.x, mesh.explosion_vector.y,
                            mesh.explosion_vector.z), dtype=np.float32),
                mesh.tissue,
                mesh.default_visible,
            ))
        self._meshes = tuple(prepared_meshes)
        # Smooth normals hide the reduced topology during direct manipulation,
        # allowing camera gestures to stay responsive on integrated graphics.
        self._motion_meshes = tuple(self._simplify_mesh(mesh, 3) for mesh in self._meshes)
        # The detailed heartbeat frames are cached after their first rasterization.
        # Direct manipulation still uses the lower motion mesh for immediate feedback.
        self._beat_meshes = tuple(self._simplify_mesh(mesh, 7) for mesh in self._meshes)
        self._image_item = self._canvas.create_image(
            0, 0, anchor="nw", state="hidden", tags=("heart-surface",)
        )
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_size = (0, 0)
        self._picking: list[_Triangle] = []
        self._whole_bounds = Rect(0, 0, 0, 0)
        self._focus_bounds = Rect(0, 0, 0, 0)
        self._shade_cache: dict[tuple[str, int, int], str] = {}
        light = Vec3(-0.32, 0.42, 1.0).normalized()
        self._light = np.asarray((light.x, light.y, light.z), dtype=np.float32)
        fill_light = Vec3(0.64, -0.12, 0.76).normalized()
        self._fill_light = np.asarray(
            (fill_light.x, fill_light.y, fill_light.z), dtype=np.float32
        )
        rim_light = Vec3(-0.62, 0.72, -0.32).normalized()
        self._rim_light = np.asarray(
            (rim_light.x, rim_light.y, rim_light.z), dtype=np.float32
        )
        self._last_signature: tuple | None = None
        self._last_motion_signature: tuple | None = None
        self._last_dynamic = False
        self._bgr_cache: dict[str, tuple[int, int, int]] = {}
        self._fiber_cache: dict[tuple[int, int], np.ndarray] = {}
        self._light_map_cache: dict[tuple[int, int], np.ndarray] = {}
        self._heartbeat_cache_base: tuple | None = None
        self._heartbeat_frames: dict[int, tuple[ImageTk.PhotoImage, float, float]] = {}

    @property
    def whole_bounds(self) -> Rect:
        return self._whole_bounds

    @property
    def focus_bounds(self) -> Rect:
        return self._focus_bounds

    def render(self, snapshot: HeartSnapshot, viewport: Rect) -> None:
        beat_bucket = int(round((snapshot.beat_scale - 1.0) / 0.005))
        offset_signature = tuple(
            (offset.part_id.value, round(offset.horizontal, 4), round(offset.vertical, 4),
             round(offset.depth, 4))
            for offset in snapshot.fragment_offsets
        )
        motion_signature = (
            round(snapshot.rotation_x, 4), round(snapshot.rotation_y, 4),
            round(snapshot.rotation_z, 4), round(snapshot.zoom, 4),
            round(snapshot.explosion, 4), offset_signature,
        )
        signature = motion_signature + (
            snapshot.selected_part,
            snapshot.realistic, beat_bucket, snapshot.spatial_mode,
            snapshot.grabbed_fragment,
            int(viewport.x), int(viewport.y), int(viewport.width), int(viewport.height),
        )
        previous_signature = self._last_signature
        transform_changed = (self._last_motion_signature is not None
                             and motion_signature != self._last_motion_signature)
        self._last_motion_signature = motion_signature
        dynamic = bool(snapshot.beating or transform_changed)
        cache_base = signature[:8] + signature[9:]
        if cache_base != self._heartbeat_cache_base:
            self._heartbeat_cache_base = cache_base
            self._heartbeat_frames.clear()
        if signature == previous_signature:
            if snapshot.beating or not self._last_dynamic:
                return
            dynamic = False
        if snapshot.beating and not transform_changed:
            cached = self._heartbeat_frames.get(beat_bucket)
            if cached is not None:
                self._show_cached_heartbeat(cached)
                self._last_signature = signature
                self._last_dynamic = True
                return
        self._last_signature = signature
        self._last_dynamic = dynamic
        if transform_changed:
            meshes = self._motion_meshes
        elif snapshot.beating:
            meshes = self._beat_meshes
        else:
            meshes = self._meshes
        size_factor = 0.305 if snapshot.spatial_mode else 0.285
        exploded_fit = 1.0 - snapshot.explosion * 0.28
        unit = min(viewport.width, viewport.height) * size_factor * snapshot.zoom * exploded_fit
        triangles: list[_Triangle] = []
        selected = snapshot.selected_part
        fragment_offsets = {
            offset.part_id: (offset.horizontal, offset.vertical, offset.depth)
            for offset in snapshot.fragment_offsets
        }
        workspace_size = (viewport.width, viewport.height)
        overview_meshes = tuple(mesh for mesh in meshes
                                if mesh.default_visible or snapshot.explosion > 0.035)
        if selected is None:
            center = (viewport.x + viewport.width * 0.50, viewport.y + viewport.height * 0.50)
            triangles.extend(self._render_instance(overview_meshes, snapshot, center, unit,
                                                    1.0, True, snapshot.grabbed_fragment,
                                                    fragment_offsets, workspace_size))
            self._whole_bounds = Rect(center[0] - unit * 1.20, center[1] - unit * 1.82,
                                      unit * 2.40, unit * 3.05)
            self._focus_bounds = Rect(0, 0, 0, 0)
            raster_bounds = self._quantized_bounds(self._whole_bounds, viewport, 24)
        else:
            heart_fraction = 0.16 if snapshot.spatial_mode else 0.30
            focus_fraction = 0.49 if snapshot.spatial_mode else 0.60
            heart_center = (viewport.x + viewport.width * heart_fraction,
                            viewport.y + viewport.height * 0.54)
            whole_unit = unit * (0.50 if snapshot.spatial_mode else 0.53)
            triangles.extend(self._render_instance(overview_meshes, snapshot, heart_center, whole_unit,
                                                    0.48, True, selected,
                                                    fragment_offsets, workspace_size))
            self._whole_bounds = Rect(heart_center[0] - whole_unit * 1.22,
                                      heart_center[1] - whole_unit * 1.82,
                                      whole_unit * 2.44, whole_unit * 3.08)
            selected_meshes = tuple(mesh for mesh in meshes if mesh.part_id == selected)
            if selected_meshes:
                focus_center = (viewport.x + viewport.width * focus_fraction,
                                viewport.y + viewport.height * 0.52)
                extent, origin = self._group_extent(selected_meshes)
                focus_unit = unit * min(2.30, max(0.82, 1.42 / max(extent, 0.18)))
                triangles.extend(self._render_instance(selected_meshes, snapshot, focus_center, focus_unit,
                                                        1.0, False, selected, {}, workspace_size,
                                                        origin))
                self._focus_bounds = Rect(focus_center[0] - focus_unit * 0.72,
                                          focus_center[1] - focus_unit * 0.72,
                                          focus_unit * 1.44, focus_unit * 1.44)
            raster_bounds = viewport

        if selected is None and (snapshot.explosion > 0.035 or fragment_offsets):
            raster_bounds = self._triangle_raster_bounds(triangles, viewport, 34)

        triangles.sort(key=lambda triangle: triangle.depth)
        self._picking = [triangle for triangle in triangles if triangle.pickable]
        cache_bucket = beat_bucket if snapshot.beating and not transform_changed else None
        raster_scale = 0.78 if transform_changed else 1.0
        self._draw(
            triangles,
            raster_bounds,
            snapshot.realistic,
            cache_bucket,
            raster_scale,
        )

    def part_at(self, x: float, y: float) -> HeartPartId | None:
        for triangle in reversed(self._picking):
            if self._point_in_triangle(x, y, triangle.points):
                return triangle.part_id
        return None

    def is_whole_heart_at(self, x: float, y: float) -> bool:
        return self._whole_bounds.contains(x, y)

    def _render_instance(
        self,
        meshes: tuple[_MeshData, ...],
        snapshot: HeartSnapshot,
        center: tuple[float, float],
        unit: float,
        opacity: float,
        pickable: bool,
        highlighted: HeartPartId | None,
        fragment_offsets: dict[HeartPartId, tuple[float, float, float]],
        workspace_size: tuple[float, float],
        local_origin: np.ndarray | None = None,
    ) -> list[_Triangle]:
        output: list[_Triangle] = []
        cos_y, sin_y = math.cos(snapshot.rotation_y), math.sin(snapshot.rotation_y)
        cos_x, sin_x = math.cos(snapshot.rotation_x), math.sin(snapshot.rotation_x)
        cos_z, sin_z = math.cos(snapshot.rotation_z), math.sin(snapshot.rotation_z)

        def object_space(vector: np.ndarray) -> np.ndarray:
            """Rotate a camera-space direction back into model space."""

            x_after_z = vector[0] * cos_z + vector[1] * sin_z
            y_after_z = -vector[0] * sin_z + vector[1] * cos_z
            z_after_z = vector[2]
            x_after_x = x_after_z
            y_after_x = y_after_z * cos_x + z_after_z * sin_x
            z_after_x = -y_after_z * sin_x + z_after_z * cos_x
            return np.asarray(
                (
                    x_after_x * cos_y - z_after_x * sin_y,
                    y_after_x,
                    x_after_x * sin_y + z_after_x * cos_y,
                ),
                dtype=np.float32,
            )

        object_key_light = object_space(self._light)
        object_fill_light = object_space(self._fill_light)
        object_rim_light = object_space(self._rim_light)
        object_view = object_space(np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
        for mesh in meshes:
            points = mesh.vertices.copy()
            if local_origin is not None:
                points -= local_origin
            else:
                points += mesh.explosion * (snapshot.explosion * 0.46)
            if mesh.tissue:
                points[:, 0] *= snapshot.beat_scale
                points[:, 1] += (snapshot.beat_scale - 1.0) * 0.16
                points[:, 2] *= snapshot.beat_scale

            rotated_x = points[:, 0] * cos_y + points[:, 2] * sin_y
            rotated_z = -points[:, 0] * sin_y + points[:, 2] * cos_y
            rotated_y = points[:, 1] * cos_x - rotated_z * sin_x
            camera_z = points[:, 1] * sin_x + rotated_z * cos_x
            camera_x = rotated_x * cos_z - rotated_y * sin_z
            camera_y = rotated_x * sin_z + rotated_y * cos_z
            fragment_offset = fragment_offsets.get(mesh.part_id) if local_origin is None else None
            if fragment_offset is not None:
                camera_z += fragment_offset[2] * 1.8
            transformed = np.column_stack((camera_x, camera_y, camera_z))
            perspective = 4.35 / np.maximum(2.40, 4.55 - camera_z)
            projected = np.column_stack((center[0] + camera_x * unit * perspective,
                                         center[1] - camera_y * unit * perspective))
            if fragment_offset is not None:
                projected[:, 0] += fragment_offset[0] * workspace_size[0]
                projected[:, 1] += fragment_offset[1] * workspace_size[1]

            base = mesh.realistic_color if snapshot.realistic else mesh.analysis_color
            is_highlighted = highlighted == mesh.part_id
            faces = mesh.faces
            visible = np.flatnonzero(mesh.face_normals @ object_view >= -0.08)

            # Normals and organic material variation are prepared once per mesh.
            # Rotating four directions is substantially cheaper than rebuilding
            # thousands of normals while a hand gesture is in progress.
            key_light = np.maximum(0.0, mesh.smooth_normals @ object_key_light)
            fill_light = np.maximum(0.0, mesh.smooth_normals @ object_fill_light)
            rim_light = np.maximum(0.0, mesh.smooth_normals @ object_rim_light)
            if snapshot.realistic:
                # Tissue scatters light; keeping most of the response diffuse
                # prevents individual raster triangles from reading as facets.
                brightness = 0.76 + key_light * 0.12 + fill_light * 0.035 + rim_light * 0.025
                material_grain = 0.018 if mesh.tissue else 0.009
                brightness *= 1.0 + mesh.surface_grain * material_grain
                specular = key_light ** (28 if mesh.tissue else 34)
                brightness += specular * (0.055 if mesh.tissue else 0.08)
            else:
                brightness = 0.30 + key_light * 0.64 + fill_light * 0.16 + rim_light * 0.10
            if is_highlighted:
                brightness = np.minimum(1.34, brightness * 1.28)
            brightness *= 0.44 + opacity * 0.56
            shade_steps = 24 if snapshot.realistic else 16
            buckets = (brightness * shade_steps).astype(np.int16)
            depths = (
                transformed[faces[:, 0], 2]
                + transformed[faces[:, 1], 2]
                + transformed[faces[:, 2], 2]
            ) / 3.0
            for face_index in visible:
                bucket = int(buckets[face_index])
                key = (base, bucket, shade_steps)
                fill = self._shade_cache.get(key)
                if fill is None:
                    fill = shade(base, bucket / shade_steps)
                    self._shade_cache[key] = fill
                outline = ""
                width = 1
                if not snapshot.realistic:
                    outline = "#0a5360" if not is_highlighted else "#27e9f3"
                elif is_highlighted:
                    outline = "#ef8b80"
                face = faces[face_index]
                screen_points = tuple(float(value) for value in projected[face].reshape(6))
                output.append(_Triangle(float(depths[face_index]), screen_points,
                                        fill, outline, width, mesh.part_id, pickable))
        return output

    def _draw(self, triangles: list[_Triangle], viewport: Rect, realistic: bool,
              heartbeat_bucket: int | None = None, raster_scale: float = 1.0) -> None:
        """Rasterize all faces in native code and update one canvas image."""

        target_width = max(2, int(viewport.width))
        target_height = max(2, int(viewport.height))
        width = max(2, int(target_width * raster_scale))
        height = max(2, int(target_height * raster_scale))
        frame = np.empty((height, width, 3), dtype=np.uint8)
        background = (10, 5, 1)
        frame[:, :] = background
        offset_x, offset_y = int(viewport.x), int(viewport.y)
        if triangles:
            coordinates = np.asarray(
                [triangle.points for triangle in triangles], dtype=np.float32
            ).reshape((-1, 3, 2))
            coordinates[:, :, 0] -= offset_x
            coordinates[:, :, 1] -= offset_y
            coordinates *= raster_scale
            coordinates = np.rint(coordinates).astype(np.int32)
            minimum_depth = triangles[0].depth
            depth_span = max(triangles[-1].depth - minimum_depth, 1e-6)
            depth_layers: list[
                dict[tuple[str, str, int], list[np.ndarray]]
            ] = [{} for _ in range(32)]
            for triangle, polygon in zip(triangles, coordinates, strict=True):
                layer_index = min(
                    31, int((triangle.depth - minimum_depth) / depth_span * 31)
                )
                style = (triangle.fill, triangle.outline, triangle.width)
                depth_layers[layer_index].setdefault(style, []).append(polygon)
            for layer in depth_layers:
                for (fill, outline, line_width), polygons in layer.items():
                    cv2.fillPoly(
                        frame, polygons, self._bgr(fill), lineType=cv2.LINE_AA
                    )
                    if outline:
                        cv2.polylines(
                            frame,
                            polygons,
                            True,
                            self._bgr(outline),
                            line_width,
                            cv2.LINE_AA,
                        )
        background_pixel = np.asarray(background, dtype=np.uint8)
        surface_mask = cv2.bitwise_not(
            cv2.inRange(frame, background_pixel, background_pixel)
        )
        if realistic:
            fiber = self._fiber_texture(width, height)
            cv2.add(frame, fiber, dst=frame, mask=surface_mask)
            frame = cv2.multiply(
                frame,
                self._continuous_light(width, height),
                dtype=cv2.CV_8U,
            )
            frame = cv2.GaussianBlur(frame, (5, 5), 0.82)
            surface_mask = cv2.GaussianBlur(surface_mask, (3, 3), 0.52)
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = surface_mask
        if raster_scale != 1.0:
            rgba = cv2.resize(
                rgba,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        rendered = Image.fromarray(rgba, "RGBA")
        if heartbeat_bucket is not None:
            self._photo = ImageTk.PhotoImage(rendered)
            self._photo_size = (width, height)
            self._heartbeat_frames[heartbeat_bucket] = (self._photo, viewport.x, viewport.y)
            self._canvas.coords(self._image_item, viewport.x, viewport.y)
            self._canvas.itemconfigure(self._image_item, image=self._photo, state="normal")
        elif self._photo is None or self._photo_size != (width, height):
            self._photo = ImageTk.PhotoImage(rendered)
            self._photo_size = (width, height)
            self._canvas.coords(self._image_item, viewport.x, viewport.y)
            self._canvas.itemconfigure(self._image_item, image=self._photo, state="normal")
        else:
            self._photo.paste(rendered)
        if self._canvas.find_withtag("hud"):
            self._canvas.tag_lower("heart-surface", "hud")

    def _show_cached_heartbeat(
        self, cached: tuple[ImageTk.PhotoImage, float, float]
    ) -> None:
        """Display a pre-rasterized heartbeat phase without rebuilding its mesh."""

        photo, x, y = cached
        self._photo = photo
        self._photo_size = (photo.width(), photo.height())
        self._canvas.coords(self._image_item, x, y)
        self._canvas.itemconfigure(self._image_item, image=photo, state="normal")
        if self._canvas.find_withtag("hud"):
            self._canvas.tag_lower("heart-surface", "hud")

    def _bgr(self, color: str) -> tuple[int, int, int]:
        cached = self._bgr_cache.get(color)
        if cached is not None:
            return cached
        value = color.lstrip("#")
        red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
        result = (blue, green, red)
        self._bgr_cache[color] = result
        return result

    def _fiber_texture(self, width: int, height: int) -> np.ndarray:
        """Return a cached, subtle myocardial fiber pattern for realistic mode."""

        key = (width, height)
        cached = self._fiber_cache.get(key)
        if cached is not None:
            return cached
        y, x = np.indices((height, width), dtype=np.float32)
        long_fibers = np.sin(x * 0.115 + y * 0.034 + np.sin(y * 0.018) * 1.8)
        cross_fibers = np.sin(x * 0.027 - y * 0.137)
        intensity = np.clip((long_fibers + 1.0) * 2.0 + (cross_fibers + 1.0) * 0.7,
                            0.0, 7.0).astype(np.uint8)
        texture = np.repeat(intensity[:, :, None], 3, axis=2)
        self._fiber_cache[key] = texture
        if len(self._fiber_cache) > 8:
            self._fiber_cache.pop(next(iter(self._fiber_cache)))
        return texture

    def _continuous_light(self, width: int, height: int) -> np.ndarray:
        """Return a cached broad light that avoids per-frame gradient work."""

        key = (width, height)
        cached = self._light_map_cache.get(key)
        if cached is not None:
            return cached
        y, x = np.ogrid[:height, :width]
        broad_key = np.exp(
            -(((x / max(width, 1) - 0.34) / 0.48) ** 2
              + ((y / max(height, 1) - 0.28) / 0.62) ** 2)
        ).astype(np.float32)
        light = 0.93 + broad_key * 0.13
        texture = np.repeat(light[:, :, None], 3, axis=2)
        self._light_map_cache[key] = texture
        if len(self._light_map_cache) > 8:
            self._light_map_cache.pop(next(iter(self._light_map_cache)))
        return texture

    @staticmethod
    def _simplify_mesh(mesh: _MeshData, divisions: int = 4) -> _MeshData:
        """Build a closed vertex-clustered mesh for motion-time rendering."""

        minimum = mesh.vertices.min(axis=0)
        extent = np.maximum(mesh.vertices.max(axis=0) - minimum, 1e-5)
        cells = extent / max(3, divisions)
        keys = np.rint((mesh.vertices - minimum) / cells).astype(np.int16)
        _unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        vertex_count = int(inverse.max()) + 1
        vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        counts = np.zeros(vertex_count, dtype=np.float32)
        np.add.at(vertices, inverse, mesh.vertices)
        np.add.at(counts, inverse, 1.0)
        vertices /= counts[:, None]
        faces = inverse[mesh.faces]
        valid = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                 & (faces[:, 0] != faces[:, 2]))
        faces = faces[valid]
        if len(faces):
            _keys, indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
            faces = faces[np.sort(indices)]
        if len(faces) < 4:
            return mesh
        faces = faces.astype(np.int32)
        face_normals, smooth_normals, surface_grain = HeartRenderer._calculate_mesh_surface(
            vertices, faces
        )
        return _MeshData(
            mesh.part_id,
            vertices,
            faces,
            face_normals,
            smooth_normals,
            surface_grain,
            mesh.analysis_color,
            mesh.realistic_color,
            mesh.explosion,
            mesh.tissue,
            mesh.default_visible,
        )

    @staticmethod
    def _calculate_mesh_surface(
        vertices: np.ndarray, faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precalculate culling, smooth lighting, and organic surface data."""

        first = vertices[faces[:, 0]]
        second = vertices[faces[:, 1]]
        third = vertices[faces[:, 2]]
        area_normals = np.cross(second - first, third - first)
        face_normals = area_normals.copy()
        face_normals /= np.maximum(
            np.linalg.norm(face_normals, axis=1), 1e-9
        )[:, None]

        vertex_normals = np.zeros_like(vertices, dtype=np.float32)
        np.add.at(vertex_normals, faces[:, 0], area_normals)
        np.add.at(vertex_normals, faces[:, 1], area_normals)
        np.add.at(vertex_normals, faces[:, 2], area_normals)
        lengths = np.linalg.norm(vertex_normals, axis=1)
        valid = lengths > 1e-9
        vertex_normals[valid] /= lengths[valid, None]
        vertex_normals[~valid, 2] = 1.0

        smooth_normals = vertex_normals[faces].sum(axis=1)
        smooth_normals /= np.maximum(
            np.linalg.norm(smooth_normals, axis=1), 1e-9
        )[:, None]
        centroids = (first + second + third) / 3.0
        surface_grain = (
            np.sin(centroids[:, 0] * 23.0 + centroids[:, 1] * 11.0)
            + np.sin(centroids[:, 1] * 31.0 - centroids[:, 2] * 17.0)
        ) * 0.5
        return face_normals, smooth_normals, surface_grain

    @staticmethod
    def _quantized_bounds(bounds: Rect, viewport: Rect, quantum: int) -> Rect:
        left = max(viewport.x, math.floor((bounds.x - 12) / quantum) * quantum)
        top = max(viewport.y, math.floor((bounds.y - 12) / quantum) * quantum)
        right = min(viewport.x + viewport.width,
                    math.ceil((bounds.x + bounds.width + 12) / quantum) * quantum)
        bottom = min(viewport.y + viewport.height,
                     math.ceil((bounds.y + bounds.height + 12) / quantum) * quantum)
        return Rect(left, top, max(2, right - left), max(2, bottom - top))

    @staticmethod
    def _triangle_raster_bounds(
        triangles: list[_Triangle], viewport: Rect, padding: int
    ) -> Rect:
        """Follow displaced fragments so the raster never looks like a fixed box."""

        if not triangles:
            return viewport
        xs = [coordinate for triangle in triangles for coordinate in triangle.points[0::2]]
        ys = [coordinate for triangle in triangles for coordinate in triangle.points[1::2]]
        left = max(viewport.x, math.floor(min(xs) - padding))
        top = max(viewport.y, math.floor(min(ys) - padding))
        right = min(viewport.x + viewport.width, math.ceil(max(xs) + padding))
        bottom = min(viewport.y + viewport.height, math.ceil(max(ys) + padding))
        return Rect(left, top, max(2, right - left), max(2, bottom - top))

    @staticmethod
    def _group_extent(meshes: tuple[_MeshData, ...]) -> tuple[float, np.ndarray]:
        vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        origin = (minimum + maximum) / 2.0
        extent = float((maximum - minimum).max())
        return extent, origin

    @staticmethod
    def _point_in_triangle(x: float, y: float, points: tuple[float, ...]) -> bool:
        x1, y1, x2, y2, x3, y3 = points
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denominator) < 1e-8:
            return False
        first = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        second = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        third = 1.0 - first - second
        return first >= 0.0 and second >= 0.0 and third >= 0.0
