"""Hardware-accelerated cardiac renderer embedded in the native Qt shell."""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass

import numpy as np
import OpenGL

# The render loop validates programs and framebuffer creation explicitly. Turning
# off PyOpenGL's per-call error wrapper removes substantial Python overhead from
# the stable hot path while preserving initialization failures.
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = False

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BACK,
    GL_BLEND,
    GL_CCW,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DITHER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_LEQUAL,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glCullFace,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDrawElements,
    glEnable,
    glEnableVertexAttribArray,
    glFrontFace,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glReadPixels,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import GL_FRAGMENT_SHADER, GL_VERTEX_SHADER
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from oloheart.domain.model import (
    CardiacPhase,
    HandObservation,
    HeartPart,
    HeartPartId,
    HeartSnapshot,
    anatomy_system_for,
)
from oloheart.infrastructure.geometry import AnatomicalMesh
from oloheart.presentation.scene_layout import placement_matrix, visible_mesh


VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uAtrialContraction;
uniform float uVentricularContraction;
uniform float uContractileGroup;
uniform float uIsTissue;
uniform vec3 uMeshCenter;
uniform float uValveMode;
uniform float uValveOpen;
uniform float uValveRadius;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec3 vObjectPosition;
out vec3 vObjectNormal;

void main() {
    vec3 position = aPosition;
    vec3 objectNormal = aNormal;
    if (uValveMode > 0.5) {
        vec2 radial = position.xz - uMeshCenter.xz;
        float radius = max(length(radial), 0.0001);
        float hinge = clamp(1.0 - radius / uValveRadius, 0.0, 1.0);
        position.xz += radial / radius * hinge * uValveRadius * 0.88 * uValveOpen;
        position.y -= hinge * uValveOpen * (uValveMode < 1.5 ? 0.16 : -0.09);
    }
    if (uIsTissue > 0.5 && uContractileGroup > 0.5) {
        float contraction = uContractileGroup < 1.5
            ? uAtrialContraction * 0.025
            : uVentricularContraction * 0.048;
        vec3 deformation = vec3(1.0 - contraction, 1.0 - contraction * 0.32, 1.0 - contraction);
        position = uMeshCenter + (position - uMeshCenter) * deformation;
        objectNormal = normalize(aNormal / deformation);
    }
    vec4 world = uModel * vec4(position, 1.0);
    vWorldPosition = world.xyz;
    vObjectPosition = position;
    vObjectNormal = aNormal;
    vNormal = normalize(transpose(inverse(mat3(uModel))) * objectNormal);
    gl_Position = uProjection * uView * world;
}
"""


FRAGMENT_SHADER = """
#version 330 core
in vec3 vWorldPosition;
in vec3 vNormal;
in vec3 vObjectPosition;
in vec3 vObjectNormal;

uniform vec3 uBaseColor;
uniform vec3 uCameraPosition;
uniform vec3 uPickColor;
uniform vec3 uMeshCenter;
uniform float uRoughness;
uniform float uSheen;
uniform float uOpacity;
uniform float uIsTissue;
uniform float uHighlighted;
uniform float uAnalytical;
uniform float uCutaway;
uniform float uActivation;
uniform float uCycleProgress;
uniform float uSectionSurface;
uniform float uConduction;
uniform int uPicking;

out vec4 fragmentColor;

void main() {
    if (uSectionSurface > 0.5 && vObjectPosition.z > 0.035) {
        discard;
    }
    if (uPicking == 1) {
        fragmentColor = vec4(uPickColor, 1.0);
        return;
    }

    float grainA = sin(vObjectPosition.y * 67.0 + vObjectPosition.z * 29.0);
    float grainB = sin(vObjectPosition.z * 61.0 - vObjectPosition.x * 31.0);
    float grainC = sin(vObjectPosition.x * 59.0 + vObjectPosition.y * 23.0);
    vec3 normal = normalize(gl_FrontFacing ? vNormal : -vNormal);
    if (uIsTissue > 0.5) {
        vec3 microNormal = vec3(grainA, grainB, grainC);
        normal = normalize(normal + microNormal * 0.038);
    }
    vec3 viewDirection = normalize(uCameraPosition - vWorldPosition);
    vec3 keyDirection = normalize(vec3(-0.48, 0.72, 1.15));
    vec3 fillDirection = normalize(vec3(0.78, -0.16, 0.62));
    vec3 rimDirection = normalize(vec3(-0.62, 0.78, -0.42));

    float key = max(dot(normal, keyDirection), 0.0);
    float fill = max(dot(normal, fillDirection), 0.0);
    float rear = max(dot(normal, rimDirection), 0.0);
    vec3 halfVector = normalize(keyDirection + viewDirection);
    float glossPower = mix(84.0, 22.0, uRoughness);
    float specular = pow(max(dot(normal, halfVector), 0.0), glossPower);
    float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    float fiber = (grainA + grainB) * 0.5;
    float cellular = grainB * grainC * 0.5;
    float tissueVariation = uIsTissue > 0.5 ? 1.0 + fiber * 0.042 + cellular * 0.12 : 1.0;
    vec3 base = uBaseColor * tissueVariation;
    vec3 radial = normalize(vec3(
        vObjectPosition.x - uMeshCenter.x,
        0.0,
        vObjectPosition.z - uMeshCenter.z
    ) + vec3(0.0001));
    float inwardSurface = 1.0 - smoothstep(
        -0.62, -0.12, dot(normalize(vObjectNormal), radial)
    );
    base = mix(base, base * vec3(1.15, 1.08, 0.98), inwardSurface * uIsTissue * 0.65);

    float anterior = smoothstep(0.08, 0.48, vObjectPosition.z);
    float interventricularGroove = exp(-pow(vObjectPosition.x * 14.0, 2.0)) * anterior;
    float atrioventricularGroove = exp(-pow((vObjectPosition.y - 0.42) * 18.0, 2.0)) * anterior;
    float anatomicalGroove = max(interventricularGroove, atrioventricularGroove);
    base *= 1.0 - anatomicalGroove * uIsTissue * 0.17;

    vec3 diffuse = base * (0.19 + key * 0.86 + fill * 0.30 + rear * 0.09);
    vec3 wetHighlight = vec3(1.0, 0.66, 0.61) * specular * (0.11 + uSheen * 0.28);
    vec3 subsurface = mix(base, vec3(0.34, 0.025, 0.018), 0.54)
                      * fresnel * (0.08 + uSheen * 0.17);
    vec3 color = diffuse + wetHighlight + subsurface;
    color += base * uActivation * 0.38;
    if (uConduction > 0.5) {
        float arrival = vObjectPosition.y > 0.52 ? 0.025
                      : (vObjectPosition.y > 0.30 ? 0.085
                      : (vObjectPosition.y > 0.0 ? 0.12 : 0.16));
        float pulse = exp(-pow((uCycleProgress - arrival) / 0.025, 2.0));
        color += vec3(0.40, 0.78, 0.42) * pulse;
    }

    if (uHighlighted > 0.5) {
        color += fresnel * vec3(0.03, 0.40, 0.43) + base * 0.08;
    }
    if (uAnalytical > 0.5) {
        color += vec3(0.0, 0.11, 0.14) + fresnel * vec3(0.0, 0.24, 0.28);
    }

    color = vec3(1.0) - exp(-color * 1.10);
    fragmentColor = vec4(color, uOpacity);
}
"""


@dataclass(slots=True)
class _GpuMesh:
    part_id: HeartPartId
    vertices: np.ndarray
    center: np.ndarray
    vao: int
    vertex_buffer: int
    index_buffer: int
    index_count: int
    analysis_color: tuple[float, float, float]
    realistic_color: tuple[float, float, float]
    explosion: np.ndarray
    tissue: bool
    default_visible: bool
    view: str
    motion: str
    opacity: float


def _identity() -> np.ndarray:
    return np.identity(4, dtype=np.float32)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = _identity()
    matrix[:3, 3] = (x, y, z)
    return matrix


def _scale(value: float) -> np.ndarray:
    matrix = _identity()
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = value
    return matrix


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = _identity()
    matrix[1, 1], matrix[1, 2] = cosine, -sine
    matrix[2, 1], matrix[2, 2] = sine, cosine
    return matrix


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = _identity()
    matrix[0, 0], matrix[0, 2] = cosine, sine
    matrix[2, 0], matrix[2, 2] = -sine, cosine
    return matrix


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = _identity()
    matrix[0, 0], matrix[0, 1] = cosine, -sine
    matrix[1, 0], matrix[1, 1] = sine, cosine
    return matrix


def _perspective(vertical_fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    factor = 1.0 / math.tan(vertical_fov * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = factor / max(aspect, 1e-6)
    matrix[1, 1] = factor
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _hex_color(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    area_normals = np.cross(second - first, third - first)
    normals = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(normals, faces[:, 0], area_normals)
    np.add.at(normals, faces[:, 1], area_normals)
    np.add.at(normals, faces[:, 2], area_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-9
    normals[valid] /= lengths[valid, None]
    normals[~valid, 2] = 1.0
    return normals


class _GpuOverlay(QWidget):
    """Transparent QWidget layer for gesture landmarks and anatomy callouts."""

    def __init__(self, renderer: "GpuHeartWidget") -> None:
        super().__init__(renderer)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        renderer = self.parentWidget()
        if isinstance(renderer, GpuHeartWidget):
            renderer._paint_overlay(painter)
        painter.end()


class GpuHeartWidget(QOpenGLWidget):
    """Render and pick anatomical meshes without exposing OpenGL to the shell."""

    part_activated = Signal(object)
    fragment_pressed = Signal(object)
    fragment_moved = Signal(float, float, float)
    fragment_released = Signal()
    rotation_requested = Signal(float, float)
    zoom_requested = Signal(float)

    def __init__(self, meshes: tuple[AnatomicalMesh, ...], parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(False)
        self._source_meshes = meshes
        self._gpu_meshes: list[_GpuMesh] = []
        self._program = 0
        self._uniforms: dict[str, int] = {}
        self._snapshot: HeartSnapshot | None = None
        self._last_mouse: QPointF | None = None
        self._press_position: QPointF | None = None
        self._drag_distance = 0.0
        self._dragging_fragment = False
        self._hand_points: tuple[tuple[float, float], ...] = ()
        self._pointer: tuple[float, float] | None = None
        self._gesture_label = ""
        self._last_overlay_observation = -1.0
        self._callout: HeartPart | None = None
        self._inspector: QWidget | None = None
        self._last_leader_anchor: QPointF | None = None
        self._last_leader_update = 0.0
        self._paint_frames = 0
        self._paint_sample_at = time.perf_counter()
        self._render_fps = 0.0
        ordered_parts = tuple(dict.fromkeys(mesh.part_id for mesh in meshes))
        self._pick_id_by_part = {part: index + 1 for index, part in enumerate(ordered_parts)}
        self._part_by_pick_id = {value: key for key, value in self._pick_id_by_part.items()}
        self._overlay = _GpuOverlay(self)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()

    @property
    def render_fps(self) -> float:
        return self._render_fps

    def set_snapshot(self, snapshot: HeartSnapshot) -> None:
        previous = self._last_leader_anchor
        self._snapshot = snapshot
        self.update()
        if self._callout is not None:
            current = self._selected_anchor()
            moved = ((previous is None) != (current is None)
                     or (previous is not None and current is not None
                         and (previous - current).manhattanLength() >= 0.75))
            now = time.perf_counter()
            if moved and now - self._last_leader_update >= 1.0 / 30.0:
                # Repaint only the old/new leader region. A full transparent
                # QWidget repaint would unnecessarily rasterize the inspector
                # at the 3D frame rate, even while the reading position is still.
                self._overlay.update(self._leader_bounds(previous).united(
                    self._leader_bounds(current)).toAlignedRect())
                self._last_leader_anchor = current
                self._last_leader_update = now

    def _leader_bounds(self, anchor: QPointF | None) -> QRectF:
        """Return a conservative dirty region around the screen-space leader."""
        if anchor is None or self._inspector is None:
            return QRectF()
        card = QRectF(self._inspector.geometry())
        end = QPointF(card.left(), max(card.top() + 28.0,
                      min(card.bottom() - 28.0, anchor.y() - 48.0)))
        return QRectF(anchor, end).normalized().adjusted(-36.0, -10.0, 10.0, 10.0)

    def set_hand_overlay(
        self,
        observation: HandObservation,
        window_width: float,
        window_height: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        stale = observation.hands == 0 or time.monotonic() - observation.captured_at > 0.7
        if stale:
            if not self._hand_points and self._pointer is None:
                return
            self._hand_points = ()
            self._pointer = None
            self._gesture_label = ""
            self._last_overlay_observation = observation.captured_at
        else:
            if observation.captured_at == self._last_overlay_observation:
                return
            self._last_overlay_observation = observation.captured_at
            self._hand_points = tuple(
                (x * window_width - origin_x, y * window_height - origin_y)
                for x, y in observation.landmarks
            )
            self._pointer = (
                observation.pointer_x * window_width - origin_x,
                observation.pointer_y * window_height - origin_y,
            )
            self._gesture_label = observation.gesture.value.upper()
        self._overlay.update()

    def set_callout(self, part: HeartPart | None) -> None:
        """Connect selected anatomy to its native inspector in either view mode."""

        self._callout = part
        self._last_leader_anchor = self._selected_anchor()
        self._overlay.update()

    def set_inspector(self, inspector: QWidget) -> None:
        """Host a scrollable data card without reducing the OpenGL viewport."""
        self._inspector = inspector
        inspector.setParent(self)
        self._layout_inspector()

    def _layout_inspector(self) -> None:
        """Keep the reading surface at the right, within resized window bounds."""
        if self._inspector is None:
            return
        width = min(340, max(280, round(self.width() * 0.27)))
        height = min(600, max(1, self.height() - 32))
        self._inspector.setGeometry(self.width() - width - 16,
                                    (self.height() - height) // 2, width, height)
        self._inspector.raise_()
        self._overlay.raise_()

    def initializeGL(self) -> None:  # noqa: N802 - Qt callback name
        self._program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        uniform_names = (
            "uModel", "uView", "uProjection", "uAtrialContraction",
            "uVentricularContraction", "uContractileGroup", "uIsTissue",
            "uBaseColor", "uCameraPosition", "uPickColor", "uRoughness",
            "uSheen", "uOpacity", "uHighlighted", "uAnalytical", "uPicking",
            "uCutaway", "uMeshCenter", "uActivation", "uCycleProgress",
            "uValveMode", "uValveOpen", "uValveRadius",
            "uSectionSurface",
            "uConduction",
        )
        self._uniforms = {
            name: glGetUniformLocation(self._program, name) for name in uniform_names
        }
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._upload_meshes()

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802 - Qt callback name
        ratio = self.devicePixelRatioF()
        glViewport(0, 0, max(1, int(width * ratio)), max(1, int(height * ratio)))
        self._overlay.setGeometry(0, 0, width, height)
        self._layout_inspector()
        self._overlay.raise_()

    def paintGL(self) -> None:  # noqa: N802 - Qt callback name
        glClearColor(0.003, 0.012, 0.025, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._program and self._snapshot is not None:
            glUseProgram(self._program)
            self._draw_scene(picking=False)
            glBindVertexArray(0)
            glUseProgram(0)
        self._paint_frames += 1
        elapsed = time.perf_counter() - self._paint_sample_at
        if elapsed >= 1.0:
            self._render_fps = self._paint_frames / elapsed
            self._paint_frames = 0
            self._paint_sample_at = time.perf_counter()

    def pick_part(self, x: float, y: float) -> HeartPartId | None:
        """Render semantic IDs into a single-sample framebuffer and read one pixel."""

        if not self.isValid() or self._snapshot is None:
            return None
        if (self._inspector is not None and self._inspector.isVisible()
                and QRectF(self._inspector.geometry()).contains(QPointF(x, y))):
            return None
        self.makeCurrent()
        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(
            QOpenGLFramebufferObject.Attachment.CombinedDepthStencil
        )
        fbo_format.setSamples(0)
        picking_fbo = QOpenGLFramebufferObject(QSize(width, height), fbo_format)
        if not picking_fbo.isValid() or not picking_fbo.bind():
            self.doneCurrent()
            return None
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)
        glDisable(GL_DITHER)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self._program)
        self._draw_scene(picking=True)
        pixel = glReadPixels(
            max(0, int(x * ratio)),
            max(0, int((self.height() - y - 1.0) * ratio)),
            1,
            1,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
        )
        glBindVertexArray(0)
        glUseProgram(0)
        picking_fbo.release()
        del picking_fbo
        glEnable(GL_DITHER)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        self.doneCurrent()
        self.update()
        values = np.frombuffer(pixel, dtype=np.uint8)
        if values.size < 3:
            return None
        identifier = int(values[0]) | (int(values[1]) << 8) | (int(values[2]) << 16)
        return self._part_by_pick_id.get(identifier)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._last_mouse = event.position()
        self._press_position = event.position()
        self._drag_distance = 0.0
        snapshot = self._snapshot
        if snapshot and snapshot.explosion >= 0.45:
            part = self.pick_part(event.position().x(), event.position().y())
            if part is not None:
                self._dragging_fragment = True
                self.fragment_pressed.emit(part)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._last_mouse is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        self._drag_distance += abs(delta.x()) + abs(delta.y())
        if self._dragging_fragment:
            self.fragment_moved.emit(
                delta.x() / max(1.0, self.width()),
                delta.y() / max(1.0, self.height()),
                0.0,
            )
        else:
            self.rotation_requested.emit(delta.x() * 0.009, delta.y() * 0.009)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        was_fragment = self._dragging_fragment
        self._dragging_fragment = False
        self.fragment_released.emit()
        if not was_fragment and self._drag_distance < 9.0:
            snapshot = self._snapshot
            self.part_activated.emit(self.pick_part(event.position().x(), event.position().y()))
        self._last_mouse = None
        self._press_position = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        direction = 1.0 if event.angleDelta().y() > 0 else -1.0
        snapshot = self._snapshot
        if snapshot and snapshot.grabbed_fragment is not None:
            self.fragment_moved.emit(0.0, 0.0, direction * 0.04)
        else:
            self.zoom_requested.emit(1.10 if direction > 0 else 0.91)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.isValid():
            self.makeCurrent()
            for mesh in self._gpu_meshes:
                glDeleteBuffers(1, [mesh.vertex_buffer])
                glDeleteBuffers(1, [mesh.index_buffer])
                glDeleteVertexArrays(1, [mesh.vao])
            self.doneCurrent()
        super().closeEvent(event)

    def _upload_meshes(self) -> None:
        grouped: dict[tuple, list[AnatomicalMesh]] = {}
        for source in self._source_meshes:
            key = (
                source.part_id,
                source.analysis_color,
                source.realistic_color,
                source.explosion_vector,
                source.tissue,
                source.default_visible,
                source.view,
                source.motion,
                source.opacity,
            )
            grouped.setdefault(key, []).append(source)

        for sources in grouped.values():
            source = sources[0]
            vertex_chunks: list[np.ndarray] = []
            normal_chunks: list[np.ndarray] = []
            index_chunks: list[np.ndarray] = []
            vertex_offset = 0
            for component in sources:
                component_vertices = np.asarray(
                    [(vertex.x, vertex.y, vertex.z) for vertex in component.vertices],
                    dtype=np.float32,
                )
                component_faces = np.asarray(
                    component.faces, dtype=np.uint32
                )[:, (0, 2, 1)].copy()
                vertex_chunks.append(component_vertices)
                normal_chunks.append(
                    _vertex_normals(component_vertices, component_faces.astype(np.int32))
                )
                index_chunks.append(component_faces.reshape(-1) + vertex_offset)
                vertex_offset += len(component_vertices)
            vertices = np.concatenate(vertex_chunks, axis=0)
            normals = np.concatenate(normal_chunks, axis=0)
            interleaved = np.column_stack((vertices, normals)).astype(np.float32)
            indices = np.concatenate(index_chunks).astype(np.uint32, copy=False)
            vao = int(glGenVertexArrays(1))
            vertex_buffer = int(glGenBuffers(1))
            index_buffer = int(glGenBuffers(1))
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            stride = 6 * np.dtype(np.float32).itemsize
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * np.dtype(np.float32).itemsize)
            )
            self._gpu_meshes.append(_GpuMesh(
                source.part_id,
                vertices,
                (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5,
                vao,
                vertex_buffer,
                index_buffer,
                int(indices.size),
                _hex_color(source.analysis_color),
                _hex_color(source.realistic_color),
                np.asarray((source.explosion_vector.x, source.explosion_vector.y,
                            source.explosion_vector.z), dtype=np.float32),
                source.tissue,
                source.default_visible,
                source.view,
                source.motion,
                source.opacity,
            ))
        glBindVertexArray(0)

    def _draw_scene(self, picking: bool) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        ratio = self.devicePixelRatioF()
        glViewport(0, 0, max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        projection, view, scale = self._scene_camera(snapshot)
        glUniformMatrix4fv(self._uniforms["uView"], 1, GL_TRUE, view)
        glUniformMatrix4fv(self._uniforms["uProjection"], 1, GL_TRUE, projection)
        glUniform3f(self._uniforms["uCameraPosition"], 0.0, 0.08, 7.0)
        glUniform1i(self._uniforms["uPicking"], 1 if picking else 0)
        glUniform1f(self._uniforms["uAtrialContraction"], snapshot.cycle.atrial_contraction)
        glUniform1f(
            self._uniforms["uVentricularContraction"],
            snapshot.cycle.ventricular_emptying,
        )
        glUniform1f(self._uniforms["uCycleProgress"], snapshot.cycle.progress)
        self._draw_mesh_group(snapshot, scale, picking)

    def _scene_camera(self, snapshot: HeartSnapshot) -> tuple[np.ndarray, np.ndarray, float]:
        """Share the exact camera between drawing, picking, and label projection."""
        projection = _perspective(math.radians(34.0), self.width() / max(self.height(), 1), 0.1, 40.0)
        view = _translation(0.0, -0.08, -7.0)
        # Selection never changes the camera or fragment transforms.
        aspect = self.width() / max(self.height(), 1)
        fit = min(1.0, aspect / 0.90)
        scale = 1.06 * snapshot.zoom * fit / (1.0 + snapshot.explosion * 0.40)
        return projection, view, scale

    def _selected_anchor(self) -> QPointF | None:
        """Project the selected structure's main visible mesh in logical pixels."""
        snapshot = self._snapshot
        if snapshot is None or self._callout is None:
            return None
        candidates = [mesh for mesh in self._gpu_meshes
                      if mesh.part_id == self._callout.identifier and visible_mesh(mesh, snapshot)]
        if not candidates:
            return None
        # Access precomputed centers only; never scan vertices in the frame loop.
        mesh = max(candidates, key=lambda item: item.index_count)
        projection, view, scale = self._scene_camera(snapshot)
        model = placement_matrix(snapshot, mesh.explosion, mesh.part_id, scale)
        clip = projection @ view @ model @ np.append(mesh.center, 1.0)
        if clip[3] <= 0:
            return None
        point = clip[:3] / clip[3]
        if not -1.0 <= point[2] <= 1.0:
            return None
        return QPointF(float((point[0] + 1.0) * self.width() * 0.5),
                       float((1.0 - point[1]) * self.height() * 0.5))

    def _draw_mesh_group(
        self,
        snapshot: HeartSnapshot,
        scale: float,
        picking: bool,
    ) -> None:
        for mesh in sorted(self._gpu_meshes, key=lambda item: item.opacity < 1.0):
            if not visible_mesh(mesh, snapshot):
                continue
            model = placement_matrix(snapshot, mesh.explosion, mesh.part_id, scale)
            glUniformMatrix4fv(self._uniforms["uModel"], 1, GL_TRUE, model)
            glUniform1f(self._uniforms["uIsTissue"], 1.0 if mesh.tissue else 0.0)
            contractile_group = 0.0
            if mesh.part_id in {HeartPartId.RIGHT_ATRIUM, HeartPartId.LEFT_ATRIUM}:
                contractile_group = 1.0
            elif mesh.part_id in {
                HeartPartId.RIGHT_VENTRICLE,
                HeartPartId.LEFT_VENTRICLE,
                HeartPartId.MYOCARDIUM,
                HeartPartId.PAPILLARY_MUSCLES,
                HeartPartId.INTERVENTRICULAR_SEPTUM,
                HeartPartId.ENDOCARDIUM,
            }:
                contractile_group = 2.0
            glUniform1f(self._uniforms["uContractileGroup"], contractile_group)
            opacity = mesh.opacity
            glDepthMask(GL_TRUE if picking or opacity >= 1.0 else GL_FALSE)
            glUniform1f(self._uniforms["uOpacity"], 1.0 if picking else opacity)
            glUniform1f(
                self._uniforms["uHighlighted"],
                1.0 if snapshot.selected_part == mesh.part_id else 0.0,
            )
            glUniform1f(self._uniforms["uAnalytical"], 0.0 if snapshot.realistic else 1.0)
            glUniform1f(self._uniforms["uCutaway"], 0.0)
            section_surface = snapshot.interior_view and mesh.part_id in {
                HeartPartId.CORONARY_ARTERIES, HeartPartId.CORONARY_VEINS,
            } and snapshot.selected_part != mesh.part_id
            glUniform1f(self._uniforms["uSectionSurface"], float(section_surface))
            glUniform1f(self._uniforms["uConduction"], float(
                snapshot.beating and mesh.part_id == HeartPartId.CARDIAC_CONDUCTION))
            glUniform1f(self._uniforms["uActivation"], self._activation(mesh.part_id, snapshot))
            color = mesh.realistic_color if snapshot.realistic else mesh.analysis_color
            glUniform3f(self._uniforms["uBaseColor"], *color)
            glUniform3f(self._uniforms["uMeshCenter"], *mesh.center)
            valve_mode = {"none": 0.0, "atrioventricular": 1.0, "semilunar": 2.0}[mesh.motion]
            valve_open = (snapshot.cycle.atrioventricular_valves_open if valve_mode == 1.0
                          else snapshot.cycle.semilunar_valves_open)
            glUniform1f(self._uniforms["uValveMode"], valve_mode)
            glUniform1f(self._uniforms["uValveOpen"], float(valve_open))
            radius = max(float(np.ptp(mesh.vertices[:, 0])) * 0.5, 0.01) if valve_mode else 1.0
            glUniform1f(self._uniforms["uValveRadius"], radius)
            roughness = 0.50 if mesh.tissue else 0.34
            sheen = 0.58 if mesh.tissue else 0.30
            glUniform1f(self._uniforms["uRoughness"], roughness)
            glUniform1f(self._uniforms["uSheen"], sheen)
            pick_id = self._pick_id_by_part[mesh.part_id]
            glUniform3f(
                self._uniforms["uPickColor"],
                (pick_id & 255) / 255.0,
                ((pick_id >> 8) & 255) / 255.0,
                ((pick_id >> 16) & 255) / 255.0,
            )
            glBindVertexArray(mesh.vao)
            glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, None)
        glDepthMask(GL_TRUE)

    @staticmethod
    def _activation(part: HeartPartId, snapshot: HeartSnapshot) -> float:
        """Return subtle phase-specific flow or conduction emphasis."""

        if not snapshot.realistic:
            return 0.0
        cycle = snapshot.cycle
        if part == HeartPartId.CORONARY_ARTERIES:
            return cycle.coronary_perfusion * 0.34
        if part in {HeartPartId.CORONARY_VEINS, HeartPartId.CORONARY_SINUS}:
            return 0.14
        if part == HeartPartId.CARDIAC_CONDUCTION:
            return 0.0
        if part in {HeartPartId.TRICUSPID_VALVE, HeartPartId.MITRAL_VALVE}:
            return 0.25 if cycle.atrioventricular_valves_open else 0.0
        if part in {HeartPartId.AORTIC_VALVE, HeartPartId.PULMONARY_VALVE}:
            return 0.25 if cycle.semilunar_valves_open else 0.0
        if part in {HeartPartId.AORTA, HeartPartId.PULMONARY_ARTERY}:
            return 0.18 if cycle.semilunar_valves_open else 0.0
        if part in {HeartPartId.VENA_CAVA, HeartPartId.PULMONARY_VEINS}:
            return 0.14 if cycle.phase in {
                CardiacPhase.RAPID_FILLING,
                CardiacPhase.DIASTASIS,
                CardiacPhase.ATRIAL_SYSTOLE,
            } else 0.0
        return 0.0

    def _paint_overlay(self, painter: QPainter) -> None:
        if not self._hand_points and self._pointer is None and self._callout is None:
            return
        anchor = self._selected_anchor()
        if anchor is not None and self._inspector is not None and self._inspector.isVisible():
            card = QRectF(self._inspector.geometry())
            if not card.contains(anchor):
                endpoint = QPointF(card.left(), max(card.top() + 28.0,
                                   min(card.bottom() - 28.0, anchor.y() - 48.0)))
                elbow = QPointF(endpoint.x() - 26.0, endpoint.y())
                painter.setPen(QPen(QColor(32, 220, 234, 190), 1.3, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(anchor, elbow)
                painter.drawLine(elbow, endpoint)
                painter.setPen(QPen(QColor("#20dcea"), 1.3))
                painter.drawEllipse(anchor, 5.0, 5.0)
                painter.setBrush(QColor("#20dcea"))
                painter.drawEllipse(anchor, 1.8, 1.8)
        pen = QPen(QColor("#16889a"), 1.2)
        painter.setPen(pen)
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        )
        for start in range(0, len(self._hand_points), 21):
            hand = self._hand_points[start:start + 21]
            if len(hand) < 21:
                continue
            for first, second in connections:
                painter.drawLine(QPointF(*hand[first]), QPointF(*hand[second]))
            painter.setBrush(QColor("#75d5df"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in hand:
                painter.drawEllipse(QPointF(*point), 2.0, 2.0)
            painter.setPen(pen)
        if self._pointer is not None:
            pointer_color = QColor("#30f29b" if self._snapshot and self._snapshot.grabbed_fragment else "#23d9e8")
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(pointer_color, 2.0))
            painter.drawEllipse(QPointF(*self._pointer), 12.0, 12.0)
            painter.setPen(pointer_color)
            painter.drawText(QPointF(self._pointer[0] + 17.0, self._pointer[1] - 14.0), self._gesture_label)
