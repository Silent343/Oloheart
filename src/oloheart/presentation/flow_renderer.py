"""Volumetric blood flow simulation and 3D biconcave red blood cells on hover."""

from __future__ import annotations

import ctypes
import math
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE, GL_TRUE,
    GL_LINES, GL_TRIANGLES, GL_POINTS, GL_PROGRAM_POINT_SIZE,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
    glUseProgram, glGetUniformLocation, glUniformMatrix4fv, glUniform1i,
    glUniform1f, glDrawArrays, glEnable, glDeleteBuffers, glDeleteVertexArrays, glDeleteProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from oloheart.domain.model import anatomy_system_for
from oloheart.infrastructure.circulation_geometry import build_flow_routes, circulation_visible
from oloheart.presentation.scene_layout import placement_matrix

VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
layout(location=2) in vec3 normal;
uniform mat4 mvp;
uniform float pixelRatio;
uniform float pointSizeScale;
out vec3 tint;
out vec3 vNormal;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = pointSizeScale * pixelRatio;
    tint = color;
    vNormal = normal;
}
"""

FRAGMENT = """#version 330 core
in vec3 tint;
in vec3 vNormal;
uniform int renderMode; // 0: plasma stream/arrow, 1: soft blood cell particle, 2: 3D biconcave erythrocyte
out vec4 frag;
void main() {
    if (renderMode == 1) {
        vec2 p = (gl_PointCoord - vec2(0.5)) * 2.0;
        float r2 = dot(p, p);
        if (r2 > 1.0) discard;
        // Biconcave disc shading on sprite particles (darker central dimple, bright rim)
        float r = sqrt(r2);
        float dimple = 0.62 + 0.48 * smoothstep(0.15, 0.72, r) - 0.25 * smoothstep(0.75, 1.0, r);
        float spec = pow(max(0.0, 1.0 - length(p - vec2(-0.28, -0.28))), 4.0) * 0.35;
        float alpha = (1.0 - smoothstep(0.72, 1.0, r)) * 0.92;
        frag = vec4(tint * dimple + vec3(1.0, 0.85, 0.82) * spec, alpha);
        return;
    }
    if (renderMode == 2) {
        // 3D lit biconcave erythrocyte mesh
        vec3 N = normalize(vNormal);
        vec3 L1 = normalize(vec3(-0.45, 0.65, 0.95));
        vec3 L2 = normalize(vec3(0.55, -0.35, 0.65));
        float diff = max(dot(N, L1), 0.0) * 0.75 + max(dot(N, L2), 0.0) * 0.35;
        float rim = pow(1.0 - abs(N.z), 2.2) * 0.42;
        float spec = pow(max(dot(N, normalize(L1 + vec3(0.0, 0.0, 1.0))), 0.0), 32.0) * 0.45;
        vec3 col = tint * (0.28 + diff) + tint * rim + vec3(1.0, 0.88, 0.86) * spec;
        frag = vec4(clamp(col, 0.0, 1.0), 0.96);
        return;
    }
    frag = vec4(tint, 0.55);
}
"""

ROUTE_LABELS = {
    "aorta": "Aorta · Torrente arterial sistémico",
    "brachiocephalic": "Tronco braquiocefálico",
    "left_carotid": "Arteria carótida común izquierda",
    "left_subclavian": "Arteria subclavia izquierda",
    "pulmonary_trunk": "Tronco pulmonar",
    "left_pulmonary_artery": "Arteria pulmonar izquierda",
    "right_pulmonary_artery": "Arteria pulmonar derecha",
    "superior_vena_cava": "Vena cava superior",
    "inferior_vena_cava": "Vena cava inferior",
    "right_superior_pulmonary_vein": "Vena pulmonar superior derecha",
    "right_inferior_pulmonary_vein": "Vena pulmonar inferior derecha",
    "left_superior_pulmonary_vein": "Vena pulmonar superior izquierda",
    "left_inferior_pulmonary_vein": "Vena pulmonar inferior izquierda",
    "right_filling": "Llenado diastólico · Aurícula → Ventrículo derecho",
    "right_ejection": "Eyección sistólica · Tracto de salida pulmonar",
    "left_filling": "Llenado diastólico · Aurícula → Ventrículo izquierdo",
    "left_ejection": "Eyección sistólica · Tracto de salida aórtico",
}


def _build_rbc_template(u_steps: int = 12, v_steps: int = 14, radius: float = 0.019) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 3D biconcave erythrocyte mesh (Evans & Fung biophysical profile)."""
    u = np.linspace(0.0, math.pi, u_steps)
    v = np.linspace(0.0, math.tau, v_steps, endpoint=False)
    ug, vg = np.meshgrid(u, v, indexing="ij")

    rho = np.sin(ug)
    rho2 = rho * rho
    z = 0.5 * np.cos(ug) * (0.207 + 2.002 * rho2 - 1.123 * (rho2 * rho2))
    x = rho * np.cos(vg)
    y = rho * np.sin(vg)

    verts = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32) * radius
    norms = np.stack((x * 0.35, y * 0.35, np.sign(z + 1e-6) * (1.0 - 0.45 * rho2)), axis=-1).reshape(-1, 3).astype(np.float32)
    norms /= np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-6)

    tris = []
    for i in range(u_steps - 1):
        for j in range(v_steps):
            jn = (j + 1) % v_steps
            p0 = i * v_steps + j
            p1 = i * v_steps + jn
            p2 = (i + 1) * v_steps + jn
            p3 = (i + 1) * v_steps + j
            tris.extend(((p0, p1, p2), (p0, p2, p3)))
    tris_arr = np.asarray(tris, dtype=np.int32)
    return verts[tris_arr].reshape(-1, 3), norms[tris_arr].reshape(-1, 3)


class BloodFlowRenderer:
    """Render volumetric blood streams and 3D red blood cells on mouse hover."""

    def __init__(self):
        self.routes = build_flow_routes()
        self.program = 0
        self.hovered_stream: dict | None = None
        self._rbc_verts, self._rbc_norms = _build_rbc_template()
        self._cached_geometry = {}
        # 28 cells per route distributed across the lumen
        self._particle_offsets = np.linspace(0.0, 1.0, 28, endpoint=False, dtype=np.float32)
        angles = np.arange(28, dtype=np.float32) * 2.3999632  # Golden angle spiral in vessel lumen
        radii = 0.016 * np.sqrt(np.linspace(0.05, 0.95, 28, dtype=np.float32))
        self._radial_cos = (radii * np.cos(angles))[:, None]
        self._radial_sin = (radii * np.sin(angles))[:, None]
        # Poiseuille laminar velocity factor: faster in center, slower at wall
        self._laminar_speed = 1.18 - 0.35 * (radii / 0.016)

        for route in self.routes:
            color = np.asarray((0.96, 0.16, 0.18) if route.oxygenated else (0.20, 0.58, 0.96), dtype=np.float32)
            points = route.points
            tangents = np.gradient(points, axis=0)
            tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-5)
            up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
            frame_u = np.cross(tangents, up)
            small = np.linalg.norm(frame_u, axis=1) < 0.15
            if np.any(small):
                frame_u[small] = np.cross(tangents[small], np.array((0.0, 1.0, 0.0), dtype=np.float32))
            frame_u /= np.maximum(np.linalg.norm(frame_u, axis=1, keepdims=True), 1e-5)
            frame_v = np.cross(tangents, frame_u)

            # Build multi-filament helical plasma stream ribbons
            stream_segments = []
            for filament in range(3):
                phase_f = filament * (math.tau / 3.0)
                theta = np.linspace(0.0, math.tau * 2.0, len(points), dtype=np.float32) + phase_f
                r_fil = 0.008 if filament > 0 else 0.0
                fil_pts = points + (frame_u * np.cos(theta)[:, None] + frame_v * np.sin(theta)[:, None]) * r_fil
                ends = fil_pts[::2]
                seg_pos = np.stack((ends[:-1], ends[1:]), axis=1).reshape(-1, 3)
                tint = color * (0.48 if filament == 0 else 0.32)
                normals = np.tile((0.0, 0.0, 1.0), (len(seg_pos), 1))
                stream_segments.append(np.column_stack((seg_pos, np.tile(tint, (len(seg_pos), 1)), normals)))
            line = np.concatenate(stream_segments, axis=0).astype(np.float32)

            arrows = []
            for sample in (32, 68):
                tip = points[sample]
                tangent = tangents[sample]
                side = frame_u[sample]
                tail = tip - tangent * 0.052
                arrows.extend((tip, tail + side * 0.020, tail - side * 0.020))
            arrow_norms = np.tile((0.0, 0.0, 1.0), (len(arrows), 1))
            arrow = np.column_stack((arrows, np.tile(color * 0.88, (len(arrows), 1)), arrow_norms)).astype(np.float32)
            dim = arrow.copy()
            dim[:, 3:6] *= 0.28
            self._cached_geometry[route.key] = (
                color,
                line,
                arrow,
                dim,
                tangents.astype(np.float32),
                frame_u.astype(np.float32),
                frame_v.astype(np.float32),
            )

    def initialize(self):
        self.program = compileProgram(
            compileShader(VERTEX, GL_VERTEX_SHADER),
            compileShader(FRAGMENT, GL_FRAGMENT_SHADER),
        )
        self.vao = int(glGenVertexArrays(1))
        self.vbo = int(glGenBuffers(1))
        self.uniforms = {
            k: glGetUniformLocation(self.program, k)
            for k in ("mvp", "renderMode", "pixelRatio", "pointSizeScale")
        }
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 36  # 9 floats: pos(3), color(3), normal(3)
        for index, offset in ((0, 0), (1, 12), (2, 24)):
            glEnableVertexAttribArray(index)
            glVertexAttribPointer(index, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glBindVertexArray(0)

    def draw(
        self,
        state,
        projection,
        view,
        scale,
        pixel_ratio,
        hover_pos: tuple[float, float] | None = None,
        viewport_size: tuple[int, int] = (1, 1),
    ):
        self.hovered_stream = None
        if not circulation_visible(state):
            return

        model = placement_matrix(state, (0, 0, 0), self.routes[0].parts[0], scale)
        mvp = projection @ view @ model
        vw, vh = max(1, viewport_size[0]), max(1, viewport_size[1])

        lines, arrows, particles, rbc_meshes = [], [], [], []
        best_hover_dist = 68.0  # Hover detection radius in logical pixels

        for route in self.routes:
            if any(anatomy_system_for(part) not in state.visible_systems for part in route.parts):
                continue
            color, line, arrow, dim, tangents, frame_u, frame_v = self._cached_geometry[route.key]
            points = route.points
            lines.append(line)
            active = route.active(state)
            arrows.append(arrow if active else dim)

            # Check mouse proximity in screen space
            hovered_route = False
            hover_center_idx = -1
            if hover_pos is not None:
                mx, my = hover_pos
                clip = (mvp @ np.column_stack((points, np.ones(len(points), dtype=np.float32))).T).T
                valid = clip[:, 3] > 0.05
                if np.any(valid):
                    ndc = clip[valid, :2] / clip[valid, 3:4]
                    sx = (ndc[:, 0] + 1.0) * (vw * 0.5)
                    sy = (1.0 - ndc[:, 1]) * (vh * 0.5)
                    dists = np.hypot(sx - mx, sy - my)
                    min_idx_local = int(np.argmin(dists))
                    min_dist = float(dists[min_idx_local])
                    if min_dist < best_hover_dist:
                        best_hover_dist = min_dist
                        hovered_route = True
                        valid_indices = np.flatnonzero(valid)
                        hover_center_idx = int(valid_indices[min_idx_local])
                        # Screen-space flow tangent for the HUD microscope callout
                        next_i = min(len(sx) - 1, min_idx_local + 1)
                        prev_i = max(0, min_idx_local - 1)
                        sdx = float(sx[next_i] - sx[prev_i])
                        sdy = float(sy[next_i] - sy[prev_i])
                        slen = max(math.hypot(sdx, sdy), 1e-5)
                        self.hovered_stream = {
                            "screen_x": float(sx[min_idx_local]),
                            "screen_y": float(sy[min_idx_local]),
                            "mouse_x": float(mx),
                            "mouse_y": float(my),
                            "dir_x": sdx / slen,
                            "dir_y": sdy / slen,
                            "oxygenated": route.oxygenated,
                            "active": active,
                            "label": ROUTE_LABELS.get(route.key, "Torrente sanguíneo"),
                            "phase": state.cycle.phase.value,
                            "bpm": state.bpm,
                        }

            if active:
                clock = state.flow_times[route.gate]
                base_rate = 0.52 if route.gate == 0 else 0.88
                samples = ((clock * base_rate * self._laminar_speed + self._particle_offsets) % 1.0) * 99.0
                indices = samples.astype(np.int32)
                weight = (samples - indices)[:, None]
                next_idx = np.minimum(indices + 1, 99)

                center_pos = points[indices] * (1.0 - weight) + points[next_idx] * weight
                u_vec = frame_u[indices]
                v_vec = frame_v[indices]
                t_vec = tangents[indices]
                positions = center_pos + u_vec * self._radial_cos + v_vec * self._radial_sin

                # Add plasma / blood cell particles
                p_normals = np.tile((0.0, 0.0, 1.0), (len(positions), 1))
                particles.append(np.column_stack((positions, np.tile(color, (len(positions), 1)), p_normals)))

                # Instantiate 3D biconcave erythrocytes (glóbulos rojos)
                # Always show subtle 3D cells in flow, and magnify/multiply them when hovered
                rbc_color = np.asarray((0.88, 0.09, 0.12) if route.oxygenated else (0.48, 0.15, 0.42), dtype=np.float32)
                step = 1 if hovered_route else 4
                for cell_i in range(0, len(positions), step):
                    idx_val = int(indices[cell_i])
                    if hovered_route and abs(idx_val - hover_center_idx) <= 26:
                        proximity = 1.0 - abs(idx_val - hover_center_idx) / 26.0
                        cell_scale = 1.35 + 1.15 * proximity
                    else:
                        cell_scale = 0.85

                    # Tumbling rotation matrix in vessel frame
                    spin = clock * 5.5 + cell_i * 1.1
                    cs, sn = math.cos(spin), math.sin(spin)
                    b1 = u_vec[cell_i] * cs + v_vec[cell_i] * sn
                    b2 = t_vec[cell_i]
                    b3 = -u_vec[cell_i] * sn + v_vec[cell_i] * cs
                    rot = np.stack((b1, b2, b3), axis=0)

                    v_trans = (self._rbc_verts * cell_scale) @ rot + positions[cell_i]
                    n_trans = self._rbc_norms @ rot
                    c_tile = np.tile(rbc_color, (len(v_trans), 1))
                    rbc_meshes.append(np.column_stack((v_trans, c_tile, n_trans)))

        glUseProgram(self.program)
        glUniformMatrix4fv(self.uniforms["mvp"], 1, GL_TRUE, mvp)
        glUniform1f(self.uniforms["pixelRatio"], pixel_ratio)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        for primitive, data, mode, pt_size in (
            (GL_LINES, lines, 0, 1.0),
            (GL_TRIANGLES, arrows, 0, 1.0),
            (GL_POINTS, particles, 1, 7.5),
            (GL_TRIANGLES, rbc_meshes, 2, 1.0),
        ):
            if not data:
                continue
            array = np.concatenate(data, axis=0).astype(np.float32, copy=False)
            glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_DYNAMIC_DRAW)
            glUniform1i(self.uniforms["renderMode"], mode)
            glUniform1f(self.uniforms["pointSizeScale"], pt_size)
            glDrawArrays(primitive, 0, len(array))

        glBindVertexArray(0)

    def dispose(self):
        if self.program:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])
            glDeleteProgram(self.program)
            self.program = 0