"""Immersive native desktop shell for OloHeart."""

from __future__ import annotations

import math
import queue
import random
import time
import tkinter as tk

from oloheart.application.controller import (
    GestureCoordinator,
    HandTracker,
    HeartController,
    InteractionAction,
)
from oloheart.domain.model import GestureKind, HandObservation, HeartPartId, HeartSnapshot
from oloheart.infrastructure.geometry import build_heart_geometry
from oloheart.presentation.renderer import HeartRenderer, Rect


class DesktopApplication:
    """A frameless native workstation; all controls are canvas primitives and gestures."""

    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )

    COLORS = {
        "background": "#01050a",
        "panel": "#030b13",
        "panel_alt": "#06111a",
        "line": "#16364a",
        "line_hot": "#00ddeb",
        "cyan": "#00e9f5",
        "cyan_soft": "#68bfc8",
        "text": "#dbe8ed",
        "muted": "#718692",
        "green": "#20ee92",
        "red": "#ff5a62",
    }

    def __init__(self, root: tk.Tk, controller: HeartController, tracker: HandTracker,
                 fullscreen: bool = True) -> None:
        self.root = root
        self.controller = controller
        self.tracker = tracker
        self.coordinator = GestureCoordinator(controller)
        self.canvas = tk.Canvas(root, background=self.COLORS["background"], highlightthickness=0,
                                cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.renderer = HeartRenderer(self.canvas, build_heart_geometry())
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.hand = HandObservation.empty(time.monotonic())
        self.camera_status = "INICIANDO ENLACE ESPACIAL"
        self.width, self.height = 1440, 900
        self.viewport = Rect(280, 76, 780, 720)
        self.part_regions: dict[HeartPartId, Rect] = {}
        self.last_frame = time.perf_counter()
        self.mouse = (0.0, 0.0)
        self.mouse_visible_until = 0.0
        self.drag_origin: tuple[float, float] | None = None
        self.drag_last: tuple[float, float] | None = None
        self.drag_distance = 0.0
        self.mouse_fragment_drag = False
        self.hovered_part: HeartPartId | None = None
        self.status_flash = "SISTEMA ANATÓMICO LISTO"
        self.status_flash_until = time.monotonic() + 3.0
        self.render_fps = 0.0
        self.vision_fps = 0.0
        self._render_frames = 0
        self._vision_frames = 0
        self._render_sample_at = time.perf_counter()
        self._vision_sample_at = time.perf_counter()
        self._last_vision_at = 0.0
        self._next_hud_at = 0.0
        self._last_hud_at = 0.0
        self._hud_dirty = True
        self._hand_overlay_dirty = True
        self._hand_overlay_visible = False
        self._stars: list[tuple[float, float, float]] = []

        root.title("OloHeart · Spatial Cardiac Intelligence")
        root.configure(background=self.COLORS["background"])
        root.minsize(1100, 680)
        root.attributes("-fullscreen", fullscreen)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Configure>", self._on_resize)
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<KeyPress-e>", lambda _event: self._toggle_explosion())
        root.bind("<KeyPress-r>", lambda _event: self._toggle_realism())
        root.bind("<KeyPress-space>", lambda _event: self._toggle_realism())
        root.bind("<KeyPress-s>", lambda _event: self._toggle_spatial_mode())
        root.bind("<KeyPress-Home>", lambda _event: self._reset_view())
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Double-Button-1>", lambda _event: self._toggle_realism())

    def start(self) -> None:
        self.tracker.start(self.post_observation, self.post_camera_status)
        self.root.after(20, self._pump_events)
        self.root.after(1, self._animate)

    def close(self) -> None:
        self.tracker.stop()
        if self.root.winfo_exists():
            self.root.destroy()

    def post_observation(self, observation: HandObservation) -> None:
        self.events.put(("hand", observation))

    def post_camera_status(self, status: str) -> None:
        self.events.put(("status", status))

    def _animate(self) -> None:
        if not self.root.winfo_exists():
            return
        now = time.perf_counter()
        delta = min(0.08, now - self.last_frame)
        self.last_frame = now
        snapshot = self.controller.tick(delta)
        self._calculate_layout(snapshot)
        self.renderer.render(snapshot, self.viewport)
        self._render_frames += 1
        sample_elapsed = now - self._render_sample_at
        if sample_elapsed >= 1.0:
            self.render_fps = self._render_frames / sample_elapsed
            self._render_frames = 0
            self._render_sample_at = now
        if now >= self._next_hud_at or (self._hud_dirty and now - self._last_hud_at >= 0.12):
            self._draw_hud(snapshot, now)
            self._hud_dirty = False
            self._last_hud_at = now
            self._next_hud_at = now + 1.0
        if (self._hand_overlay_visible and now >= self.mouse_visible_until
                and (self.hand.hands == 0 or now - self.hand.captured_at >= 0.7)):
            self._hand_overlay_dirty = True
        if self._hand_overlay_dirty:
            self._draw_hand_overlay(now)
            self._hand_overlay_dirty = False
        elapsed_ms = (time.perf_counter() - now) * 1000.0
        # Tk timers on Windows commonly wake several milliseconds late. A 26 ms
        # request keeps the perceived animation near 30 FPS while cached cardiac
        # phases prevent redundant mesh rasterization.
        self.root.after(max(1, int(26.0 - elapsed_ms)), self._animate)

    def _pump_events(self) -> None:
        latest: HandObservation | None = None
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "hand":
                latest = payload  # type: ignore[assignment]
                self._vision_frames += 1
                self._last_vision_at = time.perf_counter()
            elif kind == "status":
                self.camera_status = str(payload)
                self._hud_dirty = True
        if latest is not None:
            self.hand = latest
            self._hand_overlay_dirty = True
            self._hud_dirty = True
            result = self.coordinator.process(latest)
            if result.action == InteractionAction.ACTIVATE_POINTER:
                self._activate_at(result.pointer_x * self.width, result.pointer_y * self.height)
            elif result.action == InteractionAction.FRAGMENT_GRAB_STARTED:
                snapshot = self.controller.snapshot()
                identifier = (None if snapshot.selected_part else self.renderer.part_at(
                    result.pointer_x * self.width, result.pointer_y * self.height
                ))
                if identifier is not None and self.controller.begin_fragment_drag(identifier):
                    part = self.controller.part(identifier)
                    self._flash("FRAGMENTO TOMADO · " + (part.display_name.upper() if part else ""))
            elif result.action == InteractionAction.FRAGMENT_GRAB_MOVED:
                if self.controller.move_grabbed_fragment(
                    result.delta_x, result.delta_y, result.delta_depth
                ):
                    self._hud_dirty = True
                    self._hand_overlay_dirty = True
            elif result.action == InteractionAction.FRAGMENT_GRAB_ENDED:
                grabbed = self.controller.snapshot().grabbed_fragment
                self.controller.release_fragment()
                if grabbed is not None:
                    part = self.controller.part(grabbed)
                    self._flash("FRAGMENTO LIBERADO · " + (part.display_name.upper() if part else ""))
            elif result.action == InteractionAction.EXPLOSION_CHANGED:
                self._flash("CAPAS ANATÓMICAS RECONFIGURADAS")
            elif result.action == InteractionAction.REALISM_CHANGED:
                self._flash("TEJIDO REALISTA · LATIDO SINCRONIZADO")
            elif result.action == InteractionAction.FOCUS_WHOLE:
                self._flash("CORAZÓN COMPLETO RESTAURADO")
            elif result.action == InteractionAction.SPATIAL_MODE_CHANGED:
                state = self.controller.snapshot()
                self._flash("VISTA ESPACIAL · ESCENARIO COMPLETO" if state.spatial_mode else
                            "ESTACIÓN ANATÓMICA · PANELES ACTIVOS")
        now = time.perf_counter()
        sample_elapsed = now - self._vision_sample_at
        if sample_elapsed >= 1.0:
            self.vision_fps = self._vision_frames / sample_elapsed
            self._vision_frames = 0
            self._vision_sample_at = now
        self.root.after(20, self._pump_events)

    def _calculate_layout(self, snapshot: HeartSnapshot) -> None:
        """Keep the 3D field expansive; information panels float above it."""

        inset = 10 if snapshot.spatial_mode else 14
        top = 10 if snapshot.spatial_mode else 62
        bottom = 10 if snapshot.spatial_mode else 14
        self.viewport = Rect(inset, top, max(360, self.width - inset * 2),
                             max(360, self.height - top - bottom))

    def _draw_hud(self, snapshot: HeartSnapshot, now: float) -> None:
        self.canvas.delete("hud")
        colors = self.COLORS
        if snapshot.spatial_mode:
            self._draw_spatial_hud(snapshot, now)
            return

        left_width = max(244, min(278, self.width * 0.18))
        right_width = max(316, min(372, self.width * 0.235))
        right_x = self.width - right_width
        self._rectangle(12, 76, left_width - 10, self.height - 18,
                        "#020b12", outline=colors["line"])
        self._rectangle(right_x + 10, 76, self.width - 12, self.height - 18,
                        "#020b12", outline=colors["line"])
        self._rectangle(12, 10, 322, 57, colors["panel_alt"], outline=colors["line"])

        self._text(20, 20, "OLOHEART", colors["cyan"], ("Bahnschrift SemiBold", 16), anchor="nw")
        self._text(20, 44, "ANATOMÍA ESPACIAL", colors["muted"], ("Cascadia Mono", 8), anchor="nw")
        self._text(350, 27, "◉  CARDIAC INTELLIGENCE", colors["text"],
                   ("Bahnschrift SemiBold", 14), anchor="nw")
        mode = "TEJIDO REALISTA · LATIDO ACTIVO" if snapshot.realistic else "MATRIZ ANATÓMICA · MODO ANALÍTICO"
        mode_color = colors["red"] if snapshot.realistic else colors["cyan"]
        self._text(self.width / 2, 32, mode, mode_color,
                   ("Cascadia Mono", 9, "bold"), anchor="center")
        camera_live = time.perf_counter() - self._last_vision_at < 1.2
        self._status_indicator(self.width - 255, 25, colors["green"], "SYSTEM ONLINE")
        self._status_indicator(self.width - 122, 25,
                               colors["green"] if camera_live else colors["red"],
                               "CAMERA LIVE" if camera_live else "CAMERA WAIT")

        self._draw_left_navigation(snapshot, left_width)
        self._draw_right_panel(snapshot, right_x)
        self._draw_viewport_frame(snapshot)
        self._draw_interaction_bar(snapshot)

        if now < self.status_flash_until:
            self._rectangle(self.width * 0.34, 73, self.width * 0.66, 103,
                            "#06202a", outline=mode_color)
            self._text(self.width / 2, 88, self.status_flash,
                       mode_color, ("Cascadia Mono", 8, "bold"), anchor="center")

    def _draw_spatial_hud(self, snapshot: HeartSnapshot, now: float) -> None:
        """Draw only spatial instrumentation and an anatomy callout card."""

        colors = self.COLORS
        self.part_regions.clear()
        self._draw_viewport_frame(snapshot)
        camera_live = time.perf_counter() - self._last_vision_at < 1.2
        self._rectangle(24, 24, 246, 75, "#020c13", outline=colors["line"])
        self._text(40, 38, "OLOHEART · VISTA ESPACIAL", colors["cyan"],
                   ("Cascadia Mono", 8, "bold"), anchor="nw")
        self._text(40, 58, "MEÑIQUE: VOLVER A LA ESTACIÓN", colors["muted"],
                   ("Cascadia Mono", 6), anchor="nw")
        self._status_indicator(self.width - 132, 34,
                               colors["green"] if camera_live else colors["red"],
                               "CAMERA LIVE" if camera_live else "CAMERA WAIT")
        part = self.controller.part(snapshot.selected_part)
        if part is not None:
            self._draw_spatial_data_card(part)
        self._text(28, self.height - 25,
                   "ÍNDICE: SELECCIONAR   ·   MANO: ROTAR   ·   DOS MANOS: ZOOM   ·   👍: RECONSTRUIR",
                   colors["cyan_soft"], ("Cascadia Mono", 7), anchor="sw")
        self._text(self.width - 28, self.height - 25,
                   f"VISIÓN {self.vision_fps:02.0f} FPS  ·  RENDER {self.render_fps:02.0f} FPS",
                   colors["muted"], ("Cascadia Mono", 7), anchor="se")
        if now < self.status_flash_until:
            mode_color = colors["red"] if snapshot.realistic else colors["cyan"]
            self._rectangle(self.width * 0.34, 24, self.width * 0.66, 55,
                            "#04151d", outline=mode_color)
            self._text(self.width / 2, 39, self.status_flash, mode_color,
                       ("Cascadia Mono", 8, "bold"), anchor="center")

    def _draw_left_navigation(self, snapshot: HeartSnapshot, panel_width: float) -> None:
        colors = self.COLORS
        self._text(20, 88, "ESTRUCTURAS", colors["cyan"], ("Bahnschrift SemiBold", 11), anchor="nw")
        self._text(20, 110, "SELECCIÓN ANATÓMICA", colors["muted"], ("Cascadia Mono", 7), anchor="nw")
        self.canvas.create_line(20, 134, panel_width - 20, 134, fill=colors["line"], tags="hud")
        row_height = min(39.0, max(21.0, (self.height - 205.0) / len(self.controller.catalog)))
        self.part_regions.clear()
        y = 145.0
        for index, part in enumerate(self.controller.catalog, start=1):
            region = Rect(12, y, panel_width - 24, row_height - 4)
            self.part_regions[part.identifier] = region
            selected = snapshot.selected_part == part.identifier
            hovered = self.hovered_part == part.identifier
            if selected or hovered:
                fill = "#08262f" if selected else "#071821"
                self._rectangle(region.x, region.y, region.x + region.width, region.y + region.height,
                                fill, outline=colors["cyan"] if selected else colors["line"])
            dot = colors["cyan"] if selected else "#1f9ec2"
            self.canvas.create_oval(27, y + row_height / 2 - 2, 31, y + row_height / 2 + 2,
                                    fill=dot, outline="", tags="hud")
            label_color = colors["text"] if selected else "#91a5af"
            self._text(42, y + row_height / 2, part.display_name, label_color,
                       ("Cascadia Mono", 8, "bold" if selected else "normal"), anchor="w")
            self._text(panel_width - 24, y + row_height / 2, "›", colors["muted"],
                       ("Bahnschrift", 13), anchor="e")
            y += row_height
        self._text(20, self.height - 30, "APUNTAR + 0.7 s  PARA ABRIR", colors["muted"],
                   ("Cascadia Mono", 7), anchor="sw")

    def _draw_right_panel(self, snapshot: HeartSnapshot, panel_x: float) -> None:
        colors = self.COLORS
        x, width = panel_x + 24, self.width - panel_x - 48
        part = self.controller.part(snapshot.selected_part)
        if part is None:
            self._text(x, 92, "TELEMETRÍA SIMULADA", colors["cyan"],
                       ("Bahnschrift SemiBold", 12), anchor="nw")
            self._text(x, 115, "REFERENCIA EDUCATIVA · NO PACIENTE", colors["muted"],
                       ("Cascadia Mono", 7), anchor="nw")
            self.canvas.create_line(x, 143, x + width, 143, fill=colors["line"], tags="hud")
            self._gauge(x + width * 0.25, 203, 36, "72", "BPM", colors["cyan"])
            self._gauge(x + width * 0.75, 203, 36, "120/80", "mmHg", colors["cyan"])
            self._gauge(x + width * 0.25, 301, 36, "98%", "SpO₂", colors["green"])
            self._text(x + width * 0.75, 287, "SINCRONÍA", colors["muted"],
                       ("Cascadia Mono", 7), anchor="center")
            self._text(x + width * 0.75, 307, "NOMINAL", colors["green"],
                       ("Cascadia Mono", 10, "bold"), anchor="center")
            self.canvas.create_line(x, 354, x + width, 354, fill=colors["line"], tags="hud")
            self._text(x, 380, "ESTADO DE SUBSISTEMAS", colors["cyan"],
                       ("Bahnschrift SemiBold", 10), anchor="nw")
            rows = (("MODELO MULTICAPA", f"{len(self.controller.catalog)} ESTRUCTURAS"),
                    ("MALLA CARDÍACA", "SINCRONIZADA"),
                    ("PRIVACIDAD DE CÁMARA", "SIN VIDEO EN UI"), ("RENDER NATIVO", "ACTIVO"))
            for index, (label, value) in enumerate(rows):
                top = 410 + index * 62
                self._rectangle(x, top, x + width, top + 50, "#040d15", outline=colors["line"])
                self._text(x + 12, top + 14, label, colors["muted"], ("Cascadia Mono", 7), anchor="nw")
                self._text(x + width - 12, top + 31, value, colors["green"],
                           ("Cascadia Mono", 8, "bold"), anchor="e")
            return

        self._text(x, 90, part.display_name.upper(), colors["cyan"],
                   ("Bahnschrift SemiBold", 14), anchor="nw", width=width)
        self._text(x, 119, part.latin_name.upper(), colors["muted"],
                   ("Cascadia Mono", 7), anchor="nw")
        self._text(x, 141, part.category.value, colors["cyan_soft"],
                   ("Cascadia Mono", 7, "bold"), anchor="nw")
        self.canvas.create_line(x, 170, x + width, 170, fill=colors["line"], tags="hud")
        self._text(x, 193, part.summary, colors["text"], ("Segoe UI", 10), anchor="nw", width=width)
        y = 252
        for metric in part.metrics[:3]:
            self._rectangle(x, y, x + width, y + 70, "#040d15", outline=colors["line"])
            self._text(x + 12, y + 12, metric.label, colors["muted"],
                       ("Cascadia Mono", 7), anchor="nw")
            self._text(x + 12, y + 33, metric.value, colors["cyan"],
                       ("Cascadia Mono", 12, "bold"), anchor="nw")
            self._text(x + width - 12, y + 51, metric.reference, colors["green"],
                       ("Cascadia Mono", 6), anchor="e", width=width * 0.55)
            y += 82
        y += 4
        self._text(x, y, "FUNCIÓN", colors["cyan_soft"], ("Cascadia Mono", 7, "bold"), anchor="nw")
        self._text(x, y + 20, part.function, "#97aab3", ("Segoe UI", 9), anchor="nw", width=width)
        self._text(x, y + 82, "RELACIÓN ANATÓMICA", colors["cyan_soft"],
                   ("Cascadia Mono", 7, "bold"), anchor="nw")
        self._text(x, y + 102, part.location, "#97aab3", ("Segoe UI", 9), anchor="nw", width=width)
        self._text(x, self.height - 32, "SELECCIONA EL CORAZÓN LATERAL PARA VOLVER", colors["muted"],
                   ("Cascadia Mono", 6), anchor="sw", width=width)

    def _draw_spatial_data_card(self, part) -> None:
        """Attach a compact information card to the isolated 3D structure."""

        colors = self.COLORS
        width = max(310.0, min(374.0, self.width * 0.27))
        x = self.width - width - 34.0
        y = 112.0
        height = min(438.0, self.height - 178.0)
        focus = self.renderer.focus_bounds
        anchor_x = min(x - 24.0, focus.x + focus.width * 0.88)
        anchor_y = focus.y + focus.height * 0.48
        elbow_x = x - 54.0
        elbow_y = y + 82.0
        self.canvas.create_line(anchor_x, anchor_y, elbow_x, elbow_y, x, elbow_y,
                                fill="#1495a3", width=1, tags="hud")
        self.canvas.create_oval(anchor_x - 4, anchor_y - 4, anchor_x + 4, anchor_y + 4,
                                fill="#03131b", outline=colors["cyan"], width=1, tags="hud")
        self._rectangle(x, y, x + width, y + height, "#031019", outline=colors["cyan_soft"])
        self.canvas.create_line(x, y, x + 34, y, fill=colors["cyan"], width=2, tags="hud")
        self.canvas.create_line(x, y, x, y + 34, fill=colors["cyan"], width=2, tags="hud")
        self._text(x + 20, y + 18, "ESTRUCTURA ANALIZADA", colors["cyan_soft"],
                   ("Cascadia Mono", 7, "bold"), anchor="nw")
        self._text(x + 20, y + 43, part.display_name.upper(), colors["text"],
                   ("Bahnschrift SemiBold", 15), anchor="nw", width=width - 40)
        self._text(x + 20, y + 73, part.latin_name.upper(), colors["muted"],
                   ("Cascadia Mono", 7), anchor="nw")
        self._text(x + 20, y + 94, part.category.value, colors["cyan"],
                   ("Cascadia Mono", 7, "bold"), anchor="nw")
        self.canvas.create_line(x + 20, y + 120, x + width - 20, y + 120,
                                fill=colors["line"], tags="hud")
        self._text(x + 20, y + 138, part.summary, "#b7c8ce", ("Segoe UI", 9),
                   anchor="nw", width=width - 40)
        metric_y = y + 202
        metric_width = (width - 50) / 2
        for index, metric in enumerate(part.metrics[:2]):
            metric_x = x + 20 + index * (metric_width + 10)
            self._rectangle(metric_x, metric_y, metric_x + metric_width, metric_y + 70,
                            "#041722", outline=colors["line"])
            self._text(metric_x + 10, metric_y + 11, metric.label, colors["muted"],
                       ("Cascadia Mono", 6), anchor="nw", width=metric_width - 20)
            self._text(metric_x + 10, metric_y + 34, metric.value, colors["cyan"],
                       ("Cascadia Mono", 10, "bold"), anchor="nw", width=metric_width - 20)
        function_y = metric_y + 92
        self._text(x + 20, function_y, "FUNCIÓN", colors["cyan_soft"],
                   ("Cascadia Mono", 7, "bold"), anchor="nw")
        self._text(x + 20, function_y + 20, part.function, "#9eb1b9", ("Segoe UI", 9),
                   anchor="nw", width=width - 40)
        self._text(x + 20, y + height - 24, "👍  RECONSTRUIR CORAZÓN COMPLETO", colors["muted"],
                   ("Cascadia Mono", 6), anchor="sw")

    def _draw_viewport_frame(self, snapshot: HeartSnapshot) -> None:
        colors = self.COLORS
        v = self.viewport
        title = "CORAZÓN COMPLETO · VISTA ESPACIAL"
        if snapshot.selected_part:
            part = self.controller.part(snapshot.selected_part)
            title = f"ESTRUCTURA AISLADA · {part.display_name.upper() if part else ''}"
        title_x = v.x + (258 if snapshot.spatial_mode else max(286, self.width * 0.205))
        title_y = v.y + (20 if snapshot.spatial_mode else 14)
        self._text(title_x, title_y, title, colors["cyan"],
                   ("Cascadia Mono", 8, "bold"), anchor="nw")
        state = (f"ZOOM {snapshot.zoom:04.2f}×   ROT {math.degrees(snapshot.rotation_y) % 360:05.1f}°   "
                 f"CAPAS {snapshot.explosion:03.0%}   {self.render_fps:02.0f} FPS")
        state_y = v.y + (56 if snapshot.spatial_mode else 14)
        state_x = v.x + v.width - (14 if snapshot.spatial_mode else max(344, self.width * 0.245))
        self._text(state_x, state_y, state, colors["muted"],
                   ("Cascadia Mono", 7), anchor="ne")
        if snapshot.selected_part:
            bounds = self.renderer.whole_bounds
            if snapshot.spatial_mode:
                self._text(bounds.x + bounds.width / 2, bounds.y + bounds.height + 16,
                           "CORAZÓN COMPLETO · 👍 PARA RECONSTRUIR", colors["cyan_soft"],
                           ("Cascadia Mono", 6), anchor="center")
            else:
                self._text(bounds.x + bounds.width / 2, bounds.y + bounds.height + 16,
                           "TOCAR PARA RECONSTRUIR", colors["cyan_soft"],
                           ("Cascadia Mono", 6), anchor="center")

    def _draw_interaction_bar(self, snapshot: HeartSnapshot) -> None:
        colors = self.COLORS
        left = self.width * 0.22
        right = self.width * 0.78
        y = self.height - 80
        self.canvas.create_line(left, y, right, y, fill=colors["line"], tags="hud")
        if snapshot.explosion >= 0.45:
            hints = (("PINZA", "TOMAR PIEZA"), ("MOVER MANO", "TRASLADAR"),
                     ("ACERCAR/ALEJAR", "PROFUNDIDAD"), ("✌ VICTORIA", "RECONSTRUIR"),
                     ("MEÑIQUE", "VISTA ESPACIAL"))
        else:
            hints = (("MANO ABIERTA", "ROTAR"), ("DOS MANOS", "ZOOM"),
                     ("✌ VICTORIA", "SEPARAR CAPAS"), ("3 DEDOS", "REAL + LATIDO"),
                     ("MEÑIQUE", "VISTA ESPACIAL"))
        column_width = (right - left) / len(hints)
        for index, (gesture, action) in enumerate(hints):
            x = left + index * column_width
            if index:
                self.canvas.create_line(x, y + 11, x, y + 55, fill=colors["line"], tags="hud")
            self._text(x + 14, y + 16, gesture, colors["cyan"],
                       ("Cascadia Mono", 7, "bold"), anchor="nw")
            self._text(x + 14, y + 38, action, colors["muted"],
                       ("Cascadia Mono", 6), anchor="nw")
        sensor_state = (f"{self.camera_status[:52]} · VISIÓN {self.vision_fps:02.0f} FPS · "
                        f"{self.hand.hands} MANO{'S' if self.hand.hands != 1 else ''}")
        self._text(right - 10, y + 59, sensor_state, colors["cyan_soft"],
                   ("Cascadia Mono", 6), anchor="se")

    def _draw_hand_overlay(self, now: float) -> None:
        show_hand = bool(self.hand.hands and now - self.hand.captured_at < 0.7)
        show_mouse = bool(not show_hand and now < self.mouse_visible_until)
        if not show_hand and not show_mouse and not self._hand_overlay_visible:
            return
        self.canvas.delete("hands")
        self._hand_overlay_visible = show_hand or show_mouse
        colors = self.COLORS
        if show_hand:
            for hand_index in range(self.hand.hands):
                points = self.hand.landmarks[hand_index * 21:(hand_index + 1) * 21]
                if len(points) < 21:
                    continue
                projected = [(x * self.width, y * self.height) for x, y in points]
                for start, end in self.HAND_CONNECTIONS:
                    self.canvas.create_line(*projected[start], *projected[end], fill="#147989",
                                            width=1, tags="hands")
                for x, y in projected:
                    self.canvas.create_oval(x - 1.6, y - 1.6, x + 1.6, y + 1.6,
                                            fill=colors["cyan_soft"], outline="", tags="hands")
            px, py = self.hand.pointer_x * self.width, self.hand.pointer_y * self.height
            grabbed = self.controller.snapshot().grabbed_fragment
            pointer_color = colors["green"] if grabbed is not None else colors["cyan"]
            self.canvas.create_oval(px - 11, py - 11, px + 11, py + 11,
                                    outline=pointer_color, width=2, tags="hands")
            label = ("PINZA · " + grabbed.value.upper()) if grabbed else self.hand.gesture.value.upper()
            self.canvas.create_text(px + 16, py - 14, text=label,
                                    fill=pointer_color, font=("Cascadia Mono", 6),
                                    anchor="nw", tags="hands")
        elif show_mouse:
            x, y = self.mouse
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7,
                                    outline=colors["cyan_soft"], width=1, tags="hands")
            self.canvas.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5,
                                    fill=colors["cyan"], outline="", tags="hands")

    def _activate_at(self, x: float, y: float) -> None:
        snapshot = self.controller.snapshot()
        for identifier, region in self.part_regions.items():
            if region.contains(x, y):
                self.controller.select_part(identifier)
                self._flash("ESTRUCTURA AISLADA · " + self.controller.part(identifier).display_name.upper())
                return
        if snapshot.selected_part and self.renderer.is_whole_heart_at(x, y):
            self.controller.focus_whole_heart()
            self._flash("CORAZÓN COMPLETO RESTAURADO")
            return
        identifier = self.renderer.part_at(x, y)
        if identifier:
            self.controller.select_part(identifier)
            self._flash("ESTRUCTURA AISLADA · " + self.controller.part(identifier).display_name.upper())

    def _on_mouse_move(self, event) -> None:
        self.mouse = (event.x, event.y)
        self.mouse_visible_until = time.perf_counter() + 2.0
        self._hand_overlay_dirty = True
        hovered_part = next((part for part, region in self.part_regions.items()
                             if region.contains(event.x, event.y)), None)
        if hovered_part != self.hovered_part:
            self.hovered_part = hovered_part
            self._hud_dirty = True

    def _on_mouse_down(self, event) -> None:
        self.drag_origin = self.drag_last = (event.x, event.y)
        self.drag_distance = 0.0
        self.mouse_fragment_drag = False
        snapshot = self.controller.snapshot()
        if snapshot.explosion >= 0.45 and snapshot.selected_part is None:
            identifier = self.renderer.part_at(event.x, event.y)
            if identifier is not None and self.controller.begin_fragment_drag(identifier):
                self.mouse_fragment_drag = True
                part = self.controller.part(identifier)
                self._flash("FRAGMENTO TOMADO · " + (part.display_name.upper() if part else ""))

    def _on_mouse_drag(self, event) -> None:
        if self.drag_last is None:
            return
        dx, dy = event.x - self.drag_last[0], event.y - self.drag_last[1]
        self.drag_distance += abs(dx) + abs(dy)
        self.drag_last = (event.x, event.y)
        if self.mouse_fragment_drag:
            self.controller.move_grabbed_fragment(
                dx / max(1.0, self.viewport.width),
                dy / max(1.0, self.viewport.height),
                0.0,
            )
        else:
            self.controller.rotate(dx * 0.009, dy * 0.009)
        self._hud_dirty = True

    def _on_mouse_up(self, event) -> None:
        was_fragment_drag = self.mouse_fragment_drag
        self.controller.release_fragment()
        if self.drag_distance < 9.0:
            self._activate_at(event.x, event.y)
        self.drag_origin = self.drag_last = None
        self.mouse_fragment_drag = False
        if was_fragment_drag and self.drag_distance >= 9.0:
            self._flash("FRAGMENTO LIBERADO")

    def _on_mouse_wheel(self, event) -> None:
        if self.controller.snapshot().grabbed_fragment is not None:
            self.controller.move_grabbed_fragment(0.0, 0.0, 0.04 if event.delta > 0 else -0.04)
        else:
            self.controller.zoom(1.10 if event.delta > 0 else 0.91)
        self._hud_dirty = True

    def _on_resize(self, event) -> None:
        if event.widget is self.root:
            self.width, self.height = max(900, event.width), max(620, event.height)
            self._hud_dirty = True

    def _toggle_fullscreen(self, _event=None) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def _toggle_explosion(self) -> None:
        self.controller.toggle_explosion()
        self._flash("CAPAS ANATÓMICAS RECONFIGURADAS")

    def _toggle_realism(self) -> None:
        self.controller.toggle_realistic_heartbeat()
        state = self.controller.snapshot()
        self._flash("TEJIDO REALISTA · LATIDO SINCRONIZADO" if state.realistic else
                    "MATRIZ ANATÓMICA · LATIDO EN PAUSA")

    def _toggle_spatial_mode(self) -> None:
        self.controller.toggle_spatial_mode()
        state = self.controller.snapshot()
        self._flash("VISTA ESPACIAL · ESCENARIO COMPLETO" if state.spatial_mode else
                    "ESTACIÓN ANATÓMICA · PANELES ACTIVOS")

    def _reset_view(self) -> None:
        self.controller.reset_view()
        self._flash("VISTA ANATÓMICA RESTAURADA")

    def _flash(self, message: str) -> None:
        self.status_flash = message
        self.status_flash_until = time.monotonic() + 2.4
        self._hud_dirty = True

    def _rectangle(self, x1: float, y1: float, x2: float, y2: float, fill: str,
                   outline: str = "") -> None:
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, tags="hud")

    def _text(self, x: float, y: float, text: str, fill: str, font: tuple,
              anchor: str = "w", width: float | None = None) -> None:
        options = {"anchor": anchor, "fill": fill, "font": font, "text": text, "tags": "hud"}
        if width is not None:
            options["width"] = max(20, int(width))
        self.canvas.create_text(x, y, **options)

    def _status_indicator(self, x: float, y: float, color: str, label: str) -> None:
        self.canvas.create_oval(x, y, x + 5, y + 5, fill=color, outline="", tags="hud")
        self._text(x + 10, y + 2, label, self.COLORS["muted"],
                   ("Cascadia Mono", 6), anchor="w")

    def _gauge(self, x: float, y: float, radius: float, value: str, unit: str, color: str) -> None:
        self.canvas.create_arc(x - radius, y - radius, x + radius, y + radius,
                               start=90, extent=-300, style="arc", outline=color,
                               width=3, tags="hud")
        self.canvas.create_arc(x - radius, y - radius, x + radius, y + radius,
                               start=150, extent=-55, style="arc", outline="#16364a",
                               width=3, tags="hud")
        self._text(x, y - 4, value, self.COLORS["text"],
                   ("Cascadia Mono", 11, "bold"), anchor="center")
        self._text(x, y + 15, unit, self.COLORS["muted"],
                   ("Cascadia Mono", 6), anchor="center")
