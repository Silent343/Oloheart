"""Qt desktop composition for the hardware-accelerated OloHeart experience."""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from oloheart.application.controller import (
    GestureCoordinator,
    HandTracker,
    HeartController,
    InteractionAction,
)
from oloheart.domain.model import (
    AnatomySystem,
    HandObservation,
    HeartPart,
    HeartPartId,
    HeartSnapshot,
)
from oloheart.infrastructure.geometry import build_heart_geometry
from oloheart.presentation.gpu_renderer import GpuHeartWidget
from oloheart.presentation.scene_layout import object_drag


STYLE_SHEET = """
QMainWindow, QWidget#root {
    background: #01050a;
    color: #dbe8ed;
    font-family: "Segoe UI";
}
QWidget#inspectorContent { background: #020910; }
QFrame#header, QFrame#footer {
    background: #030b13;
    border: 1px solid #132c3c;
}
QFrame#panel {
    background: #020910;
    border: 1px solid #142f41;
    border-radius: 10px;
}
QLabel#brand {
    color: #19dbea;
    font-family: "Bahnschrift SemiBold";
    font-size: 20px;
    font-weight: 700;
}
QLabel#eyebrow {
    color: #6d8794;
    font-family: "Cascadia Mono";
    font-size: 10px;
    letter-spacing: 1px;
}
QLabel#section {
    color: #1de0ef;
    font-family: "Bahnschrift SemiBold";
    font-size: 13px;
    font-weight: 700;
}
QLabel#partTitle {
    color: #e8f1f4;
    font-family: "Bahnschrift SemiBold";
    font-size: 20px;
    font-weight: 700;
}
QLabel#latin {
    color: #22d3ee;
    font-style: italic;
    font-size: 12px;
}
QLabel#body {
    color: #aebfc7;
    font-size: 12px;
    line-height: 1.35;
}
QLabel#metricValue {
    color: #22d3ee;
    font-family: "Cascadia Mono";
    font-size: 17px;
    font-weight: 700;
}
QListWidget {
    background: transparent;
    border: none;
    outline: none;
    color: #8fa4af;
    font-family: "Cascadia Mono";
    font-size: 11px;
}
QListWidget::item {
    min-height: 29px;
    padding: 3px 9px;
    border: 1px solid transparent;
    border-radius: 6px;
}
QListWidget::item:hover {
    background: #061923;
    color: #dcecf0;
}
QListWidget::item:selected {
    background: #082631;
    border-color: #15cfe0;
    color: #ffffff;
}
QCheckBox {
    color: #a9bdc6;
    font-family: "Cascadia Mono";
    font-size: 10px;
    spacing: 7px;
    padding: 1px 0;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #315164;
    border-radius: 3px;
    background: #041019;
}
QCheckBox::indicator:checked {
    border-color: #22d3ee;
    background: #18bed0;
}
QPushButton {
    background: #07131d;
    color: #a9bdc6;
    border: 1px solid #18394c;
    border-radius: 7px;
    padding: 8px 12px;
    font-family: "Cascadia Mono";
    font-size: 10px;
    font-weight: 600;
}
QPushButton:hover {
    color: #ffffff;
    border-color: #22d3ee;
    background: #09202b;
}
QPushButton:checked {
    color: #ffffff;
    border-color: #ef5b65;
    background: #32151c;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #030b13;
    width: 7px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #1b4758;
    min-height: 30px;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class QtDesktopApplication(QMainWindow):
    """Native workstation shell that coordinates Qt, OpenGL, and hand input."""

    def __init__(
        self,
        controller: HeartController,
        tracker: HandTracker,
        fullscreen: bool = True,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.tracker = tracker
        self.coordinator = GestureCoordinator(controller)
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.hand = HandObservation.empty(time.monotonic())
        self.camera_status = "INICIANDO SEGUIMIENTO ESPACIAL"
        self.last_frame = time.perf_counter()
        self.status_message = "SISTEMA ANATÓMICO GPU LISTO"
        self.status_until = time.monotonic() + 3.0
        self._last_ui_signature: tuple | None = None
        self._last_status_update = 0.0
        self._fullscreen = fullscreen

        self.setObjectName("root")
        self.setWindowTitle("OloHeart · Anatomía cardíaca espacial")
        self.setMinimumSize(1120, 700)
        self.resize(1500, 900)
        self.setStyleSheet(STYLE_SHEET)
        self._build_interface()
        self._connect_renderer()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        if fullscreen:
            self.showFullScreen()

    def start(self) -> None:
        self.tracker.start(self.post_observation, self.post_camera_status)
        self.timer.start()

    def post_observation(self, observation: HandObservation) -> None:
        self.events.put(("hand", observation))

    def post_camera_status(self, status: str) -> None:
        self.events.put(("status", status))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.timer.stop()
        self.tracker.stop()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_E:
            self._toggle_explosion()
        elif key in (Qt.Key.Key_R, Qt.Key.Key_Space):
            self._toggle_realism()
        elif key == Qt.Key.Key_S:
            self._toggle_spatial_mode()
        elif key == Qt.Key.Key_I:
            self._toggle_interior()
        elif key == Qt.Key.Key_Home:
            self._reset_view()
        else:
            super().keyPressEvent(event)

    def _build_interface(self) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(5, 5, 5, 5)
        root_layout.setSpacing(4)
        self.setCentralWidget(root)

        self.header = QFrame(root)
        self.header.setObjectName("header")
        self.header.setFixedHeight(48)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(18, 7, 18, 7)
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        brand = QLabel("OLOHEART")
        brand.setObjectName("brand")
        subtitle = QLabel("ATLAS CARDÍACO INTERACTIVO")
        subtitle.setObjectName("eyebrow")
        brand_box.addWidget(brand)
        brand_box.addWidget(subtitle)
        header_layout.addLayout(brand_box)
        header_layout.addStretch(1)
        self.mode_label = QLabel("TEJIDO ANATÓMICO · ILUMINACIÓN PBR")
        self.mode_label.setObjectName("section")
        header_layout.addWidget(self.mode_label)
        header_layout.addStretch(1)
        self.camera_label = QLabel(self.camera_status)
        self.camera_label.setObjectName("eyebrow")
        header_layout.addWidget(self.camera_label)
        root_layout.addWidget(self.header)

        self.body = QWidget(root)
        body_layout = QHBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        root_layout.addWidget(self.body, 1)

        self.left_panel = self._build_left_panel(self.body)
        body_layout.addWidget(self.left_panel)

        center = QWidget(self.body)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(7)
        self.renderer = GpuHeartWidget(build_heart_geometry(), center)
        self.renderer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.renderer, 1)
        self.control_bar = self._build_control_bar(center)
        center_layout.addWidget(self.control_bar)
        body_layout.addWidget(center, 1)

        # The inspector floats over the continuous canvas instead of reserving
        # a second sidebar. Its contents remain available in both viewing modes.
        self.right_panel = self._build_right_panel(self.renderer)
        self.renderer.set_inspector(self.right_panel)

        self.footer = QFrame(root)
        self.footer.setObjectName("footer")
        self.footer.setFixedHeight(28)
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(16, 5, 16, 5)
        self.status_label = QLabel(self.status_message)
        self.status_label.setObjectName("eyebrow")
        self.fps_label = QLabel("GPU -- FPS · VISIÓN -- FPS")
        self.fps_label.setObjectName("eyebrow")
        footer_layout.addWidget(self.status_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.fps_label)
        root_layout.addWidget(self.footer)

    def _build_left_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("panel")
        panel.setFixedWidth(232)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(13, 15, 13, 13)
        layout.setSpacing(7)
        title = QLabel("ESTRUCTURAS CARDÍACAS")
        title.setObjectName("section")
        caption = QLabel("EXPLORA SUS ESTRUCTURAS")
        caption.setObjectName("eyebrow")
        layout.addWidget(title)
        layout.addWidget(caption)
        systems_title = QLabel("AISLAR SISTEMAS")
        systems_title.setObjectName("eyebrow")
        layout.addSpacing(4)
        layout.addWidget(systems_title)
        visible_systems = self.controller.snapshot().visible_systems
        self.system_checks: dict[AnatomySystem, QCheckBox] = {}
        for system in AnatomySystem:
            checkbox = QCheckBox(system.value, panel)
            checkbox.setChecked(system in visible_systems)
            checkbox.toggled.connect(
                lambda checked, selected_system=system: self._set_system_visibility(
                    selected_system, checked
                )
            )
            self.system_checks[system] = checkbox
            layout.addWidget(checkbox)
        structures_title = QLabel("PARTES SELECCIONABLES")
        structures_title.setObjectName("eyebrow")
        layout.addSpacing(4)
        layout.addWidget(structures_title)
        self.part_list = QListWidget(panel)
        self.part_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.part_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.part_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.part_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        for part in self.controller.catalog:
            item = QListWidgetItem(f"  •  {part.display_name}")
            item.setData(Qt.ItemDataRole.UserRole, part.identifier.value)
            item.setToolTip(part.display_name)
            self.part_list.addItem(item)
        self.part_list.itemClicked.connect(self._select_list_item)
        layout.addWidget(self.part_list, 1)
        help_label = QLabel(
            "MANO ABIERTA  ROTAR\nDOS MANOS  ZOOM\nÍNDICE  SELECCIONAR\nPINZA  MOVER FRAGMENTO\nPULGAR ABAJO  VISTA INTERIOR"
        )
        help_label.setObjectName("eyebrow")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        return panel

    def _build_control_bar(self, parent: QWidget) -> QFrame:
        bar = QFrame(parent)
        bar.setObjectName("panel")
        bar.setFixedHeight(42)
        layout = QGridLayout(bar)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(4)
        self.explosion_button = QPushButton("SEPARAR · E")
        self.realism_button = QPushButton("LATIDO · R")
        self.spatial_button = QPushButton("ESPACIAL · S")
        self.interior_button = QPushButton("VISTA INTERIOR · I")
        self.interior_button.setCheckable(True)
        self.interior_button.clicked.connect(self._toggle_interior)
        reset_button = QPushButton("RECONSTRUIR")
        clear_button = QPushButton("QUITAR SELECCIÓN")
        clear_button.clicked.connect(lambda: self._activate_part(None))
        self.explosion_button.setCheckable(True)
        self.realism_button.setCheckable(True)
        self.spatial_button.setCheckable(True)
        self.explosion_button.clicked.connect(self._toggle_explosion)
        self.realism_button.clicked.connect(self._toggle_realism)
        self.spatial_button.clicked.connect(self._toggle_spatial_mode)
        reset_button.clicked.connect(self._reset_view)
        for index, button in enumerate((self.interior_button, self.explosion_button,
                                        self.realism_button, self.spatial_button,
                                        clear_button, reset_button)):
            layout.addWidget(button, 0, index)
        return bar

    def _build_right_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(panel)
        self.info_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        content.setObjectName("inspectorContent")
        self.info_layout = QVBoxLayout(content)
        self.info_layout.setContentsMargins(18, 18, 18, 18)
        self.info_layout.setSpacing(9)
        self.part_title = QLabel("CORAZÓN HUMANO")
        self.part_title.setObjectName("partTitle")
        self.part_title.setWordWrap(True)
        self.part_latin = QLabel("Cor")
        self.part_latin.setObjectName("latin")
        self.part_latin.setWordWrap(True)
        self.part_category = QLabel("MODELO ANATÓMICO INTERACTIVO")
        self.part_category.setObjectName("eyebrow")
        self.part_category.setWordWrap(True)
        self.part_summary = QLabel(
            "Selecciona una estructura con la mano o desde el navegador izquierdo. "
            "La cámara procesa únicamente puntos de la mano; el video no se muestra."
        )
        self.part_summary.setObjectName("body")
        self.part_summary.setWordWrap(True)
        self.info_layout.addWidget(self.part_title)
        self.info_layout.addWidget(self.part_latin)
        self.info_layout.addWidget(self.part_category)
        self.info_layout.addSpacing(8)
        self.info_layout.addWidget(self.part_summary)
        self.metric_labels: list[tuple[QLabel, QLabel, QLabel]] = []
        for _ in range(3):
            metric_frame = QFrame(content)
            metric_frame.setObjectName("panel")
            metric_layout = QVBoxLayout(metric_frame)
            metric_layout.setContentsMargins(11, 8, 11, 8)
            metric_layout.setSpacing(2)
            label = QLabel("")
            label.setObjectName("eyebrow")
            label.setWordWrap(True)
            value = QLabel("")
            value.setWordWrap(True)
            value.setObjectName("metricValue")
            reference = QLabel("")
            reference.setObjectName("eyebrow")
            reference.setWordWrap(True)
            metric_layout.addWidget(label)
            metric_layout.addWidget(value)
            metric_layout.addWidget(reference)
            self.info_layout.addWidget(metric_frame)
            self.metric_labels.append((label, value, reference))
        self.function_title = QLabel("FUNCIÓN FISIOLÓGICA")
        self.function_title.setObjectName("section")
        self.function_text = QLabel("Bombeo coordinado de la circulación pulmonar y sistémica.")
        self.function_text.setObjectName("body")
        self.function_text.setWordWrap(True)
        self.location_title = QLabel("RELACIONES ANATÓMICAS")
        self.location_title.setObjectName("section")
        self.location_text = QLabel("Mediastino medio, dentro del saco pericárdico.")
        self.location_text.setObjectName("body")
        self.location_text.setWordWrap(True)
        self.info_layout.addSpacing(8)
        self.info_layout.addWidget(self.function_title)
        self.info_layout.addWidget(self.function_text)
        self.info_layout.addSpacing(6)
        self.info_layout.addWidget(self.location_title)
        self.info_layout.addWidget(self.location_text)
        self.info_layout.addItem(
            QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        scroll.setWidget(content)
        outer.addWidget(scroll)
        # Pointing dwell can scroll the same full description as the mouse wheel.
        navigation = QHBoxLayout()
        navigation.setContentsMargins(8, 4, 8, 8)
        for caption, direction in (("↑ TEXTO", -1), ("TEXTO ↓", 1)):
            button = QPushButton(caption, panel)
            button.setAccessibleName("Desplazar información " + ("arriba" if direction < 0 else "abajo"))
            button.clicked.connect(lambda checked=False, step=direction: scroll.verticalScrollBar().setValue(
                scroll.verticalScrollBar().value() + step * 160))
            navigation.addWidget(button)
        outer.addLayout(navigation)
        return panel

    def _connect_renderer(self) -> None:
        self.renderer.part_activated.connect(self._activate_part)
        self.renderer.fragment_pressed.connect(self._begin_fragment)
        self.renderer.fragment_moved.connect(self._move_fragment)
        self.renderer.fragment_released.connect(self._release_fragment)
        self.renderer.rotation_requested.connect(self.controller.rotate)
        self.renderer.zoom_requested.connect(self.controller.zoom)

    def _tick(self) -> None:
        now = time.perf_counter()
        delta = min(0.08, now - self.last_frame)
        self.last_frame = now
        self._pump_events()
        snapshot = self.controller.tick(delta)
        self.renderer.set_snapshot(snapshot)
        origin = self.renderer.mapTo(self, QPoint(0, 0))
        self.renderer.set_hand_overlay(
            self.hand,
            float(self.width()),
            float(self.height()),
            float(origin.x()),
            float(origin.y()),
        )
        signature = (
            snapshot.selected_part,
            snapshot.realistic,
            snapshot.spatial_mode,
            snapshot.interior_view,
            snapshot.explosion >= 0.45,
            tuple(sorted(system.value for system in snapshot.visible_systems)),
        )
        if signature != self._last_ui_signature:
            self._last_ui_signature = signature
            self._update_interface(snapshot)
        if now - self._last_status_update >= 0.25:
            self._last_status_update = now
            message = self.status_message if now < self.status_until else self.camera_status
            self.status_label.setText(message)
            self.camera_label.setText(self.camera_status)
            if snapshot.realistic:
                self.mode_label.setText(
                    f"{snapshot.cycle.phase.value.upper()} · {snapshot.bpm} LPM"
                )
                if snapshot.selected_part is None:
                    self._show_cycle_status(snapshot)
            self.fps_label.setText(
                f"GPU {self.renderer.render_fps:02.0f} FPS  ·  CÁMARA LOCAL  ·  SIN VIDEO EN PANTALLA"
            )

    def _pump_events(self) -> None:
        latest: HandObservation | None = None
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "hand":
                latest = payload  # type: ignore[assignment]
            elif kind == "status":
                self.camera_status = str(payload)
        if latest is None:
            return
        self.hand = latest
        result = self.coordinator.process(latest)
        if result.action == InteractionAction.ACTIVATE_POINTER:
            self._activate_from_normalized(result.pointer_x, result.pointer_y)
        elif result.action == InteractionAction.FRAGMENT_GRAB_STARTED:
            part = self._pick_from_normalized(result.pointer_x, result.pointer_y)
            if part is not None:
                self._begin_fragment(part)
        elif result.action == InteractionAction.FRAGMENT_GRAB_MOVED:
            self._move_fragment(result.delta_x, result.delta_y, result.delta_depth)
        elif result.action == InteractionAction.FRAGMENT_GRAB_ENDED:
            self._release_fragment()
        elif result.action == InteractionAction.EXPLOSION_CHANGED:
            self._flash("CAPAS ANATÓMICAS SEPARADAS")
        elif result.action == InteractionAction.REALISM_CHANGED:
            self._flash("MATERIAL BIOLÓGICO · LATIDO SINCRONIZADO")
        elif result.action == InteractionAction.SPATIAL_MODE_CHANGED:
            self._flash("VISTA ESPACIAL A PANTALLA COMPLETA")
        elif result.action == InteractionAction.INTERIOR_VIEW_CHANGED:
            self._flash("CORTE ANATÓMICO" if self.controller.snapshot().interior_view else "SUPERFICIE EXTERIOR")
        elif result.action == InteractionAction.FOCUS_WHOLE:
            self._flash("CORAZÓN COMPLETO RESTAURADO")

    def _pick_from_normalized(self, x: float, y: float) -> HeartPartId | None:
        origin = self.renderer.mapTo(self, QPoint(0, 0))
        local_x = x * self.width() - origin.x()
        local_y = y * self.height() - origin.y()
        if not (0 <= local_x < self.renderer.width() and 0 <= local_y < self.renderer.height()):
            return None
        return self.renderer.pick_part(local_x, local_y)

    def _activate_from_normalized(self, x: float, y: float) -> None:
        point = QPoint(round(x * self.width()), round(y * self.height()))
        widget = self.childAt(point)
        # The same pointing dwell can operate system checkboxes and the toolbar.
        while widget is not None and widget is not self:
            if isinstance(widget, (QCheckBox, QPushButton)):
                widget.click()
                return
            if widget is self.part_list.viewport():
                item = self.part_list.itemAt(widget.mapFrom(self, point))
                if item is not None:
                    self._select_list_item(item)
                return
            if widget is self.right_panel:
                # Reading the floating card must not select anatomy behind it.
                return
            widget = widget.parentWidget()
        self._activate_part(self._pick_from_normalized(x, y))

    def _activate_part(self, identifier: HeartPartId | None) -> None:
        if identifier is None:
            if self.controller.snapshot().selected_part is not None:
                self.controller.focus_whole_heart()
                self._flash("CORAZÓN COMPLETO RESTAURADO")
            return
        self.controller.select_part(identifier)
        part = self.controller.part(identifier)
        self._flash("ESTRUCTURA SELECCIONADA · " + (part.display_name.upper() if part else ""))

    def _begin_fragment(self, identifier: HeartPartId) -> None:
        if self.controller.begin_fragment_drag(identifier):
            part = self.controller.part(identifier)
            self._flash("FRAGMENTO TOMADO · " + (part.display_name.upper() if part else ""))

    def _move_fragment(self, horizontal: float, vertical: float, depth: float) -> None:
        self.controller.move_grabbed_fragment(
            *object_drag(self.controller.snapshot(), horizontal, vertical, depth)
        )

    def _release_fragment(self) -> None:
        grabbed = self.controller.snapshot().grabbed_fragment
        self.controller.release_fragment()
        if grabbed is not None:
            self._flash("FRAGMENTO LIBERADO")

    def _select_list_item(self, item: QListWidgetItem) -> None:
        self._activate_part(HeartPartId(item.data(Qt.ItemDataRole.UserRole)))

    def _update_interface(self, snapshot: HeartSnapshot) -> None:
        self.explosion_button.setChecked(snapshot.explosion >= 0.45)
        self.realism_button.setChecked(snapshot.realistic)
        self.spatial_button.setChecked(snapshot.spatial_mode)
        self.interior_button.setChecked(snapshot.interior_view)
        self.mode_label.setText(
            "TEJIDO REALISTA · LATIDO ACTIVO" if snapshot.realistic
            else ("CORTE ANATÓMICO" if snapshot.interior_view else "ANATOMÍA EXTERIOR")
        )
        self.left_panel.setVisible(not snapshot.spatial_mode)
        self.header.setVisible(not snapshot.spatial_mode)
        self.footer.setVisible(not snapshot.spatial_mode)
        self.control_bar.setVisible(not snapshot.spatial_mode)
        part = self.controller.part(snapshot.selected_part)
        self._show_part(part)
        self.renderer.set_callout(part)
        self.right_panel.setVisible(part is not None or not snapshot.spatial_mode)
        for system, checkbox in self.system_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(system in snapshot.visible_systems)
            checkbox.blockSignals(False)
        if part is None:
            self.part_list.clearSelection()
        else:
            for index in range(self.part_list.count()):
                item = self.part_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == part.identifier.value:
                    self.part_list.setCurrentItem(item)
                    self.part_list.scrollToItem(item)
                    break

    def _show_part(self, part: HeartPart | None) -> None:
        identifier = part.identifier if part else None
        if getattr(self, "_inspected_part", None) != identifier:
            self.info_scroll.verticalScrollBar().setValue(0)
        self._inspected_part = identifier
        if part is None:
            self.part_title.setText("CORAZÓN HUMANO")
            self.part_latin.setText("Cor")
            self.part_category.setText("EXPLORACIÓN ANATÓMICA")
            self.part_summary.setText(
                "Selecciona una estructura con la mano o desde el navegador izquierdo. "
                "Pulgar abajo o I abre el interior. La selección permanece en su posición; "
                "todas las estructuras giran juntas."
            )
            metrics = (
                ("RITMO CARDÍACO", "72 BPM", "Referencia educativa"),
                ("PRESIÓN ARTERIAL", "120/80", "mmHg"),
                ("OXIGENACIÓN", "98%", "SpO₂ simulada"),
            )
            function = "Bombeo coordinado de la circulación pulmonar y sistémica."
            location = "Mediastino medio, dentro del saco pericárdico."
        else:
            self.part_title.setText(part.display_name.upper())
            self.part_latin.setText(part.latin_name)
            self.part_category.setText(part.category.value)
            self.part_summary.setText(part.summary)
            metrics = tuple((metric.label, metric.value, metric.reference) for metric in part.metrics)
            function = part.function
            location = part.location
        for index, labels in enumerate(self.metric_labels):
            frame = labels[0].parentWidget()
            if index < len(metrics):
                label, value, reference = metrics[index]
                labels[0].setText(label)
                labels[1].setText(value)
                labels[2].setText(reference)
                frame.show()
            else:
                frame.hide()
        self.function_text.setText(function)
        self.location_text.setText(location)

    def _show_cycle_status(self, snapshot: HeartSnapshot) -> None:
        """Present the simulated pressure-driven phase without implying patient data."""

        cycle = snapshot.cycle
        metrics = (
            ("FASE DEL CICLO", cycle.phase.value.upper(), f"Ritmo simulado · {snapshot.bpm} BPM"),
            (
                "VÁLVULAS AURICULOVENTRICULARES",
                "ABIERTAS" if cycle.atrioventricular_valves_open else "CERRADAS",
                "Mitral y tricúspide",
            ),
            (
                "VÁLVULAS SEMILUNARES",
                "ABIERTAS" if cycle.semilunar_valves_open else "CERRADAS",
                "Perfusión arterial coronaria: " + ("predominio diastólico" if cycle.coronary_perfusion > 0.8 else "reducida durante sístole / transición"),
            ),
        )
        for labels, metric in zip(self.metric_labels, metrics, strict=True):
            labels[0].setText(metric[0])
            labels[1].setText(metric[1])
            labels[2].setText(metric[2])
            labels[0].parentWidget().show()
        self.function_text.setText(
            "Cuerpo → venas cavas → corazón derecho → pulmones → venas pulmonares → "
            "corazón izquierdo → aorta → cuerpo."
        )
        self.location_text.setText(
            "Secuencia educativa: aurículas, retraso auriculoventricular, ventrículos, "
            "eyección y llenado. No representa telemetría de un paciente."
        )

    def _set_system_visibility(self, system: AnatomySystem, checked: bool) -> None:
        current = system in self.controller.snapshot().visible_systems
        if checked != current:
            self.controller.toggle_system(system)
            self._flash(f"{system.value.upper()} · {'VISIBLE' if checked else 'OCULTO'}")

    def _toggle_explosion(self) -> None:
        self.controller.toggle_explosion()
        self._flash("CAPAS ANATÓMICAS RECONFIGURADAS")

    def _toggle_interior(self) -> None:
        self.controller.toggle_interior_view()
        self._flash("CORTE ANATÓMICO" if self.controller.snapshot().interior_view else "SUPERFICIE EXTERIOR")

    def _toggle_realism(self) -> None:
        self.controller.toggle_realistic_heartbeat()
        state = self.controller.snapshot()
        self._flash(
            "MATERIAL BIOLÓGICO · LATIDO SINCRONIZADO" if state.realistic
            else "ANATOMÍA ESTRUCTURAL · LATIDO EN PAUSA"
        )

    def _toggle_spatial_mode(self) -> None:
        self.controller.toggle_spatial_mode()
        state = self.controller.snapshot()
        self._flash(
            "VISTA ESPACIAL A PANTALLA COMPLETA" if state.spatial_mode
            else "ESTACIÓN ANATÓMICA RESTAURADA"
        )

    def _reset_view(self) -> None:
        self.controller.reset_view()
        self._flash("VISTA ANATÓMICA RESTAURADA")

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.showFullScreen() if self._fullscreen else self.showNormal()

    def _flash(self, message: str) -> None:
        self.status_message = message
        self.status_until = time.monotonic() + 2.4
        self.status_label.setText(message)


def configure_opengl() -> None:
    """Request one multisampled core context before QApplication is created."""

    from PySide6.QtGui import QSurfaceFormat

    surface = QSurfaceFormat()
    surface.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setStencilBufferSize(8)
    surface.setSamples(2)
    surface.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface)


def create_qt_application() -> QApplication:
    """Create the one native Qt process after the OpenGL format is fixed."""

    configure_opengl()
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("OloHeart")
    application.setOrganizationName("OloHeart")
    return application
