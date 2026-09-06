"""Desktop acceptance checks for the expanded canvas and native anatomy card.

Run explicitly with the project virtual environment. No camera is started.
Screenshots and observed frame rates are saved under artifacts/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QPushButton
from oloheart.application.controller import HeartController, NullHandTracker
from oloheart.domain.model import HeartPartId
from oloheart.presentation.qt_app import QtDesktopApplication, create_qt_application

app = create_qt_application()
controller = HeartController()
controller.toggle_realistic_heartbeat()
window = QtDesktopApplication(controller, NullHandTracker(), fullscreen=False)
window.resize(1500, 900)
window.show()
window.start()
output = Path(__file__).resolve().parents[1] / "artifacts"
output.mkdir(exist_ok=True)
records = []
previous_anchor = None


def capture(name):
    assert window.renderer.isValid() and window.renderer._program
    assert window.grab().save(str(output / f"callout-{name}.png"))
    records.append({"view": name, "fps": window.renderer.render_fps,
                    "canvas": [window.renderer.width(), window.renderer.height()],
                    "card_visible": window.right_panel.isVisible()})


def select():
    assert window.body.layout().count() == 2, "Inspector must not reserve a sidebar"
    assert window.renderer.width() > window.width() * 0.80
    assert window.renderer.height() > window.height() * 0.80
    controller.select_part(HeartPartId.MITRAL_VALVE)


def rotate():
    global previous_anchor
    assert window.renderer._callout.identifier == HeartPartId.MITRAL_VALVE
    assert window.right_panel.isVisible()
    assert window.part_title.text() == controller.part(HeartPartId.MITRAL_VALVE).display_name.upper()
    previous_anchor = window.renderer._selected_anchor()
    assert previous_anchor is not None
    assert previous_anchor.x() < window.right_panel.x()
    capture("selected")
    controller.rotate(0.5, 0.12)


def explode():
    anchor = window.renderer._selected_anchor()
    assert (anchor - previous_anchor).manhattanLength() > 4, "Leader must follow rotation"
    capture("rotated")
    controller.toggle_explosion()
    controller.select_part(HeartPartId.AORTA)


def spatial():
    capture("exploded")
    controller.toggle_spatial_mode()


def clear():
    assert window.renderer.width() > window.width() * 0.98
    assert window.right_panel.isVisible()
    assert window.renderer._callout.identifier == HeartPartId.AORTA
    # A pointing dwell on card text must not activate hidden geometry or clear it.
    point = window.right_panel.mapTo(window, QPoint(30, 45))
    x, y = point.x() / window.width(), point.y() / window.height()
    window._activate_from_normalized(x, y)
    assert controller.snapshot().selected_part == HeartPartId.AORTA
    assert window._pick_from_normalized(x, y) is None
    # The same hand activation route also reaches the card's text navigation.
    down = next(b for b in window.right_panel.findChildren(QPushButton) if b.text() == "TEXTO ↓")
    point = down.mapTo(window, down.rect().center())
    scrollbar = window.info_scroll.verticalScrollBar()
    before = scrollbar.value()
    window._activate_from_normalized(point.x() / window.width(), point.y() / window.height())
    assert scrollbar.value() == min(scrollbar.maximum(), before + 160)
    capture("spatial")
    controller.focus_whole_heart()


def resize():
    assert not window.right_panel.isVisible()
    assert window.renderer._callout is None
    capture("unobstructed")
    controller.toggle_spatial_mode()
    controller.select_part(HeartPartId.LEFT_VENTRICLE)
    window.resize(1120, 700)


def finish():
    assert window.renderer.rect().contains(window.right_panel.geometry())
    assert window.renderer._callout.identifier == HeartPartId.LEFT_VENTRICLE
    assert window.right_panel.isVisible()
    assert window.info_scroll.verticalScrollBar().value() == 0, "New selection resets reading position"
    capture("compact")
    (output / "callout-verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records), flush=True)
    window.close()
    app.quit()


def fail(kind, value, trace):
    sys.__excepthook__(kind, value, trace)
    window.close()
    app.exit(1)


sys.excepthook = fail
for delay, action in ((2500, select), (4500, rotate), (6500, explode),
                      (9000, spatial), (11000, clear), (12500, resize), (14500, finish)):
    QTimer.singleShot(delay, action)
raise SystemExit(app.exec())
