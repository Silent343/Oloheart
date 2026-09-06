"""Explicit desktop acceptance run; this script never starts the camera."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QTimer
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
records = []


def capture(name):
    assert window.renderer.isValid()
    assert window.renderer._program
    assert window.grab().save(str(output / f"anatomy-{name}.png"))
    records.append({"view": name, "fps": window.renderer.render_fps,
                    "batches": len(window.renderer._gpu_meshes)})


def section():
    capture("exterior")
    controller.toggle_interior_view()


def pericardium():
    capture("interior")
    controller.select_part(HeartPartId.PERICARDIUM)


def exploded():
    capture("pericardium")
    controller.toggle_interior_view()


def pericardial_exterior():
    capture("pericardium-exterior")
    controller.toggle_interior_view()
    controller.toggle_system(controller.part(HeartPartId.PERICARDIUM).system)
    controller.toggle_explosion()
    controller.rotate(0.34, -0.12)


def finish():
    capture("exploded")
    (output / "anatomy-verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records), flush=True)
    window.close()
    app.quit()


def fail(kind, value, trace):
    sys.__excepthook__(kind, value, trace)
    app.exit(1)


sys.excepthook = fail
QTimer.singleShot(2500, section)
QTimer.singleShot(4800, pericardium)
QTimer.singleShot(7100, exploded)
QTimer.singleShot(9200, pericardial_exterior)
QTimer.singleShot(11500, finish)
raise SystemExit(app.exec())
