"""Camera-free, deterministic diastolic/systolic and assembled-surface views."""

import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from PySide6.QtCore import QTimer
from oloheart.application.controller import HeartController, NullHandTracker
from oloheart.domain.model import CardiacPhase, AnatomySystem
from oloheart.presentation.qt_app import QtDesktopApplication, create_qt_application

app = create_qt_application()
controller = HeartController()
window = QtDesktopApplication(controller,NullHandTracker(),fullscreen=False)
window.resize(1500,900)
window.show()
folder = Path(__file__).resolve().parents[1]/"artifacts"
folder.mkdir(exist_ok=True)


def verify():
    state = controller.snapshot()
    systems = frozenset({AnatomySystem.CHAMBERS_AND_SEPTA,AnatomySystem.MYOCARDIUM,
                         AnatomySystem.VALVES_AND_SUBVALVULAR,AnatomySystem.GREAT_VESSELS})
    for name,phase,progress,tension,interior,rx,zoom in (
        ("support-diastole",CardiacPhase.DIASTASIS,.85,0,True,.42,1.36),
        ("support-isovolumetric",CardiacPhase.ISOVOLUMETRIC_CONTRACTION,.16,.5,True,.42,1.36),
        ("support-systole",CardiacPhase.VENTRICULAR_EJECTION,.35,1,True,.42,1.36),
        ("support-assembled-top",CardiacPhase.DIASTASIS,.85,0,False,.72,1.05),
        ("support-assembled-front",CardiacPhase.DIASTASIS,.85,0,False,.08,1.05),
    ):
        cycle = replace(state.cycle,phase=phase,progress=progress,ventricular_contraction=tension,
                        atrial_contraction=0,atrioventricular_valves_open=phase == CardiacPhase.DIASTASIS,
                        semilunar_valves_open=phase == CardiacPhase.VENTRICULAR_EJECTION)
        frame = replace(state,cycle=cycle,realistic=True,beating=True,visible_systems=systems,
                        interior_view=interior,rotation_x=rx,rotation_y=-.18,rotation_z=0,zoom=zoom)
        window.renderer.set_snapshot(frame)
        image = window.renderer.grabFramebuffer()
        assert not image.isNull() and image.save(str(folder/f"{name}.png"))
    print("Support views rendered: diastole, isovolumetric contraction, systole, assembled top/front.",flush=True)
    window.close()
    app.quit()


def fail(kind,value,trace):
    sys.__excepthook__(kind,value,trace)
    window.close()
    app.exit(1)


sys.excepthook = fail
QTimer.singleShot(1200,verify)
raise SystemExit(app.exec())
