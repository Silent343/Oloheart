"""Explicit GPU acceptance run for anatomy, flow and independent element picking.

This desktop test never starts a camera. It saves images for human inspection.
"""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PySide6.QtCore import QTimer
from oloheart.application.controller import HeartController, NullHandTracker
from oloheart.domain.model import AnatomicalElement, HeartPartId as I, AnatomySystem, CardiacPhase
from oloheart.infrastructure.sculpted_geometry import VALVES
from oloheart.presentation.qt_app import QtDesktopApplication, create_qt_application
from oloheart.presentation.scene_layout import placement_matrix

app = create_qt_application()
controller = HeartController()
controller.toggle_realistic_heartbeat()
window = QtDesktopApplication(controller,NullHandTracker(),fullscreen=False)
window.resize(1500,900)
window.show()
window.start()
folder = Path(__file__).resolve().parents[1]/"artifacts"
folder.mkdir(exist_ok=True)
records = []


def capture(name):
    assert window.renderer.isValid() and window.renderer._program
    assert window.grab().save(str(folder/f"atlas-{name}.png"))
    records.append({"view":name,"fps":window.renderer.render_fps,
                    "batches":len(window.renderer._gpu_meshes)})


def interior():
    capture("exterior")
    controller.toggle_interior_view()


def flow():
    capture("interior")
    controller.toggle_blood_flow()


def exploded():
    capture("flow")
    controller.toggle_explosion()
    controller.toggle_spatial_mode()


def drag():
    renderer = window.renderer
    vein = next(m for m in renderer._gpu_meshes if m.element_key == "left_superior_pulmonary_vein")
    other = next(m for m in renderer._gpu_meshes if m.element_key == "left_inferior_pulmonary_vein")
    assert renderer._pick_id_by_element[vein.element_key] != renderer._pick_id_by_element[other.element_key]
    before = controller.snapshot()
    # Remove chamber occlusion for the ID probe; normal picking must not select
    # a posterior vein through an overlying atrium.
    pick_state = replace(before,visible_systems=frozenset({AnatomySystem.GREAT_VESSELS}),
                         rotation_x=0,rotation_y=0,rotation_z=0)
    renderer.set_snapshot(pick_state)
    projection,view,scale = renderer._scene_camera(pick_state)
    model = placement_matrix(pick_state,vein.explosion,vein.part_id,scale,vein.element_key)
    clip = projection@view@model@np.asarray((*vein.center,1))
    ndc = clip[:3]/clip[3]
    picked = renderer.pick_part((ndc[0]+1)*renderer.width()/2,(1-ndc[1])*renderer.height()/2)
    assert picked is not None and picked.key == vein.element_key, f"GPU picking must identify the exact vein; got {picked}, NDC {ndc}"
    renderer.set_snapshot(before)
    reference = placement_matrix(before,other.explosion,other.part_id,1,other.element_key)
    element = AnatomicalElement(vein.part_id,vein.element_key,vein.element_name)
    controller.begin_fragment_drag(element)
    controller.move_grabbed_fragment(.13,-.06,0)
    after = controller.snapshot()
    np.testing.assert_array_equal(reference,placement_matrix(after,other.explosion,other.part_id,1,other.element_key))
    controller.release_fragment()


def selected():
    assert window.part_title.text() == "VENA PULMONAR SUPERIOR IZQUIERDA"
    assert window.renderer._selected_anchor() is not None
    capture("independent-selection")
    controller.reset_view()
    controller.toggle_interior_view()
    controller.select_part(I.PERICARDIUM)


def finish():
    capture("pericardium")
    window.timer.stop()
    # A normal-to-valve GPU probe must hit a closed leaflet and miss an open one.
    state = replace(controller.snapshot(),rotation_x=math.pi/2-.48,rotation_y=0,rotation_z=0,
                    selected_part=None,selected_element=None,flow_enabled=False,explosion=0,
                    fragment_offsets=(),visible_systems=frozenset({AnatomySystem.VALVES_AND_SUBVALVULAR}))
    renderer = window.renderer
    center = VALVES[I.MITRAL_VALVE][0]
    for opened in (False,True):
        cycle = replace(state.cycle,phase=CardiacPhase.DIASTASIS if opened else CardiacPhase.VENTRICULAR_EJECTION,
                        progress=.85 if opened else .35,atrioventricular_valves_open=opened,
                        semilunar_valves_open=not opened)
        probe = replace(state,cycle=cycle)
        renderer.set_snapshot(probe)
        projection,view,scale = renderer._scene_camera(probe)
        model = placement_matrix(probe,(0,0,0),I.MITRAL_VALVE,scale,I.MITRAL_VALVE.value)
        clip = projection@view@model@np.asarray((center.x+.09,center.y,center.z,1))
        ndc = clip[:3]/clip[3]
        hit = renderer.pick_part((ndc[0]+1)*renderer.width()/2,(1-ndc[1])*renderer.height()/2)
        assert bool(hit and hit.part_id == I.MITRAL_VALVE) != opened, f"Incorrect GPU leaflet opening: {opened}, {hit}"
    records.append({"gpu_exact_vein_pick":True,"gpu_valve_open_closed":True})
    (folder/"atlas-verification.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    print(json.dumps(records),flush=True)
    window.close()
    app.quit()


def fail(kind,value,trace):
    sys.__excepthook__(kind,value,trace)
    window.close()
    app.exit(1)


sys.excepthook = fail
for delay,action in ((2800,interior),(5200,flow),(8200,exploded),(10500,drag),(12800,selected),(15500,finish)):
    QTimer.singleShot(delay,action)
raise SystemExit(app.exec())
