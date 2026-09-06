"""Regressions for anatomical sections and non-teleporting selections."""

from dataclasses import replace
from collections import Counter
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from oloheart.application.controller import HeartController, GestureCoordinator, InteractionAction
from oloheart.domain.model import AnatomySystem, CardiacPhase, GestureKind, HandObservation, HeartPartId as I
from oloheart.infrastructure.geometry import build_heart_geometry
from oloheart.presentation.scene_layout import placement_matrix, object_drag, scene_rotation, visible_mesh


class AnatomicalSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshes = build_heart_geometry()

    def test_selection_never_changes_any_fragment_transform(self):
        controller = HeartController()
        controller.toggle_explosion()
        for _ in range(20):
            controller.tick(0.08)
        controller.begin_fragment_drag(I.LEFT_VENTRICLE)
        controller.move_grabbed_fragment(0.12, -0.07, 0.03)
        before = controller.snapshot()
        controller.select_part(I.LEFT_VENTRICLE)
        after = controller.snapshot()
        for mesh in self.meshes:
            e = mesh.explosion_vector
            np.testing.assert_array_equal(
                placement_matrix(before, (e.x,e.y,e.z), mesh.part_id, 1.0),
                placement_matrix(after, (e.x,e.y,e.z), mesh.part_id, 1.0))

    def test_camera_drag_round_trips_through_rotated_model(self):
        state = replace(HeartController().snapshot(), rotation_x=0.6, rotation_y=1.2, rotation_z=-0.3)
        dx,dy,dz = object_drag(state,0.10,-0.05,0.02)
        result = scene_rotation(state)[:3,:3] @ np.asarray((dx*4,-dy*4,dz*3))
        np.testing.assert_allclose(result,(0.40,0.20,0.06),atol=1e-6)

    def test_isovolumetric_phases_preserve_chamber_dimensions(self):
        cycle = HeartController().snapshot().cycle
        for progress in (0.125,0.16,0.195):
            state = replace(cycle,phase=CardiacPhase.ISOVOLUMETRIC_CONTRACTION,progress=progress)
            self.assertEqual(0.0,state.ventricular_emptying)
        for progress in (0.485,0.52,0.575):
            state = replace(cycle,phase=CardiacPhase.ISOVOLUMETRIC_RELAXATION,progress=progress)
            self.assertEqual(1.0,state.ventricular_emptying)

    def test_chambers_and_pericardium_have_closed_thickness_at_section_edges(self):
        for part in (I.PERICARDIUM,I.LEFT_VENTRICLE,I.RIGHT_VENTRICLE,I.LEFT_ATRIUM,I.RIGHT_ATRIUM):
            section = next(m for m in self.meshes if m.part_id == part and m.view == "interior")
            edges = Counter()
            for a,b,c in section.faces:
                edges.update(tuple(sorted(edge)) for edge in ((a,b),(b,c),(c,a)))
            self.assertTrue(all(count == 2 for count in edges.values()), part)
            height = max(v.y for v in section.vertices)-min(v.y for v in section.vertices)
            self.assertGreater(height,0.55)

    def test_surface_and_section_are_mutually_exclusive(self):
        state = HeartController().snapshot()
        surfaces = [m for m in self.meshes if m.part_id == I.LEFT_VENTRICLE and m.view != "all"]
        self.assertEqual(["exterior"],[m.view for m in surfaces if visible_mesh(m,state)])
        state = replace(state,interior_view=True)
        self.assertEqual(["interior"],[m.view for m in surfaces if visible_mesh(m,state)])
        state = replace(state,visible_systems=frozenset({AnatomySystem.GREAT_VESSELS}))
        self.assertFalse(any(visible_mesh(m,state) for m in surfaces))

    def test_valves_have_two_or_three_separate_leaflets(self):
        for part,expected in ((I.MITRAL_VALVE,2),(I.TRICUSPID_VALVE,3),
                              (I.AORTIC_VALVE,3),(I.PULMONARY_VALVE,3)):
            mesh = next(m for m in self.meshes if m.part_id == part and m.motion != "none")
            neighbors = {index:set() for index in range(len(mesh.vertices))}
            for a,b,c in mesh.faces:
                neighbors[a].update((b,c)); neighbors[b].update((a,c)); neighbors[c].update((a,b))
            remaining = set(neighbors)
            components = 0
            while remaining:
                pending = [remaining.pop()]
                components += 1
                while pending:
                    for index in neighbors[pending.pop()] & remaining:
                        remaining.remove(index)
                        pending.append(index)
            self.assertEqual(expected,components,part)

    def test_thumb_down_opens_section_once_without_moving_selection(self):
        controller = HeartController()
        controller.select_part(I.LEFT_VENTRICLE)
        coordinator = GestureCoordinator(controller)
        def hand(time):
            return HandObservation(GestureKind.THUMB_DOWN,0.95,0.5,0.5,0.5,0.5,1,1,(),time)
        coordinator.process(hand(1.0))
        self.assertEqual(InteractionAction.INTERIOR_VIEW_CHANGED,coordinator.process(hand(1.6)).action)
        self.assertEqual(InteractionAction.NONE,coordinator.process(hand(1.8)).action)
        self.assertTrue(controller.snapshot().interior_view)
        self.assertEqual(I.LEFT_VENTRICLE,controller.snapshot().selected_part)
