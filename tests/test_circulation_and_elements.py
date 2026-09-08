"""Regressions for independent picking identity and valve-gated circulation."""

import unittest
from collections import Counter
from dataclasses import replace
import numpy as np

from oloheart.application.controller import HeartController, GestureCoordinator, InteractionAction
from oloheart.domain.model import AnatomicalElement, HeartPartId as I, GestureKind, HandObservation, AnatomySystem
from oloheart.infrastructure.geometry import build_heart_geometry
from oloheart.infrastructure.circulation_geometry import build_flow_routes, circulation_visible
from oloheart.presentation.scene_layout import placement_matrix, visible_mesh


class IndependentElementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshes = build_heart_geometry()

    def test_dragging_one_pulmonary_vein_does_not_move_the_other_three(self):
        controller = HeartController()
        controller.toggle_explosion()
        for _ in range(40):
            controller.tick(.05)
        veins = [m for m in self.meshes if m.part_id == I.PULMONARY_VEINS]
        self.assertEqual(4,len({m.element_key for m in veins}))
        target = veins[0]
        before = controller.snapshot()
        element = AnatomicalElement(target.part_id,target.element_key,target.element_name)
        self.assertTrue(controller.begin_fragment_drag(element))
        controller.move_grabbed_fragment(.12,.06,.08)
        after = controller.snapshot()
        self.assertEqual(element,after.selected_element)
        self.assertEqual(target.element_name,controller.part(after.selected_element).display_name)
        for mesh in veins:
            e = mesh.explosion_vector
            args = ((e.x,e.y,e.z),mesh.part_id,1.0,mesh.element_key)
            a,b = placement_matrix(before,*args),placement_matrix(after,*args)
            if mesh.element_key == target.element_key:
                self.assertFalse(np.allclose(a,b))
            else:
                np.testing.assert_array_equal(a,b)

    def test_papillary_muscles_and_chordal_fans_have_distinct_identities(self):
        muscles = {m.element_key for m in self.meshes if m.part_id == I.PAPILLARY_MUSCLES}
        cords = {m.element_key for m in self.meshes if m.part_id == I.CHORDAE_TENDINEAE}
        self.assertEqual(5,len(muscles))
        self.assertEqual(15,len(cords))
        self.assertFalse(muscles & cords)

    def test_reconstruction_releases_and_clears_only_manual_placement(self):
        controller = HeartController()
        controller.toggle_explosion()
        target = AnatomicalElement(I.VENA_CAVA,"superior_vena_cava","Vena cava superior")
        controller.begin_fragment_drag(target)
        controller.move_grabbed_fragment(.1,0,0)
        controller.toggle_explosion()
        self.assertIsNone(controller.snapshot().grabbed_element)
        self.assertFalse(controller.snapshot().fragment_offsets)

    def test_vessel_rings_are_smooth_and_parallel_transport_does_not_flip(self):
        vessel = next(m for m in self.meshes if m.element_key == "aorta" and m.view == "exterior")
        sides = vessel.faces[0][1]
        self.assertGreaterEqual(sides,32)
        outer = np.asarray([(v.x,v.y,v.z) for v in vessel.vertices[:len(vessel.vertices)//2]])
        rings = outer.reshape(-1,sides,3)
        normals = rings[:,0]-rings.mean(axis=1)
        normals /= np.linalg.norm(normals,axis=1)[:,None]
        self.assertTrue(np.all(np.sum(normals[:-1]*normals[1:],axis=1) > 0.7))

    def test_root_windows_are_watertight_with_shared_cut_edges(self):
        for key in ("aorta","pulmonary_trunk"):
            mesh = next(m for m in self.meshes if m.element_key == key and m.view == "interior")
            edges = Counter(tuple(sorted(e)) for a,b,c in mesh.faces for e in ((a,b),(b,c),(c,a)))
            self.assertTrue(all(n == 2 for n in edges.values()),key)

    def test_ventricular_myocardium_uses_the_same_exterior_surface(self):
        for part in (I.LEFT_VENTRICLE,I.RIGHT_VENTRICLE):
            outer = next(m for m in self.meshes if m.part_id == part and m.view == "exterior")
            wall = next(m for m in self.meshes if m.element_key == f"myocardium_{part.value}" and m.view == "exterior")
            self.assertEqual(outer.vertices,wall.vertices)
            self.assertEqual(outer.deformation_center,wall.deformation_center)

    def test_selecting_one_myocardium_preserves_the_other_chamber(self):
        target = AnatomicalElement(I.MYOCARDIUM,"myocardium_left_ventricle","Miocardio izquierdo")
        state = replace(HeartController().snapshot(),selected_part=I.MYOCARDIUM,selected_element=target)
        right = next(m for m in self.meshes if m.part_id == I.RIGHT_VENTRICLE and m.view == "exterior")
        left = next(m for m in self.meshes if m.part_id == I.LEFT_VENTRICLE and m.view == "exterior")
        self.assertTrue(visible_mesh(right,state))
        self.assertFalse(visible_mesh(left,state))

    def test_mesh_identities_cover_every_catalog_part_without_cross_category_keys(self):
        ownership = {}
        for mesh in self.meshes:
            self.assertTrue(mesh.element_key)
            self.assertTrue(mesh.element_name)
            ownership.setdefault(mesh.element_key,set()).add(mesh.part_id)
        self.assertTrue(all(len(parts) == 1 for parts in ownership.values()))
        self.assertEqual(set(I),{m.part_id for m in self.meshes})


class CirculationTests(unittest.TestCase):
    def test_valve_routes_never_flow_through_closed_valves_across_cycles(self):
        controller = HeartController()
        controller.toggle_blood_flow()
        routes = build_flow_routes()
        seen = set()
        for _ in range(300):
            state = controller.tick(.01)
            seen.add(state.cycle.phase)
            for route in routes:
                if route.gate == 1:
                    self.assertEqual(state.cycle.atrioventricular_valves_open,route.active(state))
                elif route.gate == 2:
                    self.assertEqual(state.cycle.semilunar_valves_open,route.active(state))
            if not state.cycle.atrioventricular_valves_open:
                self.assertEqual(0,state.cycle.atrioventricular_opening)
            if not state.cycle.semilunar_valves_open:
                self.assertEqual(0,state.cycle.semilunar_opening)
        self.assertEqual(6,len(seen))

    def test_routes_use_the_correct_sides_and_do_not_bridge_atrial_septum(self):
        routes = {r.key:r for r in build_flow_routes()}
        self.assertFalse(routes["right_filling"].oxygenated)
        self.assertTrue(routes["left_filling"].oxygenated)
        self.assertEqual((I.RIGHT_ATRIUM,I.TRICUSPID_VALVE,I.RIGHT_VENTRICLE),routes["right_filling"].parts)
        self.assertEqual((I.LEFT_ATRIUM,I.MITRAL_VALVE,I.LEFT_VENTRICLE),routes["left_filling"].parts)
        self.assertEqual(17,len(routes))

    def test_exploded_heart_does_not_display_a_fictitious_connected_circulation(self):
        controller = HeartController()
        controller.toggle_blood_flow()
        self.assertTrue(circulation_visible(controller.snapshot()))
        controller.toggle_explosion()
        controller.tick(.08)
        self.assertFalse(circulation_visible(controller.snapshot()))

    def test_index_and_pinky_hold_toggles_flow_once(self):
        controller = HeartController()
        coordinator = GestureCoordinator(controller)
        hand = HandObservation(GestureKind.FLOW,.95,.5,.5,.5,.5,1,1,(),0)
        coordinator.process(hand)
        self.assertEqual(InteractionAction.FLOW_CHANGED,coordinator.process(replace(hand,captured_at=.6)).action)
        self.assertEqual(InteractionAction.NONE,coordinator.process(replace(hand,captured_at=1.2)).action)
        self.assertTrue(controller.snapshot().flow_enabled)

    def test_closed_gate_clocks_and_paused_animation_do_not_advance(self):
        controller = HeartController()
        controller.toggle_blood_flow()
        while controller.snapshot().cycle.progress < .14:
            controller.tick(.005)
        before = controller.snapshot().flow_times
        after = controller.tick(.005).flow_times
        self.assertGreater(after[0],before[0])
        self.assertEqual(before[1:],after[1:])
        controller.toggle_realistic_heartbeat()
        before = controller.snapshot().flow_times
        self.assertEqual(before,controller.tick(.08).flow_times)

    def test_hiding_a_system_clears_a_grabbed_element(self):
        controller = HeartController()
        controller.toggle_explosion()
        element = AnatomicalElement(I.VENA_CAVA,"superior_vena_cava","Vena cava superior")
        controller.begin_fragment_drag(element)
        controller.toggle_system(AnatomySystem.GREAT_VESSELS)
        self.assertIsNone(controller.snapshot().selected_element)
        self.assertIsNone(controller.snapshot().grabbed_element)


if __name__ == "__main__":
    unittest.main()
