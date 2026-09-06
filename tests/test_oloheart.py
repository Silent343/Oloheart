from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oloheart.application.controller import GestureCoordinator, HeartController, InteractionAction
from oloheart.domain.model import (
    AnatomySystem,
    CardiacPhase,
    GestureKind,
    HandObservation,
    HeartExperience,
    HeartPartId,
)
from oloheart.infrastructure.geometry import build_heart_geometry
from oloheart.infrastructure.vision import MediaPipeHandTracker


def observation(gesture: GestureKind, captured_at: float, hands: int = 1,
                palm: tuple[float, float] = (0.5, 0.5), span: float = 1.0,
                pointer: tuple[float, float] = (0.48, 0.46)) -> HandObservation:
    return HandObservation(gesture, 0.95, *palm, *pointer, span, hands, (), captured_at)


class HeartExperienceTests(unittest.TestCase):
    def test_zoom_is_bounded_and_state_is_smoothed(self) -> None:
        heart = HeartExperience()
        heart.zoom_by(100.0)
        for _ in range(100):
            heart.tick(0.05)
        self.assertLessEqual(heart.snapshot().zoom, HeartExperience.MAX_ZOOM)
        heart.zoom_by(0.001)
        for _ in range(100):
            heart.tick(0.05)
        self.assertGreaterEqual(heart.snapshot().zoom, HeartExperience.MIN_ZOOM)

    def test_realistic_mode_owns_the_heartbeat(self) -> None:
        heart = HeartExperience()
        self.assertFalse(heart.snapshot().beating)
        heart.toggle_realistic_heartbeat()
        heart.tick(0.07)
        self.assertTrue(heart.snapshot().realistic)
        self.assertTrue(heart.snapshot().beating)
        self.assertGreater(heart.snapshot().beat_scale, 1.0)

    def test_cardiac_cycle_orders_contraction_and_pressure_driven_valves(self) -> None:
        heart = HeartExperience()
        heart.toggle_realistic_heartbeat()

        heart.tick(0.05)
        atrial = heart.snapshot().cycle
        self.assertEqual(CardiacPhase.ATRIAL_SYSTOLE, atrial.phase)
        self.assertTrue(atrial.atrioventricular_valves_open)
        self.assertFalse(atrial.semilunar_valves_open)
        self.assertGreater(atrial.atrial_contraction, atrial.ventricular_contraction)

        heart.tick(0.06)
        isovolumetric_contraction = heart.snapshot().cycle
        self.assertEqual(
            CardiacPhase.ISOVOLUMETRIC_CONTRACTION,
            isovolumetric_contraction.phase,
        )
        self.assertFalse(isovolumetric_contraction.atrioventricular_valves_open)
        self.assertFalse(isovolumetric_contraction.semilunar_valves_open)

        heart.tick(0.06)
        ejection = heart.snapshot().cycle
        self.assertEqual(CardiacPhase.VENTRICULAR_EJECTION, ejection.phase)
        self.assertFalse(ejection.atrioventricular_valves_open)
        self.assertTrue(ejection.semilunar_valves_open)
        self.assertGreater(ejection.ventricular_contraction, 0.8)

        for _ in range(3):
            heart.tick(0.08)
        relaxation = heart.snapshot().cycle
        self.assertEqual(CardiacPhase.ISOVOLUMETRIC_RELAXATION, relaxation.phase)
        self.assertFalse(relaxation.atrioventricular_valves_open)
        self.assertFalse(relaxation.semilunar_valves_open)

        heart.tick(0.08)
        filling = heart.snapshot().cycle
        self.assertEqual(CardiacPhase.RAPID_FILLING, filling.phase)
        self.assertTrue(filling.atrioventricular_valves_open)
        self.assertFalse(filling.semilunar_valves_open)
        self.assertGreater(filling.coronary_perfusion, ejection.coronary_perfusion)

    def test_seven_anatomical_systems_are_independently_visible(self) -> None:
        heart = HeartExperience()
        self.assertEqual(7, len(AnatomySystem))
        self.assertNotIn(AnatomySystem.PERICARDIUM, heart.snapshot().visible_systems)
        heart.select(HeartPartId.PERICARDIUM)
        self.assertIn(AnatomySystem.PERICARDIUM, heart.snapshot().visible_systems)
        heart.toggle_system(AnatomySystem.CHAMBERS_AND_SEPTA)
        self.assertNotIn(AnatomySystem.CHAMBERS_AND_SEPTA, heart.snapshot().visible_systems)

    def test_fragments_move_only_in_exploded_view_and_reset_on_reconstruction(self) -> None:
        heart = HeartExperience()
        self.assertFalse(heart.begin_fragment_drag(HeartPartId.LEFT_VENTRICLE))
        heart.toggle_explosion()
        self.assertTrue(heart.begin_fragment_drag(HeartPartId.LEFT_VENTRICLE))
        self.assertTrue(heart.move_grabbed_fragment(0.12, -0.08, 0.05))
        snapshot = heart.snapshot()
        self.assertEqual(HeartPartId.LEFT_VENTRICLE, snapshot.grabbed_fragment)
        self.assertEqual(1, len(snapshot.fragment_offsets))
        self.assertAlmostEqual(0.12, snapshot.fragment_offsets[0].horizontal)
        heart.release_fragment()
        heart.toggle_explosion()
        self.assertFalse(heart.snapshot().fragment_offsets)
        self.assertIsNone(heart.snapshot().grabbed_fragment)


class CatalogAndGeometryTests(unittest.TestCase):
    def test_every_catalog_part_has_renderable_geometry(self) -> None:
        controller = HeartController()
        meshes = build_heart_geometry()
        self.assertEqual(24, len(controller.catalog))
        self.assertEqual(set(AnatomySystem), {part.system for part in controller.catalog})
        self.assertEqual({part.identifier for part in controller.catalog},
                         {mesh.part_id for mesh in meshes})
        for mesh in meshes:
            self.assertGreater(len(mesh.vertices), 5)
            self.assertGreater(len(mesh.faces), 3)
            self.assertTrue(all(max(face) < len(mesh.vertices) for face in mesh.faces))

    def test_advanced_reference_structures_are_selectable_and_renderable(self) -> None:
        controller = HeartController()
        meshes = build_heart_geometry()
        advanced = {
            HeartPartId.PERICARDIUM,
            HeartPartId.EPICARDIUM,
            HeartPartId.MYOCARDIUM,
            HeartPartId.ENDOCARDIUM,
            HeartPartId.PAPILLARY_MUSCLES,
            HeartPartId.CHORDAE_TENDINEAE,
            HeartPartId.CARDIAC_CONDUCTION,
            HeartPartId.CORONARY_SINUS,
            HeartPartId.INTERATRIAL_SEPTUM,
            HeartPartId.CORONARY_VEINS,
        }
        self.assertTrue(advanced.issubset({part.identifier for part in controller.catalog}))
        self.assertTrue(advanced.issubset({mesh.part_id for mesh in meshes}))
        self.assertGreater(sum(len(mesh.faces) for mesh in meshes), 9000)

    def test_unknown_parts_cannot_enter_the_aggregate(self) -> None:
        controller = HeartController()
        controller.select_part(HeartPartId.AORTA)
        self.assertEqual(HeartPartId.AORTA, controller.snapshot().selected_part)
        controller.focus_whole_heart()
        self.assertIsNone(controller.snapshot().selected_part)


class GestureCoordinatorTests(unittest.TestCase):
    def test_open_palm_rotates_relative_to_previous_frame(self) -> None:
        controller = HeartController()
        coordinator = GestureCoordinator(controller)
        initial = controller.snapshot().rotation_y
        coordinator.process(observation(GestureKind.OPEN_PALM, 1.0, palm=(0.40, 0.50)))
        coordinator.process(observation(GestureKind.OPEN_PALM, 1.1, palm=(0.46, 0.50)))
        for _ in range(20):
            controller.tick(0.04)
        self.assertGreater(controller.snapshot().rotation_y, initial)

    def test_stable_victory_toggles_exploded_view_only_once(self) -> None:
        controller = HeartController()
        coordinator = GestureCoordinator(controller)
        coordinator.process(observation(GestureKind.VICTORY, 1.0))
        result = coordinator.process(observation(GestureKind.VICTORY, 1.6))
        self.assertEqual(InteractionAction.EXPLOSION_CHANGED, result.action)
        self.assertEqual(InteractionAction.NONE,
                         coordinator.process(observation(GestureKind.VICTORY, 1.8)).action)
        for _ in range(30):
            controller.tick(0.04)
        self.assertGreater(controller.snapshot().explosion, 0.8)

    def test_three_fingers_switch_to_realistic_heartbeat(self) -> None:
        controller = HeartController()
        coordinator = GestureCoordinator(controller)
        coordinator.process(observation(GestureKind.THREE_FINGERS, 2.0))
        result = coordinator.process(observation(GestureKind.THREE_FINGERS, 2.6))
        self.assertEqual(InteractionAction.REALISM_CHANGED, result.action)
        self.assertTrue(controller.snapshot().realistic)

    def test_pinky_toggles_distraction_free_spatial_mode(self) -> None:
        controller = HeartController()
        coordinator = GestureCoordinator(controller)
        coordinator.process(observation(GestureKind.PINKY, 3.0))
        result = coordinator.process(observation(GestureKind.PINKY, 3.6))
        self.assertEqual(InteractionAction.SPATIAL_MODE_CHANGED, result.action)
        self.assertTrue(controller.snapshot().spatial_mode)

    def test_pinch_emits_continuous_fragment_drag_only_after_explosion(self) -> None:
        controller = HeartController()
        coordinator = GestureCoordinator(controller)
        blocked = coordinator.process(observation(GestureKind.PINCH, 4.0))
        self.assertEqual(InteractionAction.NONE, blocked.action)
        controller.toggle_explosion()
        for _ in range(20):
            controller.tick(0.04)
        start = coordinator.process(observation(
            GestureKind.PINCH, 4.1, pointer=(0.40, 0.42), span=0.82
        ))
        self.assertEqual(InteractionAction.FRAGMENT_GRAB_STARTED, start.action)
        move = coordinator.process(observation(
            GestureKind.PINCH, 4.2, pointer=(0.46, 0.45), span=0.88
        ))
        self.assertEqual(InteractionAction.FRAGMENT_GRAB_MOVED, move.action)
        self.assertAlmostEqual(0.06, move.delta_x)
        self.assertAlmostEqual(0.03, move.delta_y)
        self.assertAlmostEqual(0.06, move.delta_depth)
        end = coordinator.process(observation(GestureKind.OPEN_PALM, 4.3))
        self.assertEqual(InteractionAction.FRAGMENT_GRAB_ENDED, end.action)


class LandmarkGestureTests(unittest.TestCase):
    @staticmethod
    def _hand(extended: set[str]) -> list[SimpleNamespace]:
        points = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
        points[5] = SimpleNamespace(x=0.0, y=-0.22, z=0.0)
        for name, joint, tip, x in (("index", 6, 8, -0.18), ("middle", 10, 12, -0.05),
                                     ("ring", 14, 16, 0.09), ("pinky", 18, 20, 0.20)):
            points[joint] = SimpleNamespace(x=x, y=-0.32, z=0.0)
            tip_y = -0.68 if name in extended else -0.16
            points[tip] = SimpleNamespace(x=x, y=tip_y, z=0.0)
        points[2] = SimpleNamespace(x=0.12, y=-0.08, z=0.0)
        points[3] = SimpleNamespace(x=0.11, y=-0.12, z=0.0)
        points[4] = SimpleNamespace(x=0.08, y=-0.15, z=0.0)
        return points

    def test_landmarks_classify_primary_control_gestures(self) -> None:
        classify = MediaPipeHandTracker._classify_gesture
        self.assertEqual(GestureKind.OPEN_PALM,
                         classify(self._hand({"index", "middle", "ring", "pinky"}))[0])
        self.assertEqual(GestureKind.VICTORY,
                         classify(self._hand({"index", "middle"}))[0])
        self.assertEqual(GestureKind.THREE_FINGERS,
                         classify(self._hand({"index", "middle", "ring"}))[0])
        self.assertEqual(GestureKind.PINKY,
                         classify(self._hand({"pinky"}))[0])
        self.assertEqual(GestureKind.POINT,
                         classify(self._hand({"index"}))[0])
        self.assertEqual(GestureKind.CLOSED_FIST,
                         classify(self._hand(set()))[0])

    def test_thumb_index_contact_is_a_pinch(self) -> None:
        hand = self._hand({"index"})
        hand[4] = SimpleNamespace(x=hand[8].x + 0.01, y=hand[8].y + 0.01, z=0.0)
        self.assertEqual(GestureKind.PINCH, MediaPipeHandTracker._classify_gesture(hand)[0])


if __name__ == "__main__":
    unittest.main()
