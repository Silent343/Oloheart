"""Application use cases and gesture orchestration for OloHeart."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from oloheart.domain.model import (
    AnatomySystem,
    AnatomicalElement,
    GestureKind,
    HandObservation,
    HeartExperience,
    HeartPart,
    HeartPartId,
    HeartSnapshot,
    build_anatomy_catalog,
)


class HandTracker(Protocol):
    """Infrastructure boundary that never exposes raw camera images."""

    def start(
        self,
        on_observation: Callable[[HandObservation], None],
        on_status: Callable[[str], None],
    ) -> None: ...

    def stop(self) -> None: ...


class InteractionAction(StrEnum):
    NONE = "none"
    ACTIVATE_POINTER = "activate_pointer"
    FOCUS_WHOLE = "focus_whole"
    EXPLOSION_CHANGED = "explosion_changed"
    REALISM_CHANGED = "realism_changed"
    SPATIAL_MODE_CHANGED = "spatial_mode_changed"
    INTERIOR_VIEW_CHANGED = "interior_view_changed"
    FLOW_CHANGED = "flow_changed"
    FRAGMENT_GRAB_STARTED = "fragment_grab_started"
    FRAGMENT_GRAB_MOVED = "fragment_grab_moved"
    FRAGMENT_GRAB_ENDED = "fragment_grab_ended"


@dataclass(frozen=True, slots=True)
class InteractionResult:
    action: InteractionAction
    pointer_x: float = 0.5
    pointer_y: float = 0.5
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_depth: float = 0.0


class HeartController:
    """Facade exposing intention-oriented use cases to the desktop shell."""

    def __init__(self, heart: HeartExperience | None = None) -> None:
        self._heart = heart or HeartExperience()
        self._catalog = build_anatomy_catalog()
        self._parts_by_id = {part.identifier: part for part in self._catalog}

    @property
    def catalog(self) -> tuple[HeartPart, ...]:
        return self._catalog

    def part(self, identifier: HeartPartId | AnatomicalElement | None) -> HeartPart | None:
        if isinstance(identifier, AnatomicalElement):
            part = self._parts_by_id.get(identifier.part_id)
            return replace(part, display_name=identifier.name) if part and identifier.name else part
        return self._parts_by_id.get(identifier) if identifier else None

    def select_part(self, identifier: HeartPartId | AnatomicalElement) -> None:
        element = identifier if isinstance(identifier, AnatomicalElement) else None
        part = element.part_id if element else identifier
        if part not in self._parts_by_id:
            raise ValueError(f"Unknown heart part: {identifier}")
        self._heart.select(part, element)

    def focus_whole_heart(self) -> None:
        self._heart.focus_whole_heart()

    def rotate(self, horizontal: float, vertical: float) -> None:
        self._heart.rotate_by(horizontal, vertical)

    def zoom(self, factor: float) -> None:
        self._heart.zoom_by(factor)

    def toggle_explosion(self) -> None:
        self._heart.toggle_explosion()

    def begin_fragment_drag(self, identifier: HeartPartId | AnatomicalElement) -> bool:
        element = identifier if isinstance(identifier, AnatomicalElement) else None
        part = element.part_id if element else identifier
        if part not in self._parts_by_id:
            raise ValueError(f"Unknown heart part: {identifier}")
        return self._heart.begin_fragment_drag(part, element)

    def move_grabbed_fragment(self, horizontal: float, vertical: float, depth: float) -> bool:
        return self._heart.move_grabbed_fragment(horizontal, vertical, depth)

    def release_fragment(self) -> None:
        self._heart.release_fragment()

    def toggle_realistic_heartbeat(self) -> None:
        self._heart.toggle_realistic_heartbeat()

    def toggle_spatial_mode(self) -> None:
        self._heart.toggle_spatial_mode()

    def toggle_interior_view(self) -> None:
        self._heart.toggle_interior_view()

    def toggle_blood_flow(self) -> None:
        self._heart.toggle_blood_flow()

    def toggle_system(self, system: AnatomySystem) -> None:
        self._heart.toggle_system(system)

    def reset_view(self) -> None:
        self._heart.reset_view()

    def tick(self, delta_seconds: float) -> HeartSnapshot:
        self._heart.tick(delta_seconds)
        return self._heart.snapshot()

    def snapshot(self) -> HeartSnapshot:
        return self._heart.snapshot()


class GestureCoordinator:
    """Turns noisy hand observations into deterministic application intentions."""

    def __init__(self, controller: HeartController) -> None:
        self._controller = controller
        self._previous = HandObservation.empty()
        self._gesture = GestureKind.NONE
        self._gesture_started_at = 0.0
        self._gesture_fired = False
        self._fragment_gesture_active = False

    def process(self, observation: HandObservation) -> InteractionResult:
        previous = self._previous
        self._previous = observation
        if observation.hands == 0:
            was_pinching = self._fragment_gesture_active
            self._fragment_gesture_active = False
            self._reset_semantic_gesture(observation)
            return InteractionResult(
                InteractionAction.FRAGMENT_GRAB_ENDED if was_pinching else InteractionAction.NONE
            )

        exploded = self._controller.snapshot().explosion >= 0.45
        if observation.gesture == GestureKind.PINCH:
            self._reset_semantic_gesture(observation)
            if not exploded:
                action = (InteractionAction.FRAGMENT_GRAB_ENDED if self._fragment_gesture_active
                          else InteractionAction.NONE)
                self._fragment_gesture_active = False
                return InteractionResult(action, observation.pointer_x, observation.pointer_y)
            if not self._fragment_gesture_active:
                self._fragment_gesture_active = True
                return InteractionResult(InteractionAction.FRAGMENT_GRAB_STARTED,
                                         observation.pointer_x, observation.pointer_y)
            dx = observation.pointer_x - previous.pointer_x
            dy = observation.pointer_y - previous.pointer_y
            depth = observation.span - previous.span
            if abs(dx) > 0.18 or abs(dy) > 0.18 or abs(depth) > 0.22:
                return InteractionResult(InteractionAction.NONE)
            return InteractionResult(InteractionAction.FRAGMENT_GRAB_MOVED,
                                     observation.pointer_x, observation.pointer_y,
                                     dx, dy, depth)
        if self._fragment_gesture_active:
            self._fragment_gesture_active = False
            self._reset_semantic_gesture(observation)
            return InteractionResult(InteractionAction.FRAGMENT_GRAB_ENDED,
                                     observation.pointer_x, observation.pointer_y)

        if observation.hands >= 2 and previous.hands >= 2:
            delta = observation.span - previous.span
            if abs(delta) < 0.16:
                self._controller.zoom(max(0.88, min(1.12, 1.0 + delta * 2.4)))
        elif (
            observation.gesture in (GestureKind.OPEN_PALM, GestureKind.CLOSED_FIST)
            and previous.hands > 0
            and previous.gesture in (GestureKind.OPEN_PALM, GestureKind.CLOSED_FIST)
        ):
            dx = observation.palm_x - previous.palm_x
            dy = observation.palm_y - previous.palm_y
            if abs(dx) < 0.12 and abs(dy) < 0.12:
                self._controller.rotate(dx * 4.8, dy * 3.6)

        if observation.gesture != self._gesture:
            self._gesture = observation.gesture
            self._gesture_started_at = observation.captured_at
            self._gesture_fired = False

        dwell = observation.captured_at - self._gesture_started_at
        minimum = 0.72 if observation.gesture == GestureKind.POINT else 0.48
        if self._gesture_fired or observation.confidence < 0.66 or dwell < minimum:
            return InteractionResult(InteractionAction.NONE)

        action = InteractionAction.NONE
        if observation.gesture == GestureKind.POINT:
            action = InteractionAction.ACTIVATE_POINTER
        elif observation.gesture == GestureKind.VICTORY:
            self._controller.toggle_explosion()
            action = InteractionAction.EXPLOSION_CHANGED
        elif observation.gesture == GestureKind.THREE_FINGERS:
            self._controller.toggle_realistic_heartbeat()
            action = InteractionAction.REALISM_CHANGED
        elif observation.gesture == GestureKind.PINKY:
            self._controller.toggle_spatial_mode()
            action = InteractionAction.SPATIAL_MODE_CHANGED
        elif observation.gesture == GestureKind.FLOW:
            self._controller.toggle_blood_flow()
            action = InteractionAction.FLOW_CHANGED
        elif observation.gesture == GestureKind.THUMB_DOWN:
            self._controller.toggle_interior_view()
            action = InteractionAction.INTERIOR_VIEW_CHANGED
        elif observation.gesture == GestureKind.THUMB_UP:
            self._controller.focus_whole_heart()
            action = InteractionAction.FOCUS_WHOLE
        if action != InteractionAction.NONE:
            self._gesture_fired = True
        return InteractionResult(action, observation.pointer_x, observation.pointer_y)

    def _reset_semantic_gesture(self, observation: HandObservation) -> None:
        self._gesture = GestureKind.NONE
        self._gesture_started_at = observation.captured_at
        self._gesture_fired = False


class NullHandTracker:
    """No-op adapter used for demos and environments without a camera."""

    def start(self, on_observation, on_status) -> None:
        on_status("CÁMARA DESACTIVADA · MODO MOUSE")

    def stop(self) -> None:
        return None
