"""Composition root for the OloHeart native Windows application."""

from __future__ import annotations

import argparse
from pathlib import Path

from oloheart.application.controller import HeartController, NullHandTracker
from oloheart.infrastructure.vision import MediaPipeHandTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OloHeart spatial cardiac workstation")
    parser.add_argument("--windowed", action="store_true", help="Open in a resizable desktop window")
    parser.add_argument("--no-camera", action="store_true", help="Disable hand tracking and use mouse controls")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--smoke-test", action="store_true", help="Validate composition without opening a window")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    controller = HeartController()
    if arguments.smoke_test:
        from oloheart.infrastructure.geometry import build_heart_geometry
        meshes = build_heart_geometry()
        assert len(controller.catalog) == 24
        assert {mesh.part_id for mesh in meshes} == {part.identifier for part in controller.catalog}
        print(f"OloHeart smoke test passed: {len(meshes)} meshes, {len(controller.catalog)} structures")
        return 0

    tracker = (NullHandTracker() if arguments.no_camera else
               MediaPipeHandTracker(project_root / "models" / "hand_landmarker.task", arguments.camera))
    from oloheart.presentation.qt_app import QtDesktopApplication, create_qt_application

    qt_process = create_qt_application()
    application = QtDesktopApplication(
        controller,
        tracker,
        fullscreen=not arguments.windowed,
    )
    if arguments.windowed:
        screen = qt_process.primaryScreen().availableGeometry()
        width = min(1500, max(1120, screen.width() - 80))
        height = min(920, max(700, screen.height() - 80))
        application.resize(width, height)
        application.move(
            screen.x() + max(0, (screen.width() - width) // 2),
            screen.y() + max(0, (screen.height() - height) // 2),
        )
        application.show()
    application.start()
    return qt_process.exec()


if __name__ == "__main__":
    raise SystemExit(main())
