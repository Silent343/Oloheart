# OloHeart

> **Proprietary software — Copyright © 2026 Silent343. All rights reserved.**
> Cloning and local execution are permitted only for personal, non-commercial
> evaluation and educational demonstration. Redistribution, modification,
> commercial use, and claiming the project as one's own are prohibited. See the
> [OloHeart Proprietary Software License](LICENSE) and [ownership notice](NOTICE.md).

OloHeart is a native Windows cardiac-anatomy workstation controlled with hands,
mouse, or keyboard. It does not host a web page and it never displays the camera
feed. The camera is used only to calculate hand landmarks in memory.

The application uploads paired exterior/dissection meshes to the GPU and renders
them with continuous normals, multisampling, per-pixel tissue lighting, subtle
myocardial fibers, subsurface rim color, and wet-specular response. The meshes span
24 catalog categories and 72 independently named elements, including every chamber, great vessel and valve, the wall layers,
interatrial and interventricular septa, coronary arteries, cardiac veins and sinus, papillary muscles, chordae
tendineae, valve leaflets, ventricular trabeculae, and cardiac conduction. A selected
structure is highlighted at its anatomical position. Selection never adds a second
instance, recenters a fragment, or changes the shared rotation. Clear selection with
the toolbar, an empty-space click, or the thumb-up gesture.

## Start

Clone and start OloHeart from Command Prompt. The launcher creates the isolated
Python environment and installs the dependencies automatically on its first run:

```bat
git clone https://github.com/Silent343/Oloheart.git
cd Oloheart
start-oloheart.cmd
```

Later runs use the same `start-oloheart.cmd` command and start immediately. Any
application option can be appended to it, for example:

```bat
start-oloheart.cmd --windowed --no-camera
```

Python 3.12 or newer, Git, Windows, and an OpenGL-capable graphics adapter are
required. The camera is optional when `--no-camera` is used.

The equivalent one-time setup from PowerShell is:

```powershell
.\setup.ps1
```

Then double-click `OloHeart.vbs`. Diagnostic modes are also available:

```powershell
python run.py --windowed --no-camera
python run.py --smoke-test
```

`Esc` closes the application and `F11` toggles full screen.

## Spatial controls

| Input | Result |
| --- | --- |
| Open palm or closed fist + movement | Rotate the heart |
| Two hands moving apart/together | Zoom |
| Point and hold for 0.7 seconds | Select a structure |
| Victory gesture | Separate or reconstruct anatomical layers |
| Thumb + index pinch in exploded view | Grab, move, and release one anatomical fragment |
| Move a pinched hand closer/farther | Move the fragment through visual depth |
| Index + middle + ring fingers | Switch between analytical and realistic beating modes |
| Pinky finger only | Enter or leave the full-canvas spatial view |
| Thumb down held briefly / `I` | Toggle the anatomical interior section |
| Index + pinky held briefly / `F` | Toggle directional blood-flow paths and moving markers |
| Thumb up | Clear the anatomical selection |
| Mouse drag / wheel / click | Rotation, zoom, and selection fallback |
| `E` / `R` / `S` / `Home` | Layers, realism, spatial view, and reset fallback |

Three-finger and pinky gestures use stable finger-extension geometry rather than a
fast transient motion. The pinch is available only after layer separation and moves
one picked element at a time, rather than its entire catalog category. The four
pulmonary veins, two venae cavae, arterial branches, papillary muscles, and chordal
fans have distinct identities. Reconstructing the heart clears all manual
offsets. Lighting conditions and camera frame rate still affect recognition, so
keyboard and mouse controls remain available for precision work. In exploded view,
mouse dragging over a fragment provides the same fallback interaction.

## Anatomical scope

### Selection and reading space

The OpenGL canvas extends behind a floating native inspector on the right; the
inspector no longer consumes a fixed sidebar column. Selecting any structure
retains its full description, physiological function, anatomical relations, and
reference metrics in normal and spatial views. A dashed leader follows its main
visible mesh through rotation and exploded-view dragging without moving or
duplicating the selected anatomy. The inspector stays in a stable reading position.
A name label identifies the exact picked element beside the leader. Its text can
be scrolled with the mouse wheel or with pointing dwell on the two
text-navigation buttons. Pointer and pinch picking cannot pass through the card.
In spatial mode, clearing selection hides the card and leaves the canvas alone.

Run `.venv/Scripts/python.exe tests/verify_callout.py` for the explicit desktop
acceptance checks and screenshots. This test uses no camera.

### Atlas and circulation

The exterior ventricular envelope has a shared continuous contour and one LV apex;
the two ventricular regions retain independent picking identities. Myocardial and
epicardial exterior layers use this same contour. Curved vessels use transported
cross-section frames and 36-sided lumina. The interior opens physical windows in
the aortic and pulmonary roots. An oblique AV plane makes the mitral/tricuspid
leaflets visible, while their chordal endpoints follow leaflet excursion.

`F` opens the interior and enables the heartbeat and 17 directional flow paths.
Blue means lower oxygen content, not literally blue blood; red means higher oxygen
content. Venous return remains visible while valve-crossing paths advance only in
their appropriate phase. Arrows on inactive paths dim. The flow display is hidden
while the heart is exploded or any fragment has a manual offset, so separated
anatomy is not presented as a connected circulation. Reconstruct to resume it.

These are explanatory markers, not computed velocities or a fluid-volume solver.
The pulmonary and systemic circulation outside the heart is not geometrically
modeled. Valve, wall, and electrical animations remain educational approximations.

Run `python -m unittest discover -s tests -q` with `PYTHONPATH=src` for unit checks.
Run `.venv/Scripts/python.exe tests/verify_atlas.py` for camera-free desktop
acceptance, exact GPU picking, leaflet opening/closure probes, and screenshots in
the ignored `artifacts/` directory. FPS values from this test are local observations,
not performance guarantees with camera tracking enabled.

The interior view opens all four chambers in model coordinates. The cut rotates
with the anatomy and has real outer/inner surfaces and sealed wall-thickness edges.
The LV wall is thicker than the RV wall. Trabecular ridges, atrial pectinate ridges,
papillary groups and chordae remain associated with their chambers. Great vessels
have a luminal wall and annular ends. Two mitral and three other valve leaflets are
generated separately and deform with the simulated valve state.

Pericardium is represented as a thin surrounding sac with a transparent exterior and
an open anatomical section. Epicardium follows individual chamber surfaces. The
pericardial cavity is a potential space, not a solid organ or a separate disk.

OloHeart exposes seven independently visible systems: pericardium, myocardium,
chambers and septa, valves and subvalvular apparatus, great vessels, coronary
circulation, and the cardiac conduction system. It includes both atria, both
ventricles, their visible internal support
structures, aorta and branches, pulmonary artery, venae cavae, pulmonary veins,
four valves, both cardiac septa, pericardium, epicardium, myocardium,
endocardium, coronary arteries, cardiac veins and sinus, and the conduction system.
Realistic mode follows an educational pressure-driven sequence: atrial systole,
isovolumetric ventricular contraction, ejection, isovolumetric relaxation, rapid
filling, and diastasis. The panel reports the simulated valve state and relative
coronary perfusion. It is not a medical device and must not be used for diagnosis
or patient monitoring.

## Privacy

MediaPipe processes camera frames locally. The vision adapter emits only normalized
coordinates, gesture names, and confidence values. Pixels are never sent to the UI,
written to disk, or transmitted over a network.

## Engineering

The code follows Domain-Driven Design and Clean Architecture:

```text
presentation/native Qt + OpenGL
             ↓ intentions / snapshots
application/use-case controllers
             ↓
domain/heart aggregate + anatomy values
             ↑
infrastructure/procedural meshes + MediaPipe
```

Domain code has no GUI, camera, or numerical-library dependencies. Infrastructure
implements the camera boundary, while presentation owns GPU picking and native
drawing. Source comments and documentation are written in English.

The supplied HTML, prompt, and anatomical images are references only.
OloHeart does not execute or embed that page. The prompt expects an external
segmented `corazon.glb`; because that asset was not included, the current renderer
uses the procedural anatomy through the same presentation boundary. It is not a
photorealistic clinical scan. See [the medical-model audit](docs/MEDICAL_AUDIT.md)
for sources, modeled relationships, and explicit limitations.
