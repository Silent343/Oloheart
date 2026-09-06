# OloHeart architecture

## Boundaries

- `domain/model.py` contains the `HeartExperience` aggregate, anatomical value
  objects, identifiers, interaction values, and the educational catalog. It has no
  dependency on Tk, OpenCV, MediaPipe, or operating-system APIs.
- `application/controller.py` exposes intention-oriented use cases. Its
  `HandTracker` protocol is a dependency-inversion boundary; the gesture coordinator
  converts noisy frames into stable commands.
- `infrastructure/geometry.py` generates the independently selectable cardiac
  meshes. `vision.py` implements the hand-tracking port and guarantees that video
  pixels do not cross the adapter boundary.
- `presentation/gpu_renderer.py` owns hardware-accelerated OpenGL rendering,
  shader-based tissue lighting, and semantic color picking. `presentation/qt_app.py`
  owns the native Qt desktop layout and input wiring. The former Tk renderer remains
  isolated as a legacy diagnostic implementation and is not used by the composition root.
- `main.py` is the only composition root.

## Domain invariants

- Zoom remains between `0.58×` and `2.15×`.
- Only catalog identifiers can be selected through application use cases.
- Analytical mode is static; switching to realistic tissue owns and enables the
  heartbeat as one domain transition.
- Camera observations contain landmarks, not images.
- Semantic gestures fire once per stable gesture and must be released before they
  can fire again.
- Fragment acquisition is valid only while the exploded target is active; rebuilding
  the heart releases the active fragment and clears every manual displacement.
- Every selectable structure belongs to one of seven anatomical visibility systems.
  Hiding a system also releases or deselects any structure that belongs to it.
- Realistic mode advances through atrial systole, isovolumetric contraction,
  ventricular ejection, isovolumetric relaxation, rapid filling, and diastasis.
  Atrioventricular and semilunar valves follow pressure-compatible states.

## Rendering model

The geometry adapter refines the 24 semantic structures into paired exterior and
interior surfaces. Matching meshes are consolidated into GPU batches. Only the
active surface variant is rendered. The immutable mesh metadata records its view,
opacity and leaflet motion. `sculpted_geometry.py` owns shells, their physical cut
edges, luminal vessel walls, cusps, trabeculae and subvalvular details.
The OpenGL renderer applies the aggregate transform, separate atrial and ventricular
contraction, phase-specific valve and vessel activation,
exploded-layer offsets, camera projection, procedural fiber microvariation, and
per-pixel tissue lighting directly on the GPU. Area-weighted vertex normals,
two-sample multisampling, tone mapping, key/fill/rim illumination, wet specular
response, and subsurface-style Fresnel color make the existing topology read as a
continuous organic surface instead of individually painted polygons. Selection only
changes emphasis and information; it never creates another render instance.
Spatial mode is domain state; the presentation responds by hiding workstation
panels and attaching an anatomy callout to the isolated render.

Exploded fragments use persistent model-space offsets owned by the domain aggregate.
The scene-layout boundary converts hand/mouse deltas from camera to model space.
Rendering and picking use the same placement matrix, independent of selection.
The presentation resolves a pinch against an off-screen semantic-color GPU picking
pass, while the application
layer emits acquire, move, and release intentions. The camera fit reserves additional
space in exploded mode. Interior sections are generated in anatomical coordinates,
so they rotate with the heart instead of moving with a screen-space clip plane.

The current geometry is an anatomically informed educational representation rather
than a segmented clinical scan. The external `corazon.glb` referenced by the supplied
prototype prompt was not included. A future glTF or CT/MRI mesh adapter can implement
the same geometry boundary without changing interaction or domain code.
