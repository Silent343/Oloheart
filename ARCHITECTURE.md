# OloHeart architecture

## Boundaries

- `domain/model.py` contains the `HeartExperience` aggregate, anatomical value
  objects, identifiers, interaction values, and the educational catalog. It has no
  dependency on Tk, OpenCV, MediaPipe, or operating-system APIs.
- `application/controller.py` exposes intention-oriented use cases. Its
  `HandTracker` protocol is a dependency-inversion boundary; the gesture coordinator
  converts noisy frames into stable commands.
- `infrastructure/geometry.py` defines mesh values and reusable primitives;
  `cardiac_atlas.py` composes named anatomical elements, using shells and cusps from
  `sculpted_geometry.py`. `circulation_geometry.py` reuses the same vascular
  centerlines for directional routes. `vision.py` implements the hand-tracking port and guarantees that video
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
- An `AnatomicalElement` has a catalog category, stable element key, and display
  name. Selection and dragging carry this value through the application boundary.
  Manual displacement is keyed by `(part_id, element_key)`, not category alone.
- Three cumulative clocks track venous return, AV passage, and semilunar passage.
  Valve clocks advance only while their corresponding valve state is open. Turning
  the heartbeat off freezes all three clocks.
- Every selectable structure belongs to one of seven anatomical visibility systems.
  Hiding a system also releases or deselects any structure that belongs to it.
- Realistic mode advances through atrial systole, isovolumetric contraction,
  ventricular ejection, isovolumetric relaxation, rapid filling, and diastasis.
  Atrioventricular and semilunar valves follow pressure-compatible states.

## Rendering model

Subvalvular meshes optionally supply seven extra per-vertex attributes: the actual
leaflet opening displacement, its attachment weight, and a systolic support displacement. Only
these meshes allocate the extra GPU attributes. The shader reuses one leaflet
excursion derived from the same membrane surface for valves and chordal endpoints. Papillary tension is a
separate uniform from chamber emptying, allowing tension during isovolumetric
contraction without moving the ventricular shell. Shared endpoints carry identical
attributes; separate component keys preserve independent picking and dragging.

The geometry adapter composes 24 categories and 72 named elements into paired exterior and
interior surfaces. Matching meshes are consolidated into GPU batches. Only the
active surface variant is rendered. The immutable mesh metadata records its view,
opacity, element identity, deformation origin, valve center, radius and leaflet
motion. A component's tissue detail shares its wall identity; unrelated structures
never share a picking key just because their medical category is the same.
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
The presentation resolves a pinch against an off-screen element-color GPU picking
pass, while the application
layer emits acquire, move, and release intentions. The camera fit reserves additional
space in exploded mode. Interior sections are generated in anatomical coordinates,
so they rotate with the heart instead of moving with a screen-space clip plane.

`flow_renderer.py` owns a separate lightweight GPU program. Static path/arrow
geometry is cached; moving marker positions are vectorized. Lines, arrows and
markers require at most three additional draw calls. The flow renderer shares the
anatomy's camera and aggregate transform. It suppresses disconnected/exploded
circulation and honors the systems required by each route. It does not participate
in anatomical picking and cannot intercept a selected structure.

The name leader is projected from the exact element's main visible mesh, and the
full inspector stays on the right. A fixed normal-view camera offset reserves
reading space; selecting a different element does not recenter the scene.

The current geometry is an anatomically informed educational representation rather
than a segmented clinical scan. The external `corazon.glb` referenced by the supplied
prototype prompt was not included. A future glTF or CT/MRI mesh adapter can implement
the same geometry boundary without changing interaction or domain code.
