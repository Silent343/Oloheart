# OloHeart video reference analysis

This note records the modeling decisions derived from the four user-provided
references. The videos are visual references only; OloHeart geometry remains an
original procedural implementation.

## References

1. [Virtual tour through the heart](https://www.youtube.com/watch?v=LEWvD3ELk7E)
   demonstrates ventricular cavities, atrioventricular valve openings, chordae
   tendineae, papillary supports, trabecular walls, and blood-flow direction.
2. [Photorealistic 3D heart](https://www.youtube.com/watch?v=FgeUzVJ_BVE)
   establishes the target material language: continuous rounded silhouettes,
   low-contrast color variation, soft specular highlights, and subtle surface
   vessels rather than hard polygon outlines.
3. [Heart anatomy](https://www.youtube.com/watch?v=zl-ae3xthVE&t=3s)
   clarifies chamber proportions, septa, four valves, subvalvular support, the
   three helical myocardial orientations, and the double-pump organization.
4. [Human heart structure and function](https://www.youtube.com/watch?v=mC9tYJjmvYQ)
   adds pericardium, epicardium, myocardium, endocardium, coronary sinus, and the
   SA-to-Purkinje conduction system.

## Geometry decisions

- Preserve a continuous external silhouette in the intact view. Educational
  internal layers are hidden until selected or exploded.
- Represent the left ventricular wall as the dominant muscular volume and keep
  the right ventricle thinner and more anterior.
- Model valve leaflets separately from their annuli. Chordae form branching fans
  between the atrioventricular leaflets and papillary muscles.
- Add explicit pericardial, epicardial, myocardial, and endocardial structures.
- Use helical fiber paths for myocardium instead of baking every fiber into the
  external surface.
- Keep coronary arteries and the coronary sinus close to epicardial grooves.
- Represent the conduction system as an educational overlay with SA node, AV
  node, His bundle, bundle branches, and ventricular terminal paths.

## Polygon and performance budget

The generated model contains 65 meshes, 22 semantic structures, 5,264 vertices,
and 14,578 high-detail triangles. Hidden educational layers keep the intact view at
9,108 source triangles. Adaptive vertex clustering targets 2,694 source triangles
during hand-driven motion and 5,787 during heartbeat animation before back-face
culling.

High detail is restored after motion settles. Thirteen reusable heartbeat phases
avoid rasterizing the same systolic shapes on every cycle; a local 1440×800 QA run
stabilized at 32 FPS after the first cycle. This concentrates polygons in curved
chambers, valve annuli, great vessels, and tissue layers while preserving camera
tracking responsiveness.
