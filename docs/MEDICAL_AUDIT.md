# OloHeart medical-model audit

This audit describes the educational normal-heart model implemented in OloHeart.
It is not a clinical validation and does not make OloHeart a medical device.

## Geometry and interaction revision

The earlier audit verified catalog coverage and phase logic, but did not adequately
validate visual anatomy. A subsequent inspection found generic ventricular shapes
used for pericardium, only two tricuspid leaflets, absent semilunar leaflets, thin
vessel surfaces, and a screen-dependent ventricular cut. These were actual defects.

The current implementation replaces those forms with:

- A thin pericardial envelope and chamber-adherent epicardial surfaces.
- Paired intact and dissected walls for all four chambers, with continuous inner
  lining, a thicker LV wall, and physical cut edges.
- Three tricuspid, two mitral, and three cusps in each semilunar valve.
- Two LV and three RV papillary groups, branching chordae, and wall-attached ridges.
- Great-vessel lumina and annular wall edges.
- A model-space section and shared selection transforms; selecting a structure
  does not move or duplicate it. A moved fragment rotates with the entire model.
- Chamber deformation held constant during isovolumetric phases. Electrical
  emphasis proceeds through atrial, AV, His and ventricular regions.

Additional anatomy references consulted for this revision:

- OpenStax / Rice University, Heart Anatomy:
  https://openstax.org/books/anatomy-and-physiology/pages/19-1-heart-anatomy
- University of Minnesota Visible Heart Laboratory, The Human Heart:
  https://www.vhlab.umn.edu/atlas/physiology-tutorial/the-human-heart.shtml
- NCBI Bookshelf, Anatomy, Thorax, Pericardium:
  https://www.ncbi.nlm.nih.gov/books/NBK482256/

The supplied images showing effusion or inflammation were used to understand layer
relationships, not to introduce those diseases into the normal-heart model. The
current mesh remains procedural, with simplified attachment sites, coronary paths,
microstructure and leaflet/chordal biomechanics. It is not a photorealistic scan or
a validated hemodynamic solver. Continuous fluid simulation and patient-specific
diagnostic measurements are not implemented.

## Anatomical coverage

The 24 selectable structures are assigned to seven independently visible systems:

1. Pericardium: fibrous/serous pericardium and epicardium.
2. Myocardium: the contractile myocardial layer.
3. Chambers and septa: four chambers, both septa, and endocardial lining.
4. Valves and subvalvular apparatus: tricuspid, pulmonary, mitral, and aortic
   valves, plus papillary muscles and chordae tendineae.
5. Great vessels: venae cavae, pulmonary artery, pulmonary veins, and aorta.
6. Coronary circulation: coronary arteries, cardiac veins, and coronary sinus.
7. Conduction system: SA node, AV node, His bundle, bundle branches, and a
   simplified Purkinje network.

The interatrial septum and cardiac veins were added during this audit because the
previous model represented neither as an independently selectable structure.

## Normal flow invariant

The educational route is:

`body -> venae cavae -> right atrium -> tricuspid valve -> right ventricle ->`
`pulmonary valve -> pulmonary artery -> lungs -> pulmonary veins -> left atrium ->`
`mitral valve -> left ventricle -> aortic valve -> aorta -> body`

Coronary arteries originate from the aortic root. Most coronary venous return is
represented as cardiac veins converging on the coronary sinus and then the right
atrium.

## Cardiac-cycle invariant

At 72 beats per minute, the normalized simulation advances through:

1. Atrial systole: atria contract; atrioventricular valves are open.
2. Isovolumetric contraction: ventricles begin contracting; all four valves are
   closed.
3. Ventricular ejection: pulmonary and aortic valves are open; atrioventricular
   valves remain closed.
4. Isovolumetric relaxation: all four valves are closed.
5. Rapid filling: atrioventricular valves reopen; semilunar valves remain closed.
6. Diastasis: passive ventricular filling continues before the next atrial systole.

The shader receives separate atrial and ventricular contraction amplitudes. Valve,
great-vessel, coronary-flow, and conduction highlights follow the current phase.
Coronary highlighting is relative and intentionally stronger in diastole; it is not
a measurement of absolute flow.

## Sources and limits

- NHLBI, *How Blood Flows through the Heart*:
  https://www.nhlbi.nih.gov/health/heart/blood-flow
- NHLBI, *How the Heart Beats*:
  https://www.nhlbi.nih.gov/health/heart/heart-beats
- American Heart Association, *How the Healthy Heart Works*:
  https://www.heart.org/en/health-topics/congenital-heart-defects/about-congenital-heart-defects/how-the-healthy-heart-works
- NCBI Bookshelf, *Physiology, Cardiac Cycle*:
  https://www.ncbi.nlm.nih.gov/books/NBK459327/
- NCBI Bookshelf, *Anatomy, Thorax, Heart and Pericardial Cavity*:
  https://www.ncbi.nlm.nih.gov/books/NBK482452/

Timing fractions, surface deformation, highlight intensity, and reference values are
educational approximations. They do not model patient-specific pressures, pathology,
electromechanical delay, fluid dynamics, leaflet contact, or diagnostic ECG data.
