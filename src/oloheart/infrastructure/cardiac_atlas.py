"""Parametric cardiac atlas with explicit ownership and anatomical connections.

Anterior is +Z and anatomical left is +X. Every detachable named structure has
its own element key. Tissue detail shares the chamber's deformation origin, while
valves and subvalvular structures retain their own picking/drag identities.
"""

from __future__ import annotations

import math
from dataclasses import replace

from oloheart.domain.model import HeartPartId as I, build_anatomy_catalog
from oloheart.infrastructure.geometry import AnatomicalMesh, Vec3, _tube, _torus, _ellipsoid
from oloheart.infrastructure.sculpted_geometry import (
    CHAMBERS, VALVES, _contour, _point, _shell, _hollow_vessel, _leaflets,
)

NAMES = {part.identifier: part.display_name for part in build_anatomy_catalog()}
EXPLOSION = {
    I.RIGHT_ATRIUM: Vec3(-0.65, 0.40, -0.05),
    I.LEFT_ATRIUM: Vec3(0.65, 0.42, -0.32),
    I.RIGHT_VENTRICLE: Vec3(-0.60, -0.26, 0.20),
    I.LEFT_VENTRICLE: Vec3(0.60, -0.28, -0.05),
    I.TRICUSPID_VALVE: Vec3(-0.60, 0.12, 0.20),
    I.MITRAL_VALVE: Vec3(0.60, 0.12, -0.05),
    I.AORTIC_VALVE: Vec3(0.24, 0.52, -0.14),
    I.PULMONARY_VALVE: Vec3(-0.24, 0.55, 0.40),
}
ZERO = Vec3(0, 0, 0)
VENTRICULAR_ORIGIN = Vec3(.04,.46,-.02)
MUSCLE = ("#249ba5", "#ae3948")
LINING = ("#56aeb3", "#cd786b")
ARTERIAL = ("#2ab1bd", "#c96763")
VENOUS = ("#337fbe", "#6c87a2")
VALVULAR = ("#65c8c9", "#eed2b6")

# The vessel centerlines are also consumed by the circulation visualization.
# Endpoints terminate in the relevant atrium or ventricular outflow tract.
VESSELS = (
    (I.AORTA, "aorta", "Aorta ascendente, arco y descendente", 0.174, ARTERIAL,
     Vec3(0.20, 0.72, -0.22),
     ((0.075,0.59,0.005),(0.08,0.97,-0.015),(-0.01,1.36,-0.05),
      (0.12,1.68,-0.09),(0.40,1.73,-0.20),(0.68,1.48,-0.39),
      (0.71,0.92,-0.53),(0.65,0.10,-0.65),(0.61,-0.69,-0.61))),
    (I.AORTA, "brachiocephalic", "Tronco braquiocefálico", 0.080, ARTERIAL,
     Vec3(-0.15,0.92,-0.25), ((0.04,1.55,-0.05),(-0.06,1.91,-0.03),(-0.20,2.07,-0.02))),
    (I.AORTA, "left_carotid", "Arteria carótida común izquierda", 0.061, ARTERIAL,
     Vec3(0.10,1.04,-0.26), ((0.29,1.69,-0.13),(0.29,1.96,-0.15),(0.34,2.10,-0.15))),
    (I.AORTA, "left_subclavian", "Arteria subclavia izquierda", 0.068, ARTERIAL,
     Vec3(0.44,1.0,-0.30), ((0.53,1.63,-0.28),(0.65,1.88,-0.35),(0.80,1.98,-0.40))),
    (I.PULMONARY_ARTERY, "pulmonary_trunk", "Tronco pulmonar", 0.178, VENOUS,
     Vec3(-0.20,0.62,0.45), ((-0.19,0.63,0.31),(-0.14,0.94,0.34),
                            (-0.02,1.23,0.27),(0.17,1.37,0.02))),
    (I.PULMONARY_ARTERY, "left_pulmonary_artery", "Arteria pulmonar izquierda", 0.122, VENOUS,
     Vec3(0.66,0.63,0.08), ((0.13,1.35,0.01),(0.44,1.39,-0.07),(0.87,1.35,-0.18))),
    (I.PULMONARY_ARTERY, "right_pulmonary_artery", "Arteria pulmonar derecha", 0.120, VENOUS,
     Vec3(-0.76,0.60,-0.15), ((0.13,1.35,-0.05),(-0.13,1.25,-0.30),
                             (-0.50,1.20,-0.40),(-0.96,1.18,-0.35))),
    (I.VENA_CAVA, "superior_vena_cava", "Vena cava superior", 0.154, VENOUS,
     Vec3(-0.70,0.42,-0.24), ((-0.67,1.72,-0.22),(-0.68,1.35,-0.18),(-0.56,0.99,-0.17))),
    (I.VENA_CAVA, "inferior_vena_cava", "Vena cava inferior", 0.160, VENOUS,
     Vec3(-0.77,-0.27,-0.40), ((-0.63,-0.71,-0.42),(-0.68,-0.11,-0.38),
                             (-0.64,0.37,-0.31),(-0.52,0.59,-0.18))),
    (I.PULMONARY_VEINS, "right_superior_pulmonary_vein", "Vena pulmonar superior derecha", 0.091, ARTERIAL,
     Vec3(-0.60,0.26,-0.63), ((-0.90,0.94,-0.46),(-0.42,0.95,-0.54),(0.11,0.94,-0.50))),
    (I.PULMONARY_VEINS, "right_inferior_pulmonary_vein", "Vena pulmonar inferior derecha", 0.087, ARTERIAL,
     Vec3(-0.61,-0.04,-0.65), ((-0.86,0.66,-0.46),(-0.41,0.68,-0.53),(0.14,0.68,-0.47))),
    (I.PULMONARY_VEINS, "left_superior_pulmonary_vein", "Vena pulmonar superior izquierda", 0.089, ARTERIAL,
     Vec3(0.85,0.27,-0.60), ((0.98,1.02,-0.40),(0.77,1.00,-0.46),(0.61,0.97,-0.47))),
    (I.PULMONARY_VEINS, "left_inferior_pulmonary_vein", "Vena pulmonar inferior izquierda", 0.086, ARTERIAL,
     Vec3(0.86,-0.02,-0.63), ((1.01,0.72,-0.36),(0.79,0.70,-0.42),(0.61,0.72,-0.44))),
)


def _owned(mesh: AnatomicalMesh, key: str, name: str = "", center: Vec3 | None = None,
           **changes) -> AnatomicalMesh:
    return replace(mesh, element_key=key, element_name=name or NAMES[mesh.part_id],
                   deformation_center=center, **changes)


def _envelope_point(t: float, angle: float) -> Vec3:
    """Continuous epicardial silhouette with one LV apex and a broad RV shoulder."""
    width = _contour(t,"lv")
    organic = 1 + .023*math.cos(angle*3+t*5)*math.sin(math.pi*t)
    return Vec3(-.02+.55*t+.92*width*math.cos(angle)*organic,
                .46-1.70*t,
                -.025+.61*width*math.sin(angle)*organic+.025*math.sin(math.pi*t))


def _ventricular_envelope() -> list[AnatomicalMesh]:
    """Partition one common surface into adjacent, independently selectable walls."""
    rows,columns = 64,96
    vertices = tuple(_envelope_point(r/rows,c*math.tau/columns)
                     for r in range(rows+1) for c in range(columns))
    groups = {I.LEFT_VENTRICLE:[],I.RIGHT_VENTRICLE:[]}
    for r in range(rows):
        for c in range(columns):
            a,b = r*columns+c,r*columns+(c+1)%columns
            d,e = a+columns,b+columns
            angle = (c+.5)*math.tau/columns
            # RV occupies the anterior-right surface; LV continues to the apex.
            right = math.cos(angle) < .20*math.sin(angle)-.04 and r/rows < .88
            part = I.RIGHT_VENTRICLE if right else I.LEFT_VENTRICLE
            groups[part].extend(((a,d,b),(b,d,e)))
    meshes = []
    for part,faces in groups.items():
        used = sorted({v for face in faces for v in face})
        mapping = {v:n for n,v in enumerate(used)}
        mesh = AnatomicalMesh(part,tuple(vertices[v] for v in used),
                              tuple(tuple(mapping[v] for v in face) for face in faces),
                              *MUSCLE,EXPLOSION[part],True,True,"exterior")
        meshes.append(_owned(mesh,part.value,center=VENTRICULAR_ORIGIN))
    return meshes


def _windowed_vessel(mesh: AnatomicalMesh) -> AnatomicalMesh:
    """Clip a smooth root window, sharing edge intersections and sealing thickness."""
    from collections import Counter
    sides = mesh.faces[0][1]
    vertices = list(mesh.vertices)
    centers = []
    for first in range(0,len(vertices),sides):
        ring = vertices[first:first+sides]
        centers.append(Vec3(sum(v.x for v in ring)/sides,sum(v.y for v in ring)/sides,
                            sum(v.z for v in ring)/sides))
    vertex_centers = [centers[n//sides] for n in range(len(vertices))]
    distances = [max(v.y-1.08,vertex_centers[n].z-v.z) for n,v in enumerate(vertices)]
    intersections = {}
    faces = []
    for face in mesh.faces:
        polygon = []
        for a,b in zip(face,(*face[1:],face[0])):
            da,db = distances[a],distances[b]
            if da >= 0:
                polygon.append(a)
            if (da >= 0) != (db >= 0):
                edge = tuple(sorted((a,b)))
                if edge not in intersections:
                    weight = da/(da-db)
                    intersections[edge] = len(vertices)
                    vertices.append(vertices[a]*(1-weight)+vertices[b]*weight)
                    vertex_centers.append(vertex_centers[a]*(1-weight)+vertex_centers[b]*weight)
                polygon.append(intersections[edge])
        faces.extend((polygon[0],polygon[n],polygon[n+1]) for n in range(1,len(polygon)-1))
    count = len(vertices)
    vertices += [c+(v-c)*.82 for v,c in zip(vertices,vertex_centers)]
    edges = Counter(tuple(sorted(edge)) for a,b,c in faces for edge in ((a,b),(b,c),(c,a)))
    outer = tuple(faces)
    faces.extend((a+count,c+count,b+count) for a,b,c in outer)
    for (a,b),occurrences in edges.items():
        if occurrences == 1:
            faces.extend(((a,b,a+count),(b,b+count,a+count)))
    return replace(mesh,vertices=tuple(vertices),faces=tuple(faces),view="interior")


def _tilt_valve(point: Vec3, center: Vec3, angle: float = .48) -> Vec3:
    """Oblique AV plane; the same local frame is used for leaflet animation."""
    p = point-center
    c,s = math.cos(angle),math.sin(angle)
    return center+Vec3(p.x,p.y*c-p.z*s,p.y*s+p.z*c)


def _wall_details(part: I) -> list[AnatomicalMesh]:
    """Embed a branching trabecular relief in the endocardial wall."""
    top, height, radii, tilt, thickness = CHAMBERS[part]
    kind = "rv" if part == I.RIGHT_VENTRICLE else "lv"
    meshes = []
    center = VENTRICULAR_ORIGIN
    for n in range(19):
        angle = math.pi + (n + 0.5) * math.pi / 19
        path = []
        for k in range(16):
            t = 0.18 + (0.68 + 0.035 * math.sin(n)) * k / 15
            a = angle + 0.055 * math.sin(t * 15 + n) + 0.035 * math.cos(t*23-n)
            path.append(_point(top,height,radii,tilt,t,a,kind,thickness-0.004))
        ridge = _tube(part,path,0.017 if kind == "rv" else 0.013,LINING,
                      EXPLOSION[part],8,True,False,curve_steps=2,taper=0.50)
        meshes.append(_owned(ridge,part.value,center=center,view="interior"))
    for n in range(38):
        t = 0.23 + (n % 8) * 0.079
        angle = math.pi + 0.16 + (n // 8) * 0.52
        path = [_point(top,height,radii,tilt,t+0.025*math.sin(k*0.5+n),
                       angle+k*0.065,kind,thickness-0.008) for k in range(8)]
        ridge = _tube(part,path,0.010,LINING,EXPLOSION[part],7,True,False,taper=0.6)
        meshes.append(_owned(ridge,part.value,center=center,view="interior"))
    return meshes


def _subvalvular() -> list[AnatomicalMesh]:
    """Separate each papillary muscle and each chordal fan; anchor both ends."""
    meshes = []
    for chamber, valve, muscle_names in (
        (I.LEFT_VENTRICLE,I.MITRAL_VALVE,("anterolateral","posteromedial")),
        (I.RIGHT_VENTRICLE,I.TRICUSPID_VALVE,("anterior","posterior","septal")),
    ):
        top, height, radii, tilt, thickness = CHAMBERS[chamber]
        kind = "lv" if chamber == I.LEFT_VENTRICLE else "rv"
        side = "izquierdo" if kind == "lv" else "derecho"
        valve_center, radius, _, _ = VALVES[valve]
        for n, name in enumerate(muscle_names):
            angle = math.pi * (1.17 + n * 0.63 / max(1,len(muscle_names)-1))
            base = _point(top,height,radii,tilt,0.66,angle,kind,thickness-0.025)
            tip = Vec3(valve_center.x + 0.15*math.cos(angle),
                       -0.10 - 0.055*(n % 2), valve_center.z - 0.11 + 0.08*n)
            midpoint = base*0.48 + tip*0.52
            muscle = _tube(I.PAPILLARY_MUSCLES,[base,midpoint,tip],0.064,
                           LINING,EXPLOSION[chamber],20,True,False,curve_steps=12,taper=0.68)
            key = f"papillary_{chamber.value}_{n}"
            meshes.append(_owned(muscle,key,f"Músculo papilar {name} · ventrículo {side}",VENTRICULAR_ORIGIN))
            # Bifurcating thin cords attach along adjacent leaflet free edges.
            # The endpoint metadata lets the shader follow leaflet excursion.
            for fan in range(3):
                fork = tip*0.45 + Vec3(valve_center.x+(fan-1)*0.065,
                                     valve_center.y-0.12,valve_center.z-0.03)*0.55
                cord_key = f"chordae_{chamber.value}_{n}_{fan}"
                cord_name = f"Cuerdas tendinosas · {NAMES[valve].lower()} · fascículo {n*3+fan+1}"
                paths = [[tip,fork]]
                for twig in range(3):
                    theta = math.tau*(n+fan*0.27+twig*0.10)/len(muscle_names)
                    edge = Vec3(valve_center.x+radius*0.28*math.cos(theta),
                                valve_center.y-0.022,valve_center.z+radius*0.28*math.sin(theta))
                    paths.append([fork,_tilt_valve(edge,valve_center)])
                for path in paths:
                    cord = _tube(I.CHORDAE_TENDINEAE,path,0.0045,VALVULAR,
                                 EXPLOSION[chamber],6,False,False)
                    meshes.append(_owned(cord,cord_key,cord_name,VENTRICULAR_ORIGIN,motion="chordal",
                                         valve_center=valve_center,valve_radius=radius))
    return meshes


def _surface(x: float, y: float, back: bool = False) -> Vec3:
    """Project coronary centerlines onto the assembled epicardial contour."""
    t = max(0,min(.965,(.46-y)/1.70))
    u = (x+.02-.55*t)/max(.015,.92*_contour(t,"lv"))
    angle = math.acos(max(-.965,min(.965,u)))
    if back:
        angle = math.tau-angle
    p = _envelope_point(t,angle)
    return Vec3(p.x,p.y,p.z+(-.008 if back else .008))


def _coronaries() -> list[AnatomicalMesh]:
    meshes = []
    for part,key,name,side,back,color in (
        (I.CORONARY_ARTERIES,"lad","Arteria interventricular anterior",1,False,("#4fc4c2","#ca514b")),
        (I.CORONARY_ARTERIES,"right_coronary","Coronaria derecha y rama marginal",-1,False,("#4fc4c2","#ca514b")),
        (I.CORONARY_ARTERIES,"circumflex","Arteria circunfleja",1,True,("#4fc4c2","#ca514b")),
        (I.CORONARY_VEINS,"great_cardiac_vein","Vena cardíaca magna",1,False,("#2f91b8","#699cb4")),
        (I.CORONARY_VEINS,"middle_cardiac_vein","Vena cardíaca media",-1,True,("#2f91b8","#699cb4")),
    ):
        path = []
        vein = part == I.CORONARY_VEINS
        for k in range(25):
            t = k/24
            x = (0.015+0.48*t) if side == 1 else (-0.67+0.94*t)
            if vein:
                x += 0.035
            if key == "circumflex":
                # The circumflex follows the AV groove, not a second LAD path.
                p = _envelope_point(.025+.035*math.sin(math.pi*t),.20-2.75*t)
                path.append(Vec3(p.x,p.y,p.z-.009))
            else:
                path.append(_surface(x,0.41-1.50*t,back))
        if not vein:
            root = Vec3(0.04 if side == 1 else -0.06,0.67,-0.13 if back else 0.12)
            path = [root,*path]
        tree = [(path,0.021 if not vein else 0.018)]
        for n in range(6):
            start = 3+n*3
            a = path[start]
            branch = [a]
            direction = -1 if (n+side) % 2 else 1
            for j in range(1,10):
                t = j/9
                descent = .63 if key == "circumflex" else .24
                branch.append(_surface(a.x+direction*0.24*t,a.y-descent*t+0.025*math.sin(t*5),back))
            tree.append((branch,0.010))
        for points,radius in tree:
            mesh = _tube(part,points,radius,color,ZERO,10,False,True,curve_steps=3,taper=0.70)
            meshes.append(_owned(mesh,key,name,VENTRICULAR_ORIGIN))
    sinus = _tube(I.CORONARY_SINUS,
                  [Vec3(0.64,0.38,-0.48),Vec3(0.28,0.37,-0.61),Vec3(-0.20,0.40,-0.57),
                   Vec3(-0.47,0.54,-0.24)],0.040,VENOUS,Vec3(0,-0.10,-0.36),18,False,False,curve_steps=9)
    meshes.append(_owned(sinus,"coronary_sinus"))
    return meshes


def _conduction() -> list[AnatomicalMesh]:
    """Show SA, AV, His, bundle branches and a subendocardial Purkinje network."""
    color = ("#52c7c6","#e4c077")
    sa, av = Vec3(-0.60,0.96,-0.12), Vec3(-0.04,0.47,-0.18)
    meshes = []
    for key,name,center in (("sa_node","Nodo sinoauricular",sa),("av_node","Nodo auriculoventricular",av)):
        meshes.append(_owned(_ellipsoid(I.CARDIAC_CONDUCTION,center,Vec3(.023,.034,.018),
                                        color,ZERO,16,24,False,False),key,name))
    paths = [
        ("atrial_conduction","Conducción auricular",[sa,Vec3(-.39,.77,-.32),av]),
        ("bachmann","Haz interauricular de Bachmann",[sa,Vec3(-.15,1.03,-.20),Vec3(.40,.99,-.27)]),
        ("his_bundle","Haz de His",[av,Vec3(.04,.26,-.18),Vec3(.08,.03,-.20)]),
        ("right_bundle","Rama derecha del haz de His",[Vec3(.08,.03,-.20),Vec3(-.06,-.34,-.22),Vec3(.14,-.83,-.03)]),
        ("left_bundle","Rama izquierda del haz de His",[Vec3(.08,.03,-.20),Vec3(.26,-.41,-.27),Vec3(.46,-1.01,-.13)]),
    ]
    for part in (I.RIGHT_VENTRICLE,I.LEFT_VENTRICLE):
        top,h,radii,tilt,thickness = CHAMBERS[part]
        kind = "rv" if part == I.RIGHT_VENTRICLE else "lv"
        for n in range(6):
            a = math.pi+(n+.5)*math.pi/6
            path = [_point(top,h,radii,tilt,.90-k*.06,a+.06*math.sin(k),kind,thickness+.015)
                    for k in range(11)]
            paths.append((f"purkinje_{kind}",f"Red de Purkinje · {NAMES[part].lower()}",path))
    for key,name,path in paths:
        center = VENTRICULAR_ORIGIN if key in {"right_bundle","left_bundle","purkinje_lv","purkinje_rv"} else None
        meshes.append(_owned(_tube(I.CARDIAC_CONDUCTION,path,.0058,color,ZERO,7,
                                    False,False,curve_steps=5),key,name,center))
    return meshes


def build_cardiac_atlas() -> tuple[AnatomicalMesh, ...]:
    """Build paired exterior/interior anatomy with smooth vascular surfaces."""
    meshes = []
    envelope = _ventricular_envelope()
    meshes.extend(envelope)
    for part,(top,height,radii,tilt,thickness) in CHAMBERS.items():
        atrium = part in {I.RIGHT_ATRIUM,I.LEFT_ATRIUM}
        kind = "atrium" if atrium else ("rv" if part == I.RIGHT_VENTRICLE else "lv")
        for section in (False,True):
            wall = _shell(part,top,height,radii,tilt,thickness,MUSCLE,EXPLOSION[part],kind,section)
            deform = top if atrium else VENTRICULAR_ORIGIN
            if atrium or section:
                meshes.append(_owned(wall,part.value,center=deform))
            if not atrium:
                muscle_wall = wall if section else next(m for m in envelope if m.part_id == part)
                meshes.append(_owned(replace(muscle_wall,part_id=I.MYOCARDIUM,default_visible=False),
                                     f"myocardium_{part.value}",f"Miocardio · {NAMES[part].lower()}",deform))
            lining = _shell(I.ENDOCARDIUM,top,height,
                            (radii[0]*(1-thickness-.007),radii[1]*(1-thickness-.007)),
                            tilt,.010,LINING,EXPLOSION[part],kind,section,False)
            meshes.append(_owned(lining,f"endocardium_{part.value}",f"Endocardio · {NAMES[part].lower()}",deform))
            epicardium = _shell(I.EPICARDIUM,top,height,(radii[0]*1.007,radii[1]*1.007),
                               tilt,.006,("#82c7ba","#e8bbaa"),EXPLOSION[part],kind,section,False,.20)
            if not atrium and not section:
                surface = next(m for m in envelope if m.part_id == part)
                epicardium = replace(surface,part_id=I.EPICARDIUM,
                    vertices=tuple(VENTRICULAR_ORIGIN+(v-VENTRICULAR_ORIGIN)*1.007 for v in surface.vertices),
                    analysis_color="#82c7ba",realistic_color="#e8bbaa",opacity=.20,default_visible=False)
            meshes.append(_owned(epicardium,f"epicardium_{part.value}",f"Epicardio · {NAMES[part].lower()}",deform))
        if not atrium:
            meshes.extend(_wall_details(part))
        # Auricles are flattened, corrugated appendages, not round upper spheres.
        if atrium:
            left = part == I.LEFT_ATRIUM
            center = Vec3(.64 if left else -.59,.87,.10 if left else .11)
            auricle = _ellipsoid(part,center,Vec3(.23,.16,.12),MUSCLE,EXPLOSION[part],26,40)
            meshes.append(_owned(auricle,f"auricle_{part.value}","Orejuela izquierda" if left else "Orejuela derecha",top,view="exterior"))
            if not left:
                for n in range(8):
                    path = [_point(top,height,radii,tilt,.27+k*.04,math.pi+.15+n*.105,
                                   kind,thickness-.012) for k in range(12)]
                    mesh = _tube(part,path,.010,LINING,EXPLOSION[part],8,True,False)
                    meshes.append(_owned(mesh,part.value,center=top,view="interior"))
    # Septal cross-sections follow the shared cavity boundary and join at the apex.
    for part,top,h,radii,tilt,kind in (
        (I.INTERVENTRICULAR_SEPTUM,Vec3(.015,.47,-.055),1.63,(.080,.39),.42,"septum"),
        (I.INTERATRIAL_SEPTUM,Vec3(-.01,1.07,-.24),.61,(.037,.26),.02,"atrium"),
    ):
        for section in (False,True):
            mesh = _shell(part,top,h,radii,tilt,.92,LINING,Vec3(0,-.12,-.28),kind,section,False)
            meshes.append(_owned(mesh,part.value,center=VENTRICULAR_ORIGIN if part == I.INTERVENTRICULAR_SEPTUM else top))
    # A shallow fossa on the atrial septum is a landmark, not a patent shunt.
    fossa = _ellipsoid(I.INTERATRIAL_SEPTUM,Vec3(-.052,.77,-.17),Vec3(.014,.085,.07),
                       ("#52adb5","#d69a88"),Vec3(0,-.12,-.28),18,28,True,False)
    meshes.append(_owned(fossa,"fossa_ovalis","Fosa oval",view="interior"))
    for section in (False,True):
        sac = _shell(I.PERICARDIUM,Vec3(0,1.30,-.07),2.62,(1.13,.87),.24,.016,
                     ("#82c5cb","#dfc3b3"),ZERO,"sac",section,True,.52 if section else .18)
        meshes.append(_owned(sac,"pericardium"))
    for part,key,name,radius,color,explosion,path in VESSELS:
        tube = _tube(part,[Vec3(*p) for p in path],radius,color,explosion,36,False,True,curve_steps=12)
        intact = _hollow_vessel(tube)
        if key in {"aorta","pulmonary_trunk"}:
            intact = replace(intact,view="exterior")
            meshes.append(_owned(_windowed_vessel(tube),key,name))
        meshes.append(_owned(intact,key,name))
    for part,(center,radius,count,semilunar) in VALVES.items():
        ring = _torus(part,center,radius,.010,VALVULAR,EXPLOSION[part],tilt=0 if semilunar else .48,default_visible=False)
        meshes.append(_owned(ring,part.value))
        leaf = _leaflets(part,center,radius,count,semilunar,EXPLOSION[part])
        if not semilunar:
            leaf = replace(leaf,vertices=tuple(_tilt_valve(v,center) for v in leaf.vertices))
        meshes.append(_owned(replace(leaf,realistic_color=VALVULAR[1]),part.value,
                             valve_center=center,valve_radius=radius))
    meshes.extend(_subvalvular())
    meshes.extend(_coronaries())
    meshes.extend(_conduction())
    return tuple(meshes)
