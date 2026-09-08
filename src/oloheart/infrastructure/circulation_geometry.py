"""Illustrative circulation centerlines shared with the anatomical vessel atlas.

These paths explain direction and valve gating; they are not a CFD solution.
Blue denotes lower oxygen content, not the physical color of venous blood.
"""

from dataclasses import dataclass
import numpy as np

from oloheart.domain.model import HeartSnapshot, HeartPartId as I
from oloheart.infrastructure.geometry import Vec3, _curve_points
from oloheart.infrastructure.cardiac_atlas import VESSELS
from oloheart.infrastructure.sculpted_geometry import VALVES


@dataclass(frozen=True, slots=True)
class FlowRoute:
    key: str
    oxygenated: bool
    gate: int
    parts: tuple[I, ...]
    points: np.ndarray

    def active(self, state: HeartSnapshot) -> bool:
        """A closed valve never admits moving flow markers."""
        return (self.gate == 0 or
                (self.gate == 1 and state.cycle.atrioventricular_valves_open) or
                (self.gate == 2 and state.cycle.semilunar_valves_open))


def _route(key, oxygenated, gate, parts, path) -> FlowRoute:
    samples = _curve_points([p if isinstance(p,Vec3) else Vec3(*p) for p in path],12)
    positions = np.asarray([(p.x,p.y,p.z) for p in samples],dtype=np.float32)
    lengths = np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
    distances = np.linspace(0,lengths[-1],100)
    points = np.column_stack([np.interp(distances,lengths,positions[:,axis]) for axis in range(3)])
    return FlowRoute(key,oxygenated,gate,parts,points.astype(np.float32))


def build_flow_routes() -> tuple[FlowRoute, ...]:
    routes = []
    for part,key,_,_,_,_,path in VESSELS:
        gate = 0 if part in {I.VENA_CAVA,I.PULMONARY_VEINS} else 2
        routes.append(_route(key,part in {I.AORTA,I.PULMONARY_VEINS},gate,(part,),path))
    tri, mit, aortic, pulm = [VALVES[p][0] for p in
                             (I.TRICUSPID_VALVE,I.MITRAL_VALVE,I.AORTIC_VALVE,I.PULMONARY_VALVE)]
    routes.extend((
        _route("right_filling",False,1,(I.RIGHT_ATRIUM,I.TRICUSPID_VALVE,I.RIGHT_VENTRICLE),
               [(-.50,.86,-.12),(-.43,.64,-.04),tri,(-.43,.10,.03),(-.23,-.51,.04)]),
        _route("right_ejection",False,2,(I.RIGHT_VENTRICLE,I.PULMONARY_VALVE),
               [(-.23,-.51,.04),(-.09,-.18,.17),(-.10,.28,.29),pulm]),
        _route("left_filling",True,1,(I.LEFT_ATRIUM,I.MITRAL_VALVE,I.LEFT_VENTRICLE),
               [(.38,.91,-.29),(.38,.63,-.17),mit,(.44,.02,-.08),(.48,-.71,-.10)]),
        _route("left_ejection",True,2,(I.LEFT_VENTRICLE,I.AORTIC_VALVE),
               [(.48,-.71,-.10),(.30,-.20,-.01),(.16,.28,.025),aortic]),
    ))
    return tuple(routes)


def circulation_visible(state: HeartSnapshot) -> bool:
    """Displaced pieces no longer form a connected physiological circulation."""
    return state.flow_enabled and state.explosion < 0.035 and not state.fragment_offsets
