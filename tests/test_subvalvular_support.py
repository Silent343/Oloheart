"""Continuity regressions for the attached, independently pickable valve apparatus."""

from collections import Counter
import unittest
import numpy as np

from oloheart.domain.model import HeartPartId as I
from oloheart.infrastructure.geometry import build_heart_geometry
from oloheart.infrastructure.cardiac_atlas import _tilt_valve
from oloheart.infrastructure.sculpted_geometry import VALVES, av_leaflet_point, av_leaflet_open_point


class SubvalvularSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshes = build_heart_geometry()

    def test_support_attributes_cover_every_vertex_and_are_finite(self):
        for mesh in self.meshes:
            if mesh.motion not in {"chordal","papillary"}:
                continue
            data = np.asarray(mesh.support_motion)
            self.assertEqual((len(mesh.vertices),7),data.shape)
            self.assertTrue(np.isfinite(data).all())
            self.assertTrue(((data[:,3] >= 0)&(data[:,3] <= 1)).all())

    def test_chordal_insertions_are_on_actual_leaflet_free_margins(self):
        insertions = [m for m in self.meshes if m.motion == "chordal" and m.support_motion[-1][3] == 1]
        self.assertEqual(60,len(insertions))
        for mesh in insertions:
            point = mesh.vertices[-1]
            data = mesh.support_motion[-1]
            # Zero papillary motion at the leaflet insertion prevents detachment.
            np.testing.assert_allclose(data[4:],0,atol=1e-7)
            valve = I.MITRAL_VALVE if "left_ventricle" in mesh.element_key else I.TRICUSPID_VALVE
            center,radius,count,_ = VALVES[valve]
            candidates = [_tilt_valve(av_leaflet_point(center,radius,count,(muscle+fan%2)%count,
                          .16+.38*muscle+.08*twig+.014*(fan-1) if count == 2 else .22+.15*twig+.024*(fan-1),1),center)
                          for muscle in range(2 if count == 2 else 3) for twig in range(4) for fan in range(3)]
            self.assertLess(min((point-c).length() for c in candidates),1e-7)
            opened = [_tilt_valve(av_leaflet_open_point(center,radius,count,(muscle+fan%2)%count,
                       .16+.38*muscle+.08*twig+.014*(fan-1) if count == 2 else .22+.15*twig+.024*(fan-1),1),center)
                       for muscle in range(2 if count == 2 else 3) for twig in range(4) for fan in range(3)]
            index = min(range(len(candidates)),key=lambda i:(point-candidates[i]).length())
            np.testing.assert_allclose(np.asarray((point.x,point.y,point.z))+data[:3],
                                       (opened[index].x,opened[index].y,opened[index].z),atol=1e-7)

    def test_papillary_tip_and_chordal_root_share_position_and_systolic_displacement(self):
        for muscle in (m for m in self.meshes if m.motion == "papillary"):
            key = muscle.element_key.replace("papillary_","chordae_")
            roots = [m for m in self.meshes if m.element_key.startswith(key+"_") and m.support_motion[0][3] == 0]
            self.assertEqual(3,len(roots))
            for root in roots:
                self.assertEqual(muscle.vertices[-1],root.vertices[-2])
                np.testing.assert_allclose(muscle.support_motion[-1][4:],root.support_motion[-2][4:])
                self.assertEqual(muscle.deformation_center,root.deformation_center)

    def test_each_chordal_branch_starts_at_its_shared_trunk_fork(self):
        for trunk in (m for m in self.meshes if m.motion == "chordal" and m.support_motion[0][3] == 0):
            branches = [m for m in self.meshes if m.element_key == trunk.element_key and m.support_motion[0][3] > 0]
            self.assertEqual(4,len(branches))
            for branch in branches:
                self.assertEqual(trunk.vertices[-1],branch.vertices[-2])
                np.testing.assert_allclose(trunk.support_motion[-1],branch.support_motion[-2])

    def test_fans_do_not_duplicate_the_same_cable_geometry(self):
        trunks = [m for m in self.meshes if m.motion == "chordal" and m.support_motion[0][3] == 0]
        self.assertEqual(15,len({m.vertices for m in trunks}))

    def test_assembled_ventricular_envelope_has_no_open_base_or_apex(self):
        meshes = [m for m in self.meshes if m.part_id in {I.LEFT_VENTRICLE,I.RIGHT_VENTRICLE} and m.view == "exterior"]
        edges = Counter()
        for mesh in meshes:
            coordinates = [(round(v.x,8),round(v.y,8),round(v.z,8)) for v in mesh.vertices]
            for a,b,c in mesh.faces:
                edges.update(tuple(sorted((coordinates[x],coordinates[y]))) for x,y in ((a,b),(b,c),(c,a)))
        self.assertTrue(all(n == 2 for n in edges.values()))


if __name__ == "__main__":
    unittest.main()
