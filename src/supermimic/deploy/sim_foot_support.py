"""X2 simulator-only collision-sphere support hypothesis; no contact sensing."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def vector(text):
    return np.fromstring(text, sep=' ')


class SimFootSupport:
    """Analytic FK restricted to the unrotated, zero-anchor X2 leg XML chain."""
    def __init__(self, xml):
        root = ET.parse(xml).getroot()
        pelvis = root.find("./worldbody/body[@name='pelvis']")
        if pelvis is None:
            raise ValueError('X2 pelvis not found in XML')
        self.chains = {}
        for side in ('left', 'right'):
            node = pelvis
            chain = []
            for suffix in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'):
                node = node.find("body[@name='%s_%s_link']" % (side, suffix))
                if node is None or any(k in node.attrib for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
                    raise ValueError('unsupported X2 leg body orientation')
                joint = node.find('joint')
                if joint is None or joint.get('type', 'hinge') != 'hinge' or np.any(vector(joint.get('pos', '0 0 0'))):
                    raise ValueError('unsupported X2 joint')
                axis = vector(joint.get('axis', '0 0 1'))
                if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1.):
                    raise ValueError('unsupported joint axis')
                chain.append((joint.get('name'), vector(node.get('pos', '0 0 0')), axis))
            spheres = [(vector(g.get('pos', '0 0 0')), float(g.get('size').split()[0]))
                       for g in node.findall('geom') if g.get('class') == 'collision' and g.get('type') == 'sphere']
            if not spheres:
                raise ValueError('X2 collision spheres not found')
            self.chains[side] = (chain, spheres)
        self.side = None
        self.side_since = None
        self.accepted_relative_height = None

    def candidates(self, rwp, omega_body, position, velocity):
        result = {}
        for side, (chain, spheres) in self.chains.items():
            pos = np.zeros(3); rot = np.asarray(rwp).copy(); axes = []; origins = []
            omega = rot @ np.asarray(omega_body)
            for name, offset, axis in chain:
                pos += rot @ offset
                axes.append(rot @ axis); origins.append(pos.copy())
                rot = rot @ Rotation.from_rotvec(axis * position[name]).as_matrix()
            candidates = [(pos + rot @ center, radius) for center, radius in spheres]
            center, radius = min(candidates, key=lambda item: item[0][2] - item[1])
            point = center - [0., 0., radius]
            point_velocity = np.cross(omega, point)
            for (name, _, _), axis, origin in zip(chain, axes, origins):
                point_velocity += np.cross(axis, point-origin)*velocity[name]
            result[side] = (float(point[2]), -point_velocity)
        return result

    def observe(self, stamp, rwp, omega_body, position, velocity):
        candidates = self.candidates(rwp, omega_body, position, velocity)
        side = min(candidates, key=lambda s: candidates[s][0]); other = 'right' if side == 'left' else 'left'
        near = candidates[other][0] - candidates[side][0] < .01
        consistent = np.linalg.norm(candidates[other][1] - candidates[side][1]) <= .2
        if self.side != side:
            # If both feet produce the same support hypothesis, a change in
            # which one is microscopically lower is not a support transition.
            # Otherwise restart the existing hold, without relaxing its limit.
            if self.side is None or not (near and consistent):
                self.side_since = stamp
            self.side = side
        if stamp - self.side_since < .04:
            return None, 'hold'
        if near and not consistent:
            return None, 'disagree'
        self.accepted_relative_height = -candidates[side][0]
        return candidates[side][1], 'accepted'
