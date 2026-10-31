"""Pose algebra for offline, logging-only X2 state conversion; no ROS or actuation."""
import numpy as np


class UrdfPelvisMount:
    """Measured-joint pelvis-to-IMU FK from a device URDF and native mount TF.

    This selects geometry only. It does not establish encoder calibration,
    modify measurements, align the world frame, or register floor height.
    """
    def __init__(self, urdf, joint_names, p_torso_imu, r_torso_imu, torso='torso_link'):
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        self.rotation = Rotation
        self.names = tuple(joint_names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Invalid named joint mapping')
        self.p_mount = np.asarray(p_torso_imu, dtype=float).copy()
        self.r_mount = np.asarray(r_torso_imu, dtype=float).copy()
        if (self.p_mount.shape != (3,) or self.r_mount.shape != (3, 3)
                or not np.isfinite(self.p_mount).all() or not np.isfinite(self.r_mount).all()
                or not np.allclose(self.r_mount.T@self.r_mount, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(self.r_mount), 1., atol=1e-8)):
            raise ValueError('Invalid native mount transform')
        by_child = {}
        for joint in ET.parse(urdf).getroot().findall('joint'):
            child = joint.find('child').get('link')
            if child in by_child:
                raise ValueError('Duplicate URDF child')
            by_child[child] = joint
        chain, visited = [], set()
        child = torso
        while child != 'pelvis':
            if child in visited or child not in by_child:
                raise ValueError('Torso is not connected to pelvis')
            visited.add(child)
            joint = by_child[child]
            kind = joint.get('type')
            if kind not in ('fixed', 'revolute', 'continuous'):
                raise ValueError('Unsupported waist joint type')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            axis_node = joint.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ') if axis_node is not None else np.array([1., 0., 0.])
            if any(v.shape != (3,) or not np.isfinite(v).all() for v in (xyz, rpy, axis)):
                raise ValueError('Invalid URDF transform')
            index = None
            if kind != 'fixed':
                if joint.find('mimic') is not None or np.linalg.norm(axis) < 1e-8:
                    raise ValueError('Unsupported waist axis or mimic joint')
                index = self.names.index(joint.get('name'))
                axis = axis/np.linalg.norm(axis)
            chain.append((xyz, Rotation.from_euler('xyz', rpy).as_matrix(), axis, index))
            child = joint.find('parent').get('link')
        self.chain = tuple(reversed(chain))

    def mount(self, positions, velocities=None):
        q = np.asarray(positions, dtype=float)
        qd = np.zeros_like(q) if velocities is None else np.asarray(velocities, dtype=float)
        if q.shape != (len(self.names),) or qd.shape != q.shape or not np.isfinite(np.r_[q, qd]).all():
            raise ValueError('Invalid named joint values')
        p, r, omega = np.zeros(3), np.eye(3), np.zeros(3)
        for xyz, origin_r, axis, index in self.chain:
            p = p+r@xyz
            r = r@origin_r
            if index is not None:
                omega += (r@axis)*qd[index]
                r = r@self.rotation.from_rotvec(axis*q[index]).as_matrix()
        return p+r@self.p_mount, r@self.r_mount, omega


def gravity_level_rotation(up):
    """Minimal rotation taking static specific-force direction to world +Z."""
    v=np.asarray(up,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
        raise ValueError('Invalid gravity direction')
    v=v/np.linalg.norm(v); target=np.array([0.,0.,1.])
    c=float(v@target)
    if c < -1+1e-10:return np.diag([1.,-1.,-1.])
    cross=np.cross(v,target)
    k=np.array([[0.,-cross[2],cross[1]],[cross[2],0.,-cross[0]],[-cross[1],cross[0],0.]])
    return np.eye(3)+k+k@k/(1+c)


def pelvis_from_imu(p_world_imu, r_world_imu, p_pelvis_imu, r_pelvis_imu):
    """T_WP = T_WI inv(T_PI), using measured-joint FK for the moving mount."""
    r = np.asarray(r_world_imu) @ np.asarray(r_pelvis_imu).T
    p = np.asarray(p_world_imu) - r @ np.asarray(p_pelvis_imu)
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite pelvis transform')
    return p, r


def pelvis_angular_velocity(omega_imu, r_pelvis_imu, omega_mount_in_pelvis):
    """Remove waist-chain motion from an IMU attached above the moving waist."""
    return np.asarray(r_pelvis_imu) @ np.asarray(omega_imu) - np.asarray(omega_mount_in_pelvis)


def latest_causal(rows, now_ns, *, max_age_s, stamp_key='header_ns',
                  at_ns=None, received_times=None):
    """Select only received, nonfuture data; reject old acquisition/header times."""
    received = (np.asarray([r['received_ns'] for r in rows], dtype=np.int64)
                if received_times is None else received_times)
    at_ns = now_ns if at_ns is None else min(at_ns, now_ns)
    i = int(np.searchsorted(received, now_ns, side='right'))-1
    while i >= 0 and rows[i][stamp_key] > at_ns:
        i -= 1
    if i < 0 or (at_ns-rows[i][stamp_key])*1e-9 > max_age_s:
        raise ValueError('Missing, future or stale sample')
    return rows[i]
