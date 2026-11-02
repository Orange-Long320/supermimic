"""Validate a diagnostic relay packet without changing source timestamps."""
import json
import math

GROUP_NAMES = {
    'leg': {f'{side}_{joint}_joint' for side in ('left', 'right') for joint in
            ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')},
    'arm': {f'{side}_{joint}_joint' for side in ('left', 'right') for joint in
            ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_yaw', 'wrist_pitch', 'wrist_roll')},
    'waist': {f'waist_{axis}_joint' for axis in ('pitch', 'roll', 'yaw')},
    'head': {f'head_{axis}_joint' for axis in ('pitch', 'yaw')},
}


class JointRelayDecoder:
    def __init__(self, session, names):
        if len(session) != 32 or any(c not in '0123456789abcdef' for c in session):
            raise ValueError('invalid relay session')
        self.session = session
        self.names = set(names)
        if self.names != set().union(*GROUP_NAMES.values()) or len(names) != len(self.names):
            raise ValueError('relay profile joint set')
        self.seq = None
        self.last = {}

    def decode(self, data, now_ns):
        packet = json.loads(data)
        def integer(value):
            return type(value) is int and value >= 0
        if packet.get('schema') != 1 or packet.get('session') != self.session:
            raise ValueError('relay schema or session')
        seq, published = packet.get('sequence'), packet.get('published_ns')
        if not integer(seq) or not integer(published) or not 0 <= now_ns-published <= 50_000_000:
            raise ValueError('relay sequence or publication age')
        if self.seq is not None and seq != self.seq+1:
            self.seq = seq
            self.last.clear()
            raise ValueError('relay sequence gap')
        samples = packet.get('samples')
        if not isinstance(samples, list) or len(samples) > 4:
            raise ValueError('relay samples')
        seen, result, updates = set(), [], {}
        for row in samples:
            if not isinstance(row, dict): raise ValueError('relay sample type')
            group = row.get('group')
            if group not in ('leg', 'waist', 'head', 'arm') or group in seen:
                raise ValueError('relay group')
            seen.add(group)
            h, r, count, domain = (row.get(k) for k in ('header_ns', 'received_ns', 'source_count', 'domain'))
            if not all(integer(v) for v in (h, r, count, domain)) or not 0 < h <= r <= published:
                raise ValueError('relay source timestamps')
            if group in self.last and (h <= self.last[group][0] or count <= self.last[group][1]):
                raise ValueError('relay nonmonotonic source')
            joints = row.get('joints')
            if not isinstance(joints, list) or len(joints) != {'leg':12, 'waist':3, 'head':2, 'arm':14}[group]:
                raise ValueError('relay joint count')
            found = set()
            for j in joints:
                if (not isinstance(j, list) or len(j) != 4 or j[0] not in self.names or j[0] in found
                        or not all(type(v) in (int, float) and math.isfinite(v) for v in j[1:3])
                        or not integer(j[3])):
                    raise ValueError('relay joint values')
                found.add(j[0])
            if found != GROUP_NAMES[group]: raise ValueError('relay group joint set')
            updates[group] = (h, count)
            # Retained measurements are immutable scalar tuples. Lists from
            # json.loads would keep thousands of acyclic joint containers in
            # the cyclic collector's old-generation scan on the live bridge.
            result.append((group, {'header_ns': h, 'received_ns': r, 'domain': domain,
                                   'joints': tuple(tuple(j) for j in joints)}))
        self.seq = seq
        self.last.update(updates)
        return result
