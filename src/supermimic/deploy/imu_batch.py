"""Fail-closed validation for diagnostic, unretimestamped IMU batches."""
import json
import math


class ImuBatchDecoder:
    def __init__(self, session):
        if len(session) != 32 or any(c not in '0123456789abcdef' for c in session):
            raise ValueError('invalid IMU relay session')
        self.session = session
        self.seq = None
        self.counters = None
        self.last_stamp = 0
        self.last_received = 0

    def decode(self, data, now_ns):
        packet = json.loads(data)
        def integer(v): return type(v) is int and v >= 0
        if not isinstance(packet, dict) or packet.get('schema') != 1 or packet.get('session') != self.session:
            raise ValueError('IMU relay schema or session')
        seq, pub, rejected, overflow = (packet.get(k) for k in ('sequence', 'published_ns', 'rejected', 'overflow'))
        if not all(integer(v) for v in (seq, pub, rejected, overflow)) or not 0 <= now_ns-pub <= 50_000_000:
            raise ValueError('IMU relay publication age or sequence')
        counters = (rejected, overflow)
        if ((self.seq is not None and seq != self.seq+1) or
                (self.counters is not None and counters != self.counters)):
            self.seq, self.counters = seq, counters
            self.last_stamp = self.last_received = 0
            raise ValueError('IMU relay gap or rejected source sample')
        samples = packet.get('samples')
        if not isinstance(samples, list) or len(samples) > 64:
            raise ValueError('IMU relay sample count')
        stamp, received = self.last_stamp, self.last_received
        result = []
        for row in samples:
            if not isinstance(row, list) or len(row) != 8:
                raise ValueError('IMU relay sample shape')
            h, r = row[:2]
            if (not integer(h) or not integer(r) or not 0 < h <= r <= pub or h <= stamp or r < received
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row[2:])):
                raise ValueError('IMU relay sample values or order')
            stamp, received = h, r
            result.append(row)
        self.seq, self.counters = seq, counters
        self.last_stamp, self.last_received = stamp, received
        return result
