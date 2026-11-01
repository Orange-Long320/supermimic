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
        self.last_source_sequence = None
        self.recovered_packets = 0
        self.schema = None

    def decode(self, data, now_ns):
        packet = json.loads(data)
        def integer(v): return type(v) is int and v >= 0
        if not isinstance(packet, dict) or packet.get('schema') not in (1, 2) or packet.get('session') != self.session:
            raise ValueError('IMU relay schema or session')
        if self.schema is not None and self.schema != packet['schema']:
            raise ValueError('IMU relay schema changed')
        self.schema = packet['schema']
        if packet['schema'] == 2:
            return self._decode_replay(packet, now_ns)
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

    def _decode_replay(self, packet, now_ns):
        def integer(v): return type(v) is int and v >= 0
        seq, pub, rejected, source_seq = (packet.get(k) for k in ('sequence', 'published_ns', 'rejected', 'source_seq'))
        if not all(integer(v) for v in (seq, pub, rejected, source_seq)) or not 0 <= now_ns-pub <= 50_000_000:
            raise ValueError('IMU replay publication age or sequence')
        if self.seq is not None and seq <= self.seq:
            raise ValueError('IMU replay packet rollback')
        rows = packet.get('samples')
        if not isinstance(rows, list) or len(rows)>16:
            raise ValueError('IMU replay sample count')
        prior_id, prior_h, prior_r = None, 0, 0
        for row in rows:
            if not isinstance(row, list) or len(row)!=9:
                raise ValueError('IMU replay sample shape')
            ident, h, r = row[:3]
            if (not all(integer(v) for v in (ident,h,r)) or ident==0 or not 0<h<=r<=pub
                    or (prior_id is not None and ident!=prior_id+1) or h<=prior_h or r<prior_r
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in row[3:])):
                raise ValueError('IMU replay sample values or order')
            prior_id, prior_h, prior_r = ident, h, r
        if (rows and rows[-1][0]!=source_seq) or (not rows and source_seq!=0):
            raise ValueError('IMU replay source counter')
        previous_rejected = None if self.counters is None else self.counters[0]
        if previous_rejected is not None and rejected != previous_rejected:
            self.seq, self.counters = seq, (rejected, 0)
            self.last_source_sequence = source_seq
            self.last_stamp = self.last_received = 0
            raise ValueError('IMU replay rejected source sample')
        if self.last_source_sequence is not None and source_seq < self.last_source_sequence:
            raise ValueError('IMU replay source rollback')
        unseen = rows if self.last_source_sequence is None else [r for r in rows if r[0]>self.last_source_sequence]
        if self.last_source_sequence is not None and unseen and unseen[0][0] != self.last_source_sequence+1:
            self.seq, self.counters = seq, (rejected, 0)
            self.last_source_sequence = source_seq
            self.last_stamp = self.last_received = 0
            raise ValueError('IMU replay window cannot recover source samples')
        if unseen and (unseen[0][1] <= self.last_stamp or unseen[0][2] < self.last_received):
            raise ValueError('IMU replay nonmonotonic new sample')
        if self.seq is not None: self.recovered_packets += max(0,seq-self.seq-1)
        self.seq, self.counters = seq, (rejected, 0)
        self.last_source_sequence = source_seq
        if unseen: self.last_stamp, self.last_received = unseen[-1][1:3]
        return [r[1:] for r in unseen]
