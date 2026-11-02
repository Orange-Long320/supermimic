"""Summarize simulated support rejections and output gaps without filling them.

Events within a gap are associations, not proof of the cause of every missing
publication. All durations below use recorded IMU header times.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path


def read(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def summarize(directory):
    rows = read(directory / 'packets.jsonl')
    packets = [r for r in rows if 'stamp_ns' in r]
    events = read(directory / 'support-events.jsonl')
    times = [r['stamp_ns'] for r in events]
    stamps = [r['stamp_ns'] for r in packets]
    if len(stamps) < 2 or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('Missing or unordered output')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Unordered support events')
    output_rejections = [r for r in rows if r.get('event') == 'support_output_stale']
    rejected_stamps = [r['imu_stamp_ns'] for r in output_rejections]
    if any(b < a for a, b in zip(rejected_stamps, rejected_stamps[1:])):
        raise ValueError('Unordered publication rejections')
    gaps = []
    for before, after in zip(stamps, stamps[1:]):
        if after - before <= 50_000_000:
            continue
        lo, hi = bisect_right(times, before), bisect_right(times, after)
        count = Counter(r['reason'] for r in events[lo:hi])
        gaps.append({'after_start_s': (before - stamps[0]) * 1e-9,
                     'duration_s': (after-before) * 1e-9,
                     'support_events': dict(count),
                     'stale_publication_rejections': bisect_right(rejected_stamps, after)
                         - bisect_right(rejected_stamps, before)})
    accepted = [r['stamp_ns'] for r in events if r['reason'] == 'accepted']
    acceptance_gaps = [(b-a)*1e-9 for a, b in zip(accepted, accepted[1:])]
    stretches, current, previous_accept = [], [], None
    for event in events:
        if event['reason'] != 'accepted':
            current.append(event)
            continue
        if current:
            stretches.append({'first_rejection_s': (current[0]['stamp_ns']-stamps[0])*1e-9,
                              'last_rejection_s': (current[-1]['stamp_ns']-stamps[0])*1e-9,
                              'between_accepts_s': (None if previous_accept is None else
                                  (event['stamp_ns']-previous_accept)*1e-9),
                              'reasons': dict(Counter(r['reason'] for r in current))})
            current = []
        previous_accept = event['stamp_ns']
    return {'scope': __doc__, 'output_packets': len(stamps),
            'output_duration_s': (stamps[-1]-stamps[0])*1e-9,
            'support_reason_counts': dict(Counter(r['reason'] for r in events)),
            'publication_stale_rejections': len(output_rejections),
            'max_output_gap_s': max((b-a)*1e-9 for a, b in zip(stamps, stamps[1:])),
            'max_support_update_gap_s': max(acceptance_gaps, default=None),
            'gaps_over_50ms': gaps,
            'longest_rejection_stretches': sorted(stretches,
                key=lambda r: r['last_rejection_s']-r['first_rejection_s'], reverse=True)[:12],
            'trailing_unresolved_rejections': dict(Counter(r['reason'] for r in current))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
