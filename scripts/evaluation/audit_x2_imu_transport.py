"""Compare simultaneous native IMU readers using original headers and DDS IDs.

Sequence continuity is evidence only when the installed RMW supplies supported
publication sequence numbers from one publisher. No data interpolation.
"""
import argparse
import bisect
from collections import defaultdict
import csv
import json
from pathlib import Path

UNSUPPORTED = (1 << 64)-1


def analyze(path):
    streams = defaultdict(list)
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            streams[row['reader']].append({k: v if k in ('gid', 'reader') else int(v)
                                          for k, v in row.items()})
    if any(len(streams[k]) < 2 for k in ('native_best_effort', 'native_reliable', 'relay_reliable')):
        raise ValueError('All three readers must capture actual data')
    # Compare only the source-time interval observed by all three readers.
    start = max(min(r['header_ns'] for r in rows) for rows in streams.values())
    end = min(max(r['header_ns'] for r in rows) for rows in streams.values())
    if end <= start:
        raise ValueError('Readers have no overlapping observation window')
    trimmed = {k: [r for r in rows if start <= r['header_ns'] <= end] for k, rows in streams.items()}
    stamps = {k: sorted(set(r['header_ns'] for r in rows)) for k, rows in trimmed.items()}
    results = {}
    for name, rows in trimmed.items():
        gids = set(r['gid'] for r in rows)
        seqs = [r['publication_sequence'] for r in rows]
        supported = len(gids) == 1 and all(s not in (0, UNSUPPORTED) for s in seqs)
        events = []
        source_missing = []
        nonincreasing = 0
        for before, after in zip(rows, rows[1:]):
            gap = after['header_ns']-before['header_ns']
            if gap <= 0:
                nonincreasing += 1
            delta = after['publication_sequence']-before['publication_sequence'] if supported else None
            if supported and delta != 1:
                source_missing.append(dict(previous=before['publication_sequence'], current=after['publication_sequence'],
                                           difference=delta, header_gap_ms=gap/1e6))
            if gap > 30_000_000:
                intermediates = {k: values[bisect.bisect_right(values,before['header_ns']):
                                          bisect.bisect_left(values,after['header_ns'])]
                                 for k, values in stamps.items() if k != name}
                if any(intermediates.values()):
                    attribution = 'reader missed samples captured by another reader'
                elif supported and delta == 1:
                    attribution = 'consecutive publisher sequence across native header gap'
                else:
                    attribution = 'unresolved: no independent recovery or supported consecutive publisher sequence'
                events.append(dict(previous_header_ns=before['header_ns'],current_header_ns=after['header_ns'],
                    gap_ms=gap/1e6, publication_sequence_delta=delta,
                    callback_gap_ms=(after['callback_ns']-before['callback_ns'])/1e6,
                    source_publish_gap_ms=(after['rmw_source_ns']-before['rmw_source_ns'])/1e6,
                    callback_age_ms=(after['callback_ns']-after['header_ns'])/1e6,
                    attribution=attribution, intermediate_headers_by_reader=intermediates))
        def maximum_gap(key):
            return max((b[key]-a[key] for a,b in zip(rows,rows[1:])),default=0)/1e6
        results[name] = dict(count=len(rows), unique_headers=len(stamps[name]),
            single_publisher=len(gids)==1, publication_sequence_supported=supported,
            publisher_gids=sorted(gids), publication_sequence_discontinuities=source_missing,
            nonincreasing_headers=nonincreasing, max_header_gap_ms=maximum_gap('header_ns'),
            max_callback_gap_ms=maximum_gap('callback_ns'), max_rmw_source_gap_ms=maximum_gap('rmw_source_ns'),
            max_callback_age_ms=max((r['callback_ns']-r['header_ns'] for r in rows),default=0)/1e6,
            gaps_over_30ms=events)
    native_union = set(stamps['native_best_effort']) | set(stamps['native_reliable'])
    missing = {k: sorted(native_union-set(stamps[k])) for k in stamps}
    return dict(scope=__doc__, source_time_start_ns=start, source_time_end_ns=end,
        overlap_seconds=(end-start)*1e-9, readers=results,
        native_union_headers_missing_by_reader=missing, publishes_motor_commands=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('New output required')
    report = analyze(a.csv)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {key: value for key,value in r.items() if key not in
        ('publisher_gids','gaps_over_30ms','publication_sequence_discontinuities')} for k,r in report['readers'].items()},indent=2))
    for name,r in report['readers'].items():
        print(name, 'gaps',json.dumps(r['gaps_over_30ms'][:10]),
              'sequence discontinuities',len(r['publication_sequence_discontinuities']),
              'missing native headers',len(report['native_union_headers_missing_by_reader'][name]))


if __name__ == '__main__':
    main()
