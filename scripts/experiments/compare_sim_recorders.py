"""Compare recorded headers with an independent raw-header probe, simulation only."""
import argparse
import json
from pathlib import Path


def read_stamps(path, group=None):
    result = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if group is None or row['group'] == group:
                sec, ns = row['stamp']
                result.append(sec*10**9+ns)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--recorders', nargs='+', default=['trial-record', 'legacy-record'])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Refusing to overwrite output')
    report = {'scope': 'Header-set comparisons cannot establish source publication or physical sampling loss.',
              'groups': {}}
    groups = ('imu', 'leg', 'waist', 'head', 'arm')
    all_records = {}
    for name in a.recorders:
        data = {group: [] for group in groups}
        data['imu'] = read_stamps(a.run/name/'imu.jsonl')
        merged = a.run/name/'joints.jsonl'
        if merged.exists():
            with merged.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    sec, ns = row['stamp']
                    data[row['group']].append(sec*10**9+ns)
        else:
            # Failed recorders retain their separate, already ordered streams.
            for group in groups[1:]:
                data[group] = read_stamps(a.run/name/f'joints-{group}.jsonl', group)
        all_records[name] = data
    for group in groups:
        probe = [row[0] for row in json.loads((a.run/f'probe-{group}.json').read_text())]
        recorders = {name: all_records[name][group] for name in a.recorders}
        if not probe or any(not arr for arr in recorders.values()):
            raise ValueError(f'{group}: empty stream')
        low = max([min(probe)]+[min(arr) for arr in recorders.values()])
        high = min([max(probe)]+[max(arr) for arr in recorders.values()])
        if high <= low:
            raise ValueError(f'{group}: no common header interval')
        values = [v for v in probe if low <= v <= high]; base = set(values)
        result = {'overlap_s': (high-low)*1e-9, 'probe_count': len(values),
                  'probe_duplicates': len(values)-len(base)}
        for name, arr in recorders.items():
            values = [v for v in arr if low <= v <= high]; current = set(values)
            result[name] = {'count': len(values), 'duplicates': len(values)-len(current),
                            'shared_unique': len(base & current), 'probe_only_unique': len(base-current),
                            'recorder_only_unique': len(current-base)}
        report['groups'][group] = result
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
