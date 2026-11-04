"""Read-only post-run timing/units audit, never calibrates or publishes state."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    imu = [r for line in (a.capture/'support-events.jsonl').open()
           if (r := json.loads(line))['event'] == 'imu']
    if len(imu) < 2:
        raise ValueError('Insufficient IMU records')
    stamps = np.array([r['stamp_ns'] for r in imu], dtype=np.int64)
    dt = np.diff(stamps)*1e-9
    ages = np.array([r['recv']-r['stamp_ns']*1e-9 for r in imu])
    gyro = np.array([r['gyro'] for r in imu])
    acc = np.array([r['acceleration'] for r in imu])
    report = dict(samples=len(imu), effective_header_hz=float(1/np.mean(dt)),
                  header_gap_percentiles_ms=np.percentile(dt*1000,[0,50,95,99,100]).tolist(),
                  receive_age_percentiles_ms=np.percentile(ages*1000,[0,50,95,99,100]).tolist(),
                  gaps_above_3ms=int(sum(dt>.003)), gaps_above_30ms=int(sum(dt>.03)),
                  duplicate_or_rollback=int(sum(dt<=0)), age_above_35ms=int(sum(ages>.035)),
                  gyro_mean=gyro.mean(0).tolist(), gyro_std=gyro.std(0).tolist(),
                  acceleration_mean=acc.mean(0).tolist(), acceleration_std=acc.std(0).tolist(),
                  calibrated=False, independent_rest_declaration=False,
                  hardware_accuracy_validated=False, motor_readiness_approved=False)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
