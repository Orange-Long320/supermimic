"""Join independently projected windows only after proving complete coverage.

Projection is per input pose with immutable neighbouring source contact intent;
the chunks must share the exact source, model, helper and output clock. The
result still needs the normal independent export/physics checks.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--screening',type=Path,required=True)
    p.add_argument('--chunks',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New joined output required')
    import numpy as np
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    raw=json.loads((a.candidate/'result.json').read_text())
    with np.load(a.candidate/'x2-reference.npz') as z:
        source_frames=len(z['qpos']);source_fps=float(np.asarray(z['fps']).ravel()[0])
    original=next(row for row in json.loads(a.screening.read_text())['motions'] if row['motion']==raw['source_motion'])
    if raw['source_start_frame']!=0 or source_frames!=original['frames']:raise ValueError('Parent is not a complete screened source clip')
    expected=int(np.ceil(((source_frames-1)/source_fps)*50))*4+1
    reference_sha=sha(a.candidate/'x2-reference.npz');model_sha=sha(a.model)
    records=[]
    for chunk in a.chunks:
        report=json.loads((chunk/'result.json').read_text())
        with np.load(chunk/'x2-reference.npz') as z:q=z['qpos'].copy();fps=float(z['fps'][0]);start=int(z['dense_window_start_frame'])
        if fps!=200. or q.shape!=(report['frames'],38) or not np.isfinite(q).all():raise ValueError('Invalid chunk layout')
        if report['source_reference_sha256']!=reference_sha or report['model_sha256']!=model_sha or report.get('task_variant') is not None:raise ValueError('Chunk source/model/task mapping differs')
        if report['dense_window_start_frame']!=start or [r['index'] for r in report['rows']]!=list(range(start,start+len(q))):raise ValueError('Chunk frame provenance differs')
        if report['max_environment_penetration_m']>.001:raise ValueError('Chunk failed original geometry bound')
        records.append((start,q,report,chunk))
    records.sort(key=lambda x:x[0]);cursor=0;signature=None;rows=[];poses=[];provenance=[]
    for start,q,report,path in records:
        current=(report['restoration_helper_sha256'],report['joint_limit_factor'],report['wrist_reference_limit_deg'],report['uniform_time_scale'])
        if start!=cursor or (signature is not None and current!=signature):raise ValueError('Missing/overlapping windows or different projection settings')
        signature=current;cursor+=len(q);poses.append(q);rows.extend(report['rows'])
        provenance.append({'path':str(path),'first_frame':start,'frames':len(q),'reference_sha256':sha(path/'x2-reference.npz')})
    if cursor!=expected:raise ValueError(f'Incomplete dense coverage: {cursor}/{expected}')
    result={**{k:raw[k] for k in ['source_motion','source_start_frame','scene_boxes','joint_limit_factor','wrist_reference_limit_deg']},
        'frames':expected,'fps':200.,'dense_window_start_frame':0,'dense_window_start_time_s':0.,
        'source_reference_sha256':reference_sha,'model_sha256':model_sha,'restoration_helper_sha256':signature[0],
        'uniform_time_scale':signature[3],'complete_source_clip':True,'task_variant':None,'complete_source_prefix':False,
        'geometry_chunk_coverage_verified':True,'chunks':provenance,'rows':rows,
        'max_environment_penetration_m':max(r['penetration_m'] for r in rows),
        'maximum_joint_correction_rad':max(r['maximum_joint_correction_rad'] for r in rows),
        'corrected_frames':sum(r['restoration'] is not None for r in rows),
        'policy_evaluation':False,'training_approved':False,'scope':__doc__}
    a.out.mkdir(parents=True);np.savez_compressed(a.out/'x2-reference.npz',qpos=np.concatenate(poses),fps=np.array([200.]),dense_window_start_frame=0)
    (a.out/'result.json').write_text(json.dumps(result,indent=2));(a.out/'original-adapter-result.json').write_bytes((a.candidate/'result.json').read_bytes())
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
