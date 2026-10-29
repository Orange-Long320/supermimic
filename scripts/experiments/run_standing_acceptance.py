"""Bound a single-GPU standing run and adopt only independently validated weights."""
from pathlib import Path
import argparse,json,os,subprocess,time,hashlib,shutil,signal

def summarize(path, baseline=None):
 import numpy as np
 d=json.loads(path.read_text());issues=[];phases=[]
 if d['survived_without_reset']!=d['num_envs'] or any(d['resets']):issues.append('fall_or_reset')
 if d.get('teacher_action_override'):issues.append('action_override')
 for j,(start,target) in enumerate(d['schedule']):
  end=d['schedule'][j+1][0] if j+1<len(d['schedule']) else d['duration_s']
  rows=[r for r in d['samples'] if start+1<=r['time_s']<end]
  if not rows:continue
  v=np.array([r['velocity'] for r in rows]);t=np.array(target);err=np.abs(v.mean(0)-t)
  row=dict(start=start,target=target,mean_velocity=v.mean(0).tolist())
  if any(target):
   if (err>np.array([.05,.05,.08])).any():issues.append(f'move_tracking@{start}')
   if baseline:
    br=[r for r in baseline['samples'] if start+1<=r['time_s']<end]
    be=np.abs(np.array([r['velocity'] for r in br]).mean(0)-t)
    if (err>be+np.array([.015,.015,.03])).any():issues.append(f'move_regression@{start}')
  else:
   speed=np.linalg.norm(v[:,:,:2],axis=-1).mean(0);yaw=np.abs(v[:,:,2]).mean(0)
   arm=np.array(d['stop_arm_metrics'][str(float(start))]['joint_velocity_rms_rad_s'])
   if (speed>.06).any():issues.append(f'stop_translation@{start}')
   if (yaw>.08).any():issues.append(f'stop_yaw@{start}')
   if (arm>.12).any():issues.append(f'stop_arm@{start}')
   pos=np.array([r['root_position'] for r in rows]);delta=pos[:,:,:2]-pos[:1,:,:2];dist=np.linalg.norm(delta,axis=-1)
   q=np.array([r['root_quaternion'] for r in rows]);w,x,y,z=np.moveaxis(q,-1,0);heading=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));dh=np.arctan2(np.sin(heading-heading[:1]),np.cos(heading-heading[:1]))
   # Relative to the first recorded sample one second into the stop; no hidden warmup/reset.
   if (dist[-1]>.05).any() or (dist.max(0)>.1).any():issues.append(f'stop_position_drift@{start}')
   if (np.abs(dh[-1])>.1).any() or (np.abs(dh).max(0)>.15).any():issues.append(f'stop_heading_drift@{start}')
   row.update(planar_speed=speed.tolist(),absolute_yaw=yaw.tolist(),arm_rms=arm.tolist(),net_drift_m=dist[-1].tolist(),max_drift_m=dist.max(0).tolist())
  phases.append(row)
 return dict(passed=not issues,issues=issues,phases=phases,source=str(path))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['source','checkpoint','dataset','out','standing-pose','baseline0','baseline7']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--gpu',type=int,default=2);p.add_argument('--iterations',type=int,default=800);p.add_argument('--hours',type=float,default=1.);p.add_argument('--source-commit',required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Requested GPU is not idle')
 env=os.environ.copy();env.pop('PYTHONHOME',None);env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(a.source/'src'),SM_ROBOT='x2_ultra',MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',XDG_CACHE_HOME=str(a.source.parent.parent/'cache'))
 py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python';entry=a.source/'scripts/train_standing_continuous.py';ev=a.source/'scripts/evaluate_standing_skill.py'
 argv=[py,str(entry),'--checkpoint',str(a.checkpoint),'--dataset',str(a.dataset),'--standing-pose',str(a.standing_pose),'--out',str(a.out/'train'),'--iterations',str(a.iterations),'--hours',str(a.hours),'--sigma','.08']
 record=dict(status='initializing',pid=os.getpid(),gpu=a.gpu,source=str(a.source),source_commit=a.source_commit,started_epoch=time.time(),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),argv=argv,entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),evaluator_sha256=hashlib.sha256(ev.read_bytes()).hexdigest(),evaluations=[])
 def save():(a.out/'controller.json').write_text(json.dumps(record,indent=2)+'\n')
 save();log=open(a.out/'train.log','x');train=subprocess.Popen(argv,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record.update(status='training',train_pid=train.pid);save();checked=set();beg=time.monotonic()
 def evaluate(ck,name,seconds,seed,flags=(),baseline=None):
  out=a.out/(ck.stem+'-'+name+'.json');args=[py,str(ev),'--ckpt',str(ck),'--npz-dir',str(a.dataset),'--out',str(out),'--seconds',str(seconds),'--num-envs','8','--seed',str(seed)]+list(flags)
  with open(out.with_suffix('.log'),'x') as f:subprocess.run(args,cwd=a.source,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
  return summarize(out,baseline)
 def stop_own_training():
  if train.poll() is None:
   train.terminate()
   try:train.wait(timeout=15)
   except subprocess.TimeoutExpired:train.kill();train.wait()
 try:
  while True:
   ready=sorted((a.out/'train').glob('candidate_*.ready'),key=lambda f:int(f.stem.split('_')[-1])) if (a.out/'train').exists() else []
   eligible=[f for f in ready if int(f.stem.split('_')[-1])%80==0 and f not in checked]
   if train.poll() is not None:eligible=[f for f in ready if f not in checked][-1:]
   if eligible:
    f=eligible[-1];checked.add(f);ck=f.with_suffix('.pt');record['status']='training_and_evaluating';save()
    results=[evaluate(ck,'cold0',3,0,['--standing-only']),evaluate(ck,'cold7',3,7,['--standing-only'])]
    if all(r['passed'] for r in results):
     results.append(evaluate(ck,'hold20',20,23,['--standing-only']))
     if results[-1]['passed']:
      for seed,base in [(0,a.baseline0),(7,a.baseline7)]:results.append(evaluate(ck,f'full{seed}',66,seed,baseline=json.loads(base.read_text())))
      results.append(evaluate(ck,'extended42',42,23,['--extended-stop']))
    attempt=dict(checkpoint=str(ck),results=results,accepted=all(r['passed'] for r in results) and len(results)==6);record['evaluations'].append(attempt);save()
    if attempt['accepted']:
     stop_own_training();shutil.copy2(ck,a.out/'accepted.pt');record.update(status='accepted',accepted_checkpoint=str(a.out/'accepted.pt'),accepted_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),finished_epoch=time.time());save();break
    record['status']='training';save()
   if train.poll() is not None:
    record.update(status='not_accepted' if train.returncode==0 else 'training_failed',train_returncode=train.returncode,finished_epoch=time.time());save();break
   if time.monotonic()-beg>(a.hours*3600+600):
    stop_own_training();record.update(status='time_limit_not_accepted',finished_epoch=time.time());save();break
   time.sleep(5)
 except BaseException as error:
  stop_own_training();record.update(status='failed',error=repr(error),finished_epoch=time.time());save();raise
 finally:log.close()
if __name__=='__main__':main()
