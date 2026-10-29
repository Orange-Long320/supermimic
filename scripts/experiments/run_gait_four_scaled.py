"""Serial capacity measurements on one idle GPU, then one-hour fit at proven size."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    (a.out/'controller.pid').write_text(str(os.getpid()))
    state=dict(status='capacity',gpu=a.gpu,pid=os.getpid(),phases=[],scope='one hour optimization after capacity; 4096 then8192; possibly16384 if measured headroom permits')
    def save(): (a.out/'controller.json').write_text(json.dumps(state,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MUJOCO_GL='egl',XDG_CACHE_HOME='/home/2025-Long.xh/HHD/data/supermimic/cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for k in ['PYTHONPATH','PYTHONHOME','SM_ROBOT_SPEC','SM_SOLE_BOX_CONFIG']:env.pop(k,None)
    env['WARP_CACHE_PATH']=env['XDG_CACHE_HOME']+'/warp'
    py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    common=[py,str(a.source/'scripts/experiments/train_gait_four.py'),'--spec',str(a.source/'configs/training/gait-four-1h.json'),'--dataset',str(a.dataset),'--gpu',str(a.gpu)]
    def idle():
        while subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            state['status']='waiting_for_idle_gpu';save();time.sleep(30)
    def stage(name,args):
        idle();item=dict(name=name,argv=args,started_epoch=time.time());state['phases'].append(item);state['status']=name;save()
        with (a.out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(args,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=child.pid;save();code=child.wait()
        item.update(exit_code=code,ended_epoch=time.time());save();return code
    selected=None;previous=None
    for size in [4096,8192,16384]:
        if size==16384 and (selected is None or selected['device_peak_mib']>10500):break
        out=a.out/f'capacity_{size}'
        args=common+['--phase','capacity','--out',str(out),'--num-envs',str(size)]
        if previous:args+=['--checkpoint',previous]
        code=stage(f'capacity_{size}',args)
        if code:
            # Only a measured allocator exhaustion permits falling back to the
            # previous proven capacity; algorithm/config errors must be fixed.
            log=(a.out/f'capacity_{size}.log').read_text()
            if selected and ('out of memory' in log.lower() or 'CUDA_ERROR_OUT_OF_MEMORY' in log):break
            raise RuntimeError(f'capacity {size} failed; inspect saved log')
        proof=json.loads((out/'result.json').read_text())
        if proof['status']!='completed' or proof['updates']<20:raise RuntimeError('Capacity did not update finite models')
        if proof['device_peak_mib']>20000:break
        selected=proof;previous=proof['checkpoint'];state['selected_num_envs']=size;state['capacity_proof']=str(out/'result.json');save()
    if selected is None:raise RuntimeError('No proven capacity')
    code=stage('train_1h',common+['--phase','train','--out',str(a.out/'train_1h'),'--num-envs',str(state['selected_num_envs']),'--proof',state['capacity_proof']])
    state['status']='completed' if code==0 else 'failed';save();(a.out/'controller.exit').write_text(str(code))
    if code:raise RuntimeError('One-hour fit failed')

if __name__=='__main__':main()
