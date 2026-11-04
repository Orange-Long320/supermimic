"""Offline ROS callback replay; no ROS transport and no robot publishers."""
import sys,time,types,json,importlib.util,argparse,copy
from pathlib import Path
from types import SimpleNamespace as NS

def run():
 p=argparse.ArgumentParser();p.add_argument('bridge');p.add_argument('release');p.add_argument('capture');p.add_argument('out');p.add_argument('--fit-min-ms',type=float,default=0.);p.add_argument('--synthetic-hold-seconds',type=float,default=0.);a=p.parse_args()
 records=[json.loads(x) for x in (Path(a.capture)/'support-events.jsonl').open()]
 records=[r for r in records if r['event'] in ('imu','joint')]
 # Optional held samples test startup scheduling only, never motion accuracy.
 if a.synthetic_hold_seconds:
  templates=[next(r for r in reversed(records) if r['event']=='imu')]
  templates += [next(r for r in reversed(records) if r['event']=='joint' and r['group']==g) for g in ('leg','waist','head','arm')]
  end_stamp=max(r['stamp_ns'] for r in records);end_recv=max(r['recv'] for r in records)
  extra=[]
  for k in range(1,int(a.synthetic_hold_seconds/.002)+1):
   for rank,template in enumerate(templates):
    row=copy.deepcopy(template)
    row['stamp_ns']=end_stamp+int(k*2_000_000)
    row['recv']=end_recv+k*.002+rank*.00001
    extra.append(row)
  records += extra
 callbacks={};emitted=[];base=records[0]['recv'];clock=time.monotonic();index=0
 class Clock:
  def time(self):return base+time.monotonic()-clock
  def monotonic(self):return time.monotonic()
 class Handle:
  def __init__(self):self.message=None;self.pointer=id(self)
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def take_message(self,*args):
   message,self.message=self.message,None
   return None if message is None else (message,None)
 class Node:
  def __init__(self,name,**kw):self.context=NS(handle=None)
  def create_subscription(self,kind,topic,cb,qos):
   handle=Handle();callbacks[topic]=lambda msg:setattr(handle,'message',msg)
   return NS(handle=handle,msg_type=kind,raw=False)
  def create_publisher(self,*x):return NS(publish=lambda msg:emitted.append(json.loads(msg.data)))
  def destroy_node(self):pass
 def spin(node,timeout_sec):
  nonlocal index
  if index==len(records):raise KeyboardInterrupt()
  r=records[index];index+=1
  time.sleep(max(0,r['recv']-base-(time.monotonic()-clock)))
  t=r['stamp_ns'];header=NS(stamp=NS(sec=t//10**9,nanosec=t%10**9),frame_id=r.get('frame','replay'))
  if r['event']=='imu':
   g=r['gyro'];v=r['acceleration'];msg=NS(header=header,angular_velocity=NS(x=g[0],y=g[1],z=g[2]),linear_acceleration=NS(x=v[0],y=v[1],z=v[2]))
   callbacks['/aima/hal/imu/torso/state'](msg)
  else:
   msg=NS(header=header,state=NS(value=r['domain']),joints=[NS(name=n,position=v,velocity=r['velocity'][n]) for n,v in r['position'].items()])
   callbacks['/aima/hal/joint/'+r['group']+'/state'](msg)
 class WaitSet:
  def __init__(self,*args):pass
  def clear_entities(self):pass
  def add_subscription(self,handle):self.handle=handle
  def wait(self,timeout):sys.modules['rclpy'].spin_once(None,timeout_sec=timeout*.000000001)
  def get_ready_entities(self,kind):return [self.handle.pointer] if self.handle.message is not None else []
 modules={'rclpy.impl':{},'rclpy.impl.implementation_singleton':dict(rclpy_implementation=NS(WaitSet=WaitSet)),'rclpy':dict(init=lambda:None,ok=lambda:True,spin_once=spin,try_shutdown=lambda:None),
          'rclpy.node':dict(Node=Node),'rclpy.qos':dict(QoSProfile=lambda **kw:kw,ReliabilityPolicy=NS(BEST_EFFORT=1),DurabilityPolicy=NS(VOLATILE=1)),
          'sensor_msgs.msg':dict(Imu=object),'std_msgs.msg':dict(String=lambda **kw:NS(**kw)),
          'aimdk_msgs.msg':dict(JointStateArray=object)}
 for name,attrs in modules.items():
  module=types.ModuleType(name);module.__dict__.update(attrs);sys.modules[name]=module
 # Preserve callback timing only. Production data and timestamps are unmodified.
 spec=importlib.util.spec_from_file_location('bridge',a.bridge);bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
 bridge.time=Clock()
 # Avoid disk writer startup changing the replay clock before the first callback.
 import supermimic.deploy.imu_dead_reckoning as imu_core
 original_fit=imu_core.fit_static_initialization
 def timed_fit(*args,**kwargs):
  before=time.monotonic();result=original_fit(*args,**kwargs)
  time.sleep(max(0,a.fit_min_ms*.001-(time.monotonic()-before)))
  return result
 imu_core.fit_static_initialization=timed_fit
 import supermimic.deploy.async_sim_diagnostic_log as logging
 class Log:
  def __init__(self,out):self.counts={}
  def emit(self,key,row):
   if row.get('event')=='packet_rejected':print('REJECTED',json.dumps(row),flush=True)
   self.counts[key]=self.counts.get(key,0)+1
  def close(self):return {'success':True,'replay_in_memory':True}
 logging.AsyncDiagnosticLog=Log
 r=Path(a.release)
 sys.argv=[a.bridge,'--profile',str(r/'fk-only/profile.json'),'--model',str(r/'fk-only/fk-only.xml'),'--urdf',str(r/'robot-pc2.urdf'),'--out',a.out,'--assume-initial-rest']
 # Heavy imports and FK construction are excluded just as the first live event
 # starts after subscriptions. Set replay origin at first spin.
 original_spin=spin
 def start_spin(*args,**kwargs):
  nonlocal clock
  if index==0:clock=time.monotonic()
  return original_spin(*args,**kwargs)
 sys.modules['rclpy'].spin_once=start_spin
 try:bridge.main()
 finally:
  report=dict(offline_only=True,ros_transport_used=False,recorded_events_consumed=index,
              packets=len(emitted),synthetic_hold_seconds=a.synthetic_hold_seconds,fit_min_ms=a.fit_min_ms,
              state_stamps_ns=[p['state_stamp_ns'] for p in emitted],
              max_packet_age_ms=max(((p['published_wall']-min([p['state_stamp'],*p['joint_stamps'],*p['joint_received_wall']]))*1000 for p in emitted),default=None),
              hardware_accuracy_validated=False)
  if Path(a.out).is_dir():(Path(a.out)/'offline-replay.json').write_text(json.dumps(report,indent=2)+'\n')
  print('OFFLINE REPLAY ONLY:',json.dumps(report))

if __name__=='__main__':run()
