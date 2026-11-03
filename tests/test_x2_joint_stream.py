"""Continuous joint recorder smoke test with fake ROS messages."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


path=Path(__file__).resolve().parents[1]/'scripts/deploy/record_x2_joint_stream.py'
spec=importlib.util.spec_from_file_location('x2_joint_stream',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class StreamTests(unittest.TestCase):
    def test_continuous_recorder_flushes_on_ros_shutdown(self):
        names=['joint%d'%i for i in range(31)]; sent=[]
        class End(Exception):pass
        class RclError(Exception):pass
        class Node:
            def __init__(self,*args,**kwargs):self.callbacks=[]
            def create_subscription(self,msg,topic,callback,qos):self.callbacks.append(callback)
            def destroy_node(self):pass
        class Logger:
            def __init__(self,path):self.path=path
            def emit(self,key,value):sent.append((key,value))
            def close(self):return {'success':True,'written':{'packets':len(sent)}}
        modules={n:types.ModuleType(n) for n in ('rclpy','rclpy.node','rclpy.executors','rclpy.qos',
                 'rclpy.impl','rclpy.impl.implementation_singleton','aimdk_msgs','aimdk_msgs.msg',
                 'supermimic.deploy.async_sim_diagnostic_log')}
        rclpy=modules['rclpy'];rclpy.init=lambda:None;rclpy.try_shutdown=lambda:None;rclpy.ok=lambda:True
        def spin(node,timeout_sec):
            joints=[types.SimpleNamespace(name=n,position=.1,velocity=.2) for n in names]
            msg=types.SimpleNamespace(joints=joints,header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=10,nanosec=1)))
            node.callbacks[0](msg)
            raise End()
        rclpy.spin_once=spin;modules['rclpy.node'].Node=Node;modules['rclpy.executors'].ExternalShutdownException=End
        modules['rclpy.impl.implementation_singleton'].rclpy_implementation=types.SimpleNamespace(RCLError=RclError)
        qos=modules['rclpy.qos'];qos.QoSProfile=lambda **kw:None
        qos.ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=1);qos.DurabilityPolicy=types.SimpleNamespace(VOLATILE=1)
        modules['aimdk_msgs.msg'].JointStateArray=object;modules['supermimic.deploy.async_sim_diagnostic_log'].AsyncDiagnosticLog=Logger
        with tempfile.TemporaryDirectory() as tmp:
            startup=Path(tmp)/'startup.json';out=Path(tmp)/'stream'
            startup.write_text(json.dumps({'joint_names':names}))
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',['record','--startup',str(startup),'--out',str(out)]):
                module.main()
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['samples_enqueued'],1)
            self.assertEqual(summary['termination'],'interrupted')
            self.assertEqual(sent[0][1]['q'],[.1]*31)
            self.assertEqual(sent[0][1]['qd'],[.2]*31)
            self.assertEqual(sent[0][1]['groups']['leg']['header_ns'],10_000_000_001)


if __name__=='__main__':unittest.main()
