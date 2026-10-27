"""A saturated servo must brake before a stop under a known disturbance."""
import pytest
import torch
from supermimic.policy.joint_projection import project_joint_commands


@pytest.mark.parametrize('direction',[-1.,1.])
def test_servo_stays_inside_stop_with_unchanged_motor_limits(direction):
    q=torch.zeros(1);v=torch.zeros(1);acc=torch.zeros(1);previous=torch.zeros(1)
    target=torch.tensor([direction*1.5]);kp=torch.tensor([14.]);kd=torch.tensor([.9])
    effort=torch.tensor([24.]);inertia=torch.tensor([.01]);disturbance=direction*4.
    lower=torch.tensor([-.26]);upper=-lower;dt=.005
    positions=[];torques=[];interventions=0
    for _ in range(400):
        command,changed,_=project_joint_commands(target,q,v,acc,previous,kp,kd,effort,inertia,lower,upper)
        previous=(kp*(command-q)-kd*v).clamp(-effort,effort)
        acc=(previous+disturbance)/inertia
        v=v+dt*acc;q=q+dt*v
        positions.append(q.item());torques.append(previous.item());interventions+=int(changed.item())
    assert max(abs(x) for x in positions)<.26
    assert max(abs(x) for x in torques)<=24.
    assert interventions>0
    # The identical servo without projection reaches the forbidden stop.
    q=torch.zeros(1);v=torch.zeros(1);crossed=False
    for _ in range(400):
        force=(kp*(target-q)-kd*v).clamp(-effort,effort)
        v+=dt*(force+disturbance)/inertia;q+=dt*v
        crossed|=abs(q.item())>.26
    assert crossed
