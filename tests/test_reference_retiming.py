"""Retiming must change the future window and velocities together."""
import torch
from supermimic.data.reference_retiming import reference_clock,retimed_goal_window
from supermimic.envs.mdp.motion_library import RefState


def test_clock_preserves_events_is_monotone_and_stops():
    prep=torch.tensor([[1.2],[.6]],dtype=torch.float64)
    air=torch.tensor([[1.4],[.8]],dtype=torch.float64)
    t=torch.linspace(0,3,601,dtype=torch.float64)[None,:].expand(2,-1)
    value,rate=reference_clock(t,prep,air,.6,.5,1.8)
    assert torch.all(value[:,1:]>=value[:,:-1]-1e-12)
    assert torch.all(rate>=-1e-12)
    for k in range(2):
        events=torch.tensor([[0.,prep[k,0],prep[k,0]+.5*air[k,0],3.]],dtype=torch.float64)
        y,_=reference_clock(events,prep[k:k+1],air[k:k+1],.6,.5,1.8)
        torch.testing.assert_close(y,torch.tensor([[0.,.6,1.1,1.8]],dtype=torch.float64))
    assert torch.all(rate[:,-1]==0)


def test_clock_rate_matches_derivative_and_is_continuous_at_events():
    p=torch.tensor([[1.]],dtype=torch.float64);a=torch.tensor([[1.3]],dtype=torch.float64)
    t=torch.tensor([[.2,1.,1.65,2.1]],dtype=torch.float64);epsilon=1e-6
    y1,_=reference_clock(t+epsilon,p,a,.6,.5,1.8)
    y0,_=reference_clock(t-epsilon,p,a,.6,.5,1.8)
    _,rate=reference_clock(t,p,a,.6,.5,1.8)
    torch.testing.assert_close((y1-y0)/(2*epsilon),rate,atol=2e-6,rtol=2e-6)


def test_unit_retiming_preserves_native_timestamps_and_velocity_before_clip_end():
    t=torch.linspace(0,1.799,127,dtype=torch.float64)[None,:]
    value,rate=reference_clock(t,torch.tensor([[.6]],dtype=torch.float64),
        torch.ones((1,1),dtype=torch.float64),.6,.5,1.8)
    torch.testing.assert_close(value,t,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(rate,torch.ones_like(rate),atol=1e-12,rtol=1e-12)


def test_goal_velocity_scales_and_interpolates_without_mutating_library():
    class Library:
        clip_last=torch.tensor([100])
        default_joint_pos=torch.zeros(2)
        def ref(self,frame):
            x=frame.float()[:,None]/50
            quat=torch.zeros(len(frame),4);quat[:,0]=1
            return RefState(torch.cat([x,0*x,1+0*x],-1),quat,x.expand(-1,2),
                torch.cat([1+0*x,0*x,0*x],-1),torch.zeros(len(frame),3),torch.ones(len(frame),2))
    frames=torch.tensor([[0.,.5,1.]])
    features=retimed_goal_window(Library(),frames,torch.full_like(frames,.5)).reshape(1,3,-1)
    torch.testing.assert_close(features[0,:,0],torch.tensor([0.,.01,.02]))
    torch.testing.assert_close(features[0,:,2:4],torch.full((3,2),.5))
    torch.testing.assert_close(features[0,:,10],torch.full((3,),.5))
