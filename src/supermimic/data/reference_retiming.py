"""Continuous reference clocks and consistent future goals for design rollouts."""
import torch


def reference_clock(times,preparation,air_scale,reference_preparation,reference_flight,reference_end):
    """Monotone C1 physical-to-reference clock before the terminal hold.

    Inputs broadcast over environments and future timestamps. The three phases
    preserve their reference event times; adjacent rates use harmonic means.
    """
    prep=preparation;flight=reference_flight*air_scale
    recovery=reference_end-reference_preparation-reference_flight
    if recovery<=0:raise ValueError('Reference needs a nonempty recovery phase')
    r0=reference_preparation/prep;r1=1/air_scale
    s1=2*r0*r1/(r0+r1);s2=2*r1/(r1+1)
    early=times<prep;air=(times>=prep)&(times<prep+flight)
    start=torch.where(early,torch.zeros_like(prep),torch.where(air,prep,prep+flight))
    duration=torch.where(early,prep,torch.where(air,flight,torch.full_like(prep,recovery)))
    y0=torch.where(early,torch.zeros_like(prep),torch.where(air,
        torch.full_like(prep,reference_preparation),torch.full_like(prep,reference_preparation+reference_flight)))
    distance=torch.where(early,torch.full_like(prep,reference_preparation),torch.where(air,
        torch.full_like(prep,reference_flight),torch.full_like(prep,recovery)))
    m0=torch.where(early,r0,torch.where(air,s1,s2))
    m1=torch.where(early,s1,torch.where(air,s2,torch.ones_like(prep)))
    u=((times-start)/duration).clamp(0,1)
    value=y0+(-2*u**3+3*u**2)*distance+(u**3-2*u**2+u)*duration*m0+(u**3-u**2)*duration*m1
    rate=(-6*u**2+6*u)*distance/duration+(3*u**2-4*u+1)*m0+(3*u**2-2*u)*m1
    ended=times>=prep+flight+recovery
    return torch.where(ended,torch.full_like(value,reference_end),value),torch.where(ended,torch.zeros_like(rate),rate)


def retimed_goal_window(library,source_frames,rates):
    """Sample future poses and scale all velocity features by the same clock."""
    from supermimic.envs.mdp.motion_library import RefState,goal_features
    source_frames=torch.minimum(source_frames.clamp(min=0),library.clip_last[0])
    low=source_frames.floor().long();high=torch.minimum(low+1,library.clip_last[0])
    alpha=(source_frames-low).reshape(-1,1)
    first=library.ref(low.reshape(-1));second=library.ref(high.reshape(-1))
    blend=lambda a,b:a+(b-a)*alpha
    q1,q2=first.root_quat_w,second.root_quat_w
    q2=torch.where((q1*q2).sum(-1,keepdim=True)<0,-q2,q2)
    quat=blend(q1,q2);quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True)
    speed=rates.reshape(-1,1)
    state=RefState(blend(first.root_pos_w,second.root_pos_w),quat,
        blend(first.joint_pos,second.joint_pos),blend(first.root_lin_vel_w,second.root_lin_vel_w)*speed,
        blend(first.root_ang_vel_w,second.root_ang_vel_w)*speed,blend(first.joint_vel,second.joint_vel)*speed)
    features=goal_features(state,library.default_joint_pos)
    return features.reshape(source_frames.shape[0],-1)
