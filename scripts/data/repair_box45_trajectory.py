"""Joint trajectory optimization with native convex distances and temporal costs.

Server-only diagnostic. Keeps the box, robot, actuator limits and support-foot
targets; smooths the video orientation/upper body and optimizes all leg/root
translations together. Independent geometry/dynamics audits remain mandatory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=60)
    p.add_argument('--check-jacobian',action='store_true')
    p.add_argument('--resume',type=Path,help='Continue the same source/objective from saved solver knots')
    p.add_argument('--support-centred',action='store_true',help='Replace inherited COM drift with support-phase centres')
    p.add_argument('--knot-stride',type=int,default=5)
    p.add_argument('--foot-margin',type=float,default=0.0)
    p.add_argument('--free-torso',action='store_true',help='Optimize root orientation and waist joints for COM balance')
    p.add_argument('--natural-swing',action='store_true',help='Advance up to the box edge during lift instead of lifting vertically in place')
    p.add_argument('--standing-endpoint',type=Path,help='Checked box-top equilibrium; align only its horizontal foot-hull centre')
    p.add_argument('--constraint-rounds',type=int,default=1,help='Augmented-Lagrangian rounds for native clearance constraints')
    p.add_argument('--static-balance',action='store_true',help='Constrain COM projection inside planned support polygon with 5mm margin')
    p.add_argument('--contact-consistent',action='store_true',help='Use 0.5mm compliant compression for planted feet')
    p.add_argument('--waist-effort',action='store_true',help='Reserve 20 percent waist motor effort above gravity demand')
    p.add_argument('--free-shoulders',action='store_true',help='Allow shoulder pitch/roll counterbalance without changing contact order')
    p.add_argument('--shoulder-reference-limit-deg',type=float,help='Task-quality bound around the smoothed source/standing shoulder reference; also strengthens its posture cost')
    p.add_argument('--stance-constraints',action='store_true',help='Constrain planted foot pose, preventing contact area loss from foot tilt')
    p.add_argument('--stance-ankle-effort',action='store_true',help='Keep single-support ankle upstream gravity moment below 90 percent motor cap')
    p.add_argument('--constraint-tolerance',type=float,default=.0005)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if a.constraint_rounds<1 or a.constraint_tolerance<=0:p.error('Positive constraint rounds/tolerance required')
    if a.shoulder_reference_limit_deg is not None and (not a.free_shoulders or a.shoulder_reference_limit_deg<=0):p.error('A positive shoulder reference limit requires free shoulders')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull
    from supermimic.robots.assets import get_spec
    from supermimic.envs.box_scene import FixedBox
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
    from supermimic.robots.dimensions import PROFILE

    z=np.load(a.source);original=z['qpos'].copy();fps=float(z['fps'])
    center,half=z['box_center'],z['box_half_size']
    box=FixedBox(front_x=float(center[0]-half[0]),center_y=float(center[1]),depth=float(2*half[0]),width=float(2*half[1]),height=float(2*half[2]))
    # The prior candidate has a declared 3s hold. Optimize through its entry;
    # the repeated tail is restored after a C2 stationary endpoint is formed.
    end=len(original)-round(3*fps)-1
    if end<20:raise ValueError('Expected moving segment plus declared 3s hold')
    if a.knot_stride<1 or a.foot_margin<0:raise ValueError('Invalid knot stride or foot margin')
    indices=np.unique(np.r_[np.arange(0,end+1,a.knot_stride),end]).astype(int)
    times=indices/fps;n=len(indices);dt=a.knot_stride/fps
    if not np.allclose(np.diff(times),dt):raise ValueError('Moving endpoint must fall on 10Hz knot grid')
    spec=get_spec();box.add_to_spec(spec)
    spec.worldbody.add_geom(name='repair_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),contype=1,conaffinity=1)
    m=spec.compile();d=mujoco.MjData(m)
    box_id=m.geom('box45_collision').id;floor_id=m.geom('repair_floor').id
    bodies=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
    leg_names=[s+'_'+j+'_joint' for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    if a.free_torso:leg_names += ['waist_'+j+'_joint' for j in ('yaw','roll','pitch')]
    if a.free_shoulders:leg_names += [s+'_shoulder_'+j+'_joint' for s in ('left','right') for j in ('pitch','roll')]
    qcols=np.array([m.jnt_qposadr[m.joint(s).id] for s in leg_names],dtype=int)
    vcols=np.r_[np.arange(3),[m.jnt_dofadr[m.joint(s).id] for s in leg_names]].astype(int)
    nr=3 if a.free_torso else 0
    nv=len(vcols)+nr;root=int(m.jnt_bodyid[0])
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]!=0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    width=nv+12+2+2*len(geoms)
    targets=[];com=[]
    for q in original[indices]:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        targets.append([(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy()) for b in bodies])
        com.append(d.subtree_com[root].copy())
    com=gaussian_filter1d(np.array(com),2,axis=0,mode='nearest')
    standing=None;end_blend=np.zeros(n)
    if a.standing_endpoint:
        standing=np.load(a.standing_endpoint)['qpos'].copy()
        if standing.shape!=(m.nq,):raise ValueError('Standing dimension differs')
        def hull_center(q):
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);points=[]
            for g in geoms:
                if not m.geom(g).name.startswith('foot_'):continue
                mesh=m.geom_dataid[g];adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                points.extend(m.mesh_vert[adr:adr+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
            points=np.asarray(points);return (points[:,:2].min(0)+points[:,:2].max(0))/2
        standing[:2]+=hull_center(original[end])-hull_center(standing)
        end_blend=np.clip((times-(times[-1]-1.4))/1.4,0,1)
        end_blend=end_blend**3*(10+end_blend*(-15+6*end_blend))
        d.qpos[:]=standing;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        endpoint_feet=[(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy()) for b in bodies]
        for i,w in enumerate(end_blend):
            for j,(ep,er) in enumerate(endpoint_feet):
                tp,tr=targets[i][j];delta=(Rotation.from_matrix(tr).inv()*Rotation.from_matrix(er)).as_rotvec()
                targets[i][j]=(tp+(ep-tp)*w,(Rotation.from_matrix(tr)*Rotation.from_rotvec(delta*w)).as_matrix())
    if a.natural_swing:
        timing=json.loads((Path(__file__).resolve().parents[2]/'configs/video/box45-contacts.json').read_text())
        scale=json.loads((a.source.parent/'fit.json').read_text())['time_scale']
        ease=lambda u:(lambda t:t**3*(10+t*(-15+6*t)))(np.clip(u,0,1))
        for side,b in zip(('left','right'),bodies):
            lift,land=[timing[side][k]*scale for k in ('lift_seconds','land_seconds')]
            def sole(frame):
                d.qpos[:]=original[frame];mujoco.mj_kinematics(m,d);points=[]
                for g in geoms:
                    if int(m.geom_bodyid[g])!=b or not m.geom(g).name.startswith('foot_'):continue
                    mesh=m.geom_dataid[g];adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                    local=m.mesh_vert[adr:adr+count];points.extend(local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
                points=np.asarray(points);local=(points-d.xpos[b])@d.xmat[b].reshape(3,3)
                point=np.r_[(local[:,:2].min(0)+local[:,:2].max(0))/2,local[:,2].min()]
                return point,d.xpos[b]+d.xmat[b].reshape(3,3)@point,float(points[:,0].max())
            local,start,toe=sole(round(lift*fps));_,finish,_=sole(end)
            _,neutral,neutral_toe=sole(0)
            pre=start.copy();pre[0]=box.front_x-.02-max(toe-start[0],neutral_toe-neutral[0]);pre[2]=box.height+.035
            if pre[0]<start[0]:raise ValueError('Initial foot is already inside the swing edge margin')
            for i,t in enumerate(times):
                u=(t-lift)/(land-lift)
                if not 0<u<1:continue
                if u<.35:point=start+(pre-start)*ease(u/.35)
                elif u<.75:
                    point=pre+(finish-pre)*ease((u-.35)/.4);point[2]=pre[2]
                else:
                    point=finish.copy();point[2]+=(pre[2]-point[2])*(1-ease((u-.75)/.25))
                j=bodies.index(b);_,rotation=targets[i][j]
                targets[i][j]=(point-rotation@local,rotation)
    com_weight=40.
    if a.support_centred:
        def sole_center(frame,side):
            d.qpos[:]=original[frame];mujoco.mj_kinematics(m,d);points=[]
            for g in geoms:
                if not m.geom(g).name.startswith('foot_') or side not in m.body(int(m.geom_bodyid[g])).name:continue
                mesh=m.geom_dataid[g]
                if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                    local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
                else:
                    adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count]
                points.extend(local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
            points=np.asarray(points);near=points[points[:,2]<points[:,2].min()+.012]
            return (near[:,:2].min(0)+near[:,:2].max(0))/2
        left=sole_center(0,'left');right_ground=sole_center(0,'right')
        right=sole_center(end,'right');left_top=sole_center(end,'left')
        timing=json.loads((Path(__file__).resolve().parents[2]/'configs/video/box45-contacts.json').read_text())
        scale=json.loads((a.source.parent/'fit.json').read_text())['time_scale']
        rl,rd,ll,ld=[timing[s][k]*scale for s,k in [('right','lift_seconds'),('right','land_seconds'),('left','lift_seconds'),('left','land_seconds')]]
        ease=lambda u:(lambda t:t**3*(10+t*(-15+6*t)))(np.clip(u,0,1))
        for i,t in enumerate(times):
            if t<rl:point=(left+right_ground)/2+(left-(left+right_ground)/2)*ease(t/(rl*.9))
            elif t<rd:point=left
            elif t<ll:point=left+(right-left)*ease((t-rd)/(ll-rd))
            elif t<ld:point=right
            else:point=right+((right+left_top)/2-right)*ease((t-ld)/1.)
            com[i,:2]=point
        com_weight=500.
    contact_strength=np.zeros((n,2))
    balance_normals=[];balance_offsets=[];max_faces=0
    if a.static_balance:
        if not a.support_centred:raise ValueError('Static balance requires explicit support phases')
        foot_local={b:[] for b in bodies}
        d.qpos[:]=original[0];mujoco.mj_kinematics(m,d)
        for g in geoms:
            b=int(m.geom_bodyid[g])
            if b not in foot_local or not m.geom(g).name.startswith('foot_'):continue
            mesh=m.geom_dataid[g]
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
            else:
                adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count]
            world=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            foot_local[b].extend((world-d.xpos[b])@d.xmat[b].reshape(3,3))
        for i,t in enumerate(times):
            points=[]
            for j,(side,b) in enumerate(zip(('left','right'),bodies)):
                lift,land=[timing[side][k]*scale for k in ('lift_seconds','land_seconds')]
                u=(t-lift)/(land-lift)
                strength=1. if u<=0 or u>=1 else 1-ease(u/.08) if u<.08 else ease((u-.92)/.08) if u>.92 else 0.
                contact_strength[i,j]=strength
                if a.contact_consistent and strength>0:
                    position,rotation=targets[i][j]
                    world=np.asarray(foot_local[b])@rotation.T+position
                    surface=0. if t<(lift+land)/2 else box.height
                    correction=(world[:,2].min()-(surface-.0005))*strength*(1-end_blend[i])
                    position=position.copy();position[2]-=correction
                    targets[i][j]=(position,rotation)
                if lift<t<land:continue
                position,rotation=targets[i][j]
                world=np.asarray(foot_local[b])@rotation.T+position
                inside=(world[:,0]>=box.front_x)&(world[:,0]<=box.front_x+box.depth)&(abs(world[:,1]-box.center_y)<=box.width/2)
                mask=((~inside)&(abs(world[:,2])<.012))|(inside&(abs(world[:,2]-box.height)<.012))
                points.extend(world[mask,:2])
            hull=ConvexHull(np.asarray(points))
            balance_normals.append(hull.equations[:,:2]);balance_offsets.append(hull.equations[:,2])
            max_faces=max(max_faces,len(hull.equations))
        width+=max_faces
    elif a.contact_consistent:raise ValueError('Contact-consistent targets require static-balance support phases')
    stance_count=24 if a.stance_constraints else 0
    if stance_count:
        if not a.static_balance:raise ValueError('Stance constraints require explicit support phases')
        width+=stance_count
    ankle_count=8 if a.stance_ankle_effort else 0
    ankle_info=[]
    if ankle_count:
        if not a.static_balance:raise ValueError('Ankle moment constraints require explicit support phases')
        for side in ('left','right'):
            for axis in ('pitch','roll'):
                name=side+'_ankle_'+axis+'_joint';jid=m.joint(name).id
                ankle_info.append((jid,int(m.jnt_bodyid[jid]),PROFILE['effort_limit'][name]))
        width+=ankle_count
    waist_count=6 if a.waist_effort else 0
    if waist_count:
        gravity_model=spec.compile();gravity_model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        gravity_data=mujoco.MjData(gravity_model)
        waist_names=['waist_'+j+'_joint' for j in ('yaw','roll','pitch')]
        waist_dofs=np.array([gravity_model.jnt_dofadr[gravity_model.joint(s).id] for s in waist_names],dtype=int)
        waist_caps=np.array([PROFILE['effort_limit'][s] for s in waist_names])
        gravity_derivative=np.zeros((m.nv,m.nv));width+=waist_count
    smooth=gaussian_filter1d(original[:end+1,7:],fps*.15,axis=0,mode='nearest')
    rotation=Rotation.from_quat(original[:end+1,[4,5,6,3]])
    reference_rotation=rotation[0]
    rotvec=gaussian_filter1d((reference_rotation.inv()*rotation).as_rotvec(),fps*.15,axis=0,mode='nearest')
    base=original[indices].copy();base[:,7:]=smooth[indices]
    base[:,3:7]=(reference_rotation*Rotation.from_rotvec(rotvec[indices])).as_quat()[:,[3,0,1,2]]
    if standing is not None:
        base[:,7:]+=end_blend[:,None]*(standing[7:]-base[:,7:])
        rbase=Rotation.from_quat(base[:,[4,5,6,3]])
        correction=(rbase.inv()*Rotation.from_quat(standing[[4,5,6,3]])).as_rotvec()
        base[:,3:7]=(rbase*Rotation.from_rotvec(correction*end_blend[:,None])).as_quat()[:,[3,0,1,2]]
    initial=np.c_[original[indices,:3],np.zeros((n,nr)),original[indices][:,qcols]]
    reference=gaussian_filter1d(initial,2,axis=0,mode='nearest')
    if standing is not None:
        reference[:,:3]+=end_blend[:,None]*(standing[:3]-reference[:,:3])
        reference[:,3+nr:]+=end_blend[:,None]*(standing[qcols]-reference[:,3+nr:])
    joint_ids=np.array([m.joint(s).id for s in leg_names])
    ranges=m.jnt_range[joint_ids];mid=ranges.mean(1);radius=np.diff(ranges,axis=1)[:,0]*.9/2
    low=np.tile(np.r_[[-.5,-.7,.25],np.full(nr,-.6),mid-radius+1e-6],(n,1))
    high=np.tile(np.r_[[1.5,.7,1.7],np.full(nr,.6),mid+radius-1e-6],(n,1))
    shoulder_columns=np.array([3+nr+k for k,name in enumerate(leg_names) if 'shoulder' in name],dtype=int)
    if a.shoulder_reference_limit_deg is not None:
        deviation=np.radians(a.shoulder_reference_limit_deg)
        low[:,shoulder_columns]=np.maximum(low[:,shoulder_columns],reference[:,shoulder_columns]-deviation)
        high[:,shoulder_columns]=np.minimum(high[:,shoulder_columns],reference[:,shoulder_columns]+deviation)
        if np.any(low>=high):raise ValueError('Shoulder reference and robot ranges do not overlap')
    if standing is not None:
        endpoint=np.r_[standing[:3],np.zeros(nr),standing[qcols]]
        if np.any(endpoint<low[-1]) or np.any(endpoint>high[-1]):raise ValueError('Standing outside optimization bounds')
        low[-1]=endpoint-1e-9;high[-1]=endpoint+1e-9
    posture_scale=np.r_[np.full(3,5.),np.full(nr,5.),np.full(len(qcols),.25)]
    if a.shoulder_reference_limit_deg is not None:posture_scale[shoulder_columns]=5.
    acceleration_scale=np.r_[np.full(3,1.),np.full(nr,.2),np.full(len(qcols),.08)]/dt**2
    boundary_scale=np.r_[np.full(3,10.),np.ones(nv-3)]/dt
    total=n*width+(n-2)*nv+2*nv
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);witness=np.zeros(6)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'argv':[sys.executable,*sys.argv],'pid':os.getpid(),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest()},indent=2)+'\n')
    cache={};calls=0
    penalty=1500.**2;lagrange=np.zeros((n,2*len(geoms)+max_faces+stance_count+ankle_count+waist_count));constraint_history=[]

    def evaluate(flat):
        nonlocal calls
        if 'x' in cache and np.array_equal(flat,cache['x']):return cache['r'],cache['j']
        x=flat.reshape(n,nv);r=np.zeros(total);jac=lil_matrix((total,n*nv));violations=np.zeros_like(lagrange)
        for i in range(n):
            row=i*width;cols=slice(i*nv,(i+1)*nv)
            q=base[i].copy();q[:3]=x[i,:3];q[qcols]=x[i,3+nr:]
            mapping=np.zeros((m.nv,nv));mapping[:3,:3]=np.eye(3)
            mapping[vcols[3:],np.arange(3+nr,nv)]=1
            if nr:
                delta=x[i,3:6];angle=np.linalg.norm(delta)
                skew=np.array([[0,-delta[2],delta[1]],[delta[2],0,-delta[0]],[-delta[1],delta[0],0]])
                c1=.5 if angle<1e-5 else (1-np.cos(angle))/angle**2
                c2=1/6 if angle<1e-5 else (angle-np.sin(angle))/angle**3
                mapping[3:6,3:6]=np.eye(3)-c1*skew+c2*skew@skew
                q[3:7]=(Rotation.from_quat(base[i,[4,5,6,3]])*Rotation.from_rotvec(delta)).as_quat()[[3,0,1,2]]
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            r[row:row+nv]=(x[i]-reference[i])*posture_scale
            jac[row:row+nv,cols]=np.diag(posture_scale);row+=nv
            foot_errors=[];foot_jacs=[]
            for b,(target_p,target_r) in zip(bodies,targets[i]):
                mujoco.mj_jacBody(m,d,jp,jr,b)
                position_error=d.xpos[b]-target_p
                r[row:row+3]=position_error*1000
                jac[row:row+3,cols]=jp@mapping*1000;row+=3
                error=Rotation.from_matrix(target_r.T@d.xmat[b].reshape(3,3)).as_rotvec()
                skew=np.array([[0,-error[2],error[1]],[error[2],0,-error[0]],[-error[1],error[0],0]])
                angle=np.linalg.norm(error)
                coeff=1/12 if angle<1e-4 else 1/angle**2-(1+np.cos(angle))/(2*angle*np.sin(angle))
                inverse=np.eye(3)-.5*skew+coeff*skew@skew
                r[row:row+3]=error*100
                rotation_jac=inverse@target_r.T@jr@mapping
                jac[row:row+3,cols]=rotation_jac*100;row+=3
                foot_errors.append(np.r_[position_error,error]);foot_jacs.append(np.r_[jp@mapping,rotation_jac])
            mujoco.mj_jacSubtreeCom(m,d,jp,root)
            root_com_jac=jp.copy()
            r[row:row+2]=(d.subtree_com[root,:2]-com[i,:2])*com_weight
            jac[row:row+2,cols]=jp[:2]@mapping*com_weight;row+=2
            com_jac=jp[:2]@mapping
            collision_column=0
            for g in geoms:
                is_foot=m.geom(g).name.startswith('foot_')
                foot_margin=a.foot_margin*(1-end_blend[i])-.0008*end_blend[i] if standing is not None else a.foot_margin
                if a.contact_consistent and is_foot:
                    j=bodies.index(int(m.geom_bodyid[g]));strength=max(contact_strength[i,j],end_blend[i])
                    foot_margin=a.foot_margin*(1-strength)-.0008*strength
                for obstacle,margin in [(box_id,foot_margin if is_foot else .006),(floor_id,max(.001,foot_margin) if is_foot and standing is None else foot_margin if is_foot else .006)]:
                    dist=mujoco.mj_geomDistance(m,d,g,obstacle,.05,witness)
                    gap=margin-dist;violations[i,collision_column]=gap
                    shifted=gap+lagrange[i,collision_column]/penalty
                    if shifted>0:
                        r[row]=shifted*np.sqrt(penalty)
                        if abs(dist)>1e-9:
                            normal=(witness[3:]-witness[:3])/dist
                            mujoco.mj_jac(m,d,jp,jr,witness[:3],int(m.geom_bodyid[g]))
                            jac[row,cols]=(normal@jp@mapping)*np.sqrt(penalty)
                    row+=1;collision_column+=1
            for k in range(max_faces):
                if k<len(balance_normals[i]):
                    normal=balance_normals[i][k]
                    gap=float(normal@d.subtree_com[root,:2]+balance_offsets[i][k]+.005)
                    shifted=gap+lagrange[i,collision_column]/penalty
                    violations[i,collision_column]=gap
                    if shifted>0:
                        r[row]=shifted*np.sqrt(penalty);jac[row,cols]=normal@com_jac*np.sqrt(penalty)
                else:violations[i,collision_column]=-1.
                row+=1;collision_column+=1
            if stance_count:
                tolerance=np.r_[np.full(3,.0015),np.full(3,.005)]
                units=np.r_[np.ones(3),np.full(3,.05)]
                for j in range(2):
                    active=contact_strength[i,j]>=1-1e-8
                    for k in range(6):
                        for sign in (1.,-1.):
                            gap=(sign*foot_errors[j][k]-tolerance[k])*units[k] if active else -1.
                            violations[i,collision_column]=gap;shifted=gap+lagrange[i,collision_column]/penalty
                            if active and shifted>0:r[row]=shifted*np.sqrt(penalty);jac[row,cols]=sign*foot_jacs[j][k]*units[k]*np.sqrt(penalty)
                            row+=1;collision_column+=1
            if ankle_count:
                mass=float(m.body_subtreemass[root]);gravity=m.opt.gravity
                def skew_of(vector):
                    x,y,z=vector;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                for k,(jid,body,cap) in enumerate(ankle_info):
                    side=k//2
                    active=contact_strength[i,side]>=1-1e-8 and contact_strength[i,1-side]<1-1e-8
                    if active:
                        subtree_mass=float(m.body_subtreemass[body]);upstream_mass=mass-subtree_mass
                        lever=mass*d.subtree_com[root]-subtree_mass*d.subtree_com[body]-upstream_mass*d.xanchor[jid]
                        mujoco.mj_jacSubtreeCom(m,d,jp,body);lever_jac=mass*root_com_jac-subtree_mass*jp
                        mujoco.mj_jac(m,d,jp,jr,d.xanchor[jid],body);lever_jac-=upstream_mass*jp
                        axis=d.xaxis[jid];moment=np.cross(lever,gravity)
                        fraction=-float(axis@moment)/cap
                        derivative=-(moment@(-skew_of(axis)@jr)+axis@(-skew_of(gravity)@lever_jac))@mapping/cap
                    for sign in (1.,-1.):
                        gap=(sign*fraction-.9)*.05 if active else -1.
                        violations[i,collision_column]=gap;shifted=gap+lagrange[i,collision_column]/penalty
                        if active and shifted>0:r[row]=shifted*np.sqrt(penalty);jac[row,cols]=sign*derivative*.05*np.sqrt(penalty)
                        row+=1;collision_column+=1
            if waist_count:
                gravity_data.qpos[:]=q;gravity_data.qvel[:]=0;gravity_data.qacc[:]=0
                mujoco.mj_inverse(gravity_model,gravity_data)
                torque=gravity_data.qfrc_inverse[waist_dofs]/waist_caps
                mujoco.mjd_inverseFD(gravity_model,gravity_data,1e-6,False,gravity_derivative,None,None,None,None,None,None)
                derivative=gravity_derivative.T[waist_dofs]@mapping/waist_caps[:,None]
                for k in range(3):
                    for sign in (1.,-1.):
                        gap=(sign*torque[k]-.8)*.05
                        violations[i,collision_column]=gap;shifted=gap+lagrange[i,collision_column]/penalty
                        if shifted>0:r[row]=shifted*np.sqrt(penalty);jac[row,cols]=sign*derivative[k]*.05*np.sqrt(penalty)
                        row+=1;collision_column+=1
            assert row==(i+1)*width
        row=n*width
        for i in range(1,n-1):
            r[row:row+nv]=(x[i-1]-2*x[i]+x[i+1])*acceleration_scale
            for offset,mult in [(-1,1),(0,-2),(1,1)]:
                jac[row:row+nv,slice((i+offset)*nv,(i+offset+1)*nv)]=np.diag(acceleration_scale*mult)
            row+=nv
        for first,last in [(0,1),(n-2,n-1)]:
            r[row:row+nv]=(x[last]-x[first])*boundary_scale
            jac[row:row+nv,slice(first*nv,(first+1)*nv)]=-np.diag(boundary_scale)
            jac[row:row+nv,slice(last*nv,(last+1)*nv)]=np.diag(boundary_scale);row+=nv
        calls+=1
        if calls%5==0:print(f'evaluation={calls} cost={np.dot(r,r)/2:.6f} max_residual={abs(r).max():.6f}',flush=True)
        cache.update(x=flat.copy(),r=r,j=jac.tocsr(),violations=violations)
        return r,cache['j']

    epsilon=np.minimum(1e-8,(high-low)/4)
    start=np.clip(initial,low+epsilon,high-epsilon).ravel()
    if a.resume:
        saved=np.load(a.resume)
        if 'x' in saved:
            prior_hash=str(saved['source_sha256'])
            np.testing.assert_array_equal(saved['indices'],indices)
            start=saved['x'].copy()
        else:
            prior_hash=json.loads((a.resume.parent/'launch.json').read_text())['source_sha256']
            prior=saved['qpos'][indices]
            delta=(Rotation.from_quat(base[:,[4,5,6,3]]).inv()*Rotation.from_quat(prior[:,[4,5,6,3]])).as_rotvec() if nr else np.empty((n,0))
            start=np.c_[prior[:,:3],delta,prior[:,qcols]].ravel()
        if prior_hash!=hashlib.sha256(a.source.read_bytes()).hexdigest():raise ValueError('Resume source differs')
        if start.shape!=initial.ravel().shape:raise ValueError('Resume dimension differs')
        projected=np.clip(start,(low+epsilon).ravel(),(high-epsilon).ravel())
        (a.out/'resume-projection.json').write_text(json.dumps({'maximum_initial_guess_projection':float(abs(projected-start).max()),'note':'Initial guess only; final trajectory is optimized under bounds and independently audited'},indent=2)+'\n')
        start=projected
    if a.check_jacobian:
        r,jac=evaluate(start);checks=[]
        for i in [0,round((n-1)*.6),round((n-1)*.7),n-1]:
            for column in ((2,3,4,5,20,24) if a.free_shoulders and nr else (2,3,4,5,20) if nr else (0,2,3,7,14)):
                c=i*nv+column;plus=start.copy();minus=start.copy();plus[c]+=1e-6;minus[c]-=1e-6
                numeric=(evaluate(plus)[0]-evaluate(minus)[0])/2e-6
                analytic=jac[:,c].toarray().ravel()
                checks.append(float(np.linalg.norm(numeric-analytic)/max(1.,np.linalg.norm(numeric))))
        report={'max_relative_column_error':max(checks),'passed':max(checks)<.02}
        (a.out/'jacobian-check.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
        if not report['passed']:raise SystemExit('Jacobian check failed')
        return
    for outer in range(a.constraint_rounds):
        result=least_squares(lambda x:evaluate(x)[0],start,jac=lambda x:evaluate(x)[1],bounds=(low.ravel(),high.ravel()),max_nfev=a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6,verbose=1,tr_solver='lsmr',x_scale='jac')
        evaluate(result.x);gap=cache['violations'].copy()
        violation=float(max(0,gap.max()))
        constraint_history.append({'round':outer,'maximum_scaled_constraint_violation':violation,'native_clearance_violation_m':float(max(0,gap[:,:2*len(geoms)].max())),'support_polygon_violation_m':float(max(0,gap[:,2*len(geoms):2*len(geoms)+max_faces].max())) if max_faces else 0.,'waist_effort_violation_fraction':float(max(0,gap[:,-waist_count:].max())/.05) if waist_count else 0.,'cost':result.cost,'nfev':result.nfev})
        print('constraint_round',constraint_history[-1],flush=True)
        if violation<a.constraint_tolerance:break
        lagrange=np.maximum(0,lagrange+penalty*gap);penalty*=10
        start=result.x.copy();cache.clear()
    solved=result.x.reshape(n,nv);knots=base.copy();knots[:,:3]=solved[:,:3];knots[:,qcols]=solved[:,3+nr:]
    if nr:knots[:,3:7]=(Rotation.from_quat(base[:,[4,5,6,3]])*Rotation.from_rotvec(solved[:,3:6])).as_quat()[:,[3,0,1,2]]
    np.savez_compressed(a.out/'solver-knots.npz',x=result.x,indices=indices,source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest())
    if standing is not None:np.savez(a.out/'standing-target.npz',qpos=standing,source_sha256=hashlib.sha256(a.standing_endpoint.read_bytes()).hexdigest())
    dense_times=np.arange(end+1)/fps
    bc=([(1,np.zeros(m.nq-4)),(2,np.zeros(m.nq-4))],[(1,np.zeros(m.nq-4)),(2,np.zeros(m.nq-4))])
    scalars=np.c_[knots[:,:3],knots[:,7:]]
    scalar_curve=make_interp_spline(times,scalars,k=5,bc_type=bc)(dense_times)
    rb=([(1,np.zeros(3)),(2,np.zeros(3))],[(1,np.zeros(3)),(2,np.zeros(3))])
    knot_angles=(reference_rotation.inv()*Rotation.from_quat(knots[:,[4,5,6,3]])).as_rotvec()
    angles=make_interp_spline(times,knot_angles,k=5,bc_type=rb)(dense_times)
    quat=(reference_rotation*Rotation.from_rotvec(angles)).as_quat()[:,[3,0,1,2]]
    q=np.c_[scalar_curve[:,:3],quat,scalar_curve[:,3:]]
    q=np.r_[q,np.repeat(q[-1:],round(3*fps),axis=0)]
    velocity=differentiate_qpos(m,q,fps)
    np.savez_compressed(a.out/'aligned-diagnostic.npz',qpos=q,fps=fps,box_center=center,box_half_size=half)
    export_qpos_motion(m,q,a.out/'motion-fk.npz',fps,qvel=velocity,source=a.source)
    report={'cost':result.cost,'optimality':result.optimality,'nfev':result.nfev,'solver_success':bool(result.success),'constraint_history':constraint_history,'frames':len(q),'peak_root_acceleration_m_s2':float(np.linalg.norm(np.diff(q[:,:3],n=2,axis=0)*fps**2,axis=1).max()),'peak_joint_acceleration_rad_s2':float(abs(np.diff(q[:,7:],n=2,axis=0)*fps**2).max()),'box':{'front_x':box.front_x,'center_y':box.center_y,'depth':box.depth,'width':box.width,'height':box.height,'friction':box.friction},'training_approved':False}
    if a.shoulder_reference_limit_deg is not None:
        report['shoulder_reference_limit_deg']=a.shoulder_reference_limit_deg
        report['shoulder_peak_reference_deviation_deg']=float(np.degrees(abs(solved[:,shoulder_columns]-reference[:,shoulder_columns])).max())
    (a.out/'repair.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
