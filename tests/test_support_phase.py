from supermimic.control.support_phase import SupportMeasurement, SupportStage, SupportStageConfig, SupportStageMonitor


def sample(t, speed=.02, omega=.05, loads=(120., 120.), slip=(.01,.01)):
    return SupportMeasurement(t,speed,omega,loads,slip)


def test_move_brake_land_hold_and_hysteresis():
    m=SupportStageMonitor(SupportStageConfig(weight_n=400.))
    assert m.update(0.,True,None)==SupportStage.MOVING
    assert m.update(.1,False,sample(.1,speed=.4,loads=(100.,0.)))==SupportStage.BRAKING
    assert m.update(.2,False,sample(.2))==SupportStage.LANDING
    assert m.update(.6,False,sample(.6))==SupportStage.LANDING
    assert m.update(.71,False,sample(.71))==SupportStage.HOLD
    assert m.update(.72,False,sample(.72,speed=.1,loads=(30.,30.),slip=(.04,.04)))==SupportStage.HOLD
    assert m.update(.73,False,sample(.73,loads=(0.,120.)))==SupportStage.LANDING


def test_missing_stale_and_recovery_require_new_stability_window():
    m=SupportStageMonitor(SupportStageConfig(weight_n=400.))
    assert m.update(0.,False,None)==SupportStage.UNVERIFIED
    assert m.update(.1,False,sample(.1))==SupportStage.LANDING
    assert m.update(.7,False,sample(.1))==SupportStage.UNVERIFIED
    assert m.update(.71,False,sample(.71))==SupportStage.LANDING
    assert m.update(1.22,False,sample(1.22))==SupportStage.HOLD


def test_nonfinite_force_is_not_support():
    m=SupportStageMonitor(SupportStageConfig(weight_n=400.))
    assert m.update(0.,False,sample(0.,loads=(float('nan'),120.)))==SupportStage.UNVERIFIED
