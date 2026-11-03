# X2 preparation pose on hardware, 2026-11-03

Purpose: run the preparation-only takeover (`manual_supervised_unvalidated_trial`,
hanging fall arrest, feet loaded, operator on the emergency stop) with the 500 Hz
IMU state chain, moving to the `x2_rl_deploy` example initial pose and holding it.
No policy handover, no policy targets, no automatic host-loss protection.

All runs were executed inside the rootless `lxh` container on PC2. The robot was
suspended and the operator held the emergency stop. Nothing here is a validation
result: preparation/handover dynamics, command limits and the state pipeline are
still unvalidated.

## Why the first attempts never reached the motors

Four earlier power-on attempts produced no motor command at all. The blocking
cause was not host CPU contention and not the camera/perception stack:

| Observation | Measurement |
| --- | --- |
| Five device streams, light subscriber | leg/waist/arm/IMU ≈495 Hz, head 331 Hz, per-stream max gap ≤18 ms |
| Four-group stamp spread, light subscriber | p50 2.0 ms, p99 20.0 ms, only 0.31 % over 25 ms |
| Same spread as seen by the producer | 32 % of cycles over the 25 ms gate (`skipped_skew` 348/1081) |
| Producer publish rate / gaps | 32.2 Hz, gaps p99 262 ms, max 1180 ms |
| IMU receive age inside the producer | p50 38.7 ms, p95 40.2 ms, clipped at depth 20 × 2 ms |
| Consumer-side packet age (worker's view) | p50 4-5 ms, p99 36 ms, max 42 ms, 0 rejects |

Three separate defects were behind this:

1. The skew gate compared the *newest* joint samples against an IMU sample that
   is 17-40 ms older, so a third of the cycles were discarded and bursts of
   skips produced second-long holes in the published stream.
2. The producer could not drain 5 streams × ~500 Hz with one `spin_once` per
   loop iteration, so the IMU receive queue sat at its 20-message cap and every
   packet inherited a ~39 ms age. The consumer's 50 ms budget, plus the
   publisher's 10 ms command tick, could not be met.
3. The publisher latched its target lease without printing the reason, so a
   failed lease looked identical to "no target was ever needed".

## Changes

* `scripts/deploy/publish_pelvis_packet.py`
  * Coherent snapshot: take the IMU sample as the observation instant and use
    each joint group's nearest sample to it. The five stamps then spread by one
    joint sample period instead of tracking the IMU dispatch delay.
  * `--subscribe-depth` for the receive queues. A deep queue on a saturated
    subscriber is a fixed lag, not fresh data: depth 3 turns 39 ms into 2 ms.
  * Reject a publish whose observation instant repeats the previous one.
* `scripts/deploy/x2_11500_worker.py`
  * Drop packets that were already late when dispatch resumed, and drop repeated
    observation instants, instead of latching a fault. Stale state is never
    used; the controller's tick-gap guard remains the hard limit.
* `scripts/deploy/x2_hardware_publisher.py`
  * Print a latched target-lease fault once, so "nothing was sent" is visible.

## State chain after the changes

Producer, 22 s, `--subscribe-depth 3`: 48.7 Hz publish, 1077 packets,
`skipped_duplicate` 0, `skipped_skew` 2, `skipped_age` 0, IMU receive age
p50 2.4 ms / p99 6.3 ms.

Consumer view over the same window: packet age p50 4.6 ms / p99 11.4 ms /
max 24.5 ms, publish gap p50 20.2 ms / p99 24.5 ms / max 36.7 ms, stamp spread
p99 4.2 ms, zero packets rejected by the worker gate.

## Trial result (run-r15, plan `46cc03dcafc9547c`)

Producer 48.1 Hz, 1712 packets, `invalid` 1, `skipped_age` 2, `skipped_skew` 2.
Worker: no fault. Publisher: no lease fault, no fallback.

Joint trajectory (read-only recording, 48 Hz, 26 s):

| t (s) | left elbow | left shoulder pitch | left knee | left hip pitch | waist pitch |
| --- | --- | --- | --- | --- | --- |
| 0.0 | −0.376 | −0.401 | 0.015 | −0.383 | 0.349 |
| 11.7 | −1.100 | 0.264 | 0.054 | −0.267 | 0.077 |
| 14.3 | −1.163 | 0.327 | 0.061 | −0.247 | 0.057 |
| 26.0 | −1.163 | 0.327 | 0.061 | −0.247 | 0.056 |
| target | −1.200 | 0.400 | 0.100 | −0.050 | 0.000 |

The pose was reached at ~14 s and held unchanged for 12 s. Residuals: elbow
0.037, shoulder pitch 0.073, knee 0.039, waist pitch 0.056, **left hip pitch
0.197** (it stopped at −0.247 instead of −0.050; the right hip pitch residual is
0.101). Peak measured joint rate 0.84 rad/s over a 20 ms window on the elbow,
against a commanded quintic peak of ~0.25 rad/s; the arm tracks in bursts
(stick-slip) rather than smoothly, so the recorded peak is not a command
violation but it is not the commanded rate either.

## Follow-up runs (run-r16, run-r17, plan `b12accef075f73b0`, `d1e33e7fb1c32d49`)

run-r16: the producer aborted inside its 3 s calibration window
(`Calibration window is moving, invalid, or has an IMU timestamp gap`). The
shallow receive queue plus the startup load can leave a >30 ms IMU stamp gap in
that window, and the calibration gate rejects it. No packet was published, so no
motor command was sent and the robot stayed where it was. This is intermittent
(1 of 3 runs) and is not yet handled: the gate should wait for a clean window
instead of exiting.

run-r17: 47.4 Hz, 1685 packets, `invalid` 0, `skipped_duplicate` 5,
`imu_discontinuity_skipped` 15, no worker fault, no lease fault. The robot was
already at the target pose, so travel was only 0.035-0.039 rad per joint and the
run is a hold test rather than a motion test.

Residual to the example initial pose after run-r17 (every joint within 0.076 rad):

| joint | residual (rad) |
| --- | ---: |
| right knee | 0.075 |
| left / right shoulder roll | 0.067 / 0.066 |
| left knee | 0.064 |
| right elbow | 0.050 |
| left elbow | 0.046 |
| left hip pitch | 0.024 |

The 0.197 rad left hip pitch residual in run-r15 was a convergence lag, not a
stall: the HAL keeps applying the last command with its stiffness after the
publisher exits, and the hip finished converging from -0.247 to -0.074 between
runs. Any residual measured at the instant the publisher is killed understates
what the joint would reach.

## Known open items

* Left hip pitch does not reach the commanded pose under load; the left/right
  asymmetry (knee, hip) is consistent with a joint zero that is still
  unverified.
* The pose is held by the software path only while the publisher lives. When the
  trial's bounded window ends or the lease expires, the publisher sends its
  fallback: zero stiffness with damping. That is what happened in run-r14 and
  why the arms dropped at up to 2.8 rad/s; it is designed behaviour, not a hold.
* No automatic host-loss protection; a human holds the e-stop.
* The 500 Hz state chain is a diagnostic producer; its pose is not a verified
  estimate.

## Policy handover attempts (run-p1 .. run-p5)

The handover chain itself works: `/supermimic/x2/policy_request` with the plan
hash is accepted by the worker, the phase moves to `handover`, and the publisher
acknowledges. Four attempts reached that point; none reached `active`.

What each attempt faulted on, and what it taught:

| run | fault | meaning |
| --- | --- | --- |
| p1 | `Measured joint outside contract` | shoulder roll sits 0.05 rad from its bound; arm swing crossed it |
| p2 | `Tilt ... deviation 0.2013 > 0.200` | body attitude changes 0.20 rad during the arm swing |
| p3 | `Target step exceeds command contract` (+0.1502, -0.1703 vs 0.15) | preparation pose is not the policy's stance |
| p4 | `Tilt ... deviation 0.20036 > 0.200` | same sway, 0.0004 rad over |
| p5 | `Measured joint outside contract` (right ankle -0.8106 vs lower -0.803) | leg is pinned at the model limit in a deep squat |

Two real findings:

1. **The preparation pose was not the policy's stance.** `startup.position`
   (knees 0.100, ankles -0.050, elbows -1.200) differs from `action_default`
   (knees 0.669, ankles -0.363, elbows -0.600) by up to 0.600 rad. The policy's
   observations and actions are defined around `action_default`, so handing over
   from the startup pose asks for a 0.57 rad knee move at handover. The contract
   builder now takes `--prep-target action_default` and records the choice as
   `prep_target` / `prep_target_source` in the plan.
2. **The robot has been squatting since the first fallback.** Zero-stiffness
   fallback at the end of run-p1 let the legs sag; every later run started from
   that pose and the preparation could not lift it back:

   | run | knees L/R (rad) | ankle pitch L/R (rad) |
   | --- | --- | --- |
   | r15 | +0.015 / +0.136 | +0.167 / -0.019 |
   | r17 | +0.147 / +0.144 | +0.011 / -0.036 |
   | p1 | +0.163 / +0.175 | -0.010 / -0.071 |
   | p2 | +1.115 / +1.202 | -0.797 / -0.804 |
   | p3-p5 | ~+1.0 / ~+1.1 | -0.804 / -0.799 |

   The ankles sit on the model limit (-0.803), and the measured ankle stayed
   there while the preparation target asked for -0.363: with the body's weight on
   the feet, the preparation gains cannot extend the legs, so the loaded joints
   do not track while the unloaded arms do. The evidence file now records this
   start pose as an unvalidated item.

Correction to an earlier reading: the ankle is 0.0015 rad outside its modelled
range, not 0.15. `-0.803` is the model's own lower limit, so the joint is at the
limit rather than through it.

Next attempt needs the robot out of the squat first: unload the feet (raise the
harness) so the legs can extend, or use the native motion controller to stand it
up before the trial. Raising `--max-tilt-delta` to 0.35 and preparing to
`action_default` are already in place.

## Preparation to the policy's starting pose (run-m1 .. run-m4)

The policy's action offset `action_default` is the stance its observations are
defined around, so the preparation now targets it (`--prep-target action_default`).

Two failures happened before the pose was reached, both in the trial harness
rather than the robot:

* run-m2: the publisher latched `Source state lease expired` because packet age
  reached 56 ms, above the 50 ms lease, and every later command was rejected as
  `Command step outside contract`. The arms and waist therefore never moved.
* run-m3: the ad-hoc preflight used `pgrep -f '<pattern>'`, whose own command line
  contains the pattern, so it matched itself and aborted before anything ran.

Fix: the producer now publishes only packets up to 35 ms old (`--max-age 0.035`),
leaving the publisher's 10 ms tick inside the 50 ms lease. That is a tightening
on the producer side; no gate was widened.

run-m4 (plan `ce638b2c1e9efa4f`) reached the pose with no publisher fault at all:

| joint | start | end | target | residual |
| --- | ---: | ---: | ---: | ---: |
| left knee | +0.528 | +0.610 | +0.669 | -0.059 |
| right knee | +0.586 | +0.609 | +0.669 | -0.060 |
| left ankle pitch | -0.338 | -0.386 | -0.363 | -0.023 |
| right ankle pitch | -0.437 | -0.407 | -0.363 | -0.044 |
| left hip pitch | -0.358 | -0.390 | -0.312 | -0.078 |
| left elbow | -0.129 | -0.602 | -0.600 | -0.002 |
| left shoulder pitch | -0.488 | +0.110 | +0.200 | -0.090 |
| waist pitch | +0.349 | +0.063 | +0.000 | +0.063 |

Worst residual over all 31 joints is the shoulder roll (0.135 / 0.113), the joint
whose encoder zero is still unverified. Peak measured rate 0.45 rad/s.

## Open blocker for handover: the policy's elbow baseline

Offline, with the nominal observation and the real `SkillGoal`, the policy asks
for an elbow action of about +4.5, i.e. target -2.50 rad, against a command
lower bound of -2.356. This does not depend on the goal: a zero goal gives
+4.50, a quarter-magnitude goal +4.52, the real goal +4.67, the synthetic fixture
goal +4.87. So it is a baseline offset in the policy output, not a goal or
observation-assembly error.

Until that offset is explained (trained default pose, joint sign convention, or
exported action scale), a handover will fault on the elbows regardless of the
preparation pose.

## Session 20261103 afternoon: the handover fault was a bounded-window bug

Runs `run-p26` … `run-q4` in
`trials/11500_prep_imu_20261102_r1/` on PC2 (container `lxh`). Every run below
is an explicitly unvalidated manual trial with the operator at the e-stop.

### What actually stopped the earlier policy runs

`run-p26`, `run-p28` and `run-p29` did not fail on the policy. The publisher hit
`target lease fault: Bounded manually supervised trial ended` 30 s after its
first command, fell back to zero stiffness, and the joints sagged. `--unbounded`
only cleared the contract log line: `plan.json` still carried
`manual_supervision.max_trial_seconds = 30`, which is what the publisher reads.
Fixed in `build_x2_11500_trial.py` (the plan now carries `None` plus
`unbounded: true`) and accepted in `x2_hardware_publisher.validated_contract`
only for that explicit combination. `run-q4` confirms it: 7150 policy ticks with
no fallback.

### The policy is not the problem

- `goal[:31] + action_default` equals the preparation target to 6e-4 rad, so the
  goal construction matches the exported policy.
- Re-running the policy offline on the observation the live worker dumped
  (`run-p30/obs.json`) reproduces the live action bit for bit (max diff 0.0).
- The reference trajectory frame is upright (0.0 deg); the fitted dataset is not
  leaning, so a lean on the robot is a robot-side condition.

### The robot-side condition: the reference stance sags under load

With the SDK preparation gains the ankle sits 0.25–0.35 rad below its target and
the body leans ~20 deg. Scaling the hip/knee/ankle preparation stiffness by 3.5
(kd with kp, `--prep-leg-kp-scale`) moves it to:

| quantity | SDK gains | scaled 3.5 |
| --- | --- | --- |
| worst joint error at end of preparation | 0.45 rad | 0.178 rad |
| ankle pitch error | 0.40 rad | 0.11–0.18 rad |
| body tilt after preparation | 20.0 deg | 19.3 deg |
| peak joint rate | 1.25 rad/s | 0.43 rad/s |

The tilt does not follow the joint error: the remaining lean is the PD
steady-state sag, so an I-term (or much higher gains) is still needed to reach
the reference attitude. Not implemented yet.

### `run-q4`: the policy does straighten the body

`SM_PREP_SECONDS=20 SM_ACTIVE_SECONDS=180`, contract with
`--prep-leg-kp-scale 3.5 --handover-max-tilt 0.40 --max-tilt-delta 0.6
--blend-seconds 6 --max-target-step 0.03 --unbounded`. Handover acknowledged at
14:09:43 and the policy stayed in `phase active` with no tilt, angular-speed,
state-age or lease fault:

| time under policy | body tilt |
| --- | --- |
| 0 s (handover) | 19.4 deg |
| ~40 s | 7.5 deg |
| ~120 s | 4–5 deg (last 10 s mean 5.0, max 14.7) |

The policy still saturates the ankle by up to 3.2 rad, which is expected: the
policy-phase ankle stiffness is 14.3 (soft, so the policy asks past the limit to
produce torque) and the reference deployment performs no range check at all.
Saturation is a normal actuator condition here, not a contract violation.

The joint recorder was stopped before writing (`run-q4/joints.log` is empty);
the pelvis attitude above comes from `state/packets.jsonl`. A joint-level record
of a long policy run is still missing.

### Standing caveats

- The operator stopped the run at ~145 s of policy time, so this is one run, not
  a repeatable result.
- Encoder zero remains unverified; the shoulder-roll residual (0.13–0.45) is the
  largest one and it is the joint whose zero is not confirmed.
- `handover_max_tilt` exists in the plan but was set to 0.40 rad, i.e. above the
  0.34 rad the preparation actually reaches, so it did not gate anything in this
  run.
