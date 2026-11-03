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
