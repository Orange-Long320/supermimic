"""Open-loop checks on fresh physical replay, without fitting or policy updates."""

import torch

from supermimic.locomotion.task import measured_velocity, velocity_reward


@torch.no_grad()
def validate_velocity_model(wm, buffer, nominal_height, default_joint_pos, dt, samples=256, horizons=(1, 8, 32)):
    """Use the same episode-safe starts and recorded executed actions at every horizon.

    The caller must collect a fresh buffer after loading the checkpoint and must
    not update from that buffer. This is an action-replay check, not a closed-loop
    policy or generalization score. Quantities follow [vx m/s, vy m/s, wz rad/s].
    """
    if not horizons or min(horizons) < 1 or samples < 1:
        raise ValueError("Positive sample count and horizons required")
    mask = buffer.window_mask(max(horizons)) & buffer.velocity_env_mask[:, None]
    available = int(mask.sum())
    if available == 0:
        raise RuntimeError("No complete velocity windows for the longest horizon")
    e, t = buffer._sample_from_mask(mask, samples)
    state = buffer.s[e, t].clone()
    initial = measured_velocity(state)
    native_reward = torch.zeros(samples, device=buffer.device)
    model_reward = torch.zeros_like(native_reward)
    checks = []
    for offset in range(max(horizons)):
        time = (t + offset) % buffer.T
        state = wm.step(state, buffer.a_exec[e, time], buffer.xi[e, time])
        previous = buffer.a[e, (time - 1) % buffer.T]
        previous = torch.where(buffer.ep_start[e, time, None], 0., previous)
        model_reward += velocity_reward(
            state, buffer.command[e, time], buffer.a[e, time], previous,
            wm.heads.soft_limits, nominal_height, default_joint_pos,
        )
        native_reward += buffer.r[e, time] / dt
        horizon = offset + 1
        if horizon in horizons:
            truth = buffer.s[e, (t + horizon) % buffer.T]
            prediction, actual = measured_velocity(state), measured_velocity(truth)
            error = prediction - actual
            checks.append({
                "horizon": horizon, "seconds": horizon * dt,
                "velocity_rmse": error.square().mean(0).sqrt().tolist(),
                "velocity_bias": error.mean(0).tolist(),
                "predicted_velocity": prediction.mean(0).tolist(),
                "actual_velocity": actual.mean(0).tolist(),
                "initial_velocity": initial.mean(0).tolist(),
                "model_reward_rate": float(model_reward.mean() / horizon),
                "native_reward_rate": float(native_reward.mean() / horizon),
            })
    return {"available_windows": available, "samples_with_replacement": samples,
            "same_starts_for_all_horizons": True, "checks": checks}
