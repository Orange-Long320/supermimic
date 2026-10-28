"""Experimental early-contact control using calibrated native short previews.

Only ordinary position actions change. Dynamics and acceptance thresholds stay
in the independent native execution environment.
"""
import time

import numpy as np
from scipy.optimize import minimize

from supermimic.robots.dimensions import PROFILE


class NativeContactController:
    def __init__(self, preview, trust=.05, iterations=12):
        self.preview = preview
        self.trust = trust; self.iterations = iterations
        names = PROFILE['joint_names']
        self.indices = [i for i, n in enumerate(names) if any(s in n for s in ['hip_', 'knee_', 'ankle_'])]
        self.conversion = np.array([PROFILE['effort_limit'][names[i]]/PROFILE['stiffness'][names[i]]/preview.scale[i]
                                    for i in self.indices])
        self.lower, self.upper = preview.model.jnt_range[1:].T
        self.speed = np.array([PROFILE['velocity_limit'][n] for n in names])

    def project(self, qpos, qvel, observation, goal, action, current_time):
        started = time.monotonic(); nq = self.preview.model.nq
        count = len(self.indices); cache = {}; evaluations = 0

        def evaluate(x):
            nonlocal evaluations
            key = x.tobytes()
            if key not in cache:
                change = np.zeros(len(PROFILE['joint_names']))
                change[self.indices] = x*self.conversion
                prediction = self.preview.predict(qpos, qvel, observation, current_time,
                    correction=change, initial_goal=goal)
                state = prediction['states']; q = state[:, 7:nq]
                hard = max(float((self.lower-q).max()), float((q-self.upper).max()), 0.)
                foot = max(-.004-float(prediction['heights'][:, :2].min()), 0.)
                body = max(.006-float(prediction['heights'][:, 2].min()), 0.)
                speed = max(float((np.abs(state[:, nq+6:])/self.speed).max())-1, 0.)
                constraints = np.r_[((prediction['heights'][:, :2]+.004)/.005).ravel(),
                    ((prediction['heights'][:, 2]-.006)/.01).ravel(),
                    ((q-self.lower-.001)/.05).ravel(), ((self.upper-.001-q)/.05).ravel(),
                    (1-np.abs(state[:, nq+6:])/self.speed).ravel()]
                quality = (hard > 5e-5, max(hard-5e-5, 0.), body > 0, body, speed > 0, speed, foot, float(x@x))
                cache[key] = (prediction, constraints, quality)
                evaluations += 1
            return cache[key]

        zero = np.zeros(count); baseline, constraints, baseline_quality = evaluate(zero)
        reconstruction_error = float(np.abs(baseline['actions'][0]-np.asarray(action)).max())
        if reconstruction_error > 3e-5:
            raise ValueError(f'Preview does not reproduce the live baseline command: {reconstruction_error}')
        if constraints.min() >= 0:
            result = None; selected = zero
        else:
            result = minimize(lambda x: .5*float(x@x), zero, jac=lambda x: x,
                method='SLSQP', bounds=[(-self.trust, self.trust)]*count,
                constraints=[{'type': 'ineq', 'fun': lambda x: evaluate(x)[1]}],
                options={'maxiter': self.iterations, 'ftol': 1e-8, 'eps': 1e-3})
            # Failed optimisation is still a proposal: compare its physical
            # forecast against the unmodified command before native execution.
            selected = result.x if np.isfinite(result.x).all() and evaluate(result.x)[2] < baseline_quality else zero
        prediction, selected_constraints, selected_quality = evaluate(selected)
        change = np.zeros(len(PROFILE['joint_names']))
        change[self.indices] = selected*self.conversion
        projected = np.asarray(action)+change
        report = {'time_s': current_time, 'elapsed_s': time.monotonic()-started, 'preview_evaluations': evaluations,
            'optimizer_success': bool(result.success) if result is not None else True,
            'optimizer_message': str(result.message) if result is not None else 'Baseline forecast satisfies preview constraints',
            'accepted_correction': bool(np.any(selected)), 'normalized_effort_corrections': selected.tolist(),
            'baseline_minimum_foot_height_m': float(baseline['heights'][:, :2].min()),
            'selected_minimum_foot_height_m': float(prediction['heights'][:, :2].min()),
            'baseline_predicted_hard_violation_rad': float(max((self.lower-baseline['states'][:, 7:nq]).max(),
                (baseline['states'][:, 7:nq]-self.upper).max(), 0.)),
            'selected_predicted_hard_violation_rad': float(max((self.lower-prediction['states'][:, 7:nq]).max(),
                (prediction['states'][:, 7:nq]-self.upper).max(), 0.)),
            'selected_minimum_scaled_constraint': float(selected_constraints.min()),
            'baseline_action_reconstruction_error': reconstruction_error,
            'scope': 'Local predictive proposal only; native execution and full physical acceptance remain authoritative'}
        return projected, prediction, report
