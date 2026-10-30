"""Evaluation-only EMA of the trainable standing expert; never an acting policy."""

import copy
import math

import torch


class StandingEMA:
    def __init__(self, actor, decay=0.99):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('EMA decay must be finite and in [0, 1)')
        self.decay = float(decay)
        self.updates = 0
        self.model = copy.deepcopy(actor).requires_grad_(False).eval()

    @torch.no_grad()
    def update(self, actor, *, changed):
        """Call after a completed BPTT update; rejected and no-op steps are skipped."""
        if not changed:
            return
        current = dict(actor.named_parameters())
        for name, averaged in self.model.named_parameters():
            if self.updates == 0:
                averaged.copy_(current[name])
            else:
                averaged.lerp_(current[name], 1. - self.decay)
        # Scales and other buffers describe the policy, not trainable estimates.
        current_buffers = dict(actor.named_buffers())
        for name, averaged in self.model.named_buffers():
            averaged.copy_(current_buffers[name])
        self.updates += 1

    def metadata(self):
        return dict(decay=self.decay, updates=self.updates, scope='standing_skill',
                    evaluation_only=True, initialization='copy first effective actor update',
                    update_unit='effective BPTT actor optimizer step')

    def policy_state(self, online_state):
        """Export a normal complete policy while retaining the exact frozen base."""
        exported = {key: value.detach().cpu().clone() for key, value in online_state.items()}
        actor_state = self.model.state_dict()
        expected = {key.removeprefix('standing_skill.') for key in exported if key.startswith('standing_skill.')}
        if expected != set(actor_state):
            raise ValueError('EMA and online standing expert state layouts differ')
        for key, value in actor_state.items():
            exported['standing_skill.' + key] = value.detach().cpu().clone()
        return exported


def validate_pair(online, averaged, online_sha256):
    """Reject comparisons with mismatched training provenance or a changed base."""
    raw_meta, ema_meta = online['policy_ema'], averaged['policy_ema']
    if raw_meta['variant'] != 'online' or ema_meta['variant'] != 'ema':
        raise ValueError('Expected online and EMA variants')
    if ema_meta.get('paired_online_sha256') != online_sha256:
        raise ValueError('EMA export belongs to a different online checkpoint')
    for ck in (online, averaged):
        meta = ck['supermimic_stop']
        if meta['algorithm'] != 'supermimic_bptt' or meta['actor_updates'] < 2:
            raise ValueError('At least two effective BPTT updates are required for an EMA comparison')
        if ck['policy_ema']['updates'] != meta['actor_updates']:
            raise ValueError('EMA update count does not match effective actor updates')
    for key in ('iteration', 'source_sha256', 'reference_sha256', 'entry_sha256', 'config', 'source_commit'):
        if online['supermimic_stop'][key] != averaged['supermimic_stop'][key]:
            raise ValueError(f'EMA provenance differs: {key}')
    for key in ('config', 'standing_skill', 'standing_reference'):
        if online[key] != averaged[key]:
            raise ValueError(f'EMA policy configuration differs: {key}')
    for key in ('decay', 'updates', 'scope', 'initialization', 'update_unit', 'evaluation_only'):
        if raw_meta[key] != ema_meta[key]:
            raise ValueError(f'EMA metadata differs: {key}')
    for section in ('policy', 'wm'):
        if online[section].keys() != averaged[section].keys():
            raise ValueError(f'EMA state layout differs: {section}')
        for key, value in online[section].items():
            other = averaged[section][key]
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError(f'Non-finite state: {section}.{key}')
            if section != 'policy' or not key.startswith('standing_skill.'):
                if not torch.equal(value, other):
                    raise ValueError(f'EMA changed a shared frozen tensor: {section}.{key}')
