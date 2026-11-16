"""Causal adaptive tails, reliability gating and spike-preserving accumulation."""
import numpy as np
import pandas as pd
from numba import njit
from ..development.tails import ExtrapolatedProfiles

@njit(cache=True)
def personal(values, dates, n, t, score, parent_scale, floor):
    if n == 0:
        return (0.0, 0.0, 0.0, parent_scale, score, score)
    v = values[:n]
    age = t - dates[:n]
    w = np.exp(-np.log(2.0) * age / 60.0)
    sw = w.sum()
    sw2 = (w * w).sum()
    neff = min(sw, sw * sw / max(sw2, 1e-300))
    ordered = np.sort(v)
    q75 = np.quantile(ordered, 0.75)
    excess = 0.0
    mass = 0.0
    for j in range(n):
        if v[j] >= q75:
            excess += w[j] * (v[j] - q75)
            mass += w[j]
    b0 = excess / max(mass, 1e-300)
    mixing = neff / (neff + 30.0)
    b = max(mixing * b0 + (1 - mixing) * parent_scale, floor)
    median = np.median(v)
    scale = max(1.4826 * np.median(np.abs(v - median)), parent_scale, floor)
    drift = 0.0
    if n >= 20:
        drift = abs(np.median(v[n - 10:n]) - np.median(v[max(0, n - 50):n - 10])) / scale
    maximum = ordered[-1]
    if score < ordered[0]:
        logp = 0.0
    elif score > maximum:
        at_max = 0.0
        for j in range(n):
            if v[j] == maximum:
                at_max += w[j]
        logp = np.log((1 + at_max) / (1 + sw)) - (score - maximum) / b
    else:
        low = ordered[0]
        high = maximum
        for j in range(n):
            if v[j] <= score and v[j] > low:
                low = v[j]
            if v[j] > score and v[j] < high:
                high = v[j]
        pl = 1.0
        ph = 1.0
        for j in range(n):
            if v[j] >= low:
                pl += w[j]
            if v[j] >= high:
                ph += w[j]
        pl /= 1 + sw
        ph /= 1 + sw
        p = pl if high == low else pl + (ph - pl) * (score - low) / (high - low)
        logp = np.log(max(p, 1e-300))
    lower = np.quantile(ordered, 0.05) - 3 * b
    upper = np.quantile(ordered, 0.95) + 3 * b
    return (logp, neff, drift, b, lower, upper)

@njit(cache=True)
def stream_kernel(uid, roles, days, scores, pg, pr, nr, role_scale, hist, stamps, count, current_role, last_day, cusum, ab_values, ab_days, ab_count, floor, update):
    output = np.empty((len(scores), 6))
    traces = np.empty((len(scores), 6))
    for i in range(len(scores)):
        u = uid[i]
        role = roles[i]
        t = days[i]
        if current_role[u] != role:
            count[u] = 0
            current_role[u] = role
            cusum[u, :] = 0.0
            ab_count[u] = 0
            last_day[u] = -1000000
        n = count[u]
        logpu, neff, drift, b, lower, upper = personal(hist[u], stamps[u], n, t, scores[i], role_scale[i], floor)
        if n == 0:
            logpu = pr[i]
        wu = neff / (neff + 30.0) / (1 + drift * drift)
        wu_simple = neff / (neff + 30.0)
        wr = nr[i] / (nr[i] + 200.0)
        parent = wr * pr[i] + (1 - wr) * pg[i]
        evidence = np.empty(3)
        evidence[0] = max(0.0, -wu * logpu - (1 - wu) * parent)
        evidence[1] = max(0.0, -wu_simple * logpu - (1 - wu_simple) * parent)
        evidence[2] = max(0.0, -wu * logpu - (1 - wu) * pg[i])
        delta = t - last_day[u]
        if delta <= 0:
            raise ValueError('Duplicate or nonchronological user-day')
        decay = np.exp(-delta / 7.0)
        for k in range(3):
            cusum[u, k] = max(0.0, decay * cusum[u, k] + min(evidence[k], 8.0) - 2.0)
        output[i, 0] = max(evidence[0], cusum[u, 0])
        output[i, 1] = max(evidence[1], cusum[u, 1])
        output[i, 2] = max(evidence[2], cusum[u, 2])
        output[i, 3] = evidence[0]
        output[i, 4] = cusum[u, 0]
        na = ab_count[u]
        keep = 0
        for j in range(na):
            if t - ab_days[u, j] < 7:
                ab_values[u, keep] = ab_values[u, j]
                ab_days[u, keep] = ab_days[u, j]
                keep += 1
        ab_values[u, keep] = -logpu
        ab_days[u, keep] = t
        keep += 1
        ab_count[u] = keep
        output[i, 5] = max(-logpu, np.mean(ab_values[u, :keep]))
        traces[i, 0] = neff
        traces[i, 1] = wu
        traces[i, 2] = drift
        traces[i, 3] = cusum[u, 0]
        traces[i, 4] = evidence[0]
        traces[i, 5] = n
        if update:
            value = scores[i] if n == 0 else min(max(scores[i], lower), upper)
            if n == 90:
                hist[u, :89] = hist[u, 1:90]
                stamps[u, :89] = stamps[u, 1:90]
                n = 89
            hist[u, n] = value
            stamps[u, n] = t
            count[u] = n + 1
        last_day[u] = t
    return (output, traces)

class AdaptiveContext:

    def __init__(self, update=True):
        self.update = update

    def fit(self, calibration):
        self.anchor = ExtrapolatedProfiles().fit(calibration)
        self.last_calibration = calibration.date.max()
        self.last_transform = self.last_calibration
        self.user_map = {str(u): i for i, u in enumerate(sorted(calibration.user.unique()))}
        self.role_map = {str(r): i for i, r in enumerate(sorted(calibration.role.unique()))}
        n = len(self.user_map)
        self.hist = np.zeros((n, 90))
        self.stamps = np.zeros((n, 90), np.int64)
        self.count = np.zeros(n, np.int64)
        self.current_role = np.full(n, -1, np.int64)
        self.last_day = np.full(n, -1000000, np.int64)
        self.cusum = np.zeros((n, 3))
        self.ab_values = np.zeros((n, 90))
        self.ab_days = np.zeros((n, 90), np.int64)
        self.ab_count = np.zeros(n, np.int64)
        ordered = calibration.sort_values(['user', 'date'])
        for user, part in ordered.groupby('user', sort=False):
            u = self.user_map[str(user)]
            role = str(part.role.iloc[-1])
            changed = part.role.astype(str).ne(role).to_numpy()
            begin = np.flatnonzero(changed)[-1] + 1 if changed.any() else 0
            part = part.iloc[begin:].tail(90)
            length = len(part)
            self.hist[u, :length] = part.score.to_numpy(float)
            self.stamps[u, :length] = part.date.to_numpy('datetime64[D]').astype(np.int64)
            self.count[u] = length
            self.current_role[u] = self.role_map[role]
        return self

    def transform(self, targets):
        if targets.empty:
            return ({}, pd.DataFrame(index=targets.index))
        if targets.date.min() <= self.last_transform:
            raise ValueError('Each batch must follow all previous batches strictly')
        if targets[['user', 'date']].duplicated().any() or not np.isfinite(targets.score).all():
            raise ValueError('Invalid target observations')
        for user in sorted(set(map(str, targets.user)) - set(self.user_map)):
            self.user_map[user] = len(self.user_map)
        for role in sorted(set(map(str, targets.role)) - set(self.role_map)):
            self.role_map[role] = len(self.role_map)
        needed = len(self.user_map) - len(self.count)
        if needed:
            for name, width, fill in [('hist', 90, 0), ('stamps', 90, 0), ('cusum', 3, 0), ('ab_values', 90, 0), ('ab_days', 90, 0)]:
                old = getattr(self, name)
                setattr(self, name, np.vstack([old, np.full((needed, width), fill, dtype=old.dtype)]))
            for name, fill in [('count', 0), ('current_role', -1), ('last_day', -1000000), ('ab_count', 0)]:
                old = getattr(self, name)
                setattr(self, name, np.r_[old, np.full(needed, fill, dtype=old.dtype)])
        c = self.anchor.components(targets)
        order = np.lexsort((targets.date.to_numpy(), targets.user.astype(str).to_numpy()))
        inverse = np.argsort(order)
        users = targets.user.astype(str).map(self.user_map).to_numpy(np.int64)[order]
        roles = targets.role.astype(str).map(self.role_map).to_numpy(np.int64)[order]
        dates = targets.date.to_numpy('datetime64[D]').astype(np.int64)[order]
        scales = np.array([self.anchor.roles.get(str(r), (None, self.anchor.global_scale))[1] for r in targets.role])[order]
        values, traces = stream_kernel(users, roles, dates, targets.score.to_numpy(float)[order], c['log_global'][order], c['log_role'][order], c['n_role'][order], scales, self.hist, self.stamps, self.count, self.current_role, self.last_day, self.cusum, self.ab_values, self.ab_days, self.ab_count, self.anchor.floor, self.update)
        self.last_transform = targets.date.max()
        names = ['M5', 'A_no_stability', 'A_no_role', 'A_no_temporal', 'A_no_spike', 'AB']
        result = {name: values[inverse, i] for i, name in enumerate(names)}
        trace = pd.DataFrame(traces[inverse], columns=['n_eff', 'weight_user', 'instability', 'cusum', 'instant_evidence', 'history_count'])
        return (result, trace)