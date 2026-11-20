"""GTA VI trophy rarity forecast — a count-based cohort forecaster.

Model (written from scratch for this competition; standard library only).

The rates the contest scores are ratios, earners / players-tracked, and the two
sides move for different reasons, so this program models counts, not rates:

* Players arrive in cohorts.  Exophase only refreshes a game's page every few
  hours, so the history is a staircase: long flat stretches, then a jump in
  players-tracked (a refresh) that brings a batch of newly tracked players.
  Each cohort joins at its refresh with some age (how long its players had
  already been playing when first counted) and then ages.
* A trophy is earned after a random play-time.  For trophy i the share of a
  cohort of age a holding it is an earn curve F_i(a): either a Weibull
  F = 1 - exp(-(max(0, a - d*tau) / tau) ** k) (k < 1 a long slow tail,
  k > 1 an S-curve, d a flat foot before any earning) or a lognormal
  F = Phi((ln a - ln tau) / sigma) (a wide spread of player speeds).
  tau_i is the trophy's timescale (early/easy trophies are fast); the shape,
  the first cohort's age a0 and the newcomers' age at first count anew are
  fitted globally from the case's history, tau_i per trophy, on the logit
  scale against the observed rates at every distinct read.  The best model
  of each family is kept and the forecast averages them, weighted by fit,
  because the two families fit a two-day history almost equally well but
  extrapolate differently over four more days.
* Players at the target are projected with two growth curves fitted to the
  tracked-player counts (a power law in (t + c), offset fitted, and a
  saturating exponential), blended in log space and weighted by how well
  each fits the history in log space; their disagreement widens the spread.
  When the history is flat (a single distinct read, as on launch night) no
  growth can be fitted, so the program falls back to a refresh mixture:
  with probability p the page refreshes before the target and players jump
  by a lognormal multiplier (median 6x, the size of the launch night's
  first refresh, which the rules document), else they stay put.
* The forecast is a scenario distribution, not a point: future player growth
  (5 log-points) crossed with the earn-timescale uncertainty (5 multipliers),
  each scenario's rates computed from the cohort convolution, then weighted
  quantiles per trophy, widened a little more for Poisson count noise and a
  model floor.  So the 5% quantile sits near the heavy-dilution outcome and
  the 95% near the no-refresh / fast-progression outcome, which is what the
  pinball loss on the logit scale rewards.

Why this beats persistence: on launch night the tracked base jumped 638 ->
3,868 players at the first refresh while the newcomers had earned almost
nothing yet (the refreshed page's total trophies earned, 4,276, is barely
more than the 3,230 newcomers' one near-automatic trophy each plus a little
progression).  Rates therefore fall roughly like 638/3868 for most trophies
at a refresh, and a forecaster that says "rates stay" is wrong by ~1.6 logit
units on almost every trophy.  This model prices that dilution in, while
still letting existing players earn more over a multi-day horizon.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(p) for p in LEVELS]

# --- small numeric helpers -------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x)
    return math.log(p / (100.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        e = math.exp(-l)
        return 100.0 / (1.0 + e)
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    """ISO-8601 timestamp -> epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


# --- case parsing ----------------------------------------------------------

def build_states(case):
    """Collapse the history into distinct reads (the page is a staircase).

    Returns a dict with the distinct states, each (hours since release,
    players tracked, per-trophy earner counts or None), plus checkpoint /
    target times in the same hours and the horizon in hours.
    """
    game = case.get("game") or {}
    rel = game.get("release_date") or "2026-11-19"
    try:
        release = parse_ts(rel + "T00:00:00Z") if len(rel) == 10 else parse_ts(rel)
    except ValueError:
        release = None
    history = case.get("history") or []
    if release is None and history:
        release = parse_ts(history[0].get("slot") or history[0]["at"])
    if release is None:
        release = 0.0

    states = []  # (t_hours, N, counts list, total_earned)
    last_key = None
    for read in history:
        rates = read.get("rates") or []
        n = read.get("players_tracked") or 0
        key = (n, tuple(-1.0 if r is None else r for r in rates))
        if key == last_key:
            continue
        last_key = key
        slot = read.get("slot") or read.get("at")
        t = (parse_ts(slot) - release) / 3600.0 if slot else 0.0
        counts = []
        for r in rates:
            counts.append(None if r is None else int(round(r * n / 100.0)))
        states.append((t, n, counts, read.get("trophies_earned")))
    t_at = (parse_ts(case["at"]) - release) / 3600.0
    t_target = (parse_ts(case["target"]) - release) / 3600.0
    horizon_h = case.get("horizon_minutes", 0) / 60.0
    return {
        "states": states, "t_at": t_at, "t_target": t_target,
        "horizon_h": horizon_h, "release": release,
    }


# --- earn-curve fit ---------------------------------------------------------

# tau grid (hours) for the per-trophy timescale, log-spaced.
def _log_grid(lo, hi, n):
    step = (math.log(hi) - math.log(lo)) / (n - 1)
    return [math.exp(math.log(lo) + step * i) for i in range(n)]


TAU_FIT = _log_grid(0.005, 1000000.0, 64)
TAU_SEL = [TAU_FIT[i] for i in range(0, 64, 5)]  # 13 points, same range

# Flat-history priors (launch night: one distinct read, so the global shape
# is unidentified).  a0 = 12h: the first tracked cohort is the NZ/Australia
# unlock wave, most of a day into the game by the first read; a larger a0
# also matches the documented first refresh best (the existing cohort had
# barely progressed in 4.7h).  anew = 0.4h: players first counted at a
# refresh have barely played (the first refresh's total trophies earned
# shows newcomers brought almost nothing but the one near-automatic
# trophy).  Shape: a plain exponential earn curve, no delay.
FLAT_SHAPE = ("w", 1.0, 0.0)
FLAT_GLOBALS = (12.0, 0.4)
# Rich-history global grid: earn-curve shape, first-cohort age a0,
# newcomer age anew.  Shapes: Weibull ("w", k, delay-fraction) — k < 1 a
# long slow tail, k > 1 an S-curve, delay a flat foot before any earning —
# and lognormal ("l", sigma), whose heavy tail fits a wide spread of
# player speeds.
SHAPE_GRID = ([("w", k, d) for k in (0.5, 1.0, 1.5, 2.0) for d in (0.0, 0.5)]
              + [("l", s, 0.0) for s in (0.8, 1.5, 2.5)])
A0_GRID = (4.0, 8.0, 12.0)
ANEW_GRID = (0.25, 0.5, 1.0, 2.0)


def _cohorts(states, a0, anew):
    """Non-empty join cohorts: (size, age offset at its first read, join time).

    Cohort 0 is the first read's players, aged a0 then; later cohorts are
    the jumps in players-tracked, aged anew when first counted.  The single
    offset is the cohort's mean age at first count: the batch's members
    joined over the gap since the previous refresh, so anew near half that
    gap is the natural value, and the grid search picks it.
    """
    out = []
    prev_n = 0
    for j, (_t, n, _c, _tot) in enumerate(states):
        dn = n - prev_n
        if dn > 0:
            out.append((dn, a0 if j == 0 else anew, states[j][0]))
        prev_n = n
    return out


def _surv(age, shape, scale):
    """Share of a cohort of age `age` still without the trophy.

    shape ("w", k, delay): Weibull, F = 1 - exp(-(max(0, age-d*scale)/scale)^k).
    shape ("l", sigma, 0): lognormal, F = Phi((ln age - ln scale)/sigma).
    """
    if age <= 0.0:
        return 1.0
    if shape[0] == "w":
        _f, k, delay = shape
        a = age - delay * scale
        if a <= 0.0:
            return 1.0
        return math.exp(-((a / scale) ** k))
    _f, sigma, _d = shape
    return 0.5 * math.erfc((math.log(age) - math.log(scale)) / (sigma * math.sqrt(2.0)))


def _survival_matrix(states, cohorts, shape, scale):
    """V[s] = sum of cohort sizes still without the trophy at state s.

    V[s] / N[s] is the predicted share without it, so the predicted rate is
    1 - V[s] / N[s].  One scale, all states.
    """
    out = []
    for (t_s, n_s, _c, _tot) in states:
        v = 0.0
        for (dn, off, t_c) in cohorts:
            age = (t_s - t_c) + off
            if age <= 0.0:
                continue
            v += dn * _surv(age, shape, scale)
        out.append(v)
    return out


def _fit_taus(states, cohorts, shape, tau_grid, trophies_n, only=None):
    """Best scale per trophy (logit-SSE over states) on the given grid.

    Returns (scales, total_sse, per_trophy_sse).  Rates are clamped to a
    half-player floor so a zero count still fits to a small positive rate.
    `only` restricts the fit to a subset of trophy indices (used while
    grid-searching the globals, for speed); other entries get scale 1.0.
    """
    s_count = len(states)
    n_last = states[-1][1] if states else 1
    weights = [math.sqrt(max(s[1], 1) / max(n_last, 1)) for s in states]
    # observed logits per trophy per state, None where the read lacked it
    ids = list(range(trophies_n)) if only is None else list(only)
    obs = {}
    for i in ids:
        row = []
        for (t_s, n_s, counts, _tot) in states:
            c = counts[i] if i < len(counts) else None
            if c is None or n_s <= 0:
                row.append(None)
            else:
                rate = 100.0 * c / n_s
                rate = min(99.99, max(0.5 / n_s, rate))
                row.append(logit_pct(rate))
        obs[i] = row
    # predicted logits per scale per state
    pred = []
    for tau in tau_grid:
        v = _survival_matrix(states, cohorts, shape, tau)
        row = []
        for s_idx, (t_s, n_s, _c, _tot) in enumerate(states):
            if n_s <= 0:
                row.append(logit_pct(0.01))
            else:
                rate = 100.0 * (1.0 - v[s_idx] / n_s)
                row.append(logit_pct(max(0.0001, min(99.99, rate))))
        pred.append(row)
    taus = [1.0] * trophies_n
    total_sse = 0.0
    per_trophy = [0.0] * trophies_n
    for i in ids:
        best_tau, best_sse = tau_grid[0], None
        for g, tau in enumerate(tau_grid):
            sse = 0.0
            for s_idx in range(s_count):
                o = obs[i][s_idx]
                if o is None:
                    continue
                d = pred[g][s_idx] - o
                sse += weights[s_idx] * d * d
            if best_sse is None or sse < best_sse:
                best_sse, best_tau = sse, tau
        taus[i] = best_tau
        per_trophy[i] = best_sse if best_sse is not None else 0.0
        total_sse += best_sse if best_sse is not None else 0.0
    return taus, total_sse, per_trophy


def fit_models(states, trophies_n):
    """Fit the earn-curve models for the history.

    Returns a list of (weight, shape, a0, anew, scales): the best model per
    earn-curve family (Weibull and lognormal), weighted by how well each
    fits, so the forecast hedges the family the history cannot tell apart
    (the two families fit a two-day history almost equally well but
    extrapolate differently over four more days).  Flat history (one
    distinct read): the single prior model, weight 1.
    """
    if len(states) < 2:
        a0, anew = FLAT_GLOBALS
        cohorts = _cohorts(states, a0, anew)
        taus, _sse, _per = _fit_taus(states, cohorts, FLAT_SHAPE, TAU_FIT,
                                     trophies_n)
        return [(1.0, FLAT_SHAPE, a0, anew, taus)]
    # subset of trophies spread over the last-read rate range, for speed
    last_counts = states[-1][2]
    order = sorted(range(trophies_n),
                   key=lambda i: (last_counts[i] is None, last_counts[i] or 0))
    step = max(1, len(order) // 12)
    subset = sorted(order[::step][:12])
    best_per_family = {}
    for shape in SHAPE_GRID:
        for a0 in A0_GRID:
            for anew in ANEW_GRID:
                cohorts = _cohorts(states, a0, anew)
                _taus, _sse, per = _fit_taus(states, cohorts, shape, TAU_SEL,
                                             trophies_n, only=subset)
                sub = sum(per[i] for i in subset)
                fam = shape[0]
                if fam not in best_per_family or sub < best_per_family[fam][0]:
                    best_per_family[fam] = (sub, shape, a0, anew)
    sse_min = min(v[0] for v in best_per_family.values())
    models = []
    for fam, (sub, shape, a0, anew) in sorted(best_per_family.items()):
        cohorts = _cohorts(states, a0, anew)
        taus, _sse, _per = _fit_taus(states, cohorts, shape, TAU_FIT,
                                     trophies_n)
        w = math.exp(-(sub - sse_min) / 10.0)
        models.append((w, shape, a0, anew, taus))
    w_sum = sum(m[0] for m in models)
    return [(w / w_sum, shape, a0, anew, taus)
            for (w, shape, a0, anew, taus) in models]


# --- player-growth projection ----------------------------------------------

def _fit_power_law(times, ns):
    """log N = a + b*log(t+c); returns (a, b, c, sse_log) or None.

    The offset c matters: the launch's growth is a power law in (t + c)
    with c the effective age of the player base at release, and the wrong
    offset badly biases the extrapolation, so c is fitted on a grid.
    """
    best = None
    for c in (1.0, 4.0, 8.0, 24.0, 72.0):
        xs, ys = [], []
        for t, n in zip(times, ns):
            if n > 0 and t + c > 0:
                xs.append(math.log(t + c))
                ys.append(math.log(n))
        if len(xs) < 2:
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            continue
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx
        a = my - b * mx
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[3]:
            best = (a, b, c, sse)
    return best


def _fit_saturating(times, ns):
    """N = Ninf*(1-exp(-(t+1)/tau)); returns (Ninf, tau, sse_log) or None."""
    best = None
    for tau_n in (24.0, 48.0, 96.0, 192.0, 384.0, 768.0):
        gs = [1.0 - math.exp(-(t + 1.0) / tau_n) for t in times]
        num = sum(n * g for n, g in zip(ns, gs))
        den = sum(g * g for g in gs)
        if den <= 0:
            continue
        ninf = num / den
        sse = sum((math.log(n) - math.log(max(ninf * g, 1.0))) ** 2
                  for n, g in zip(ns, gs) if n > 0 and ninf * g > 0)
        if best is None or sse < best[2]:
            best = (ninf, tau_n, sse)
    return best


def project_players(states, t_target, horizon_h):
    """Median players at target, log-spread, power-law slope b and offset c.

    Returns (n_med, sigma_log, b, c).  With fewer than 3 distinct player
    counts no curve can be fitted: (None, ...) signals the refresh mixture.
    The two growth curves are blended in log space, weighted by how well
    each fits the history in log space, so a saturating history leans
    saturating and a power-law history leans power-law.
    """
    pts = [(t, n) for (t, n, _c, _tot) in states if n > 0]
    # distinct (time, N) points only
    uniq = []
    for t, n in pts:
        if not uniq or uniq[-1][1] != n:
            uniq.append((t, n))
    if len(uniq) < 3:
        return None, 0.0, 0.7, 1.0
    times = [t for t, _n in uniq]
    ns = [n for _t, n in uniq]
    t_last = times[-1]
    n_last = ns[-1]
    pl = _fit_power_law(times, ns)
    sat = _fit_saturating(times, ns)
    horizon_days = max(horizon_h / 24.0, 0.0)
    cands = []  # (log N at target, log-sse, is_power)
    if pl is not None:
        a, b, c, sse = pl
        b_c = min(2.0, max(0.0, b))
        n_pow = math.exp(a + b_c * math.log(t_target + c))
        cands.append((math.log(max(n_pow, n_last)), sse, True, b_c, c))
    else:
        b_c, c = 0.7, 1.0
    if sat is not None:
        ninf, tau_n, sse = sat
        n_sat = ninf * (1.0 - math.exp(-(t_target + 1.0) / tau_n))
        cands.append((math.log(max(n_sat, n_last)), sse, False, b_c, c))
    if not cands:
        return None, 0.0, 0.7, 1.0
    pow_c = next((x for x in cands if x[2]), None)
    sat_c = next((x for x in cands if not x[2]), None)
    if pow_c is not None and sat_c is not None:
        # softmax over the log-space fit errors (temperature 0.2)
        w_pow = 1.0 / (1.0 + math.exp((pow_c[1] - sat_c[1]) / 0.2))
        log_med = w_pow * pow_c[0] + (1.0 - w_pow) * sat_c[0]
        disagree = abs(pow_c[0] - sat_c[0])
    elif pow_c is not None:
        log_med, disagree = pow_c[0], 0.0
    else:
        log_med, disagree = sat_c[0], 0.0
    sigma = 0.15 + 0.10 * math.log2(1.0 + horizon_days) + 0.25 * disagree
    b, c = (pow_c[3], pow_c[4]) if pow_c is not None else (0.7, 1.0)
    return math.exp(log_med), sigma, b, c


# --- scenario forecast -------------------------------------------------------

# Player-growth scenario offsets (in units of sigma_log) and their weights.
N_Z = (-1.5, -0.75, 0.0, 0.75, 1.5)
N_W = tuple(math.exp(-0.5 * z * z) for z in N_Z)
# Earn-timescale multipliers and their weights (lognormal, sd 0.25).
TAU_M = (0.7, 0.85, 1.0, 1.18, 1.43)
TAU_W = tuple(math.exp(-(math.log(m) ** 2) / (2 * 0.25 * 0.25)) for m in TAU_M)
# Refresh-mixture multipliers (flat history only) and weights around M_MED.
M_MED = 6.0
M_SIGMA = 0.45
M_GRID = (3.0, 4.5, 6.0, 8.0, 11.0)
M_W = tuple(math.exp(-((math.log(m) - math.log(M_MED)) ** 2) / (2 * M_SIGMA * M_SIGMA))
            for m in M_GRID)
# Extra logit spread: a model floor plus Poisson count noise (added later).
SIGMA_FLOOR = 0.10
JOIN_STEPS = 12


def _earn_frac(age, scale, shape):
    if age <= 0.0:
        return 0.0
    return 1.0 - _surv(age, shape, scale)


def refresh_prob(horizon_h, stale_h):
    """Probability the page refreshes within the horizon.

    A refresh is more likely the longer the horizon and the staler the page
    already is: Exophase refreshes a game's counts every few hours, so a
    page unchanged for hours is due one, while a freshly changed page is not.
    """
    h = max(horizon_h, 0.0)
    s = max(stale_h, 0.0)
    p = 1.0 - (1.0 - h / (h + 2.0)) * math.exp(-s / 4.0)
    return min(0.95, max(0.05, p))


def _weighted_quantiles(pairs, levels):
    """pairs: list of (weight, value). Returns the weighted quantile per level."""
    pairs = sorted(pairs, key=lambda p: p[1])
    total = sum(w for w, _v in pairs)
    if total <= 0:
        return [pairs[0][1]] * len(levels) if pairs else [0.0] * len(levels)
    out = []
    for lv in levels:
        thresh = lv * total
        cum = 0.0
        val = pairs[-1][1]
        for w, v in pairs:
            cum += w
            if cum >= thresh:
                val = v
                break
        out.append(val)
    return out


def forecast_case(case):
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    info = build_states(case)
    states = info["states"]
    t_target = info["t_target"]
    horizon_h = info["horizon_h"]
    if not states or n_tr == 0:
        rows = [{"id": t.get("id", i + 1),
                 "q": [round4(x) for x in (0.01, 0.1, 1.0, 5.0, 20.0)]}
                for i, t in enumerate(trophies)]
        return {"trophies": rows}

    models = fit_models(states, n_tr)
    n_last = states[-1][1]
    t_last = states[-1][0]
    n_med, sigma_log, b_pow, c_pow = project_players(states, t_target, horizon_h)

    # Scenario rates per trophy, merged over the fitted models: each
    # model's scenarios carry its model weight, so the forecast hedges the
    # earn-curve family instead of betting on one.
    pairs_per_trophy = [[] for _ in trophies]
    all_scenarios = []  # (weight, N_target) over every model, for med_n
    for (mw, shape, a0, anew, taus) in models:
        # Cohorts already counted, with their age at the target.
        existing = [(dn, (t_target - t_c) + off)
                    for (dn, off, t_c) in _cohorts(states, a0, anew)]

        # Future join schedule: N(t) follows the fitted power-law shape
        # from N_last to N_med, split into JOIN_STEPS cohorts.
        def future_cohorts(n_target, anew=anew):
            if n_target <= n_last:
                return []
            denom = ((t_target + c_pow) / (t_last + c_pow)) ** b_pow - 1.0
            if denom <= 0:
                denom = 1.0
            out = []
            prev = n_last
            for j in range(1, JOIN_STEPS + 1):
                t_j = t_last + (j / JOIN_STEPS) * (t_target - t_last)
                frac = (((t_j + c_pow) / (t_last + c_pow)) ** b_pow - 1.0) / denom
                n_j = n_last + (n_target - n_last) * frac
                if n_j > prev:
                    out.append((n_j - prev, anew + (t_target - t_j)))
                prev = n_j
            return out

        # Scenario list: (weight, N_target, future cohorts, tau multiplier).
        scenarios = []
        if n_med is None:
            # Flat history: refresh mixture.  p_refresh grows with horizon
            # and with how stale the page already is at the checkpoint.
            stale = max(0.0, info["t_at"] - states[0][0])
            p_ref = refresh_prob(horizon_h, stale)
            for m_w, m in zip(TAU_W, TAU_M):
                scenarios.append(((1.0 - p_ref) * m_w, n_last, [], m))
                for mwx, mult in zip(M_W, M_GRID):
                    n_t = n_last * mult
                    fut = [(n_t - n_last, anew)]
                    scenarios.append((p_ref * mwx * m_w, n_t, fut, m))
            # normalise the M weights (relative within the refresh arm)
            mw_sum = sum(M_W)
            scenarios = [(w / mw_sum if fut else w, nt, fut, m)
                         for (w, nt, fut, m) in scenarios]
            # the no-refresh arm kept weight (1-p)*m_w; refresh sums to p*m_w
        else:
            for zw, z in zip(N_W, N_Z):
                n_t = n_last * math.exp(math.log(max(n_med / n_last, 1.0))
                                        + z * sigma_log)
                n_t = max(n_last, n_t)  # players only grow
                fut = future_cohorts(n_t)
                for m_w, m in zip(TAU_W, TAU_M):
                    scenarios.append((zw * m_w, n_t, fut, m))
        w_sum = sum(w for w, _nt, _f, _m in scenarios)
        scenarios = [(w / w_sum, nt, fut, m) for (w, nt, fut, m) in scenarios]
        for (w, n_t, _f, _m) in scenarios:
            all_scenarios.append((mw * w, n_t))
        for i, trophy in enumerate(trophies):
            tau_i = taus[i]
            for (w, n_t, fut, m) in scenarios:
                tau = tau_i * m
                earners = 0.0
                for (dn, age) in existing + fut:
                    earners += dn * _earn_frac(age, tau, shape)
                rate = 100.0 * earners / n_t if n_t > 0 else 0.01
                pairs_per_trophy[i].append(
                    (mw * w, min(100.0, max(0.0001, rate))))

    rows = []
    for i, trophy in enumerate(trophies):
        pairs = pairs_per_trophy[i]
        qs = _weighted_quantiles(pairs, LEVELS)
        # median-scenario earner count, for the Poisson-noise widening
        med_rate = qs[2]
        med_n = _weighted_quantiles(all_scenarios, [0.5])[0]
        e_med = max(med_rate * med_n / 100.0, 1.0)
        sigma_extra = math.sqrt(SIGMA_FLOOR ** 2 + 1.0 / e_med)
        out_q = []
        for lv_q, z in zip(qs, Z):
            lq = logit_pct(lv_q) + z * sigma_extra
            out_q.append(min(100.0, max(0.01, inv_logit_pct(lq))))
        # enforce non-decreasing after rounding
        rounded = [round4(q) for q in out_q]
        for j in range(1, 5):
            if rounded[j] < rounded[j - 1]:
                rounded[j] = rounded[j - 1]
        rows.append({"id": trophy["id"], "q": rounded})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
