"""Numba possession engine. Flat arrays in, box score / score / possession counts (+ optional log) out.

Every probability is sigmoid(logit(base) + sum(beta * difference)) where differences are in physical
units (cm) or frozen-reference z-scores. Inputs carry no group labels of any kind.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from dynasty_sim.engine import params as EP
from dynasty_sim.engine import schema as S


@njit(cache=True)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def _fat(e, prm):
    f = 1.0 - prm[EP.P_FAT_FACTOR] * (1.0 - e) / 0.1
    return max(f, prm[EP.P_FAT_FLOOR])


@njit(cache=True)
def _pick(w, n):
    tot = 0.0
    for i in range(n):
        tot += w[i]
    r = np.random.random() * tot
    acc = 0.0
    for i in range(n):
        acc += w[i]
        if r < acc:
            return i
    return n - 1


@njit(cache=True)
def _foul_limit(period):
    if period == 0:
        return 2
    if period == 1:
        return 3
    if period < 4:
        return 5
    return 6


@njit(cache=True)
def _free_throws(n, p):
    made = 0
    last = False
    for _ in range(n):
        last = np.random.random() < p
        if last:
            made += 1
    return made, last


@njit(cache=True)
def _subs(pl, t, on, energy, fouls, order, depth, period, prm):
    lim = _foul_limit(period)
    for j in range(5):
        p = on[t, j]
        val_p = pl[t, p, S.C_OVR] * _fat(energy[t, p], prm)
        need = energy[t, p] < prm[EP.P_SUB_ENERGY] or fouls[t, p] >= lim
        best = -1
        best_v = -1e9
        for r in range(depth[t]):
            b = order[t, r]
            busy = False
            for k in range(5):
                if on[t, k] == b:
                    busy = True
            if busy or fouls[t, b] >= lim or energy[t, b] < 0.85:
                continue
            v = pl[t, b, S.C_OVR] * _fat(energy[t, b], prm)
            if v > best_v:
                best_v = v
                best = b
        if best >= 0 and (need or best_v > prm[EP.P_SUB_MARGIN] * val_p):
            on[t, j] = best


@njit(cache=True)
def _rebound(off, base_logit, is_three, prm, rb, on, box, wk):
    mo = 0.0
    md = 0.0
    for j in range(5):
        mo += rb[off, j]
        md += rb[1 - off, j]
    zs = prm[EP.P_Z_SAT]
    p = _sig(base_logit + prm[EP.P_ORB_SLOPE] * zs * np.tanh((mo - md) / 5.0 / zs) - prm[EP.P_ORB_THREE] * is_three)
    if np.random.random() < p:
        for j in range(5):
            wk[j] = np.exp(prm[EP.P_REB_PICK] * prm[EP.P_REB_SAT] * np.tanh(rb[off, j] / prm[EP.P_REB_SAT]))
        box[off, on[off, _pick(wk, 5)], S.ST_ORB] += 1
        return True
    for j in range(5):
        wk[j] = np.exp(prm[EP.P_REB_PICK] * prm[EP.P_REB_SAT] * np.tanh(rb[1 - off, j] / prm[EP.P_REB_SAT]))
    box[1 - off, on[1 - off, _pick(wk, 5)], S.ST_DRB] += 1
    return False


@njit(cache=True)
def _shot(k, prm, diff, dq, z_touch, dstr, hm, clutch_term, rel_s, off_home, gfoul, drive, usage_rel):
    """Return (p_make, p_block, p_shooting_foul) for shot type k (0 rim, 1 mid, 2 three)."""
    size = 0.0
    if k == 0:
        cr, cq, coff = prm[EP.P_C_REACH_RIM], prm[EP.P_C_QUICK_RIM], 0.0
        lgm, bt, bc = prm[EP.P_LG_MAKE_RIM], prm[EP.P_BETA_TOUCH_RIM], prm[EP.P_BETA_C_RIM]
        lgb, lgs = prm[EP.P_LG_BLK_RIM], prm[EP.P_LG_SF_RIM]
        size = prm[EP.P_SIZE_RIM] * (rel_s - prm[EP.P_REL_MU]) / 10.0
        size = min(max(size, -prm[EP.P_SIZE_CLIP]), prm[EP.P_SIZE_CLIP])
    elif k == 1:
        cr, cq, coff = prm[EP.P_C_REACH_MID], prm[EP.P_C_QUICK_MID], 0.0
        lgm, bt, bc = prm[EP.P_LG_MAKE_MID], prm[EP.P_BETA_TOUCH_MID], prm[EP.P_BETA_C_MID]
        lgb, lgs = prm[EP.P_LG_BLK_MID], prm[EP.P_LG_SF_MID]
    else:
        cr, cq, coff = prm[EP.P_C_REACH_THREE], prm[EP.P_C_QUICK_THREE], prm[EP.P_C_OFF_THREE]
        lgm, bt, bc = prm[EP.P_LG_MAKE_THREE], prm[EP.P_BETA_TOUCH_THREE], prm[EP.P_BETA_C_THREE]
        lgb, lgs = prm[EP.P_LG_BLK_THREE], prm[EP.P_LG_SF_THREE]
    ts = prm[EP.P_TOUCH_SAT]
    zt = ts * np.tanh(z_touch / ts)                       # diminishing returns on shooting skill
    c = _sig(cr * diff + cq * dq + coff)
    pm = _sig(lgm + bt * zt - bc * (c - _sig(coff)) + hm + clutch_term + size
              - prm[EP.P_USAGE_COST] * max(0.0, usage_rel - 1.0))            # a heavily used shooter is defended harder
    pb = _sig(lgb + prm[EP.P_BLK_SLOPE] * diff / 10.0)
    zs = prm[EP.P_Z_SAT]
    lg_ps = lgs + prm[EP.P_SF_STR] * zs * np.tanh(dstr / zs) - prm[EP.P_SF_QUICK] * dq
    if k < 2:
        lg_ps += prm[EP.P_SF_DRIVE] * drive                # drives draw contact; catch-and-shoot threes do not
    ps = _sig(lg_ps)
    ps *= gfoul
    if off_home:
        ps *= prm[EP.P_HOME_FOUL]
    return pm, pb, min(ps, 0.9)


@njit(cache=True)
def _log(log, n, period, clock, off, ev, a, b, pts, flags):
    if n < log.shape[0]:
        log[n, 0] = period
        log[n, 1] = clock
        log[n, 2] = off
        log[n, 3] = ev
        log[n, 4] = a
        log[n, 5] = b
        log[n, 6] = pts
        log[n, 7] = flags
        return n + 1
    return n


@njit(cache=True)
def _game(pl, npl, coach, prm, box, score, poss, log):
    """Simulate one game. pl [2,13,NF]; team 0 is home. Returns (log rows written, periods played)."""
    use_log = log.shape[0] > 0
    nlog = 0
    energy = np.ones((2, S.MAX_ROSTER))
    fouls = np.zeros((2, S.MAX_ROSTER), np.int64)
    tfoul = np.zeros(2, np.int64)
    on = np.zeros((2, 5), np.int64)
    order = np.zeros((2, S.MAX_ROSTER), np.int64)
    depth = np.zeros(2, np.int64)
    for t in range(2):
        ovr = np.full(S.MAX_ROSTER, -1e9)
        navail = 0
        for i in range(npl[t]):
            if pl[t, i, S.C_AVAIL] > 0.5:
                ovr[i] = pl[t, i, S.C_OVR]
                navail += 1
        idx = np.argsort(-ovr)
        for i in range(S.MAX_ROSTER):
            order[t, i] = idx[i]
        depth[t] = min(navail, max(5, int(coach[t, S.CO_DEPTH])))
        for j in range(5):
            on[t, j] = order[t, j]

    qk = np.zeros((2, 5))
    mr = np.zeros((2, 5))
    rel = np.zeros((2, 5))
    rb = np.zeros((2, 5))
    hd = np.zeros((2, 5))
    zw = np.zeros((2, 5))
    uw = np.zeros(5)
    wk = np.zeros(5)
    onc = np.zeros(S.MAX_ROSTER, np.bool_)
    pm_a = np.zeros(3)
    pb_a = np.zeros(3)
    ps_a = np.zeros(3)
    ev_a = np.zeros(3)
    pts_k = np.array([2.0, 2.0, 3.0])

    gpace = np.exp(prm[EP.P_PACE_GAME_SD] * np.random.randn())
    gshoot = prm[EP.P_SHOOT_GAME_SD] * np.random.randn()
    gfoul = np.exp(prm[EP.P_FOUL_GAME_SD] * np.random.randn())
    elapsed = 0.0
    period = 0
    period_end = 720.0
    off = 0
    fresh = True
    plays = 0
    while plays < 2000:
        if elapsed >= period_end - 1e-9:
            if period >= 3 and score[0] != score[1]:
                break
            if period >= 12:
                break
            period += 1
            period_end += 720.0 if period < 4 else 300.0
            tfoul[0] = 0
            tfoul[1] = 0
            fresh = True
            off = period & 1
        plays += 1
        de = 1 - off
        for t in range(2):
            _subs(pl, t, on, energy, fouls, order, depth, period, prm)
            for a in range(1, 5):                      # insertion sort on-court by height
                v = on[t, a]
                h = pl[t, v, S.C_HEIGHT]
                b = a - 1
                while b >= 0 and pl[t, on[t, b], S.C_HEIGHT] > h:
                    on[t, b + 1] = on[t, b]
                    b -= 1
                on[t, b + 1] = v
            for j in range(5):
                p = on[t, j]
                f = _fat(energy[t, p], prm)
                row = pl[t, p]
                q = row[S.C_QUICK] * f
                m = row[S.C_REACH] + row[S.C_VERT] * np.sqrt(f)
                qk[t, j] = q
                mr[t, j] = m
                rel[t, j] = 0.92 * row[S.C_REACH] + 0.4 * row[S.C_VERT] * f
                rb[t, j] = 0.45 * (m - prm[EP.P_MR_MU]) / prm[EP.P_MR_SD] + 0.35 * row[S.C_STR] \
                    + 0.20 * row[S.C_VERTZ]
                hd[t, j] = 0.45 * row[S.C_VISION] + 0.35 * q + 0.20 * row[S.C_HAND]
                zw[t, j] = row[S.C_WING]

        mq_o = 0.0
        mq_d = 0.0
        for j in range(5):
            mq_o += qk[off, j]
            mq_d += qk[de, j]
        if fresh:
            mean_t = gpace * prm[EP.P_PACE_MEAN] * np.exp(-prm[EP.P_PACE_QUICK_W] * (mq_o + mq_d) / 5.0
                                                  - prm[EP.P_PACE_PREF_W] * (coach[off, S.CO_PACE] + coach[de, S.CO_PACE]))
            poss[off] += 1
        else:
            mean_t = prm[EP.P_SECOND_LEN]
        tlen = np.random.gamma(prm[EP.P_PACE_SHAPE], mean_t / prm[EP.P_PACE_SHAPE])
        tlen = min(max(tlen, 1.5), 24.0)
        if elapsed + tlen > period_end:
            tlen = period_end - elapsed
        clock = period_end - elapsed
        next_off = de
        next_fresh = True
        done = False

        for j in range(5):
            r = pl[off, on[off, j]]
            ux = prm[EP.P_USAGE_TOUCH] * r[S.C_TOUCH] + prm[EP.P_USAGE_VISION] * r[S.C_VISION] + prm[EP.P_USAGE_TEMPER] * r[S.C_TEMPER]
            uw[j] = np.exp(prm[EP.P_USAGE_SAT] * np.tanh(ux / prm[EP.P_USAGE_SAT]))

        # 0. intentional foul on a poor free-throw shooter: late, close, fouling team not ahead
        if fresh and period >= 3 and period_end - elapsed <= prm[EP.P_HACK_SECS] and score[de] <= score[off] \
                and score[off] - score[de] <= prm[EP.P_HACK_MARGIN]:
            hj = 0
            hp = 2.0
            for j in range(5):
                pj = _sig(prm[EP.P_LG_FT] + prm[EP.P_BETA_TOUCH_FT] * prm[EP.P_TOUCH_SAT]
                          * np.tanh(pl[off, on[off, j], S.C_TOUCH] / prm[EP.P_TOUCH_SAT]))
                if pj < hp:
                    hp = pj
                    hj = j
            if hp < prm[EP.P_HACK_P]:
                hpid = on[off, hj]
                fp = on[de, int(np.random.random() * 5)]
                fouls[de, fp] += 1
                box[de, fp, S.ST_PF] += 1
                tfoul[de] += 1
                tlen *= 0.15
                if use_log:
                    nlog = _log(log, nlog, period, clock, off, S.EV_NSF, fp, 1, 0, 1)
                if tfoul[de] >= prm[EP.P_BONUS_AT]:
                    made, last = _free_throws(2, hp)
                    box[off, hpid, S.ST_FTA] += 2
                    box[off, hpid, S.ST_FTM] += made
                    box[off, hpid, S.ST_PTS] += made
                    score[off] += made
                    if not last and _rebound(off, prm[EP.P_LG_ORB_FT], 0.0, prm, rb, on, box, wk):
                        next_off = off
                        next_fresh = False
                else:
                    next_off = off
                    next_fresh = False
                done = True

        # 1. non-shooting foul
        if not done and np.random.random() < gfoul * prm[EP.P_NSF_BASE] * np.exp(prm[EP.P_NSF_AGGR] * coach[de, S.CO_FOUL]):
            fj = int(np.random.random() * 5)
            fp = on[de, fj]
            fouls[de, fp] += 1
            box[de, fp, S.ST_PF] += 1
            tfoul[de] += 1
            tlen *= 0.3
            if use_log:
                nlog = _log(log, nlog, period, clock, off, S.EV_NSF, fp, 0, 0, 0)
            if tfoul[de] >= prm[EP.P_BONUS_AT]:
                sj = _pick(uw, 5)
                sp = on[off, sj]
                pft = _sig(prm[EP.P_LG_FT] + prm[EP.P_BETA_TOUCH_FT] * prm[EP.P_TOUCH_SAT]
                           * np.tanh(pl[off, sp, S.C_TOUCH] / prm[EP.P_TOUCH_SAT]))
                made, last = _free_throws(2, pft)
                box[off, sp, S.ST_FTA] += 2
                box[off, sp, S.ST_FTM] += made
                box[off, sp, S.ST_PTS] += made
                score[off] += made
                if not last and _rebound(off, prm[EP.P_LG_ORB_FT], 0.0, prm, rb, on, box, wk):
                    next_off = off
                    next_fresh = False
            else:
                next_off = off
                next_fresh = False
            done = True

        # 2. turnover
        bh = 0
        if not done:
            for j in range(1, 5):
                if hd[off, j] > hd[off, bh]:
                    bh = j
            dmean = 0.0
            for j in range(5):
                dmean += qk[de, j] + 0.5 * zw[de, j]
            zs = prm[EP.P_Z_SAT]
            delta = zs * np.tanh((dmean / 5.0 - hd[off, bh]) / zs)
            p_tov = _sig(prm[EP.P_LG_TOV] + prm[EP.P_TOV_SLOPE] * delta
                         - prm[EP.P_TOV_TEMPER] * pl[off, on[off, bh], S.C_TEMPER])
            if np.random.random() < p_tov:
                box[off, on[off, bh], S.ST_TOV] += 1
                sb = -1
                if np.random.random() < min(0.95, 2.0 * prm[EP.P_STEAL_SHARE] * _sig(prm[EP.P_STEAL_SLOPE] * delta)):
                    for j in range(5):
                        wk[j] = np.exp(qk[de, j] + 0.5 * zw[de, j])
                    sb = _pick(wk, 5)
                    box[de, on[de, sb], S.ST_STL] += 1
                if use_log:
                    nlog = _log(log, nlog, period, clock, off, S.EV_TOV, on[off, bh], sb, 0, 0)
                done = True

        # 3-7. shot
        if not done:
            sj = _pick(uw, 5)
            sp = on[off, sj]
            srow = pl[off, sp]
            dj = sj
            dp = on[de, dj]
            rp = 0
            for j in range(1, 5):
                if mr[de, j] > mr[de, rp]:
                    rp = j
            hm = gshoot + (prm[EP.P_HOME_MAKE] if off == 0 else 0.0) \
                - prm[EP.P_SCORE_FLOW] * np.tanh((score[off] - score[de]) / 15.0)
            clutch = 0.0
            if period >= 3 and period_end - elapsed <= 300.0 and abs(score[0] - score[1]) <= 5:
                clutch = prm[EP.P_CLUTCH] * srow[S.C_TEMPER]
            diff_std = mr[de, dj] - rel[off, sj] - prm[EP.P_REACH_OFF]
            diff_rim = 0.5 * (mr[de, dj] + mr[de, rp]) - rel[off, sj] - prm[EP.P_REACH_OFF]
            rs, zs = prm[EP.P_REACH_SAT], prm[EP.P_Z_SAT]
            diff_std = rs * np.tanh(diff_std / rs)
            diff_rim = rs * np.tanh(diff_rim / rs)
            dq = zs * np.tanh((qk[de, dj] - qk[off, sj]) / zs)
            dstr = srow[S.C_STR] - pl[de, dp, S.C_STR]
            pft = _sig(prm[EP.P_LG_FT] + prm[EP.P_BETA_TOUCH_FT] * prm[EP.P_TOUCH_SAT]
                       * np.tanh(srow[S.C_TOUCH] / prm[EP.P_TOUCH_SAT]))
            usage_rel = uw[sj] * 5.0 / max(uw.sum(), 1e-9)
            best = -1e9
            for k in range(3):
                pm, pb, ps = _shot(k, prm, diff_rim if k == 0 else diff_std, dq, srow[S.C_TOUCH], dstr,
                                   hm, clutch, rel[off, sj], off == 0, gfoul, hd[off, sj], usage_rel)
                pm_a[k] = pm
                pb_a[k] = pb
                ps_a[k] = ps
                prior = 0.0
                if k == 1:
                    prior = prm[EP.P_EV_MID]
                elif k == 2:
                    prior = prm[EP.P_EV_THREE]
                ev_a[k] = ((1.0 - pb) * (pts_k[k] * pm + ps * pts_k[k] * pft) + coach[off, k] + prior) / prm[EP.P_TAU]
                if ev_a[k] > best:
                    best = ev_a[k]
            tot = 0.0
            for k in range(3):
                ev_a[k] = np.exp(ev_a[k] - best)
                tot += ev_a[k]
            r = np.random.random() * tot
            kk = 2
            acc = 0.0
            for k in range(3):
                acc += ev_a[k]
                if r < acc:
                    kk = k
                    break
            is3 = 1.0 if kk == 2 else 0.0
            npts = 3 if kk == 2 else 2

            if np.random.random() < pb_a[kk]:                              # blocked
                bj = dj
                if kk == 0 and np.random.random() < 0.6:
                    bj = rp
                box[off, sp, S.ST_FGA] += 1
                box[off, sp, S.ST_TPA] += is3
                if kk == 0:
                    box[off, sp, S.ST_RIM] += 1
                box[de, on[de, bj], S.ST_BLK] += 1
                if use_log:
                    nlog = _log(log, nlog, period, clock, off, S.EV_BLK, sp, on[de, bj], 0, kk)
                if _rebound(off, prm[EP.P_LG_ORB], is3, prm, rb, on, box, wk):
                    next_off = off
                    next_fresh = False
            else:
                fouled = np.random.random() < ps_a[kk]
                made = np.random.random() < pm_a[kk]
                if fouled:
                    fouls[de, dp] += 1
                    box[de, dp, S.ST_PF] += 1
                    tfoul[de] += 1
                if made:
                    box[off, sp, S.ST_FGA] += 1
                    if kk == 0:
                        box[off, sp, S.ST_RIM] += 1
                    box[off, sp, S.ST_FGM] += 1
                    box[off, sp, S.ST_TPA] += is3
                    box[off, sp, S.ST_TPM] += is3
                    box[off, sp, S.ST_PTS] += npts
                    score[off] += npts
                    if kk == 0:
                        lg_a = prm[EP.P_LG_AST_RIM]
                    elif kk == 1:
                        lg_a = prm[EP.P_LG_AST_MID]
                    else:
                        lg_a = prm[EP.P_LG_AST_THREE]
                    for j in range(5):
                        wk[j] = 0.0 if j == sj else np.exp(prm[EP.P_AST_PICK] * pl[off, on[off, j], S.C_VISION])
                    aj = _pick(wk, 5)
                    if np.random.random() < _sig(lg_a + prm[EP.P_AST_VISION] * pl[off, on[off, aj], S.C_VISION]):
                        box[off, on[off, aj], S.ST_AST] += 1
                    if use_log:
                        nlog = _log(log, nlog, period, clock, off, S.EV_MAKE, sp, dp, npts, kk + 10 * int(fouled))
                    if fouled:
                        m1, last = _free_throws(1, pft)
                        box[off, sp, S.ST_FTA] += 1
                        box[off, sp, S.ST_FTM] += m1
                        box[off, sp, S.ST_PTS] += m1
                        score[off] += m1
                        if not last and _rebound(off, prm[EP.P_LG_ORB_FT], 0.0, prm, rb, on, box, wk):
                            next_off = off
                            next_fresh = False
                elif fouled:
                    nft = 3 if kk == 2 else 2
                    made_ft, last = _free_throws(nft, pft)
                    box[off, sp, S.ST_FTA] += nft
                    box[off, sp, S.ST_FTM] += made_ft
                    box[off, sp, S.ST_PTS] += made_ft
                    score[off] += made_ft
                    if use_log:
                        nlog = _log(log, nlog, period, clock, off, S.EV_SFOUL, sp, dp, made_ft, kk)
                    if not last and _rebound(off, prm[EP.P_LG_ORB_FT], 0.0, prm, rb, on, box, wk):
                        next_off = off
                        next_fresh = False
                else:
                    box[off, sp, S.ST_FGA] += 1
                    box[off, sp, S.ST_TPA] += is3
                    if kk == 0:
                        box[off, sp, S.ST_RIM] += 1
                    if use_log:
                        nlog = _log(log, nlog, period, clock, off, S.EV_MISS, sp, dp, 0, kk)
                    if _rebound(off, prm[EP.P_LG_ORB], is3, prm, rb, on, box, wk):
                        next_off = off
                        next_fresh = False

        # clock and energy
        elapsed += tlen
        scale = tlen / 14.4
        for t in range(2):
            for i in range(S.MAX_ROSTER):
                onc[i] = False
            for j in range(5):
                p = on[t, j]
                onc[p] = True
                box[t, p, S.ST_SEC] += tlen
                d = prm[EP.P_ENERGY_DRAIN] * (1.0 - prm[EP.P_DRAIN_END] * pl[t, p, S.C_END]) \
                    * np.sqrt(pl[t, p, S.C_MASS] / 100.0) * scale
                energy[t, p] = max(0.0, energy[t, p] - d)
            for i in range(npl[t]):
                if not onc[i]:
                    energy[t, i] = min(1.0, energy[t, i] + prm[EP.P_BENCH_REC] * scale)
        off = next_off
        fresh = next_fresh
    return nlog, period


@njit(cache=True)
def sim_game_single(pl, npl, coach, prm, seed, box, score, poss, log):
    np.random.seed(seed)
    return _game(pl, npl, coach, prm, box, score, poss, log)


@njit(parallel=True, cache=True)
def sim_games(PL, NPL, COACH, prm, seeds, BOX, SCORE, POSS):
    """PL [G,2,13,NF], NPL [G,2], COACH [G,2,NCO], seeds [G] uint32 -> fills BOX [G,2,13,NSTAT], SCORE [G,2], POSS [G,2]."""
    for g in prange(PL.shape[0]):
        np.random.seed(seeds[g])
        _game(PL[g], NPL[g], COACH[g], prm, BOX[g], SCORE[g], POSS[g], np.zeros((0, S.NLOG), np.float32))
