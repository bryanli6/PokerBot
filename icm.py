"""Exact Malmuth-Harville ICM, tuned for an 8-max table under a 1 ms decision budget.

Why this exists
---------------
Scoring in NO LIMIT '26 is by finishing place (default 50/30/20, zero for 4th-8th),
and at the hand cap survivors split the remaining points by ICM on their stacks.
That makes chip EV the wrong objective: the chips you risk are worth more than the
chips you win, so a call that gains chips on average can still lose points on average.

The gap between "equity needed to profit in chips" and "equity needed to profit in
points" is the risk premium. This module computes it exactly.

Performance notes
-----------------
n is always <= 10, so pure Python with locally-bound names beats NumPy here by a
wide margin (NumPy's per-call dispatch overhead alone exceeds our whole budget).
Single-seat equity is O(n^2); all-seat equity is O(n^3). At n=8 that is ~64 and
~512 inner iterations respectively. Both are measured in bench.py.

No imports at module scope on purpose: this file must be importable inside the
bot's microVM with zero side effects and zero cold-start cost.
"""

DEFAULT_PAYOUTS = (50.0, 30.0, 20.0)


def icm_equity_seat(stacks, idx, payouts=DEFAULT_PAYOUTS):
    """Expected points for seat `idx` given current `stacks`.

    `stacks` is a sequence of chip counts indexed by seat; zero or negative means
    busted and is excluded. Returns a float in points, not chips.

    Handles up to 3 paid places exactly (the default structure). Deeper payout
    ladders fall back to the generic recursion in `icm_equity_all`.
    """
    if len(payouts) > 3:
        return icm_equity_all(stacks, payouts)[idx]

    n = len(stacks)
    if idx < 0 or idx >= n:
        return 0.0

    s_i = stacks[idx]
    if s_i <= 0:
        return 0.0

    # Hot path: iterate indices and skip busted seats inline rather than building a
    # `live` list. At ~8 seats the allocation cost of that list dominated the
    # arithmetic, and the resulting gen-0 GC churn showed up as multi-millisecond
    # tail stalls -- fatal in the upload test match, which has no time bank.
    total = 0.0
    nlive = 0
    for j in range(n):
        s = stacks[j]
        if s > 0:
            total += s
            nlive += 1
    if total <= 0.0:
        return 0.0

    s_i = float(s_i)
    np_ = len(payouts)

    # Sole survivor takes first place outright.
    if nlive == 1:
        return float(payouts[0])

    # P(1st)
    eq = payouts[0] * (s_i / total)
    if np_ == 1:
        return eq

    # P(2nd): some j wins, then i is "first" among the remainder.
    p2 = 0.0
    for j in range(n):
        if j == idx:
            continue
        s_j = stacks[j]
        if s_j <= 0:
            continue
        rem = total - s_j
        if rem > 0.0:
            p2 += (s_j / total) * (s_i / rem)
    eq += payouts[1] * p2
    if np_ == 2 or nlive == 2:
        return eq

    # P(3rd): j wins, k takes second, i is "first" among what is left.
    p3 = 0.0
    for j in range(n):
        if j == idx:
            continue
        s_j = stacks[j]
        if s_j <= 0:
            continue
        rem_j = total - s_j
        if rem_j <= 0.0:
            continue
        pj = s_j / total
        for k in range(n):
            if k == idx or k == j:
                continue
            s_k = stacks[k]
            if s_k <= 0:
                continue
            rem_jk = rem_j - s_k
            if rem_jk > 0.0:
                p3 += pj * (s_k / rem_j) * (s_i / rem_jk)
    eq += payouts[2] * p3
    return eq


def icm_equity_all(stacks, payouts=DEFAULT_PAYOUTS):
    """Expected points for every seat. O(n^3) at 3 paid places.

    Use only when you genuinely need the whole vector; at decision time you almost
    always want `icm_equity_seat` for your own seat.
    """
    n = len(stacks)
    if len(payouts) <= 3:
        return [icm_equity_seat(stacks, i, payouts) for i in range(n)]

    # Generic ladder: recursive enumeration over finishing orders. Exponential in
    # the number of paid places, so it is deliberately off the hot path.
    out = [0.0] * n
    live = [i for i in range(n) if stacks[i] > 0]

    def recurse(remaining, place, prob):
        if place >= len(payouts) or not remaining:
            return
        tot = 0.0
        for i in remaining:
            tot += stacks[i]
        if tot <= 0.0:
            return
        pay = payouts[place]
        for i in remaining:
            p = prob * (stacks[i] / tot)
            out[i] += pay * p
            if place + 1 < len(payouts):
                recurse([x for x in remaining if x != i], place + 1, p)

    recurse(live, 0, 1.0)
    return out


def required_equity(stacks, idx, risk, payouts=DEFAULT_PAYOUTS, win_from=None):
    """Minimum win probability that makes risking `risk` chips break even in POINTS.

    This is the number that actually governs a call-off. Compare your hand's raw
    equity against it; if equity is lower, folding is correct even when calling
    gains chips.

    `risk` is the number of chips you put at risk. `win_from` optionally names the
    seat the chips come from (a heads-up all-in); when omitted the won chips are
    drawn proportionally from the other live seats, which is the right model for a
    multiway pot.

    Returns a float in [0, 1], or 1.0 when the spot is unwinnable in points.
    """
    cur = [float(s) for s in stacks]
    s_i = cur[idx]
    if s_i <= 0.0:
        return 1.0

    risk = float(risk)
    if risk <= 0.0:
        return 0.0
    if risk > s_i:
        risk = s_i

    e_fold = icm_equity_seat(cur, idx, payouts)

    # Losing branch: we ship `risk` chips out. Distribute them the same way the
    # winning branch draws them in, so the two branches stay symmetric.
    lose = list(cur)
    lose[idx] = s_i - risk
    if lose[idx] <= 0.0:
        # We bust. Busting is NOT zero equity -- we finish in the place equal to
        # the number of seats currently live, and collect whatever that place pays.
        # Getting this wrong understates e_lose and so overstates the equity we
        # need, which shows up most starkly heads-up: with equal stacks the true
        # requirement is exactly 50% (there is no ladder left to climb), but
        # treating a bust as zero equity demands 80%.
        nlive = 0
        for v in cur:
            if v > 0:
                nlive += 1
        place = nlive - 1  # 0-indexed: k live seats, we finish kth
        e_lose = float(payouts[place]) if 0 <= place < len(payouts) else 0.0
    else:
        _redistribute(lose, idx, risk, win_from, sign=+1)
        e_lose = icm_equity_seat(lose, idx, payouts)

    win = list(cur)
    win[idx] = s_i + risk
    _redistribute(win, idx, risk, win_from, sign=-1)
    e_win = icm_equity_seat(win, idx, payouts)

    denom = e_win - e_lose
    if denom <= 1e-12:
        return 1.0
    p = (e_fold - e_lose) / denom
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def _redistribute(stacks, idx, amount, win_from, sign):
    """Move `amount` chips between seat `idx` and the rest of the table.

    sign=+1 gives the chips to the others (we lost), sign=-1 takes them (we won).
    Chips are capped at what each opponent actually has, which matters at a short
    table where a big stack cannot lose more than the shortest stack covers.
    """
    if win_from is not None and 0 <= win_from < len(stacks):
        if sign > 0:
            stacks[win_from] += amount
        else:
            stacks[win_from] = max(0.0, stacks[win_from] - amount)
        return

    others = [j for j in range(len(stacks)) if j != idx and stacks[j] > 0]
    if not others:
        return
    tot = 0.0
    for j in others:
        tot += stacks[j]
    if tot <= 0.0:
        return
    for j in others:
        share = amount * (stacks[j] / tot)
        if sign > 0:
            stacks[j] += share
        else:
            stacks[j] = max(0.0, stacks[j] - share)


def risk_premium(stacks, idx, risk, payouts=DEFAULT_PAYOUTS, win_from=None):
    """Extra equity, above the pot-odds break-even, that points-scoring demands.

    Positive means play tighter than chip EV suggests. It spikes near the 4th-place
    bubble, where the 20-to-0 cliff makes survival worth more than chips.
    """
    cur = [float(s) for s in stacks]
    s_i = cur[idx]
    if s_i <= 0.0:
        return 0.0
    risk = min(float(risk), s_i)
    if risk <= 0.0:
        return 0.0
    # Chip-EV break-even for a symmetric all-in is 50%; the ICM requirement above
    # that is the premium.
    return required_equity(cur, idx, risk, payouts, win_from) - 0.5
