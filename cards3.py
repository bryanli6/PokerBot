"""Three-card preflop equity, and the post-flop discard/expose choice.

The variant (RULES.md): three hole cards, normal preflop betting, then after the
flop every player simultaneously discards one card face down and exposes one of
the other two. The exposed card stays live and public forever; the third stays
hidden to showdown. Six legal combinations.

Two problems this module solves.

1. PREFLOP EQUITY WITH THREE CARDS. preflop.py is a two-card table, and it is
   wrong here in two compounding ways: you get to pick your best two after
   seeing the flop (optionality, pushes equity up), and so does everyone else
   (pushes it down, harder). Measured against full simulation, naively taking
   the best two-card subset overstates equity by 3-7 points, and far worse for
   redundant holdings -- 777 reads as 66% but is really 51%, because all three
   subsets are the same pair and the optionality is an illusion. We correct
   with a model fitted to simulation.

2. WHICH TWO TO KEEP, AND WHICH TO SHOW. Keeping is an equity question. Showing
   is an information question, and they are chosen together.
"""
import itertools

import preflop

# Fitted on 70 random holdings x 2500 simulated hands each, where every player
# keeps the two cards making their best five on the flop:
#     true ~ a*max_subset + b*mean_subset + c*trips + d*pair + e
# Residual stdev 0.9-1.6%, against 1.9-2.3% for the uncorrected proxy, and the
# worst case improves from 12.5% to 6.0%.
_COEF = {
    1: (0.7143, 0.2654, 0.0, 0.0107, -0.0383),
    3: (0.7267, 0.1118, 0.0, -0.0002, 0.0014),
    7: (0.5404, 0.1229, 0.0, 0.0209, 0.0197),
}


def _coef(n):
    """Interpolate between the three fitted opponent counts."""
    if n <= 1:
        return _COEF[1]
    if n >= 7:
        return _COEF[7]
    if n == 3:
        return _COEF[3]
    lo, hi = (1, 3) if n < 3 else (3, 7)
    t = (n - lo) / (hi - lo)
    a, b = _COEF[lo], _COEF[hi]
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def preflop3(cards, n_opponents):
    """Equity for a three-card holding against n opponents who also hold three.

    Pure arithmetic over at most three dict lookups: nanoseconds, no sampling,
    safe to call inside the 1 ms budget.
    """
    try:
        if not cards or len(cards) < 2:
            return None
        if len(cards) == 2:
            return preflop.equity(cards[0], cards[1], n_opponents)
        subs = [preflop.equity(a, b, n_opponents)
                for a, b in itertools.combinations(cards, 2)]
        if not subs:
            return None
        mx = max(subs)
        avg = sum(subs) / len(subs)
        ranks = {c[0] for c in cards}
        trips = 1.0 if len(ranks) == 1 else 0.0
        pair = 1.0 if len(ranks) == 2 else 0.0
        a, b, c, d, e = _coef(n_opponents)
        v = a * mx + b * avg + c * trips + d * pair + e
        if trips:
            # The fit learned nothing about trips: they are ~0.06% of random
            # three-card holdings, so a 70-hand calibration sample contained
            # none. Measured directly, 777 is truly 51.1% heads-up where the
            # proxy says 66.0%, and AAA is 77.8% against a proxy of 85.4%.
            # Trips are the worst holding in this variant relative to how
            # strong they look: all three two-card subsets are the same pair,
            # so the discard is forced and the optionality that makes other
            # three-card hands valuable simply is not there. This flat penalty
            # splits the two measurements; trips are rare enough that a tighter
            # fit is not worth the build time.
            v -= 0.06
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
    except Exception:  # noqa: BLE001
        return None


# Rank order for the expose decision.
_RANK = {r: i for i, r in enumerate("23456789TJQKA")}


def rank_of(card):
    try:
        return _RANK.get(card[0].upper(), 0)
    except Exception:  # noqa: BLE001
        return 0


def choose_expose(kept):
    """Of the two retained cards, which to make public. Returns its index in `kept`.

    Expose the LOWER card. The exposed card stays live, so this costs no equity
    -- it only shapes what opponents believe. We continue mostly with strong
    holdings, so looking weak is worth more than looking strong: it buys calls
    when we are ahead and discourages opponents from folding to our bets.

    It also happens to feed the Empty Bluffer side prize, which counts hands won
    without showdown after exposing a 2-5. That is a bonus, not the reason.
    """
    try:
        if len(kept) < 2:
            return 0
        return 0 if rank_of(kept[0]) <= rank_of(kept[1]) else 1
    except Exception:  # noqa: BLE001
        return 0
