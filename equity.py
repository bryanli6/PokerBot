"""Hand-strength estimation for the three-card discard/expose variant.

The important thing this does that a naive hold'em port does not: it consumes
opponents' EXPOSED cards. After the flop every player shows one card that stays
live and public. A rollout that deals opponents two uniformly random cards is
simply wrong once you can see one of them -- the exposed card is both a dead
card removed from the deck and a card that specific opponent is KNOWN to hold.
Modelling that is strictly more accurate, costs nothing extra per trial, and is
the main edge available in this variant.
"""
import random
import time

import cards3
import preflop
import revealmodel as rm
try:
    import preflop3
    _HAVE_P3 = True
except Exception:  # noqa: BLE001
    _HAVE_P3 = False

try:
    import eval7
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False

RANKS = "23456789TJQKA"
SUITS = "shdc"
_DECK = [r + s for r in RANKS for s in SUITS]
_RANK_IDX = {r: i for i, r in enumerate(RANKS)}


def _rank_of_card(card):
    try:
        return _RANK_IDX[card[0].upper()]
    except Exception:  # noqa: BLE001
        return None
_CARD = {}
if _HAVE:
    for _c in _DECK:
        _CARD[_c] = eval7.Card(_c)

_RNG = random.Random(0xA5EED)
CHECK_EVERY = 8
MAX_TRIALS = 1200


def _norm(c):
    """Normalize a card string to the table's spelling."""
    try:
        return c[0].upper() + c[1].lower()
    except Exception:  # noqa: BLE001
        return None


def exposed_by_seat(state):
    """{seat: card} for every opponent showing a card, plus the full dead set."""
    out = {}
    dead = set()
    try:
        me = state.seat
        for s in state.seats:
            sc = getattr(s, "shown_cards", ()) or ()
            for c in sc:
                n = _norm(c)
                if n:
                    dead.add(n)
                    if s.seat != me:
                        out.setdefault(s.seat, n)
    except Exception:  # noqa: BLE001
        pass
    return out, dead


def live_opponents(state):
    """Seats still contesting the pot, excluding us."""
    try:
        me = state.seat
        return [s.seat for s in state.seats
                if s.seat != me and getattr(s, "status", "") in ("active", "all_in")]
    except Exception:  # noqa: BLE001
        return []


def rollout(hero, board, opp_seats, exposed, budget_ns, t0=None):
    """Equity vs the given opponents, honoring each one's exposed card.

    `opp_seats` is a list of seat ids; `exposed` maps seat -> card for those
    whose card is public. An opponent with a known card gets dealt only ONE
    random card; the rest get two.
    """
    if not _HAVE or not hero or len(hero) < 2:
        return None, 0
    if t0 is None:
        t0 = time.perf_counter_ns()

    try:
        hero_c = [_CARD[_norm(c)] for c in hero]
        known = {_norm(c) for c in hero}
        board_c = []
        for c in board or ():
            n = _norm(c)
            if n in _CARD:
                board_c.append(_CARD[n])
                known.add(n)
        # Each opponent's known card is fixed, not sampled.
        fixed = []
        for s in opp_seats:
            e = exposed.get(s)
            if e and e in _CARD:
                fixed.append(_CARD[e])
                known.add(e)
            else:
                fixed.append(None)
    except Exception:  # noqa: BLE001
        return None, 0

    rest_names = [c for c in _DECK if c not in known]
    rest = [_CARD[c] for c in rest_names]
    n_opp = len(opp_seats)
    if n_opp < 1:
        return None, 0
    need_board = 5 - len(board_c)
    # Opponents with an exposed card get their hidden card SAMPLED FROM THE
    # REVEAL MODEL, not uniformly: in the real field an exposed K is KK 39% of
    # the time. The rest get two uniform cards.
    known_idx = []     # positions in `fixed` with an exposed card
    exp_rank = []
    for i, s_ in enumerate(opp_seats):
        e = exposed.get(s_)
        if e and e in _CARD:
            known_idx.append(i)
            exp_rank.append(_rank_of_card(e))
    n_unknown = n_opp - len(known_idx)
    need = need_board + 2 * n_unknown
    if need > len(rest) or need < 0:
        return None, 0
    by_rank = [[] for _ in range(13)]
    for nm, cd in zip(rest_names, rest):
        r = _rank_of_card(nm)
        if r is not None:
            by_rank[r].append(cd)

    ev = eval7.evaluate
    smp = _RNG.sample
    rnd = _RNG.random
    choice = _RNG.choice
    perf = time.perf_counter_ns
    won = 0.0
    trials = 0

    while True:
        for _ in range(CHECK_EVERY):
            d = smp(rest, need) if need else []
            used = set(d)
            full = board_c + d[:need_board] if need_board else board_c
            hv = ev(hero_c + full)
            best = 0
            ties = 0
            o = need_board
            ki = 0
            for i, f in enumerate(fixed):
                if f is not None:
                    er = exp_rank[ki]
                    ki += 1
                    hc = None
                    if er is not None:
                        for _try in range(4):
                            r = rm.sample_hidden_rank(er, rnd())
                            pool = by_rank[r]
                            if pool:
                                cand = choice(pool)
                                if cand not in used:
                                    hc = cand
                                    break
                    if hc is None:
                        # fall back to any unused card
                        for _try in range(8):
                            cand = choice(rest)
                            if cand not in used:
                                hc = cand
                                break
                        if hc is None:
                            hc = rest[0]
                    used.add(hc)
                    hand = [f, hc]
                else:
                    hand = [d[o], d[o + 1]]
                    o += 2
                ov = ev(hand + full)
                if ov > best:
                    best = ov
                    ties = 1
                elif ov == best:
                    ties += 1
            if hv > best:
                won += 1.0
            elif hv == best:
                won += 1.0 / (ties + 1)
            trials += 1
        if perf() - t0 >= budget_ns or trials >= MAX_TRIALS:
            break

    if trials == 0:
        return None, 0
    return won / trials, trials


def river_exact(hero, board, opp_card, dead):
    """Exact equity on the river against ONE opponent with one exposed card.

    With the board complete, the only unknown is the opponent's hidden card,
    drawn from the deck minus everything visible. Enumerating that pool gives
    the true equity with zero sampling error, for ~44 evaluations -- the hero's
    own hand value is computed once. Returns None if the spot does not qualify.
    """
    if not _HAVE or len(hero) < 2 or len(board) != 5 or not opp_card:
        return None
    try:
        known = {_norm(c) for c in hero} | {_norm(c) for c in board} | set(dead)
        oc = _norm(opp_card)
        if oc is None or oc not in _CARD:
            return None
        known.add(oc)
        board_c = [_CARD[_norm(c)] for c in board]
        hv = eval7.evaluate([_CARD[_norm(c)] for c in hero] + board_c)
        ocard = _CARD[oc]
        er = _rank_of_card(oc)
        ev = eval7.evaluate
        wins = 0.0
        wsum = 0.0
        for c in _DECK:
            if c in known:
                continue
            # Weight each candidate hidden card by how likely the field is to
            # hold it given the exposed card, instead of uniformly.
            r = _rank_of_card(c)
            w = rm.P_HIDDEN[er][r] if (er is not None and r is not None) else 1.0
            ov = ev([ocard, _CARD[c]] + board_c)
            if hv > ov:
                wins += w
            elif hv == ov:
                wins += 0.5 * w
            wsum += w
        if wsum <= 0.0:
            return None
        return wins / wsum
    except Exception:  # noqa: BLE001
        return None


def turn_enum(hero, board, opp_card, dead, budget_ns, t0=None, rivers=3):
    """Turn equity vs one opponent with one exposed card.

    Stratified: EVERY candidate hidden card is visited (that is where the
    variance lives -- it decides whether they hold a pair, a draw, or air), and
    for each a few random rivers are sampled. The first version did the
    opposite, exhausting rivers for a handful of hiddens, and measured worse
    than a plain rollout. Returns (equity, fraction_of_hidden_cards_covered).
    """
    if not _HAVE or len(hero) < 2 or len(board) != 4 or not opp_card:
        return None, 0.0
    if t0 is None:
        t0 = time.perf_counter_ns()
    try:
        known = {_norm(c) for c in hero} | {_norm(c) for c in board} | set(dead)
        oc = _norm(opp_card)
        if oc is None or oc not in _CARD:
            return None, 0.0
        known.add(oc)
        pool_names = [c for c in _DECK if c not in known]
        pool = [_CARD[c] for c in pool_names]
        hidden = list(zip(pool_names, pool))
        _RNG.shuffle(hidden)
        er = _rank_of_card(oc)
        board_c = [_CARD[_norm(c)] for c in board]
        hero_c = [_CARD[_norm(c)] for c in hero]
        ocard = _CARD[oc]
        ev = eval7.evaluate
        perf = time.perf_counter_ns
        choice = _RNG.choice
        wins = 0.0
        n = 0.0
        covered = 0
        npool = len(pool)
        for hname, hc in hidden:
            hr = _rank_of_card(hname)
            w = rm.P_HIDDEN[er][hr] if (er is not None and hr is not None) else 1.0
            for _ in range(rivers):
                rc = choice(pool)
                if rc is hc:
                    rc = pool[(pool.index(rc) + 1) % npool]
                full = board_c + [rc]
                hv = ev(hero_c + full)
                ov = ev([ocard, hc] + full)
                if hv > ov:
                    wins += w
                elif hv == ov:
                    wins += 0.5 * w
                n += w
            covered += 1
            if perf() - t0 >= budget_ns:
                break
        if n == 0:
            return None, 0.0
        return wins / n, covered / max(1, len(hidden))
    except Exception:  # noqa: BLE001
        return None, 0.0


def estimate(state, budget_ns, t0=None):
    """(equity, standard_error) for the current spot. Never raises."""
    try:
        cards = [c for c in (getattr(state, "cards", ()) or ())]
        board = [c for c in (getattr(state, "board", ()) or ())]
        opp = live_opponents(state)
        n_opp = len(opp) if opp else 1

        if not board:
            # Preflop: three cards, everyone holds three. Generated table first
            # (2,000 trials/cell, ~+/-1.1%), fitted model as fallback.
            e = preflop3.equity(cards, n_opp) if _HAVE_P3 else None
            if e is None:
                e = cards3.preflop3(cards, n_opp)
                return e, 0.04
            return e, 0.011

        exposed, dead = exposed_by_seat(state)

        # River, one opponent, their exposed card known: enumerate instead of
        # sampling. Exact, and cheaper than the rollout it replaces.
        if len(board) == 5 and len(opp) == 1 and opp[0] in exposed:
            e = river_exact(cards, board, exposed[opp[0]], dead)
            if e is not None:
                return e, 0.0
        if len(board) == 4 and len(opp) == 1 and opp[0] in exposed:
            e, cov = turn_enum(cards, board, exposed[opp[0]], dead, budget_ns, t0)
            if e is not None:
                # Residual error shrinks with coverage; full coverage is exact.
                return e, (0.0 if cov >= 0.999 else 0.03 * (1.0 - cov) ** 0.5)

        e, trials = rollout(cards, board, opp, exposed, budget_ns, t0)
        if e is None or trials == 0:
            return cards3.preflop3(cards, n_opp), 0.15
        se = (e * (1.0 - e) / trials) ** 0.5
        return e, se
    except Exception:  # noqa: BLE001
        return None, 1.0


def range_discount(to_call, pot):
    """Shade equity down facing a large bet: our rollout assumed random hands."""
    try:
        if to_call <= 0 or pot <= 0:
            return 1.0
        ratio = to_call / pot
        if ratio <= 0.25:
            return 1.0
        if ratio >= 1.5:
            return 0.80
        return 1.0 - 0.16 * ((ratio - 0.25) / 1.25)
    except Exception:  # noqa: BLE001
        return 1.0


def select_rollout(cards, board, opp_seats, exposed, budget_ns, t0=None):
    """Score the three keep-pairs against ONE shared sample of opponents+runout.

    Each trial deals a single opponent set and board completion, then evaluates
    all three hero subsets against it. Versus three independent rollouts this is
    (3 + N) evaluations per trial instead of 3*(1 + N), and because every subset
    faces the same cards the comparison is paired: sampling noise largely cancels
    when we rank them. Returns [equity_of_pair_01, _02, _12] and the trial count.
    """
    if not _HAVE or len(cards) < 3:
        return None, 0
    if t0 is None:
        t0 = time.perf_counter_ns()
    try:
        hc = [_CARD[_norm(c)] for c in cards[:3]]
        known = {_norm(c) for c in cards[:3]}
        board_c = []
        for c in board or ():
            n = _norm(c)
            if n in _CARD:
                board_c.append(_CARD[n])
                known.add(n)
        fixed = []
        for s in opp_seats:
            e = exposed.get(s)
            if e and e in _CARD:
                fixed.append(_CARD[e])
                known.add(e)
            else:
                fixed.append(None)
    except Exception:  # noqa: BLE001
        return None, 0
    rest = [_CARD[c] for c in _DECK if c not in known]
    need_board = 5 - len(board_c)
    need_opp = sum(1 if f is not None else 2 for f in fixed)
    need = need_board + need_opp
    if need > len(rest) or not fixed:
        return None, 0
    pairs = ((hc[0], hc[1]), (hc[0], hc[2]), (hc[1], hc[2]))
    won = [0.0, 0.0, 0.0]
    ev = eval7.evaluate
    smp = _RNG.sample
    perf = time.perf_counter_ns
    trials = 0
    while True:
        for _ in range(CHECK_EVERY):
            d = smp(rest, need)
            full = board_c + d[:need_board] if need_board else board_c
            best = 0
            ties = 0
            o = need_board
            for f in fixed:
                if f is not None:
                    ov = ev([f, d[o]] + full)
                    o += 1
                else:
                    ov = ev([d[o], d[o + 1]] + full)
                    o += 2
                if ov > best:
                    best = ov
                    ties = 1
                elif ov == best:
                    ties += 1
            for k in range(3):
                hv = ev(list(pairs[k]) + full)
                if hv > best:
                    won[k] += 1.0
                elif hv == best:
                    won[k] += 1.0 / (ties + 1)
            trials += 1
        if perf() - t0 >= budget_ns or trials >= MAX_TRIALS:
            break
    if trials == 0:
        return None, 0
    return [w / trials for w in won], trials


def best_selection(state, budget_ns, t0=None):
    """Choose (discard_index, reveal_index) at the select phase.

    Keeping is an equity question, answered by the paired rollout above. Showing
    is an information question, answered in cards3. Returns indexes into
    state.cards as dealt.
    """
    if t0 is None:
        t0 = time.perf_counter_ns()
    cards = [c for c in (getattr(state, "cards", ()) or ())]
    if len(cards) < 3:
        return 2, 0
    board = [c for c in (getattr(state, "board", ()) or ())]
    opp = live_opponents(state) or [-1]
    exposed, _ = exposed_by_seat(state)

    eqs, trials = select_rollout(cards, board, opp, exposed, budget_ns, t0)
    pair_idx = ((0, 1), (0, 2), (1, 2))
    if eqs is None:
        # Clock already gone: keep the two making the best current five.
        try:
            bc = [_CARD[_norm(b)] for b in board]
            scores = [eval7.evaluate([_CARD[_norm(cards[i])], _CARD[_norm(cards[j])]] + bc)
                      for i, j in pair_idx]
        except Exception:  # noqa: BLE001
            return 2, 0
        k = max(range(3), key=lambda x: scores[x])
    else:
        k = max(range(3), key=lambda x: eqs[x])
    keep_i, keep_j = pair_idx[k]
    discard_idx = 3 - keep_i - keep_j
    which = cards3.choose_expose([cards[keep_i], cards[keep_j]])
    reveal_idx = keep_i if which == 0 else keep_j
    return discard_idx, reveal_idx
