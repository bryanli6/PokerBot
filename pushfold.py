"""ICM shove / call-off arithmetic and hand-cap awareness. Pure Python, stdlib only.

Owner: pushfold chat. It does NOT rank hands: the caller supplies equities (from the preflop
table / equity module) and fold probabilities (from `profiles.fold_to_raise_rate`), and this
module turns them into points.

Hot-path cost: `hero_icm` is O(n^2) for the 3-paid-place default (about 10-20 us at 8
seats); `shove_ev` with 7 candidate callers makes 16 calls, ~250 us worst case. Call
`shove_ev` once per decision, not per candidate hand.

API
    hero_icm(stacks, hero, payouts) -> float            points for `hero` if the table stopped now
    icm_all(stacks, payouts) -> list[float]             every seat (slower; tests / cap logic)
    shove_ev(...) -> ShoveResult                        EV of shoving vs folding, in points
    call_required_equity(...) -> float                  ICM bar for calling an all-in
    effective_blinds(state) -> (sb, bb, ante)           next level's blinds when it starts next hand
    cap_view(state) -> CapView                          hands left, blind cost to the cap, survival
"""

from collections.abc import Sequence

__all__ = [
    "hero_icm", "icm_all", "shove_ev", "call_required_equity", "ShoveResult",
    "effective_blinds", "cap_view", "CapView", "stacks_and_bets",
]


# ---- ICM ---------------------------------------------------------------------------------

def hero_icm(stacks: Sequence[int], hero: int, payouts: Sequence[float]) -> float:
    """Malmuth-Harville equity of one seat. Exact. Seats with 0 chips are out.

    Closed form for up to three paid places (the event default 50/30/20); the general
    DP for anything longer."""
    sh = stacks[hero]
    if sh <= 0:
        return 0.0
    alive = [s for s in stacks if s > 0]
    paid = len(payouts)
    while paid > 0 and payouts[paid - 1] == 0:  # 50/30/20/0/0/0/0/0 pays three places
        paid -= 1
    places = paid if paid < len(alive) else len(alive)
    if places == 0:
        return 0.0
    if places > 3:
        return icm_all(stacks, payouts)[hero]
    total = float(sum(alive))
    p1 = payouts[0]
    ev = p1 * sh / total
    if places == 1:
        return ev
    p2 = payouts[1]
    others = [float(s) for i, s in enumerate(stacks) if s > 0 and i != hero]
    m = len(others)
    second = 0.0
    if places == 2:
        for sj in others:
            second += sj / (total - sj)
        return ev + p2 * sh * second / total
    p3 = payouts[2]
    third = 0.0
    for j in range(m):
        sj = others[j]
        rem = total - sj
        second += sj / rem
        inner = 0.0
        for k in range(m):
            if k != j:
                sk = others[k]
                inner += sk / (rem - sk)
        third += sj * inner / rem
    return ev + (p2 * second + p3 * third) * sh / total


def outcome_points(stacks: Sequence[int], hero: int, payouts: Sequence[float], alive_before: int) -> float:
    """Hero's points after a hypothetical hand: ICM on the new stacks, or, if hero is left
    with nothing, the payout of the place hero busts in (busting is not zero equity:
    heads-up it is second place). `alive_before` = seats with chips before the hand."""
    if stacks[hero] > 0:
        return hero_icm(stacks, hero, payouts)
    place = alive_before - 1  # zero-based: busting from k live seats = place k
    return float(payouts[place]) if 0 <= place < len(payouts) else 0.0


def alive_count(stacks: Sequence[int], bets: Sequence[int] | None = None) -> int:
    if bets is None:
        return sum(1 for s in stacks if s > 0)
    return sum(1 for s, b in zip(stacks, bets) if s + b > 0)


def icm_all(stacks: Sequence[int], payouts: Sequence[float]) -> list[float]:
    """Every seat's equity: the SDK's DP, duplicated so this module has no import."""
    players = [p for p in range(len(stacks)) if stacks[p] > 0]
    total = sum(stacks[p] for p in players)
    equity = [0.0] * len(stacks)
    places = min(len(payouts), len(players))
    chance = {0: 1.0}
    for place in range(places):
        following: dict[int, float] = {}
        for mask, probability in chance.items():
            remaining = total - sum(stacks[p] for i, p in enumerate(players) if mask >> i & 1)
            for i, player in enumerate(players):
                if mask >> i & 1:
                    continue
                share = probability * stacks[player] / remaining
                equity[player] += share * payouts[place]
                following[mask | 1 << i] = following.get(mask | 1 << i, 0.0) + share
        chance = following
    return equity


# ---- Shove arithmetic --------------------------------------------------------------------

class ShoveResult:
    __slots__ = ("fold_ev", "shove_ev", "p_no_call", "required", "best_caller")

    def __init__(self, fold_ev, shove_ev, p_no_call, required, best_caller):
        self.fold_ev = fold_ev          # points if we fold now
        self.shove_ev = shove_ev        # points if we shove now, given equities and call probs
        self.p_no_call = p_no_call      # probability everyone folds
        self.required = required        # {seat: equity needed vs that seat if it surely calls}
        self.best_caller = best_caller  # the seat with the lowest required equity

    @property
    def edge(self) -> float:
        return self.shove_ev - self.fold_ev

    def __repr__(self) -> str:
        return (f"ShoveResult(fold={self.fold_ev:.3f} shove={self.shove_ev:.3f} "
                f"p_no_call={self.p_no_call:.3f} required={self.required})")


def stacks_and_bets(state) -> tuple[list[int], list[int]]:
    """(chips behind, chips bet this street) per seat, from `state.seats`."""
    stacks = []
    bets = []
    for s in state.seats:
        stacks.append(s.stack)
        bets.append(s.bet)
    return stacks, bets


def shove_ev(
    hero: int,
    stacks: Sequence[int],
    bets: Sequence[int],
    pot: int,
    payouts: Sequence[float],
    callers: Sequence[tuple[int, float, float]],
    pot_taker: int | None = None,
) -> ShoveResult:
    """Points for shoving our whole stack now versus folding.

    stacks, bets: chips behind and chips bet this street for every seat (busted seats 0).
    pot: all chips committed this hand including current bets (`state.pot`).
    callers: (seat, p_call, equity_vs_seat) for every seat that can still call; equity is
        our all-in equity against the range that seat calls with. Multiway calls are folded
        into single-caller cases: P(j calls alone) = p_j * prod(1 - p_k), k != j, and the
        leftover multiway mass is spread over the callers in proportion.
    pot_taker: who collects the pot if we fold (the current aggressor; default: the seat
        with the largest bet other than us, else the big blind guess = the largest bet).
    """
    n = len(stacks)
    hero_total = bets[hero] + stacks[hero]
    alive = alive_count(stacks, bets)
    # Fold: our bets are gone; the pot goes to the taker (approximation: it is contested
    # between the others, but hero's equity only depends on it weakly).
    if pot_taker is None:
        best = -1
        pot_taker = hero
        for i in range(n):
            if i != hero and bets[i] > best and (stacks[i] > 0 or bets[i] > 0):
                best = bets[i]
                pot_taker = i
    fold_stacks = list(stacks)
    if pot_taker != hero:
        fold_stacks[pot_taker] += pot
    fold_ev = hero_icm(fold_stacks, hero, payouts)

    # Everyone folds.
    win_stacks = list(stacks)
    win_stacks[hero] = stacks[hero] + pot
    allfold_ev = hero_icm(win_stacks, hero, payouts)

    p_no_call = 1.0
    for _, p, _ in callers:
        p_no_call *= 1.0 - p
    # Single-caller weights.
    weights = []
    for j, p, _ in callers:
        w = p
        for k, q, _ in callers:
            if k != j:
                w *= 1.0 - q
        weights.append(w)
    single = sum(weights)
    leftover = 1.0 - p_no_call - single  # multiway mass
    if leftover > 1e-12:
        if single > 1e-12:
            scale = (single + leftover) / single
            weights = [w * scale for w in weights]
        else:  # Every caller is certain: split the mass by call probability.
            total_p = sum(p for _, p, _ in callers)
            weights = [(p / total_p) * (1.0 - p_no_call) for _, p, _ in callers]

    ev = p_no_call * allfold_ev
    required = {}
    best_caller = None
    best_req = 2.0
    for (j, _, eq), w in zip(callers, weights):
        j_total = bets[j] + stacks[j]
        matched = j_total if j_total < hero_total else hero_total
        excess = hero_total - matched           # refunded to hero if j is shorter
        pot_after = pot + (hero_total - bets[hero]) + (matched - bets[j])
        # Hero wins.
        ws = list(stacks)
        ws[hero] = pot_after
        ws[j] = j_total - matched
        win_j = hero_icm(ws, hero, payouts)
        # Hero loses (busting = the payout of the place we go out in).
        ls = list(stacks)
        ls[hero] = excess
        ls[j] = j_total - matched + pot_after - excess
        lose_j = outcome_points(ls, hero, payouts, alive)
        ev += w * (eq * win_j + (1.0 - eq) * lose_j)
        span = win_j - lose_j
        req = (fold_ev - lose_j) / span if span > 1e-12 else 1.0
        required[j] = req
        if req < best_req:
            best_req = req
            best_caller = j
    return ShoveResult(fold_ev, ev, p_no_call, required, best_caller)


def call_required_equity(
    hero: int,
    stacks: Sequence[int],
    bets: Sequence[int],
    pot: int,
    payouts: Sequence[float],
    villain: int,
    to_call: int,
) -> float:
    """Equity we need to call `to_call` more chips against `villain`'s all-in (or any bet
    that puts us all in). Treats the hand as ending at this call (no further betting), the
    standard all-in call-off model. Returns 0..1; above 1 means never."""
    call = to_call if to_call < stacks[hero] else stacks[hero]
    hero_in = bets[hero] + call
    # Villain's bet beyond what we can match is refunded.
    excess = bets[villain] - hero_in if bets[villain] > hero_in else 0
    pot_after = pot + call - excess
    # Fold now: villain takes the pot.
    fs = list(stacks)
    fs[villain] += pot
    fold_ev = hero_icm(fs, hero, payouts)
    # Call and win.
    ws = list(stacks)
    ws[hero] = stacks[hero] - call + pot_after
    ws[villain] = stacks[villain] + excess
    win_ev = hero_icm(ws, hero, payouts)
    # Call and lose (busting pays the place we go out in).
    ls = list(stacks)
    ls[hero] = stacks[hero] - call
    ls[villain] = stacks[villain] + excess + pot_after
    lose_ev = outcome_points(ls, hero, payouts, alive_count(stacks, bets))
    span = win_ev - lose_ev
    if span <= 1e-12:
        return 1.0
    return (fold_ev - lose_ev) / span


# ---- Blind levels and the hand cap -------------------------------------------------------

def effective_blinds(state) -> tuple[int, int, int]:
    """(small, big, ante) to size stacks with: the next level's blinds when the level changes
    after this hand, otherwise the current ones."""
    if state.hands_until_next_level <= 1:
        levels = state.config.blinds
        nxt = state.blind_level + 1
        if nxt >= len(levels):
            nxt = len(levels) - 1
        level = levels[nxt]
        return level.small_blind, level.big_blind, level.ante
    b = state.blinds
    return b.small_blind, b.big_blind, b.ante


class CapView:
    __slots__ = ("hands_left", "alive", "cost", "survives", "leader", "fold_lock_ev")

    def __init__(self, hands_left, alive, cost, survives, leader, fold_lock_ev):
        self.hands_left = hands_left      # hands after this one before the table stops
        self.alive = alive                # seats with chips
        self.cost = cost                  # expected blind+ante cost per seat to the cap
        self.survives = survives          # per seat: stack > cost (folding every hand works)
        self.leader = leader              # seat with the most chips
        self.fold_lock_ev = fold_lock_ev  # hero's points if everyone just folds to the cap

    def __repr__(self) -> str:
        return (f"CapView(hands_left={self.hands_left} alive={self.alive} cost={self.cost:.0f} "
                f"leader={self.leader} fold_lock_ev={self.fold_lock_ev:.3f})")


def cap_view(state, stacks: Sequence[int] | None = None) -> CapView:
    """What the hand cap means right now. Cost model: each remaining hand charges every live
    seat (sb + bb) / alive + ante, walking the blind schedule hand by hand."""
    cfg = state.config
    hands_left = cfg.hand_cap - state.hand_number
    if hands_left < 0:
        hands_left = 0
    if stacks is None:
        stacks = [s.stack + s.bet for s in state.seats]
    alive = sum(1 for s in stacks if s > 0)
    levels = cfg.blinds
    last = len(levels) - 1
    level = state.blind_level
    until = state.hands_until_next_level
    per = cfg.hands_per_level
    cost = 0.0
    if alive > 0:
        for _ in range(hands_left):
            until -= 1
            if until <= 0:
                until = per
                if level < last:
                    level += 1
            b = levels[level if level <= last else last]
            cost += (b.small_blind + b.big_blind) / alive + b.ante
    survives = [s > cost for s in stacks]
    leader = max(range(len(stacks)), key=lambda i: stacks[i]) if stacks else 0
    eroded = [max(int(s - cost), 0) if s > 0 else 0 for s in stacks]
    if all(e == 0 for e in eroded):
        eroded = list(stacks)
    fold_lock_ev = outcome_points(eroded, state.seat, cfg.payouts, alive)
    return CapView(hands_left, alive, cost, survives, leader, fold_lock_ev)
