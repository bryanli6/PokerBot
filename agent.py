"""Bay Area variant bot: three hole cards, discard-and-expose after the flop.

Strategy in one line: estimate equity honestly, compare it against the ICM bar
rather than pot odds, and act only on the lower confidence bound of that
estimate.

Why ICM and not chip EV. Scoring is by finishing place (default 50/30/20, zero
for 4th-8th) and survivors split by ICM at the hand cap. Chips risked are worth
more than chips won, so the equity needed to profit in POINTS exceeds the
pot-odds break-even -- by ~6% at even 8-handed stacks and ~18% on the 4-place
bubble, where the 20-to-0 cliff sits. icm.required_equity() computes that bar
exactly; most entries will play chip EV and spew there.

Variant specifics:
  * preflop equity uses a model fitted to simulation, because the obvious
    "best two-card subset" proxy overstates by 3-7 points (and ~15 for trips);
  * the select phase picks the keep-pair by rollout and exposes the lower card;
  * rollouts consume opponents' exposed cards as both dead cards and known
    holdings.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import gc  # noqa: E402
import time  # noqa: E402

import poker_agent_sdk  # noqa: E402
from poker_agent_sdk import Action, GameState  # noqa: E402

import cards3  # noqa: E402,F401  (imported for import-time warmup)
import equity  # noqa: E402
import openrange  # noqa: E402
import preflop3  # noqa: E402
import pushfold  # noqa: E402
import icm  # noqa: E402
import memcodec as mc  # noqa: E402

# --- tunables ---------------------------------------------------------------
SAFETY_FRACTION = 0.45     # of decision_ms; we stop well short of the deadline
SELECT_FRACTION = 0.30     # the select phase has its own clock; keep it tighter
ROLLOUT_SHARE = 0.60       # of what remains, the rest reserved for ICM
CONFIDENCE_Z = 1.0         # act on the ~84% lower bound of our own estimate
VALUE_BET_EQUITY = 0.55    # equity needed to bet when checking is free
RAISE_MARGIN = 0.20        # above the bar before raising rather than calling
VALUE_FRACTION = 0.65      # bet size as a fraction of pot
SHORT_STACK_BB = 12.0      # below this, play push/fold
DEAD_MIN_OBS = 8           # check+fold observations before a seat counts as dead
OPEN_SIZE_BB = 2.5         # standard open
ICM_FULL_PLAYERS = 4       # full risk premium at/below this many left (the bubble)
ICM_EARLY_WEIGHT = 0.65    # share of the premium applied when the table is still full
THREEBET_PCT = 0.94        # re-raise a raise with the top 6% of hands
CALL_RAISE_PCT = 0.82      # flat a single raise with the top 18%
CAP_WATCH_HANDS = 30       # start reading the hand cap this many hands out
CAP_LOCK_RATIO = 0.92      # lock down when folding to the cap keeps this share of our equity
BLUFF_F2B = 0.50           # bluff-bet when every live opponent folds to bets at least this often
BLUFF_MAX_OPP = 2          # ...and there are at most this many of them
BLUFF_FRACTION = 0.65      # bluff size as a share of pot
STATION_F2B = 0.35         # below this fold rate a seat is a station: value only, thinner
DEAD_STEAL_MAX_BB = 2.5    # only auto-steal when the amount to call is blinds-sized


def _budget_ns(state):
    try:
        ms = float(getattr(state.config, "decision_ms", 1) or 1)
    except Exception:  # noqa: BLE001
        ms = 1.0
    return int(ms * 1e6 * SAFETY_FRACTION)


def _fallback(state):
    """The engine's own fallback: check if free, else fold, else first legal."""
    try:
        la = state.legal_actions
        if not la:
            return Action.check()
        kinds = {a.kind for a in la}
        if "check" in kinds:
            return Action.check()
        if "fold" in kinds:
            return Action.fold()
        return Action(la[0].kind, la[0].min_amount)
    except Exception:  # noqa: BLE001
        return Action.check()


def _safe_raise(state, target):
    ra = state.can("raise")
    if ra is None:
        return Action.call() if state.can("call") else _fallback(state)
    try:
        lo, hi = int(ra.min_amount), int(ra.max_amount)
        t = int(target)
    except Exception:  # noqa: BLE001
        return _fallback(state)
    if hi < lo:
        return _fallback(state)
    if t < lo:
        t = lo
    elif t > hi:
        t = hi
    return Action.raise_to(t)


def _stacks(state):
    """Effective stacks for ICM: stack PLUS current bet.

    An all-in opponent has stack 0 because their chips are in the pot, which is
    indistinguishable from busted. Feeding that to ICM makes every opponent look
    eliminated, collapses the equity denominator, and returns a required equity
    of 1.0 -- "never call anything". Caught in simulation when the bot folded
    pocket aces to a heads-up shove.
    """
    out = []
    try:
        for s in state.seats:
            out.append(max(0, int(s.stack or 0) + int(s.bet or 0)))
    except Exception:  # noqa: BLE001
        return []
    return out


def _my_index(state, stacks):
    try:
        for i, s in enumerate(state.seats):
            if s.seat == state.seat:
                return i
    except Exception:  # noqa: BLE001
        pass
    return 0


def _payouts(state):
    """A round may draw its own split of 100, so never hardcode 50/30/20."""
    try:
        p = tuple(float(x) for x in state.config.payouts)
        if p and sum(p) > 0:
            return p
    except Exception:  # noqa: BLE001
        pass
    return icm.DEFAULT_PAYOUTS


def _record(state, st):
    """Fold this turn's events into per-seat counters, including whether each
    action was taken while FACING a bet -- the stat that drives bet frequency.
    Bets are tracked per seat per street from the event stream itself."""
    try:
        bets = {}
        maxbet = 0
        for ev in state.events:
            k = ev.kind
            s = ev.seat
            if k == "hand":
                st["hand_no"] = (st["hand_no"] + 1) & 0xFFFF
                bets = {}
                maxbet = 0
                continue
            if k == "board":
                bets = {}
                maxbet = 0
                continue
            if s is None:
                continue
            if k == "blind":
                bets[s] = ev.amount
                if ev.amount > maxbet:
                    maxbet = ev.amount
                continue
            if k in ("fold", "check", "call", "raise"):
                facing = maxbet > bets.get(s, 0)
                if facing:
                    mc.bump(st, s, mc.FACED)
                    if k == "fold":
                        mc.bump(st, s, mc.F2B)
                if k == "raise":
                    mc.bump(st, s, mc.RAISES)
                    bets[s] = ev.amount
                    if ev.amount > maxbet:
                        maxbet = ev.amount
                elif k == "call":
                    mc.bump(st, s, mc.CALLS)
                    bets[s] = maxbet
                elif k == "fold":
                    mc.bump(st, s, mc.FOLDS)
                else:
                    mc.bump(st, s, mc.CHECKS)
            elif k in ("show", "reveal"):
                mc.bump(st, s, mc.SHOWN)
            elif k == "stack":
                mc.bump(st, s, mc.SEEN)
    except Exception:  # noqa: BLE001
        pass


def _live_opp_seats(state):
    try:
        return [s.seat for s in state.seats
                if s.seat != state.seat and s.status == "active"]
    except Exception:  # noqa: BLE001
        return []


def _is_dead(st, seat):
    """A seat that has never raised or called, only checked and folded.

    That is exactly the engine's fallback -- what a crashed, timed-out, or
    unwritten bot plays every hand. Round 1 fields are full of them. Against a
    dead seat any bet wins the pot outright, because it folds to a bet and only
    ever checks when checking is free.
    """
    try:
        r, c, f, ch = st["counts"][seat][:4]
        return (r + c) == 0 and (f + ch) >= DEAD_MIN_OBS
    except Exception:  # noqa: BLE001
        return False


def _all_opponents_dead(state, st):
    """True when every opponent still able to act is a dead seat.

    All-in opponents cannot fold, so they do not count as exploitable; if one
    is present we fall through to normal play rather than betting into a
    forced showdown with air.
    """
    try:
        me = state.seat
        seen = 0
        for s in state.seats:
            if s.seat == me:
                continue
            if s.status == "all_in":
                return False
            if s.status != "active":
                continue
            seen += 1
            if not _is_dead(st, s.seat):
                return False
        return seen > 0
    except Exception:  # noqa: BLE001
        return False


def _my_raises_this_round(state):
    """How many times we have already raised in the current betting round."""
    try:
        n = 0
        for ev in reversed(state.history):
            if ev.kind == "board":
                break
            if ev.kind == "raise" and ev.seat == state.seat:
                n += 1
        return n
    except Exception:  # noqa: BLE001
        return 0


def _last_aggressor(state):
    """Seat of the most recent raise this hand, or None."""
    try:
        for ev in reversed(state.history):
            if ev.kind == "raise" and ev.seat is not None and ev.seat != state.seat:
                return ev.seat
    except Exception:  # noqa: BLE001
        pass
    return None


def _aggressor_discount(state, st, to_call, pot):
    """How much to shade our equity when facing a bet, by WHO bet.

    Our rollouts assume opponents hold random hands. A seat that bets rarely
    is not betting a random hand -- it is betting the top of its range. The
    round-1 bot bet only above 62% equity; calling it at pot odds with 35% was
    how the aggressive build bled out. Aggression frequency comes from the
    per-seat counters in table memory; with no sample we assume a moderately
    tight opponent, which is the safe prior in this field.
    """
    size = equity.range_discount(to_call, pot)
    seat = _last_aggressor(state)
    if seat is None:
        return size
    try:
        r, c, f, ch = st["counts"][seat][:4]
        n = r + c + f + ch
        if n < 6:
            agg = 0.18                       # prior: fairly tight
        else:
            agg = r / n
    except Exception:  # noqa: BLE001
        agg = 0.18
    # Maps bet frequency to a multiplier: a seat that raises 5% of the time
    # gets 0.62 (its bets are near-nutted); 35%+ is maniac territory, no shade.
    if agg <= 0.05:
        who = 0.62
    elif agg >= 0.35:
        who = 1.0
    else:
        who = 0.62 + (agg - 0.05) * (0.38 / 0.30)
    return min(size, who)


def _cap_lock(state, stacks_eff, idx, payouts):
    """True when the hand cap is close and folding to it preserves our equity.

    At the cap survivors split by ICM on stacks, so a seat that outlasts the
    blinds by folding has nothing to gain from a confrontation and a whole
    finishing place to lose. Only evaluated in the last CAP_WATCH_HANDS hands.
    """
    try:
        left = int(state.config.hand_cap) - int(state.hand_number)
        if left > CAP_WATCH_HANDS or left < 0:
            return False
        cv = pushfold.cap_view(state, stacks_eff)
        if not cv.survives[idx]:
            return False
        now = icm.icm_equity_seat(stacks_eff, idx, payouts)
        if now <= 1e-9:
            return False
        return (cv.fold_lock_ev / now) >= CAP_LOCK_RATIO
    except Exception:  # noqa: BLE001
        return False


def _icm_bar(state, need, pot_odds):
    """Blend the ICM requirement toward pot odds while the table is still full.

    ICM assumes equal-skill opponents. Against a weaker field, a full risk
    premium at 8-handed is a leak: it refuses chip-EV spots that build the
    stack we need to actually WIN, and 1st pays 50 of the 100 points. Round 1
    we cashed 52% of the time and won 7.4% -- laddering into the money broke.
    Near the bubble the premium is real and we take all of it.
    """
    try:
        left = int(state.players_remaining)
    except Exception:  # noqa: BLE001
        left = 8
    if need <= pot_odds:
        return pot_odds
    w = 1.0 if left <= ICM_FULL_PLAYERS else ICM_EARLY_WEIGHT
    return pot_odds + w * (need - pot_odds)


def _preflop_counts(state, bb):
    """(n_raised, n_committed, n_behind) from the seat table.

    n_raised:    opponents whose bet exceeds the big blind (someone raised)
    n_committed: opponents with any chips in, i.e. likely to see a flop
    n_behind:    opponents still to act -- active with a bet at or below the
                 blind, which includes the blinds themselves and any limpers
                 (a slight overcount that only makes us tighter multiway)
    """
    n_raised = n_committed = n_behind = 0
    try:
        me = state.seat
        for s in state.seats:
            if s.seat == me or s.status not in ("active", "all_in"):
                continue
            b = int(s.bet or 0)
            if b > bb + 0.5:
                n_raised += 1
            if b > 0:
                n_committed += 1
            if s.status == "active" and b <= bb + 0.5:
                n_behind += 1
    except Exception:  # noqa: BLE001
        pass
    return n_raised, n_committed, n_behind


class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        t0 = time.perf_counter_ns()
        try:
            return self._decide(state, memory, t0)
        except Exception:  # noqa: BLE001
            return _fallback(state), (memory or b"")

    def _preflop(self, state, st, to_call, pot, bb, stacks, idx, payouts, out_mem, locked=False):
        """Range-based preflop. Returns (Action, memory) or None to fall through."""
        pct = openrange.percentile(state.cards)
        if pct is None:
            return None
        n_raised, n_committed, n_behind = _preflop_counts(state, bb)
        my_stack = stacks[idx] if 0 <= idx < len(stacks) else 0
        spr = my_stack / bb if bb > 0 else 100.0
        ra = state.can("raise")

        def open_raise():
            if ra is None:
                return Action.call() if state.can("call") else _fallback(state)
            if spr <= SHORT_STACK_BB:
                return _safe_raise(state, ra.max_amount)       # push/fold territory
            # Open to a standard size, one extra blind per limper already in.
            limpers = max(0, n_committed - 2)
            return _safe_raise(state, int(bb * (OPEN_SIZE_BB + limpers)))

        # ---- nobody has raised ----------------------------------------------
        if n_raised == 0:
            thr = openrange.open_threshold(n_behind)
            if spr <= SHORT_STACK_BB:
                thr -= 0.12          # short: shove wider, fold equity is the point
            if locked:
                thr = max(thr, 0.95)  # cap lock-down: premiums only
            if pct >= thr:
                return open_raise(), out_mem
            if to_call <= 0:
                return (Action.check() if state.can("check") else _fallback(state)), out_mem
            # Completing from the small blind with a playable hand is cheap.
            if to_call <= bb * 0.6 and pct >= thr - 0.15 and state.can("call"):
                return Action.call(), out_mem
            return (Action.fold() if state.can("fold") else _fallback(state)), out_mem

        # ---- facing a raise ---------------------------------------------------
        # Measure against the players actually in the pot, not the whole table.
        n_in = max(1, n_committed)
        eq = preflop3.equity(state.cards, n_in)
        if eq is None:
            return None
        denom = pot + to_call
        pot_odds = (to_call / denom) if denom > 0 else 1.0
        try:
            need = icm.required_equity(stacks, idx, to_call, payouts)
        except Exception:  # noqa: BLE001
            need = pot_odds
        if need >= 0.999:
            need = pot_odds
        bar = _icm_bar(state, need, pot_odds)

        # Commitment guard: pot odds on each small re-raise look fine, but the
        # sum of them is our stack. Past a quarter of it, only a real hand
        # continues, whatever the odds say.
        if to_call > 0.25 * my_stack and pct < 0.88 and eq < bar + 0.15:
            return (Action.fold() if state.can("fold") else _fallback(state)), out_mem

        # A tight seat's raise is a strong hand: demand a strong hand back.
        agg_mult = _aggressor_discount(state, st, to_call, pot)
        if agg_mult < 0.80:
            if pct < 0.90:
                return (Action.fold() if state.can("fold") else _fallback(state)), out_mem
        eq *= agg_mult

        # We already raised and got re-raised: that is a monster or a maniac.
        # Do not re-raise again without the very top of the range.
        if _my_raises_this_round(state) >= 1 and pct < 0.985:
            if eq >= bar + 0.12 and state.can("call") and to_call <= 0.33 * my_stack:
                return Action.call(), out_mem
            return (Action.fold() if state.can("fold") else _fallback(state)), out_mem

        if pct >= THREEBET_PCT and ra is not None:
            if spr <= SHORT_STACK_BB * 1.5:
                return _safe_raise(state, ra.max_amount), out_mem
            return _safe_raise(state, int(to_call * 3 + pot)), out_mem
        if eq >= bar and (pct >= CALL_RAISE_PCT or eq >= bar + 0.10):
            if state.can("call"):
                return Action.call(), out_mem
        return (Action.fold() if state.can("fold") else _fallback(state)), out_mem

    def _decide(self, state, memory, t0):
        budget = _budget_ns(state)
        perf = time.perf_counter_ns

        st = mc.decode(memory)
        _record(state, st)
        out_mem = mc.encode(st)

        if not state.legal_actions:
            return _fallback(state), out_mem

        # --- the variant's own decision --------------------------------------
        if state.phase == "select":
            sel = state.can("select")
            if sel is not None:
                try:
                    spent = perf() - t0
                    sel_budget = int(_budget_ns(state) / SAFETY_FRACTION * SELECT_FRACTION)
                    d, r = equity.best_selection(
                        state, max(sel_budget - spent, 60_000), t0
                    )
                    d, r = int(d), int(r)
                    if d == r or not (0 <= d <= 2) or not (0 <= r <= 2):
                        d, r = 2, 0
                    return Action("select", 0, (d, r)), out_mem
                except Exception:  # noqa: BLE001
                    return Action("select", 0, (2, 0)), out_mem
            return _fallback(state), out_mem

        if state.phase != "betting":
            # Unknown future phase: take its documented default.
            la = state.legal_actions[0]
            return Action(la.kind, la.min_amount), out_mem

        if perf() - t0 > budget:
            return _fallback(state), out_mem

        stacks = _stacks(state)
        if not stacks:
            return _fallback(state), out_mem
        idx = _my_index(state, stacks)
        payouts = _payouts(state)

        try:
            to_call = max(0, int(state.to_call or 0))
            pot = max(0, int(state.pot or 0))
        except Exception:  # noqa: BLE001
            to_call, pot = 0, 0

        try:
            bb = max(1.0, float(state.blinds.big_blind))
        except Exception:  # noqa: BLE001
            bb = 20.0

        # --- dead-seat steal -------------------------------------------------
        # Every remaining opponent folds to any bet: take the pot for the
        # minimum, with any three cards, before spending the budget on equity.
        if to_call <= DEAD_STEAL_MAX_BB * bb and _all_opponents_dead(state, st):
            ra = state.can("raise")
            if ra is not None:
                return Action.raise_to(int(ra.min_amount)), out_mem
            if to_call <= 0 and state.can("check"):
                return Action.check(), out_mem
            if state.can("call"):
                return Action.call(), out_mem

        locked = _cap_lock(state, stacks, idx, payouts)

        # --- preflop: range-based, not equity-vs-everyone ----------------
        if not state.board and len(state.cards) == 3:
            r = self._preflop(state, st, to_call, pot, bb, stacks, idx, payouts, out_mem, locked)
            if r is not None:
                return r

        spent = perf() - t0
        remaining = budget - spent
        if remaining < 0:
            remaining = 0
        eq, se = equity.estimate(state, int(remaining * ROLLOUT_SHARE), t0)
        if eq is None:
            return _fallback(state), out_mem

        conf = eq - CONFIDENCE_Z * se
        if to_call > 0:
            conf *= _aggressor_discount(state, st, to_call, pot)
        if conf < 0.0:
            conf = 0.0

        my_stack = stacks[idx] if 0 <= idx < len(stacks) else 0
        spr = my_stack / bb

        # --- free to continue ------------------------------------------------
        if to_call <= 0:
            ra = state.can("raise")
            live = _live_opp_seats(state)
            f2bs = [mc.fold_to_bet(st, s) for s in live] or [0.60]
            min_f2b = min(f2bs)
            station = min_f2b <= STATION_F2B
            value_thr = 0.80 if locked else (0.50 if station else VALUE_BET_EQUITY)
            if ra is not None and conf >= value_thr:
                return _safe_raise(state, int(pot * VALUE_FRACTION)), out_mem
            # The field folds to 69% of bets. Heads-up or against two folders,
            # a 65%-pot bet is +EV as a pure bluff once they fold >39%; we ask
            # for more margin and never bluff a station.
            if (ra is not None and not locked and not station
                    and 0 < len(live) <= BLUFF_MAX_OPP and min_f2b >= BLUFF_F2B
                    and conf >= 0.12):
                return _safe_raise(state, int(pot * BLUFF_FRACTION)), out_mem
            if state.can("check"):
                return Action.check(), out_mem
            return _fallback(state), out_mem

        # --- facing a bet ----------------------------------------------------
        denom = pot + to_call
        pot_odds = (to_call / denom) if denom > 0 else 1.0

        try:
            need = icm.required_equity(stacks, idx, to_call, payouts)
        except Exception:  # noqa: BLE001
            need = pot_odds
        if need >= 0.999:
            need = pot_odds      # degenerate ICM view; trust pot odds instead
        bar = _icm_bar(state, need, pot_odds)
        if locked:
            bar += 0.15

        if conf < bar:
            return (Action.fold() if state.can("fold") else _fallback(state)), out_mem

        ra = state.can("raise")
        if ra is not None and conf >= bar + RAISE_MARGIN:
            if spr <= SHORT_STACK_BB:
                return _safe_raise(state, ra.max_amount), out_mem
            return _safe_raise(state, int(pot * VALUE_FRACTION)), out_mem

        if state.can("call"):
            return Action.call(), out_mem
        return _fallback(state), out_mem


def _warmup():
    """Pay every first-call cost at import, where the startup allowance covers it."""
    try:
        probe = [2000, 1800, 1500, 1200, 900, 600, 400, 200]
        for s in (0, 3, 7):
            icm.icm_equity_seat(probe, s)
            icm.required_equity(probe, s, 500)
        mc.decode(mc.encode(mc.fresh(8)))
        cards3.preflop3(["As", "Kd", "7c"], 3)
        equity.estimate_preflop_probe = None
        try:
            import preflop3  # noqa: F401  (forces the 22k-entry table build here)
            preflop3.equity(["As", "Kd", "7c"], 3)
        except Exception:  # noqa: BLE001
            pass
        equity.rollout(["As", "Kd"], ["7h", "2c", "Ts"], [1, 2], {1: "Qh"}, 200_000)
    except Exception:  # noqa: BLE001
        pass


_warmup()

try:
    gc.collect()
    gc.freeze()      # keep forked copies sharing these pages
    gc.disable()     # gen-0 pauses cost more than the collector saves here
except Exception:  # noqa: BLE001
    pass
