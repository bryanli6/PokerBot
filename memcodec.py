"""Per-table memory: fixed-layout, fast, and incapable of raising on decode.

The poison-pill problem
-----------------------
From the rules: "memory that makes act raise comes back unchanged on every later
turn at that table." So a single decode exception does not cost one decision, it
costs *every remaining decision at that table* -- the bot plays the fallback
(check-if-free-else-fold) for up to 250 hands and almost certainly busts.

Decode therefore catches everything and falls back to a fresh state. There is no
input, corrupt or truncated or hostile, that can make `decode` raise.

Layout
------
A fixed struct, not JSON or pickle. Reasons: it is ~20x faster to parse, its size
is constant so it can never approach the 64 KB cap, and a fixed size means a
truncated buffer is detected by a length check rather than by a parser blowing up.

  header: magic(2) version(1) hand_no(2) n_seats(1)        =   6 bytes
  seats:  10 x [raises, calls, folds, checks, seen, shown,
                faced_bet, folded_to_bet]                     = 160 bytes
                                                              -----------
                                                              166 bytes

Well under the ~16 KB the rules describe as costing nothing measurable.
"""
import struct

MAGIC = b"P6"
VERSION = 2
MAX_SEATS = 10
_U16_MAX = 65535

_HEADER = struct.Struct("<2sBHB")
_SEATS = struct.Struct("<" + "H" * (8 * MAX_SEATS))
_SIZE = _HEADER.size + _SEATS.size

# Per-seat counter slots.
RAISES, CALLS, FOLDS, CHECKS, SEEN, SHOWN, FACED, F2B = range(8)


def fresh(n_seats=MAX_SEATS):
    """A new, empty table state."""
    if not isinstance(n_seats, int) or n_seats < 1 or n_seats > MAX_SEATS:
        n_seats = MAX_SEATS
    return {
        "hand_no": 0,
        "n_seats": n_seats,
        "counts": [[0] * 8 for _ in range(MAX_SEATS)],
    }


def decode(blob):
    """Parse `blob` into a state dict. Never raises; returns `fresh()` on anything odd."""
    try:
        if not blob or len(blob) < _SIZE:
            return fresh()
        magic, version, hand_no, n_seats = _HEADER.unpack_from(blob, 0)
        if magic != MAGIC or version != VERSION:
            return fresh()
        if n_seats < 1 or n_seats > MAX_SEATS:
            n_seats = MAX_SEATS
        flat = _SEATS.unpack_from(blob, _HEADER.size)
        counts = [list(flat[i * 8:(i + 1) * 8]) for i in range(MAX_SEATS)]
        return {"hand_no": hand_no, "n_seats": n_seats, "counts": counts}
    except Exception:
        # Deliberately broad: a decode failure here would poison the whole table.
        return fresh()


def encode(state):
    """Serialize `state` to bytes. Never raises; returns an empty-state blob on error."""
    try:
        hand_no = state["hand_no"]
        if hand_no > _U16_MAX:
            hand_no = _U16_MAX
        n_seats = state["n_seats"]
        if n_seats < 1 or n_seats > MAX_SEATS:
            n_seats = MAX_SEATS

        counts = state["counts"]
        flat = []
        for i in range(MAX_SEATS):
            row = counts[i] if i < len(counts) else (0,) * 8
            for j in range(8):
                v = row[j] if j < len(row) else 0
                # Saturate rather than overflow: struct would raise on >65535.
                if v < 0:
                    v = 0
                elif v > _U16_MAX:
                    v = _U16_MAX
                flat.append(int(v))

        return _HEADER.pack(MAGIC, VERSION, hand_no, n_seats) + _SEATS.pack(*flat)
    except Exception:
        return _HEADER.pack(MAGIC, VERSION, 0, MAX_SEATS) + _SEATS.pack(
            *([0] * (8 * MAX_SEATS))
        )


def bump(state, seat, slot, by=1):
    """Increment one counter, ignoring out-of-range seats (anonymized IDs drift)."""
    if 0 <= seat < MAX_SEATS and 0 <= slot < 8:
        row = state["counts"][seat]
        v = row[slot] + by
        row[slot] = _U16_MAX if v > _U16_MAX else v


def aggression(state, seat):
    """Raises as a share of voluntary actions. 0.5 when there is no sample yet.

    Returned as a neutral prior rather than None so callers never branch on it in
    the hot path.
    """
    if not (0 <= seat < MAX_SEATS):
        return 0.5
    r, c, f, ch = state["counts"][seat][:4]
    denom = r + c + f + ch
    if denom < 8:  # too small a sample to mean anything
        return 0.5
    return r / denom


def vpip(state, seat):
    """Share of observed hands in which the seat put money in voluntarily."""
    if not (0 <= seat < MAX_SEATS):
        return 0.5
    row = state["counts"][seat]
    seen = row[SEEN]
    if seen < 8:
        return 0.5
    return (row[RAISES] + row[CALLS]) / seen


def fold_to_bet(state, seat, prior=0.60, weight=4.0):
    """Share of bets this seat has folded to, shrunk toward the field prior.

    The round-1 field median is 69%; the prior sits a little under it so a
    seat we know nothing about is not assumed to be a pushover. `weight` is the
    prior's strength in pseudo-observations.
    """
    if not (0 <= seat < MAX_SEATS):
        return prior
    row = state["counts"][seat]
    faced = row[FACED]
    return (row[F2B] + prior * weight) / (faced + weight)


SIZE = _SIZE
