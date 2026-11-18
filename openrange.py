"""Preflop opening ranges for three-card hold'em.

Why this exists: comparing equity-vs-seven-random-hands against pot odds is a
fold machine. A strong three-card hand has ~25% equity against seven opponents
and the bar to call a blind is ~40%, so the bot folded almost everything it was
not already in for free -- no opens, no steals, no initiative. Round 1 result:
25% third-place finishes, 7.4% firsts. Laddering into the money with nothing.

Standard answer: rank hands by a stable strength metric and open the top X%
from each position, wider the later you act. Percentile is computed once at
import from the generated 3-card table.
"""
import preflop3

# key -> percentile in [0,1] by heads-up equity. 1.0 = best hand.
_PCT = {}


def _build():
    items = sorted(preflop3.TABLE.items(), key=lambda kv: kv[1][0])
    n = len(items)
    for i, (k, _) in enumerate(items):
        _PCT[k] = i / max(1, n - 1)


_build()


def percentile(cards):
    """Strength percentile of a three-card holding, or None."""
    try:
        if len(cards) != 3:
            return None
        key = tuple(sorted(preflop3._norm(c) for c in cards))
        return _PCT.get(key)
    except Exception:  # noqa: BLE001
        return None


# Open the top (1 - threshold) of hands given how many opponents still act
# after us. Later position -> more behind have folded -> open wider.
#   0 behind -> top 50%   (blind vs blind)
#   2 behind -> top 34%   (button)
#   3 behind -> top 28%   (cutoff)
#   7 behind -> top 15%   (under the gun, 8-handed)
_OPEN = {0: 0.50, 1: 0.58, 2: 0.66, 3: 0.72, 4: 0.77, 5: 0.81, 6: 0.85}


def open_threshold(n_behind):
    if n_behind < 0:
        n_behind = 0
    if n_behind > 6:
        n_behind = 6
    return _OPEN[n_behind]
