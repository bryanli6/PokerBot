# ICM-Aware Poker Bot

A tournament poker agent built for the **Bay Area Regional of No Limit**, a nationwide pokerbot tournament hosted by Sequoia Capital, Citadel Securities and OpenAI (November 2026). Built in Python with Claude as an AI coding assistant.

**Result:** _[add your placement, e.g. "placed X of Y teams"]_

## The variant

A new form of no-limit hold'em:

- Each player gets **three** hole cards instead of two.
- After the flop, every player simultaneously **discards one** card face down and **exposes one** of the other two. The exposed card stays public for the rest of the hand.
- Scoring is by **finishing place** (default 50/30/20 points for the top three, 0 for 4th–8th), not by chips won.

## Strategy

The bot estimates its equity honestly, compares it with a tournament-adjusted bar instead of plain pot odds, and acts only on a cautious version of that estimate.

- **ICM instead of chip EV.** Because only finishing place scores, chips you risk are worth more than chips you win. `icm.py` computes exact Malmuth-Harville ICM to find the equity actually needed to profit in points. That bar is roughly 6 points higher than pot odds at a full table and about 18 points higher near the 4th-place cutoff.
- **Monte Carlo equity with exposed cards.** `equity.py` simulates random runouts and treats opponents' exposed cards as known, both as dead cards and as part of their hands.
- **Reading the hidden card.** `revealmodel.py` is an empirical model, fitted to about 36,000 showdowns from the tournament's hand history, that predicts an opponent's hidden card from the one they exposed.
- **Acting on a lower confidence bound.** Decisions use roughly the 84% lower bound of the equity estimate, so noisy simulations don't trigger bad calls.
- **Discard and expose choice.** `cards3.py` simulates the six legal discard/expose combinations and keeps the strongest pair, then exposes the lower of the two kept cards.
- **Preflop tables.** `preflop3.py` holds precomputed three-card preflop equities from simulation, and `openrange.py` sets opening ranges by position.
- **Short-stack play.** `pushfold.py` switches to all-in-or-fold decisions with ICM arithmetic when the stack drops below about 12 big blinds.
- **Safety.** `memcodec.py` stores per-table memory in a fixed format that can't crash on decode, and the agent always stops well before the decision time limit.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Main agent: decision logic and tunable parameters |
| `equity.py` | Monte Carlo equity estimation using exposed cards |
| `icm.py` | Exact ICM calculation for tournament payouts |
| `pushfold.py` | Short-stack shove and call-off math |
| `cards3.py` | Three-card preflop equity and the discard/expose choice |
| `preflop3.py`, `preflop.py` | Precomputed preflop equity tables (generated) |
| `openrange.py` | Preflop opening ranges by position |
| `revealmodel.py` | Model of an opponent's hidden card from their exposed card (generated) |
| `memcodec.py` | Crash-proof per-table memory encoding |

## Running it

The bot runs inside the No Limit tournament's starter kit, which provides `poker_agent_sdk` and the game engine. The starter kit isn't included in this repo because it's the organizers' code. The only other dependency is `eval7` (see `requirements.txt`).
