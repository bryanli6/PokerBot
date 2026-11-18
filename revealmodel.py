"""Empirical model of an opponent's HIDDEN card given their EXPOSED card.
GENERATED from 36,393 real showdowns in the round hand history -- do not edit.

Why: the field keeps its two best cards and exposes one, so the exposed card
is strongly predictive of the hidden one. Exposing a K means KK 39% of the
time (random: ~6%). Exposing a 2-5 mostly marks a default-(2,0) bot whose
hidden card really is random. Sampling the hidden card from this table instead
of uniformly is the single largest correction to opponent equity available.

P_HIDDEN[exposed_rank][hidden_rank], ranks indexed 0..12 = 2..A.
P_SUITED[exposed_rank] = probability the hidden card shares the exposed suit.
Caveat: estimated from hands that REACHED showdown, which are stronger than
hands that folded; appropriate when the opponent is still contesting the pot.
"""
RANKS = "23456789TJQKA"
P_HIDDEN = (
    (0.1873, 0.0432, 0.0532, 0.0502, 0.0432, 0.0373, 0.0402, 0.0492, 0.0661, 0.071, 0.084, 0.1058, 0.1694),  # exposed 2
    (0.0264, 0.2172, 0.049, 0.0423, 0.0423, 0.0356, 0.0414, 0.049, 0.0741, 0.0699, 0.0774, 0.1159, 0.1595),  # exposed 3
    (0.015, 0.0242, 0.2038, 0.0702, 0.058, 0.0449, 0.0449, 0.0418, 0.0526, 0.0672, 0.094, 0.1086, 0.1747),  # exposed 4
    (0.0207, 0.0194, 0.0246, 0.2724, 0.0536, 0.0424, 0.0463, 0.043, 0.0516, 0.0667, 0.0851, 0.1121, 0.162),  # exposed 5
    (0.0105, 0.0145, 0.0248, 0.031, 0.3153, 0.059, 0.0516, 0.0527, 0.0641, 0.0561, 0.0846, 0.0926, 0.1433),  # exposed 6
    (0.0105, 0.0129, 0.0212, 0.0212, 0.0314, 0.3514, 0.0621, 0.0562, 0.0601, 0.0723, 0.0806, 0.0918, 0.1283),  # exposed 7
    (0.0139, 0.0071, 0.0169, 0.0144, 0.0242, 0.0302, 0.3649, 0.0726, 0.0649, 0.0804, 0.0744, 0.0889, 0.1472),  # exposed 8
    (0.0116, 0.0106, 0.013, 0.0182, 0.0227, 0.0268, 0.022, 0.3828, 0.0763, 0.0863, 0.088, 0.0998, 0.142),  # exposed 9
    (0.0096, 0.0124, 0.0138, 0.0119, 0.0127, 0.0211, 0.0311, 0.0325, 0.3851, 0.0962, 0.1116, 0.1158, 0.146),  # exposed T
    (0.0115, 0.0111, 0.0081, 0.0132, 0.0162, 0.0205, 0.0278, 0.0295, 0.0387, 0.3787, 0.116, 0.1466, 0.182),  # exposed J
    (0.0127, 0.0132, 0.0134, 0.0158, 0.0152, 0.0209, 0.0229, 0.0247, 0.0437, 0.0541, 0.3974, 0.1558, 0.2101),  # exposed Q
    (0.0126, 0.0177, 0.0147, 0.0169, 0.0231, 0.0222, 0.0282, 0.0352, 0.0505, 0.0616, 0.0838, 0.3892, 0.2443),  # exposed K
    (0.0252, 0.0266, 0.0277, 0.0322, 0.0288, 0.0344, 0.0369, 0.0417, 0.0554, 0.075, 0.0973, 0.1256, 0.3932),  # exposed A
)

P_SUITED = (0.294, 0.279, 0.25, 0.246, 0.245, 0.22, 0.224, 0.213, 0.223, 0.218, 0.201, 0.196, 0.187)

# Blend toward uniform. The table is estimated from hands that REACHED
# showdown, which are stronger than the average hand still in; full weight is
# right when the opponent is betting at us, too pessimistic when they check.
# Validated at 1.0 against 3,000 real showdowns: the uniform model over-predicted
# hero equity by 16 points, 0.65 by 10; even full weight is slightly optimistic,
# because showdown hands are the ones that kept betting.
MODEL_WEIGHT = 1.0
_U = 1.0 / 13.0
P_HIDDEN = tuple(
    tuple(MODEL_WEIGHT * p + (1.0 - MODEL_WEIGHT) * _U for p in row) for row in P_HIDDEN
)

_CUM = [None] * 13
for _i, _row in enumerate(P_HIDDEN):
    _c = []
    _acc = 0.0
    for _p in _row:
        _acc += _p
        _c.append(_acc)
    _CUM[_i] = _c


def sample_hidden_rank(exposed_rank, u):
    """Rank index 0..12 for the hidden card, given exposed rank and a uniform u."""
    c = _CUM[exposed_rank]
    for i in range(13):
        if u <= c[i]:
            return i
    return 12
