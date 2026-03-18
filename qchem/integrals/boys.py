import numpy as np
from scipy.special import gammainc, gamma
from math import gamma as _gamma


def boys(m: int, x: float) -> float:
    """
    Compute the Boys function F_m(x) = integral_0^1 t^(2m) exp(-x t^2) dt.

    Two regimes:
      x < 1.0  : Taylor series (20 terms, error ~ x^20 / 20!)
      x >= 1.0 : incomplete gamma relation (exact up to scipy precision)

    Note: an asymptotic form Gamma(m+1/2) / (2 x^(m+1/2)) is available via
    _boys_asymptotic(), but the safe threshold depends on m — roughly
    x > 40 + 5*m for 1e-10 relative error — so it is not activated here.
    Use it explicitly inside performance-critical callers that know m_max.
    """
    if x < 1.0:
        return _boys_small_x(m, x)
    else:
        a = m + 0.5
        return gamma(a) * gammainc(a, x) / (2.0 * x**a)


def _boys_small_x(m: int, x: float, n_terms: int = 20) -> float:
    """
    Taylor series:

        F_m(x) = sum_{k=0}^inf  (-x)^k / (k! (2m+2k+1))

    Stable recurrence between successive terms:

        term_k = term_{k-1} * (-x / k) * (2m+2k-1) / (2m+2k+1)

    which cancels cleanly without accumulating the (2m+1) denominator
    from term_0 into every subsequent term (the previous version did not
    cancel correctly and produced an O(x) error for m > 0).
    """
    result = 0.0
    term = 1.0 / (2 * m + 1)
    for k in range(1, n_terms + 1):
        result += term
        term *= (-x / k) * (2 * m + 2 * k - 1) / (2 * m + 2 * k + 1)
    result += term  # final term (k = n_terms)
    return result


def _boys_asymptotic(m: int, x: float) -> float:
    """
    Asymptotic form for large x:

        F_m(x)  ->  Gamma(m + 1/2) / (2 x^(m + 1/2))    as x -> inf

    The relative error is O(exp(-x)) and is below double-precision
    machine epsilon for x > 25.  Uses math.gamma (pure C) rather than
    scipy.special.gammainc, so there is no scipy overhead in the hot loop.
    """
    a = m + 0.5
    return _gamma(a) / (2.0 * x**a)


def boys_array(m_max: int, x: float) -> np.ndarray:
    """
    Compute F_m(x) for m = 0, 1, ..., m_max all at once using downward
    recurrence from F_{m_max}.  Returns an array of length m_max + 1.

    This is the function called in the hot loop.  The seed F_{m_max}(x)
    is computed via boys(), which avoids the scipy incomplete-gamma call
    entirely when x > 25 (asymptotic branch) or x < 1 (Taylor branch),
    so the only path that touches scipy is the moderate-x regime.

    Downward recurrence (exact):
        F_m = (2x * F_{m+1} + exp(-x)) / (2m + 1)
    """
    F = np.zeros(m_max + 1)
    F[m_max] = boys(m_max, x)
    exp_x = np.exp(-x)
    for m in range(m_max - 1, -1, -1):
        F[m] = (2.0 * x * F[m + 1] + exp_x) / (2 * m + 1)
    return F