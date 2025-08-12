import numpy as np


def IH_powerLaw_stressBased(t_exp, sigma_exp, A, alpha, beta, log=False):
    """
        Model #1: Stress-based power law.
        Sensible limits for the parameters:
        0 <= A <= 20
        0.5 <= alpha <= 2.5
        0.01 <= beta <= 1
    """
    if log:
        return -A + alpha * np.log(sigma_exp) + beta * np.log(t_exp) + np.log(100)
    else:
        return np.exp(-A) * (sigma_exp ** alpha) * (t_exp ** beta) * 100  # convert to percentage
    