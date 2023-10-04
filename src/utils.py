import numpy as np


def min_max_normalize(s, min_val=0, max_val=1):
    """Min-max normalizes a signal.

    Parameters
    ----------
    s : list
        Signal that will be normalized.
    min_val : float
        Minimum value of the resulting signal.
    max_val : float
        Maximum value of the resulting signal.

    Returns
    -------
    s_norm : list
        Normalized version of s.
    """
    s_norm = min_val + (s - np.nanmin(s)) * (max_val - min_val) / \
        (np.nanmax(s) - np.nanmin(s))

    return s_norm