"""Distance utilities for spatial and space-time coordinate systems."""

from __future__ import annotations
import numpy as np


def coord2dist(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix.

    Parameters
    ----------
    c1 : (n1, d) array
    c2 : (n2, d) array

    Returns
    -------
    (n1, n2) distance matrix
    """
    c1, c2 = np.atleast_2d(c1), np.atleast_2d(c2)
    diff = c1[:, None, :] - c2[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))
