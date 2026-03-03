"""Distance utilities for spatial and space-time coordinate systems.

Performance notes
-----------------
* ``coord2dist`` delegates to ``scipy.spatial.distance.cdist`` (compiled C)
  which avoids a temporary ``(n1, n2, d)`` Python-level array.
* For very large one-sided queries (one point vs many) a fast-path using
  ``np.linalg.norm`` is used.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist as _cdist


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
    c1 = np.ascontiguousarray(np.atleast_2d(c1), dtype=np.float64)
    c2 = np.ascontiguousarray(np.atleast_2d(c2), dtype=np.float64)
    # Fast path: single query point → avoids cdist overhead
    if c1.shape[0] == 1:
        return np.linalg.norm(c2 - c1, axis=1).reshape(1, -1)
    if c2.shape[0] == 1:
        return np.linalg.norm(c1 - c2, axis=1).reshape(-1, 1)
    return _cdist(c1, c2, metric="euclidean")
