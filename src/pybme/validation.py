"""Cross-validation utilities."""

from __future__ import annotations
import numpy as np
from .predict import bme_predict


def cross_validate(ch, zh, cs=None, soft_pdfs=None,
                   model="exponential", params=None,
                   nhmax=20, nsmax=8, dmax=np.inf, order=0):
    """Leave-one-out cross-validation on hard data.

    Returns
    -------
    dict with ``'predicted'``, ``'predicted_var'``, ``'actual'``,
    ``'errors'``, ``'rmse'``, ``'mae'``
    """
    ch, zh = np.atleast_2d(ch), np.asarray(zh)
    nh = len(zh)
    pred = np.zeros(nh)
    vari = np.zeros(nh)
    for i in range(nh):
        r = bme_predict(
            ch[i:i + 1], np.delete(ch, i, 0), np.delete(zh, i),
            cs, soft_pdfs, model, params,
            nhmax, nsmax, dmax, order, n_grid=80,
        )[0]
        pred[i], vari[i] = r.mean, r.variance
    err = zh - pred
    return {
        "predicted": pred,
        "predicted_var": vari,
        "actual": zh,
        "errors": err,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
