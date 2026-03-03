"""Mean-trend (drift) estimation and removal for BME / universal kriging."""

from __future__ import annotations
import numpy as np
from .soft_data import SoftPDF


def design_matrix(coords: np.ndarray, order) -> np.ndarray:
    """Polynomial design matrix.

    Parameters
    ----------
    coords : (n, d) coordinate array
    order  : ``None`` / ``NaN`` → empty (simple kriging),
             ``0`` → constant (ordinary kriging),
             ``1`` → linear,  ``2`` → quadratic.
    """
    coords = np.atleast_2d(coords)
    n, d = coords.shape
    if order is None or (isinstance(order, float) and np.isnan(order)):
        return np.empty((n, 0))
    order = int(order)
    if order == 0:
        return np.ones((n, 1))
    if order == 1:
        return np.column_stack([np.ones(n)] + [coords[:, i] for i in range(d)])
    if order == 2:
        cols = [np.ones(n)] + [coords[:, i] for i in range(d)]
        for i in range(d):
            for j in range(i, d):
                cols.append(coords[:, i] * coords[:, j])
        return np.column_stack(cols)
    raise ValueError(f"order must be NaN, 0, 1, or 2; got {order}")


def estimate_trend(ch, zh, cs, soft_pdfs, ck, order, mean_prior):
    """Estimate and subtract the local mean trend.

    Returns
    -------
    zh_dt    : de-trended hard values
    sp_dt    : de-trended soft PDFs
    mk       : estimated mean at the estimation point
    trend_h  : trend at hard locations
    trend_s  : trend at soft locations
    """
    if order is None or (isinstance(order, float) and np.isnan(order)):
        zh_dt = zh - mean_prior
        sp_dt = [
            SoftPDF(sp.z_grid - mean_prior, sp.pdf_values.copy(), sp.pdf_type)
            for sp in soft_pdfs
        ]
        return zh_dt, sp_dt, mean_prior, np.full(len(zh), mean_prior), np.zeros(len(soft_pdfs))

    Xh = design_matrix(ch, order)
    Xk = design_matrix(np.asarray(ck).reshape(1, -1), order)

    soft_means = (
        np.array([sp.moments()[0] for sp in soft_pdfs]) if soft_pdfs else np.array([])
    )
    if len(soft_pdfs) > 0 and len(cs) > 0:
        Xs = design_matrix(cs, order)
        X_all = np.vstack([Xh, Xs])
        z_all = np.concatenate([zh, soft_means])
    else:
        X_all, z_all = Xh, zh

    try:
        beta = np.linalg.lstsq(X_all, z_all, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(Xh.shape[1])

    trend_h = Xh @ beta if Xh.shape[1] > 0 else np.zeros(len(zh))
    mk = float((Xk @ beta).item()) if Xk.shape[1] > 0 else 0.0
    if len(soft_pdfs) > 0 and len(cs) > 0:
        trend_s = design_matrix(cs, order) @ beta
    else:
        trend_s = np.array([])

    zh_dt = zh - trend_h
    sp_dt = [
        SoftPDF(sp.z_grid - trend_s[i], sp.pdf_values.copy(), sp.pdf_type)
        for i, sp in enumerate(soft_pdfs)
    ]
    return zh_dt, sp_dt, mk, trend_h, trend_s
