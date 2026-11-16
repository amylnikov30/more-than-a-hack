"""Evidence model (§4.2): logistic regression for P(H | features), as clipped log-odds.

l_t = clip(logit P(H), -CLIP, +CLIP), so one frame can never force a switch. Fitted by Newton's
method with a small ridge, on standardised features (training-fold mean and scale); a missing
feature falls back to the training mean, i.e. contributes nothing. Deterministic.

The fitted model and the decoder constants live in ha_model.json next to this file.
"""

import json
import os

import numpy as np

from possession import config

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ha_model.json")


def fit(x, y, l2=config.L2, steps=config.NEWTON_STEPS):
    """{mean, scale, weights} for P(y = 1 | x); x (N, F) may hold NaN, y (N,) of 0 / 1."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    z = _design(x, mean, scale)
    w = np.zeros(z.shape[1])
    penalty = l2 * np.eye(len(w))
    penalty[0, 0] = 0.0
    # z is finite by construction; numpy 2.0 with macOS Accelerate raises spurious FP warnings
    # in large matmuls, so they are silenced here and the result is checked instead
    with np.errstate(all="ignore"):
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-(z @ w)))
            grad = z.T @ (p - y) / len(y) + penalty @ w
            hess = (z * (p * (1 - p))[:, None]).T @ z / len(y) + penalty + 1e-9 * np.eye(len(w))
            w -= np.linalg.solve(hess, grad)
    if not np.isfinite(w).all():
        raise FloatingPointError("logistic fit diverged")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": w.tolist()}


def _design(x, mean, scale):
    z = (np.where(np.isfinite(x), x, mean) - mean) / scale
    return np.column_stack([np.ones(len(x)), z])


def log_odds(model, x, clip=config.CLIP):
    """Clipped logit P(H) per row; NaN rows (off live frames) give 0."""
    x = np.asarray(x, dtype=float)
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    with np.errstate(all="ignore"):   # see fit()
        out = _design(x, mean, scale) @ np.asarray(model["weights"])
    out = np.clip(out, -clip, clip)
    return np.where(np.isfinite(x).any(axis=1), out, 0.0)


def load(path=MODEL_PATH):
    with open(path) as handle:
        return json.load(handle)


def save(model, path=MODEL_PATH):
    with open(path, "w") as handle:
        json.dump(model, handle, indent=1, sort_keys=True)
        handle.write("\n")
