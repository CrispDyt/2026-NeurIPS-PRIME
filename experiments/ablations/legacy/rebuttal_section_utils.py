"""Opt-in utilities for the Lorenz63 recurrence-section rebuttal experiment.

This module does not implement PRIME training.  It only provides event
extraction, matched controls, and extractor diagnostics used by the rebuttal
harness.  The existing data-driven PRIME extractor remains in
``Ab_Lorenz_2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:
    import torch
    from geomloss import SamplesLoss

    _HAS_SINKHORN = True
except Exception:
    torch = None
    SamplesLoss = None
    _HAS_SINKHORN = False

try:
    from sklearn.neighbors import NearestNeighbors

    _HAS_KNN = True
except Exception:
    NearestNeighbors = None
    _HAS_KNN = False


REBUTTAL_SECTION_MODES = (
    "data_driven",
    "lorenz_u0_up_10",
    "lorenz_u2_up_40",
    "random_matched",
)


@dataclass(frozen=True)
class EventExtraction:
    hits: np.ndarray
    raw_detection_count: int
    gap_removed_count: int


def apply_minimum_gap(candidates: Iterable[int], minimum_gap: int) -> EventExtraction:
    """Greedily retain sorted event indices separated by at least minimum_gap."""

    minimum_gap = int(minimum_gap)
    if minimum_gap < 1:
        raise ValueError(f"minimum_gap must be >= 1, got {minimum_gap}")

    raw = np.asarray(list(candidates), dtype=np.int64)
    if raw.ndim != 1:
        raise ValueError("candidates must be one-dimensional")
    if len(raw) == 0:
        return EventExtraction(np.zeros((0,), dtype=np.int64), 0, 0)

    raw = np.unique(np.sort(raw))
    kept = []
    removed = 0
    last = -10**18
    for value in raw:
        value = int(value)
        if value - last >= minimum_gap:
            kept.append(value)
            last = value
        else:
            removed += 1
    return EventExtraction(
        hits=np.asarray(kept, dtype=np.int64),
        raw_detection_count=int(len(raw)),
        gap_removed_count=int(removed),
    )


def directed_crossing_events(
    observations: np.ndarray,
    column: int,
    threshold: float,
    minimum_gap: int,
    min_t: int = 0,
) -> EventExtraction:
    """Extract left-index upward crossings exactly as specified in the rebuttal.

    An event occurs at integer index ``t`` iff
    ``u[t] < threshold`` and ``u[t + 1] >= threshold``.
    """

    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"observations must have shape (T, d), got {values.shape}")
    column = int(column)
    if column < 0 or column >= values.shape[1]:
        raise ValueError(f"column={column} out of range for shape {values.shape}")
    if len(values) < 2:
        return EventExtraction(np.zeros((0,), dtype=np.int64), 0, 0)

    series = values[:, column]
    mask = (series[:-1] < float(threshold)) & (series[1:] >= float(threshold))
    raw = np.flatnonzero(mask).astype(np.int64)
    raw = raw[raw >= int(min_t)]
    return apply_minimum_gap(raw, minimum_gap=minimum_gap)


def sample_matched_random_events(
    *,
    T: int,
    min_t: int,
    n_events: int,
    minimum_gap: int,
    seed: int,
) -> EventExtraction:
    """Sample exactly n_events from legal left event indices with a hard gap.

    Legal indices are ``min_t, ..., T-2`` because all section events use the
    left index of a pair ``(t, t+1)``.  The order-statistic transformation gives
    an exact-size sample without rejection or a potentially lossy fallback.
    """

    T = int(T)
    min_t = int(min_t)
    n_events = int(n_events)
    minimum_gap = int(minimum_gap)
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")
    if min_t < 0 or min_t > T - 2:
        raise ValueError(f"min_t={min_t} leaves no legal event index for T={T}")
    if n_events < 0:
        raise ValueError(f"n_events must be nonnegative, got {n_events}")
    if minimum_gap < 1:
        raise ValueError(f"minimum_gap must be >= 1, got {minimum_gap}")
    if n_events == 0:
        return EventExtraction(np.zeros((0,), dtype=np.int64), 0, 0)

    n_legal = (T - 1) - min_t
    compressed_size = n_legal - (n_events - 1) * (minimum_gap - 1)
    if compressed_size < n_events:
        max_events = 1 + (n_legal - 1) // minimum_gap
        raise RuntimeError(
            "Cannot construct random_matched with the requested exact count: "
            f"n_events={n_events}, legal_indices={n_legal}, "
            f"minimum_gap={minimum_gap}, maximum_feasible={max_events}"
        )

    rng = np.random.default_rng(int(seed))
    compressed = np.sort(
        rng.choice(compressed_size, size=n_events, replace=False).astype(np.int64)
    )
    offsets = np.arange(n_events, dtype=np.int64) * (minimum_gap - 1)
    hits = min_t + compressed + offsets

    if len(hits) != n_events:
        raise AssertionError("random_matched event count changed unexpectedly")
    if len(hits) > 1 and np.any(np.diff(hits) < minimum_gap):
        raise AssertionError("random_matched minimum-gap invariant was violated")
    if hits[0] < min_t or hits[-1] > T - 2:
        raise AssertionError("random_matched produced an illegal event index")

    return EventExtraction(
        hits=hits,
        raw_detection_count=int(n_events),
        gap_removed_count=0,
    )


def build_return_graph_atoms(
    augmented: np.ndarray,
    hits: np.ndarray,
    *,
    state_dim: int,
    alpha_pair_state: float,
    alpha_pair_delay: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return whitened source states, target states, and graph atoms."""

    augmented = np.asarray(augmented, dtype=np.float64)
    hits = np.asarray(hits, dtype=np.int64)
    if augmented.ndim != 2:
        raise ValueError("augmented coordinates must be two-dimensional")
    if len(hits) < 2:
        empty = np.zeros((0, augmented.shape[1]), dtype=np.float64)
        return empty, empty.copy(), np.zeros((0, 2 * augmented.shape[1]))
    if np.any(hits < 0) or np.any(hits >= len(augmented)):
        raise ValueError("hit index outside augmented trajectory")

    recurrence_states = augmented[hits]
    mean = recurrence_states.mean(axis=0)
    std = recurrence_states.std(axis=0) + 1e-6
    whitened = (recurrence_states - mean[None, :]) / std[None, :]
    state_dim = int(state_dim)
    if state_dim < 1 or state_dim > whitened.shape[1]:
        raise ValueError(
            f"state_dim={state_dim} is invalid for augmented dim {whitened.shape[1]}"
        )
    whitened[:, :state_dim] *= float(alpha_pair_state)
    whitened[:, state_dim:] *= float(alpha_pair_delay)
    source = whitened[:-1]
    target = whitened[1:]
    graph = np.concatenate([source, target], axis=1)
    return source, target, graph


def split_half_sinkhorn(
    graph_atoms: np.ndarray,
    *,
    blur: float,
    device: str,
) -> float:
    """Sinkhorn divergence between chronological halves of graph atoms."""

    graph_atoms = np.asarray(graph_atoms, dtype=np.float32)
    midpoint = len(graph_atoms) // 2
    if midpoint < 2 or len(graph_atoms) - midpoint < 2:
        return float("nan")
    if not _HAS_SINKHORN:
        raise RuntimeError(
            "GeomLoss/PyTorch are required for split-half Sinkhorn diagnostics"
        )
    first = torch.tensor(graph_atoms[:midpoint], dtype=torch.float32, device=device)
    second = torch.tensor(graph_atoms[midpoint:], dtype=torch.float32, device=device)
    loss = SamplesLoss(
        loss="sinkhorn", p=2, blur=float(blur), backend="tensorized", debias=True
    )
    return float(loss(first, second).detach().cpu())


def heldout_knn_next_return(
    source: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    train_fraction: float,
) -> Dict[str, float]:
    """Chronological held-out kNN prediction of the next recurrence state."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) != len(target):
        raise ValueError("source and target must have equal lengths")
    if not _HAS_KNN:
        raise RuntimeError("scikit-learn is required for held-out kNN diagnostics")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")

    n_pairs = len(source)
    split = int(np.floor(n_pairs * float(train_fraction)))
    if split < 3 or n_pairs - split < 2:
        return {
            "knn_next_state_nrmse": float("nan"),
            "knn_k_effective": 0,
            "knn_train_pairs": int(split),
            "knn_test_pairs": int(max(0, n_pairs - split)),
        }

    k_effective = min(max(1, int(k)), split)
    model = NearestNeighbors(n_neighbors=k_effective)
    model.fit(source[:split])
    neighbor_ids = model.kneighbors(source[split:], return_distance=False)
    prediction = target[:split][neighbor_ids].mean(axis=1)
    truth = target[split:]
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    reference_scale = float(
        np.sqrt(np.mean((truth - target[:split].mean(axis=0, keepdims=True)) ** 2))
    )
    nrmse = rmse / (reference_scale + 1e-12)
    return {
        "knn_next_state_nrmse": float(nrmse),
        "knn_k_effective": int(k_effective),
        "knn_train_pairs": int(split),
        "knn_test_pairs": int(n_pairs - split),
    }


def compute_extractor_diagnostics(
    *,
    augmented: np.ndarray,
    hits: np.ndarray,
    minimum_gap: int,
    raw_detection_count: int,
    gap_removed_count: int,
    sinkhorn_blur: float,
    sinkhorn_device: str,
    knn_k: int,
    knn_train_fraction: float,
    state_dim: int,
    alpha_pair_state: float,
    alpha_pair_delay: float,
) -> Dict[str, Any]:
    """Compute the complete requested extractor diagnostic record."""

    hits = np.asarray(hits, dtype=np.int64)
    intervals = np.diff(hits).astype(np.float64)
    source, target, graph = build_return_graph_atoms(
        augmented,
        hits,
        state_dim=state_dim,
        alpha_pair_state=alpha_pair_state,
        alpha_pair_delay=alpha_pair_delay,
    )

    if len(intervals):
        mean_k = float(np.mean(intervals))
        std_k = float(np.std(intervals))
        median_k = float(np.median(intervals))
        min_k = float(np.min(intervals))
        max_k = float(np.max(intervals))
        cv_k = float(std_k / (mean_k + 1e-12))
    else:
        mean_k = std_k = median_k = min_k = max_k = cv_k = float("nan")

    result: Dict[str, Any] = {
        "N_hits": int(len(hits)),
        "N_return_pairs": int(len(intervals)),
        "mean_K": mean_k,
        "median_K": median_k,
        "std_K": std_k,
        "CV_K": cv_k,
        "min_K": min_k,
        "max_K": max_k,
        "minimum_gap": int(minimum_gap),
        "raw_detection_count": int(raw_detection_count),
        "gap_removed_count": int(gap_removed_count),
        "graph_state_dim": int(state_dim),
        "graph_alpha_pair_state": float(alpha_pair_state),
        "graph_alpha_pair_delay": float(alpha_pair_delay),
        "split_half_graph_sinkhorn": split_half_sinkhorn(
            graph, blur=sinkhorn_blur, device=sinkhorn_device
        ),
    }
    result.update(
        heldout_knn_next_return(
            source,
            target,
            k=knn_k,
            train_fraction=knn_train_fraction,
        )
    )
    return result
