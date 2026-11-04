"""Opt-in scheduling utilities for the PRIME mechanism-isolation experiment.

This module contains no model or optimizer code.  It only constructs the
transition/source schedules requested by the rebuttal experiment so that the
existing PRIME trainer remains the single optimization implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from rebuttal_section_utils import sample_matched_random_events


MECHANISM_SCHEDULE_MODES = (
    "recurrence_correct",
    "recurrence_fixed_median",
    "recurrence_shuffled",
    "uniform_fixed_median",
    "uniform_empirical_lag",
    "random_event_matched",
)

MECHANISM_LOSS_MODES = (
    "full_graph_plus_marginals",
    "graph_only",
    "marginals_only",
    "dim_only",
)


@dataclass(frozen=True)
class TransitionSchedule:
    """One Stage-2 transition block.

    ``source_indices`` and ``target_indices`` select data atoms.  The model is
    rolled out for ``model_horizons`` observation steps.  For the recurrence
    clock controls the target atoms remain the next empirical recurrence atoms,
    while only the model-side clock is changed, matching the existing clock
    ablation semantics.
    """

    mode: str
    source_indices: np.ndarray
    target_indices: np.ndarray
    data_horizons: np.ndarray
    model_horizons: np.ndarray
    reference_pair_indices: np.ndarray
    uses_recurrence_sources: bool
    state_paired_k: bool

    @property
    def pair_count(self) -> int:
        return int(len(self.source_indices))

    @property
    def total_unrolled_observation_steps(self) -> int:
        return int(np.sum(self.model_horizons, dtype=np.int64))

    def validate(self, *, T: int, min_t: int) -> None:
        arrays = (
            self.source_indices,
            self.target_indices,
            self.data_horizons,
            self.model_horizons,
            self.reference_pair_indices,
        )
        if any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("All transition-schedule arrays must be one-dimensional")
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Transition-schedule lengths disagree: {sorted(lengths)}")
        if self.pair_count < 1:
            raise ValueError("A transition schedule must contain at least one pair")
        if np.any(self.source_indices < int(min_t)):
            raise ValueError("A source index precedes the legal delay-history boundary")
        if np.any(self.target_indices >= int(T)):
            raise ValueError("A target index lies beyond the training trajectory")
        if np.any(self.target_indices <= self.source_indices):
            raise ValueError("Every data target must occur after its source")
        if np.any(self.data_horizons != self.target_indices - self.source_indices):
            raise ValueError("data_horizons must equal target_indices - source_indices")
        if np.any(self.model_horizons < 1):
            raise ValueError("All model rollout horizons must be positive")


def rounded_median_k(empirical_k: np.ndarray) -> int:
    """Return the pre-specified fixed K: round(median(training recurrence K))."""

    values = np.asarray(empirical_k, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("empirical_k must be a non-empty one-dimensional array")
    if np.any(values < 1):
        raise ValueError("empirical recurrence intervals must be positive")
    return max(1, int(round(float(np.median(values)))))


def loss_term_weights(
    mode: str,
    *,
    marginal_source_weight: float,
    marginal_target_weight: float,
    defect_weight: float,
) -> Tuple[float, float, float, float]:
    """Map an opt-in loss mode to graph/source/target/defect weights."""

    if mode not in MECHANISM_LOSS_MODES:
        raise ValueError(
            f"Unknown mechanism loss mode {mode!r}; expected {MECHANISM_LOSS_MODES}"
        )
    if mode == "full_graph_plus_marginals":
        return (
            1.0,
            float(marginal_source_weight),
            float(marginal_target_weight),
            float(defect_weight),
        )
    if mode == "graph_only":
        return 1.0, 0.0, 0.0, 0.0
    if mode == "marginals_only":
        return (
            0.0,
            float(marginal_source_weight),
            float(marginal_target_weight),
            float(defect_weight),
        )
    return 0.0, 0.0, 0.0, 0.0


def schedule_properties(mode: str) -> Dict[str, object]:
    """Reader-facing mechanism labels used by the final table."""

    if mode not in MECHANISM_SCHEDULE_MODES:
        raise ValueError(
            f"Unknown mechanism schedule mode {mode!r}; "
            f"expected {MECHANISM_SCHEDULE_MODES}"
        )
    return {
        "recurrence_sources": mode
        in {
            "recurrence_correct",
            "recurrence_fixed_median",
            "recurrence_shuffled",
        },
        "state_paired_K": mode
        in {
            "recurrence_correct",
            "random_event_matched",
        },
        "source_description": {
            "recurrence_correct": "Algorithm-2 recurrence events",
            "recurrence_fixed_median": "Algorithm-2 recurrence events",
            "recurrence_shuffled": "Algorithm-2 recurrence events",
            "uniform_fixed_median": "uniform legal observation times",
            "uniform_empirical_lag": "uniform legal observation times",
            "random_event_matched": "minimum-gap matched random events",
        }[mode],
    }


class MechanismScheduleSampler:
    """Deterministic per-seed sampler for opt-in mechanism schedules."""

    def __init__(
        self,
        *,
        recurrence_hits: np.ndarray,
        T: int,
        min_t: int,
        minimum_gap: int,
        seed: int,
    ) -> None:
        hits = np.sort(np.asarray(recurrence_hits, dtype=np.int64))
        if hits.ndim != 1 or len(hits) < 4:
            raise ValueError("At least four recurrence hits are required")
        if np.any(hits < int(min_t)) or np.any(hits >= int(T)):
            raise ValueError("recurrence_hits contain illegal indices")
        empirical_k = np.diff(hits)
        if np.any(empirical_k < 1):
            raise ValueError("recurrence_hits must be strictly increasing")

        self.recurrence_hits = hits
        self.empirical_k = empirical_k.astype(np.int64)
        self.T = int(T)
        self.min_t = int(min_t)
        self.minimum_gap = int(minimum_gap)
        self.fixed_median_k = rounded_median_k(self.empirical_k)
        self.uniform_max_horizon = int(np.max(self.empirical_k))
        self.uniform_source_pool = np.arange(
            self.min_t,
            self.T - self.uniform_max_horizon,
            dtype=np.int64,
        )
        if len(self.uniform_source_pool) < len(self.empirical_k):
            raise ValueError(
                "The common uniform-source pool is too small to draw one "
                "transition block without replacement"
            )

        # Source draws and horizon permutations deliberately use separate
        # streams.  Consequently, uniform_fixed_median and
        # uniform_empirical_lag receive the same source plans (given the same
        # sampled reference blocks), while empirical K is assigned
        # independently of those sources.
        seed_sequence = np.random.SeedSequence(int(seed))
        source_seed, horizon_seed = seed_sequence.spawn(2)
        self.source_rng = np.random.default_rng(source_seed)
        self.horizon_rng = np.random.default_rng(horizon_seed)

        random_events = sample_matched_random_events(
            T=self.T,
            min_t=self.min_t,
            n_events=len(hits),
            minimum_gap=self.minimum_gap,
            seed=int(seed) + 1,
        )
        self.random_hits = np.asarray(random_events.hits, dtype=np.int64)
        if len(self.random_hits) != len(self.recurrence_hits):
            raise RuntimeError(
                "Matched random events did not preserve the recurrence event count"
            )

    def _reference_block(self, pair_count: int, reference_k0: int) -> Tuple[np.ndarray, ...]:
        n = int(pair_count)
        k0 = int(reference_k0)
        if n < 1:
            raise ValueError("pair_count must be positive")
        if k0 < 0 or k0 + n > len(self.empirical_k):
            raise ValueError(
                f"reference block [{k0}, {k0 + n}) exceeds "
                f"{len(self.empirical_k)} empirical intervals"
            )
        pair_ids = np.arange(k0, k0 + n, dtype=np.int64)
        source = self.recurrence_hits[pair_ids]
        target = self.recurrence_hits[pair_ids + 1]
        horizons = self.empirical_k[pair_ids]
        return pair_ids, source, target, horizons

    def _sample_unique_legal_sources(self, horizons: np.ndarray) -> np.ndarray:
        """Draw sources independently of K from one common legal-time pool."""

        values = np.asarray(horizons, dtype=np.int64)
        if np.any(values > self.uniform_max_horizon):
            raise ValueError(
                "A requested horizon exceeds the predeclared common-pool maximum"
            )
        if len(values) > len(self.uniform_source_pool):
            raise ValueError("Too few common legal source times for this block")
        return np.asarray(
            self.source_rng.choice(
                self.uniform_source_pool,
                size=len(values),
                replace=False,
            ),
            dtype=np.int64,
        )

    def sample(
        self,
        mode: str,
        *,
        pair_count: int,
        reference_k0: int,
    ) -> TransitionSchedule:
        if mode not in MECHANISM_SCHEDULE_MODES:
            raise ValueError(
                f"Unknown mechanism schedule mode {mode!r}; "
                f"expected {MECHANISM_SCHEDULE_MODES}"
            )

        pair_ids, recurrence_source, recurrence_target, reference_k = (
            self._reference_block(pair_count, reference_k0)
        )

        if mode == "recurrence_correct":
            source = recurrence_source
            target = recurrence_target
            model_k = reference_k.copy()
            paired = True
            uses_recurrence = True
        elif mode == "recurrence_fixed_median":
            source = recurrence_source
            target = recurrence_target
            model_k = np.full(
                len(reference_k), self.fixed_median_k, dtype=np.int64
            )
            paired = False
            uses_recurrence = True
        elif mode == "recurrence_shuffled":
            source = recurrence_source
            target = recurrence_target
            permutation = self.horizon_rng.permutation(len(reference_k))
            if len(permutation) > 1 and np.array_equal(
                permutation, np.arange(len(permutation))
            ):
                permutation = np.roll(permutation, 1)
            model_k = reference_k[permutation]
            paired = False
            uses_recurrence = True
        elif mode == "uniform_fixed_median":
            model_k = np.full(
                len(reference_k), self.fixed_median_k, dtype=np.int64
            )
            source = self._sample_unique_legal_sources(model_k)
            target = source + model_k
            paired = False
            uses_recurrence = False
        elif mode == "uniform_empirical_lag":
            permutation = self.horizon_rng.permutation(len(reference_k))
            model_k = reference_k[permutation]
            source = self._sample_unique_legal_sources(model_k)
            target = source + model_k
            paired = False
            uses_recurrence = False
        else:
            random_source = self.random_hits[pair_ids]
            random_target = self.random_hits[pair_ids + 1]
            source = random_source
            target = random_target
            model_k = random_target - random_source
            paired = True
            uses_recurrence = False

        data_k = target - source
        schedule = TransitionSchedule(
            mode=mode,
            source_indices=np.asarray(source, dtype=np.int64),
            target_indices=np.asarray(target, dtype=np.int64),
            data_horizons=np.asarray(data_k, dtype=np.int64),
            model_horizons=np.asarray(model_k, dtype=np.int64),
            reference_pair_indices=np.asarray(pair_ids, dtype=np.int64),
            uses_recurrence_sources=bool(uses_recurrence),
            state_paired_k=bool(paired),
        )
        schedule.validate(T=self.T, min_t=self.min_t)
        return schedule
