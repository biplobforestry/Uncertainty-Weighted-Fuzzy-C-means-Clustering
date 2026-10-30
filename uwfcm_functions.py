# -*- coding: utf-8 -*-
r"""
@author: b.dey
email: biplobforestry@gmail.com

uwfcm_functions.py
uncertainty-weighted fuzzy c-means
Dependencies: numpy, pandas. Nothing else is imported.

Method
  - Per-cell uncertainty (precision only, no accuracy budget):
        u^2 = (EF_j |x|)^2 + (LOD / 3)^2
    EF_j = relative precision of ion j, estimated from its own record
    (robust scatter of consecutive differences in the top decile of the signal,
    divided by the signal level there). LOD/3 because LOD = 3 sigma of the
    background. One continuous formula, no below-LOD branch.
  - A cell with no signal, or no usable LOD (missing or <= 0), is unmeasured:
    u = NaN in the matrix and weight W = 0 in the fit. Nothing is imputed.
  - Objects = ions, features = time points. Each ion profile is divided by its
    L2 norm over measured cells; u is divided by the same norm (u' = u / ||x||).
    Time points with no measured ion are removed.
  - Weight per cell W = 1/u'^2, then divided by the ion's mean weight
    (balance_objects), so every ion brings the same total weight.
  - FCM with W in the distance only:
        d2[k, j] = sum_f W[j, f] (x[j, f] - v[k, f])^2
        v[k, f]  = sum_j mu[k, j]^m W[j, f] x[j, f] / sum_j mu[k, j]^m W[j, f]
"""

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import pandas as pd

EPS = 1e-12
ROBUST_MAD = 1.4826
LOD_SIGMAS = 3.0


# %% Uncertainty 
def estimate_relative_precision(x, top_quantile=0.90, minimum_pairs=20, fallback=0.02):
    """EF per row of x (ions x time): robust sd of consecutive differences in the
    top decile of the signal, over the median signal there. Upper bound."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape[0], float(fallback))
    for row in range(x.shape[0]):
        series = x[row][np.isfinite(x[row])]
        if series.size < 4 * minimum_pairs:
            continue
        diff = np.diff(series)
        level = 0.5 * (series[:-1] + series[1:])
        keep = np.isfinite(diff) & np.isfinite(level) & (level > 0)
        diff, level = diff[keep], level[keep]
        if diff.size < minimum_pairs:
            continue
        high = level >= np.quantile(level, top_quantile)
        if high.sum() < minimum_pairs:
            continue
        band = diff[high]
        sigma = ROBUST_MAD * np.median(np.abs(band - np.median(band))) / np.sqrt(2.0)
        centre = np.median(level[high])
        if centre > 0 and np.isfinite(sigma) and sigma > 0:
            out[row] = float(sigma / centre)
    return out


def per_ion_relative_precision(signal, time_column, fallback_error_fraction=0.02):
    """EF_j for each ion column of `signal` (time x ions), in column order."""
    ions = [name for name in signal.columns if name != time_column]
    values = signal[ions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
    relative = estimate_relative_precision(values, fallback=fallback_error_fraction)
    return np.where(np.isfinite(relative) & (relative > 0), relative, fallback_error_fraction)


def precision_uncertainty(x, lod, relative, k=LOD_SIGMAS):
    """u = sqrt((EF |x|)^2 + (LOD/k)^2) for x, lod shaped (ions, time); NaN where
    the LOD is missing or <= 0."""
    x = np.asarray(x, dtype=float)
    lod = np.asarray(lod, dtype=float)
    floor = np.where(np.isfinite(lod) & (lod > 0), lod / float(k), np.nan)
    with np.errstate(invalid="ignore"):
        u = np.sqrt((relative[:, None] * np.abs(x)) ** 2 + floor ** 2)
    return np.where(np.isfinite(u) & (u > 0), u, np.nan)


def uncertainty_from_limits(signal, detection, time_column):
    """Uncertainty matrix (time column + ions) from signal and LOD frames with
    identical columns. Unmeasured cells (no signal or unusable LOD) are NaN.
    Returns (u_df, report); report["error_fraction_per_ion"] maps ion -> EF_j."""
    if list(detection.columns) != list(signal.columns):
        raise ValueError("Signal and detection-limit frames must carry the same columns")
    relative = per_ion_relative_precision(signal, time_column)
    ions = [name for name in signal.columns if name != time_column]
    x = signal[ions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mdl = detection[ions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    unusable = (~np.isfinite(mdl)) | (mdl <= 0)
    missing_signal = ~np.isfinite(x)

    values = precision_uncertainty(x.T, mdl.T, relative, k=LOD_SIGMAS).T
    values = np.where(unusable | missing_signal, np.nan, values)

    out = pd.DataFrame(values, columns=ions)
    out.insert(0, time_column, signal[time_column].reset_index(drop=True))
    report = {
        "error_fraction_median": float(np.nanmedian(relative)),
        "error_fraction_per_ion": {str(ion): float(v) for ion, v in zip(ions, relative)},
        "lod_divisor_k": LOD_SIGMAS,
        "unmeasured_cells": int((unusable | missing_signal).sum()),
    }
    return out, report


# %% 
class Orientation(str, Enum):
    """Which axis is clustered. Only IONS (objects = ions, features = time) is used."""
    IONS = "ions"


class Dataset:
    """Signal and uncertainty, both time x ion, with the time axis."""

    def __init__(self, signal, uncertainty, *, time, name="dataset"):
        if list(signal.columns) != list(uncertainty.columns) or signal.shape != uncertainty.shape:
            raise ValueError("signal and uncertainty must have identical columns and shape")
        self.name = name
        self.ions = [str(col) for col in signal.columns]
        self.signal = signal.to_numpy(dtype=float)
        self.uncertainty = uncertainty.to_numpy(dtype=float)
        self.time = pd.Series(pd.to_datetime(time, errors="coerce")).reset_index(drop=True)


@dataclass
class WorkingView:
    """The fitted matrices: x' and u' (ions x kept time points), W, and labels."""
    x: np.ndarray
    u: np.ndarray
    weights: np.ndarray
    object_labels: list
    feature_labels: list
    weight_diagnostics: object = None    # kept for call compatibility; not used


def l2_transform(x, u, mask):
    """Divide each row of x and u by the L2 norm of its measured cells."""
    norm = np.sqrt(np.sum(np.where(mask, x, 0.0) ** 2, axis=1))
    norm = np.where(norm > EPS, norm, 1.0)
    divisor = norm[:, None]
    return x / divisor, u / divisor


def compute_weights(u, mask, balance_objects=False):
    """W = 1/u^2 on measured cells, 0 elsewhere; optionally divided by each
    object's mean weight so every object carries the same total weight."""
    safe = np.where(mask, u, 1.0)
    weights = np.where(mask, 1.0 / (safe * safe), 0.0)
    if balance_objects:
        scale = weights.mean(axis=1, keepdims=True)
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError("an object has no measured cell")
        weights = weights / scale
    return weights


def working_view(dataset, orientation, transform="l2", *, balance_objects=False,
                 missing_as_unmeasured=True):
    """Ions x time view of `dataset` for the fit.

    measured = finite x and finite u > 0. Time points with no measured ion are
    dropped. Unmeasured cells get the placeholders x = 0, u = 1, which carry
    W = 0 and so cannot affect distances or centres. Then L2 transform with
    u' = u / ||x|| and W = 1/u'^2 (balanced per ion).
    Returns (view, None); the second item mirrors the library's repair record,
    which is a no-op here because every unusable u cell is an unmeasured cell.
    """
    if orientation is not Orientation.IONS or transform != "l2" or not missing_as_unmeasured:
        raise ValueError("only Orientation.IONS, transform 'l2', missing_as_unmeasured=True")
    x = dataset.signal.T.copy()
    u = dataset.uncertainty.T.copy()
    mask = np.isfinite(x) & np.isfinite(u) & (u > 0)

    keep = mask.any(axis=0)
    labels = [stamp.isoformat() if pd.notna(stamp) else str(pos)
              for pos, stamp in enumerate(dataset.time)]
    if not keep.all():      # index only when needed: it changes memory order, hence sum order
        x, u, mask = x[:, keep], u[:, keep], mask[:, keep]
        labels = [lab for lab, k in zip(labels, keep) if k]

    x = np.where(mask, x, 0.0)
    u = np.where(mask, u, 1.0)
    x, u = l2_transform(x, u, mask)
    weights = compute_weights(u, mask, balance_objects=balance_objects)
    view = WorkingView(x=x, u=u, weights=weights, object_labels=list(dataset.ions),
                       feature_labels=labels)
    return view, None


# %% Uncertainty-weighted FCM
@dataclass(frozen=True)
class FCMParameters:
    """Fit settings. balance_objects is recorded; the weights already carry it."""
    c: int
    m: float
    balance_objects: bool = False
    max_iter: int = 300
    tolerance: float = 1e-6
    seed: int = 24


@dataclass
class FCMResult:
    """One fitted model: centres (c x features), memberships (c x objects)."""
    centers: np.ndarray
    memberships: np.ndarray
    objective_history: list
    converged: bool
    n_iter: int
    parameters: FCMParameters
    object_labels: list
    feature_labels: list

    @property
    def objective(self):
        return self.objective_history[-1] if self.objective_history else float("nan")

    def is_valid_partition(self):
        return 1 < np.unique(np.argmax(self.memberships, axis=0)).size <= self.parameters.c


@dataclass
class RestartReport:
    """Best restart plus the objectives of all valid restarts."""
    best: FCMResult
    objectives: list
    n_distinct_optima: int
    n_converged: int
    n_attempted: int
    n_valid: int


def update_memberships(distances, m):
    """mu[k, j] = 1 / sum_l (d[k, j] / d[l, j])^(1/(m-1)), ratio form."""
    exponent = 1.0 / (m - 1.0)
    ratio = (distances[:, None, :] / distances[None, :, :]) ** exponent
    memberships = 1.0 / np.maximum(ratio.sum(axis=1), EPS)
    return memberships / np.maximum(memberships.sum(axis=0, keepdims=True), EPS)


def fit(x, weights, parameters, object_labels, feature_labels):
    """One FCM run from a random membership start drawn with parameters.seed.

    W x and W are stored side by side so the centre update and the expanded
    distance  d2 = sum W x^2 - 2 (W x) v + W v^2  are one matrix product each.
    Stops when max |d mu| < tolerance; centres are recomputed from the final mu.
    """
    x = np.ascontiguousarray(x, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    n_objects, n_features = x.shape
    c, m = parameters.c, parameters.m

    combined = np.empty((n_objects, 2 * n_features), dtype=float)
    np.multiply(weights, x, out=combined[:, :n_features])
    combined[:, n_features:] = weights
    wxx = np.einsum("jf,jf->j", combined[:, :n_features], x)   # sum_f W x^2

    def centers_of(mu):
        both = (mu ** m) @ combined
        return both[:, :n_features] / np.maximum(both[:, n_features:], EPS)

    def distances_of(v):
        right = np.empty((2 * n_features, v.shape[0]), dtype=float)
        np.multiply(v.T, -2.0, out=right[:n_features])
        np.multiply(v.T, v.T, out=right[n_features:])
        d2 = wxx[:, None] + combined @ right
        return np.maximum(d2.T, EPS)

    rng = np.random.default_rng(parameters.seed)
    memberships = rng.random((c, n_objects))
    memberships /= memberships.sum(axis=0, keepdims=True)

    history, converged, iteration = [], False, 0
    for iteration in range(1, parameters.max_iter + 1):
        previous = memberships
        centers = centers_of(memberships)
        distances = distances_of(centers)
        memberships = update_memberships(distances, m)
        history.append(float(np.sum((memberships ** m) * distances)))
        if np.max(np.abs(memberships - previous)) < parameters.tolerance:
            converged = True
            break
    centers = centers_of(memberships)
    return FCMResult(centers=centers, memberships=memberships, objective_history=history,
                     converged=converged, n_iter=iteration, parameters=parameters,
                     object_labels=list(object_labels), feature_labels=list(feature_labels))


def fit_best(x, u, parameters, *, object_labels, feature_labels, n_init=20,
             weights=None, weight_diagnostics=None):
    """n_init restarts with seeds seed, seed+1, ...; keeps the lowest objective.
    `u` and `weight_diagnostics` are accepted for call compatibility; the fit
    uses `weights` (W from working_view). Runs with < 2 occupied clusters are
    skipped. Distinct optima = unique objectives rounded to 9 decimals."""
    best, objectives, n_converged, n_valid = None, [], 0, 0
    for offset in range(n_init):
        result = fit(x, weights, replace(parameters, seed=parameters.seed + offset),
                     object_labels, feature_labels)
        if not result.is_valid_partition():
            continue
        n_valid += 1
        n_converged += int(result.converged)
        objectives.append(result.objective)
        if best is None or result.objective < best.objective:
            best = result
    if best is None:
        raise RuntimeError("no restart produced a valid partition")
    return RestartReport(best=best, objectives=objectives,
                         n_distinct_optima=len(np.unique(np.round(objectives, 9))),
                         n_converged=n_converged, n_attempted=n_init, n_valid=n_valid)


# %% Indices 
def fpc(memberships):
    """Fuzzy partition coefficient: mean over objects of sum_k mu^2."""
    u = np.asarray(memberships, dtype=float)
    return float(np.mean(np.sum(u * u, axis=0)))


def partition_entropy(memberships):
    """Partition entropy: -sum mu log mu / n_objects."""
    u = np.asarray(memberships, dtype=float)
    return float(-np.sum(u * np.log(np.maximum(u, EPS))) / u.shape[1])


def centroid_separation(centers):
    """Largest |Pearson r| between two centroid series (flat pairs: 1 if both
    flat, 0 if one flat; pairs with < 3 finite points skipped)."""
    centers = np.asarray(centers, dtype=float)
    worst = float("nan")
    for first in range(centers.shape[0]):
        for second in range(first + 1, centers.shape[0]):
            a, b = centers[first], centers[second]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < 3:
                continue
            a, b = a[both], b[both]
            flat_a, flat_b = np.ptp(a) == 0.0, np.ptp(b) == 0.0
            if flat_a and flat_b:
                pair = 1.0
            elif flat_a or flat_b:
                pair = 0.0
            else:
                pair = abs(float(np.corrcoef(a, b)[0, 1]))
                if not np.isfinite(pair):
                    continue
            worst = pair if not np.isfinite(worst) else max(worst, pair)
    return worst


def pairwise_weighted_distances(x, u):
    """d[j, k] = sqrt(sum_f (x_jf - x_kf)^2 / (u_jf^2 + u_kf^2)); u = inf gives
    a cell zero weight."""
    x = np.asarray(x, dtype=float)
    variance = np.asarray(u, dtype=float) ** 2
    n_objects = x.shape[0]
    distances = np.zeros((n_objects, n_objects), dtype=float)
    for j in range(n_objects):
        residual = x[j] - x[j + 1:]
        denominator = variance[j] + variance[j + 1:]
        row = np.einsum("kf,kf->k", residual * residual, 1.0 / np.maximum(denominator, EPS))
        distances[j, j + 1:] = row
        distances[j + 1:, j] = row
    return np.sqrt(distances)


def silhouette_samples(distance_matrix, labels):
    """Per-object silhouette on a precomputed distance matrix (singleton = 0)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.full(labels.size, np.nan, dtype=float)
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= labels.size:
        return scores
    for index in range(labels.size):
        own_mask = labels == labels[index]
        own_mask[index] = False
        if not own_mask.any():
            scores[index] = 0.0
            continue
        cohesion = float(distance_matrix[index, own_mask].mean())
        separation = min(float(distance_matrix[index, labels == other].mean())
                         for other in unique if other != labels[index])
        denominator = max(cohesion, separation)
        scores[index] = 0.0 if denominator <= 0 else (separation - cohesion) / denominator
    return scores


# %% Cluster map helpers 
@dataclass
class EllipseGeometry:
    """Covariance ellipse in map units."""
    centre: tuple
    width: float
    height: float
    angle: float
    n_sd: float
    count: int


def covariance_ellipse(points, n_sd=2.0, min_points=3):
    """n_sd covariance ellipse of a 2-D point set; None for < min_points."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < min_points:
        return None
    centre = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return None
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    values = np.maximum(values, 0.0)
    width, height = 2.0 * n_sd * np.sqrt(values)
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    return EllipseGeometry(centre=(float(centre[0]), float(centre[1])), width=float(width),
                           height=float(height), angle=float(angle), n_sd=float(n_sd),
                           count=int(points.shape[0]))


@dataclass
class ClusterMap:
    """Map coordinates, labels, ellipses and silhouettes, with summary counts."""
    coordinates: np.ndarray
    explained: np.ndarray
    labels: np.ndarray
    ellipses: dict = field(default_factory=dict)
    silhouette: np.ndarray = field(default_factory=lambda: np.empty(0))
    distance_space: str = "euclidean"
    covariate_correlation: float = None
    n_sd: float = 2.0

    @property
    def mean_silhouette(self):
        return float(np.nanmean(self.silhouette)) if self.silhouette.size else float("nan")

    @property
    def misplaced(self):
        return int(np.sum(self.silhouette < 0)) if self.silhouette.size else 0

    @property
    def counts(self):
        return {int(k): int(np.sum(self.labels == k)) for k in np.unique(self.labels)}


# %% Diel composites
@dataclass(frozen=True)
class TimeWindow:
    """Named period [start, end) with excluded sub-periods [a, b) and an
    optional hour-of-day filter [h0, h1)."""
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    exclude: tuple = ()
    hours: tuple = None

    def mask(self, time):
        stamps = pd.to_datetime(pd.Series(time).reset_index(drop=True))
        keep = (stamps >= pd.Timestamp(self.start)) & (stamps < pd.Timestamp(self.end))
        for low, high in self.exclude:
            keep &= ~((stamps >= pd.Timestamp(low)) & (stamps < pd.Timestamp(high)))
        if self.hours is not None:
            minutes = stamps.dt.hour * 60 + stamps.dt.minute
            keep &= (minutes >= self.hours[0] * 60) & (minutes < self.hours[1] * 60)
        return keep.to_numpy(dtype=bool)


@dataclass
class WindowSet:
    """Ordered windows plus overlap warnings."""
    windows: list
    warnings: list = field(default_factory=list)

    @classmethod
    def build(cls, windows):
        warnings = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                left, right = windows[i], windows[j]
                if (pd.Timestamp(left.start) < pd.Timestamp(right.end)
                        and pd.Timestamp(right.start) < pd.Timestamp(left.end)):
                    warnings.append(f"windows {left.name!r} and {right.name!r} overlap in time; "
                                    "samples in the overlap contribute to both composites")
        return cls(windows=list(windows), warnings=warnings)


@dataclass
class DielResult:
    """values: long table (window, cluster, bin, centre, lower, upper, n, in_scope);
    coverage: rows and days per window; warnings."""
    values: pd.DataFrame
    coverage: pd.DataFrame
    warnings: list


def diel_composite(series, window_set, *, bin_minutes=60, statistic="median",
                   min_samples_per_bin=3, spread="iqr"):
    """Median and 25-75th percentile of each series column per clock bin
    (floored) within each window; bins with < min_samples_per_bin are NaN."""
    if statistic != "median" or spread != "iqr" or 1440 % bin_minutes != 0:
        raise ValueError("only statistic='median', spread='iqr', bin_minutes dividing 1440")
    stamps = pd.to_datetime(series["time"]).reset_index(drop=True)
    cluster_columns = [col for col in series.columns if col != "time"]
    minute_of_day = (stamps.dt.hour * 60 + stamps.dt.minute).to_numpy()
    bin_index = (minute_of_day // bin_minutes).astype(int)
    all_bins = np.arange(1440 // bin_minutes)

    records, warnings = [], list(window_set.warnings)
    for window in window_set.windows:
        mask = window.mask(stamps)
        if not mask.any():
            warnings.append(f"window {window.name!r} selects no rows and is empty")
            continue
        window_bins = bin_index[mask]
        n_days = int(stamps[mask].dt.normalize().nunique())
        if n_days < 2:
            warnings.append(f"window {window.name!r} spans {n_days} day(s); a diel composite over "
                            "fewer than 2 days is a single day's time series, not a composite")
        if window.hours is None:
            in_scope = set(all_bins.tolist())
        else:
            in_scope = {int(b) for b in all_bins
                        if window.hours[0] * 60 <= b * bin_minutes < window.hours[1] * 60}

        for column in cluster_columns:
            values = pd.to_numeric(series[column], errors="coerce").to_numpy()[mask]
            for bin_id in all_bins:
                sample = values[(window_bins == bin_id) & np.isfinite(values)]
                count = int(sample.size)
                if count < min_samples_per_bin:
                    centre = lower = upper = np.nan
                else:
                    centre = float(np.median(sample))
                    lower = float(np.percentile(sample, 25))
                    upper = float(np.percentile(sample, 75))
                start_minute = int(bin_id * bin_minutes)
                records.append({"window": window.name, "cluster": column,
                                "bin_minutes_of_day": start_minute,
                                "bin_label": f"{start_minute // 60:02d}:{start_minute % 60:02d}",
                                "centre": centre, "lower": lower, "upper": upper,
                                "n": count, "in_scope": int(bin_id) in in_scope})

    values_frame = pd.DataFrame(records)
    if not values_frame.empty:
        in_scope_cells = values_frame[values_frame["in_scope"]]
        thin = in_scope_cells[in_scope_cells["n"] < min_samples_per_bin]
        if len(thin):
            affected = sorted(thin["window"].unique().tolist())
            warnings.append(
                f"{len(thin)} of {len(in_scope_cells)} in-scope window-cluster-bin cells "
                f"had fewer than {min_samples_per_bin} samples and are blank "
                f"(windows: {', '.join(affected)}). Bins outside a window's hour filter "
                "are excluded from this count.")

    coverage = pd.DataFrame([{"window": w.name, "n_rows": int(w.mask(stamps).sum()),
                              "n_days": int(stamps[w.mask(stamps)].dt.normalize().nunique())}
                             for w in window_set.windows])
    return DielResult(values=values_frame, coverage=coverage, warnings=warnings)
