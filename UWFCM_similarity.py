# -*- coding: utf-8 -*-
r"""
@author: b.dey
email: biplobforestry@gmail.com

UWFCM_similarity.py

Scores each c = 5 UWFCM cluster against the three stress windows of its
species, with two independent pieces of evidence:
  1. spectral similarity  - cosine and 1 - Bray-Curtis between the cluster
                            spectrum and each window's reference spectrum;
  2. temporal enrichment  - z-score of the cluster centroid inside each window.


Input  : Data\{species}_signal_LOD.csv
         Output\tables\{species}_UWFCM_c5.npz                (script 1)
         Output\tables\UWFCM_c5_l2_m1.6_seed24.xlsx          (script 1)
Output : Output\tables\UWFCM_c5_similarity.xlsx
         Output\figures\UWFCM_c5_similarity_plate_{species}.png/.svg
         Output\figures\UWFCM_c5_bray_vs_cosine.png/.svg
"""

# %% Imports
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.spatial.distance import braycurtis


try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    code_dir = os.getcwd()

# %% Paths and settings

base_dir = r"C:\My_PC\2025_Campaign\Manuscript\npj Clim Atmos Sci"
data_dir = os.path.join(base_dir, "Data")
table_dir = os.path.join(code_dir, "Output", "tables")
fig_dir = os.path.join(code_dir, "Output", "figures")
os.makedirs(fig_dir, exist_ok=True)

fit_excel = os.path.join(table_dir, "UWFCM_c5_l2_m1.6_seed24.xlsx")
out_excel = os.path.join(table_dir, "UWFCM_c5_similarity.xlsx")

species_list = ["Chestnut", "Linden"]
c = 5
clusters = [f"C{k + 1}" for k in range(c)]
stress_order = ["Ac", "Ht", "HtO3"]


cos_high, cos_moderate = 0.15, 0.07          
z_high, z_moderate = 1.0, 0.5               

cluster_colors = ["#009E73", "#445462", "#F2CF1D", "#CC79A7", "#E69F00"]
stress_markers = {"Ac": "o", "Ht": "s", "HtO3": "^"}

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.6})

# %% Stress windows (copied from the old scripts)
ts = lambda s: pd.to_datetime(s, format="%m/%d/%Y %H:%M")
stress_events = {
 "CN_Ac": {"species":"Chestnut","include":(ts("07/04/2025 12:01"),ts("07/11/2025 17:10")),"exclude":[(ts("07/08/2025 06:23"),ts("07/08/2025 06:32")),(ts("07/09/2025 04:55"),ts("07/09/2025 05:00"))]},
 "CN_Ht": {"species":"Chestnut","include":(ts("07/11/2025 17:11"),ts("07/18/2025 04:30")),"exclude":[(ts("07/16/2025 04:55"),ts("07/16/2025 15:00"))]},
 "CN_HtO3": {"species":"Chestnut","include":(ts("07/21/2025 09:01"),ts("07/25/2025 05:00")),"exclude":[(ts("07/23/2025 06:53"),ts("07/23/2025 07:00"))]},
 "LD_Ac": {"species":"Linden","include":(ts("07/25/2025 18:00"),ts("07/30/2025 11:30")),"exclude":[(ts("07/28/2025 06:54"),ts("07/28/2025 07:08"))]},
 "LD_Ht": {"species":"Linden","include":(ts("07/30/2025 11:31"),ts("08/05/2025 06:00")),"exclude":[(ts("08/02/2025 06:45"),ts("08/02/2025 15:00"))]},
 "LD_HtO3": {"species":"Linden","include":(ts("08/18/2025 05:31"),ts("08/21/2025 05:00")),"exclude":[(ts("08/20/2025 04:56"),ts("08/20/2025 05:00"))]},
}


def window_mask(time, species, stress):
    """Boolean mask of one stress window; include and exclude inclusive at both ends."""
    ev = stress_events[f"{'CN' if species == 'Chestnut' else 'LD'}_{stress}"]
    time = pd.Series(pd.to_datetime(time))
    mask = time.between(ev["include"][0], ev["include"][1], inclusive="both")
    for ex0, ex1 in ev["exclude"]:
        mask &= ~time.between(ex0, ex1, inclusive="both")
    return mask.to_numpy()


def l2(v):
    norm = np.sqrt(np.sum(v ** 2))
    return v / norm if norm > 0 else np.full_like(v, np.nan)


# %% Centroid series from the workbook
centroid_df = pd.read_excel(fit_excel, sheet_name="centroid_timeseries")
centroid_df["time"] = pd.to_datetime(centroid_df["time"])

# %% Score each species
score_rows, event_rows, spectra_tables = [], [], []

for species in species_list:
    fit = np.load(os.path.join(table_dir, f"{species}_UWFCM_c{c}.npz"))
    U = fit["memberships"]                              # (c, n_ions)
    ions = [str(i) for i in fit["ions"]]

    frame = pd.read_csv(os.path.join(data_dir, f"{species}_signal_LOD.csv"))
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    assert all(i in frame.columns for i in ions), "ion missing in raw flux"

    raw = frame[ions].to_numpy(dtype=float)

    with np.errstate(invalid="ignore"):
        logx = np.log1p(np.where(raw < 0, np.nan, raw))

    # global median log1p profile over all measured cells (base of cluster spectra)
    base_vec = np.nanmedian(logx, axis=0)

    # ---- reference spectrum per window: mean log1p over the window's measured cells ----
    refs = {}
    n_window = {}
    for stress in stress_order:
        mask = window_mask(frame["time"], species, stress)
        # an ion unmeasured through a whole window gives NaN (not 0) and leaves the comparison
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            refs[stress] = np.nanmean(logx[mask], axis=0)
        n_window[stress] = int(mask.sum())

    # ---- cluster spectrum: membership x base profile (no noise weight) ----
    cluster_spec = {cl: U[k] * base_vec for k, cl in enumerate(clusters)}

    # per-ion spectra kept as a table (all ions, un-normalised)
    spec_tab = pd.DataFrame({"species": species, "amu": [float(i) for i in ions],
                             "global_median_log1p": base_vec})
    for stress in stress_order:
        spec_tab[f"ref_{stress}_mean_log1p"] = refs[stress]
    for k, cl in enumerate(clusters):
        spec_tab[f"membership_{cl}"] = U[k]
    spectra_tables.append(spec_tab)

    # ---- spectral similarity over the ions finite in both vectors ----
    for cl in clusters:
        for stress in stress_order:
            a, b = cluster_spec[cl], refs[stress]
            ok = np.isfinite(a) & np.isfinite(b)
            aa, bb = l2(a[ok]), l2(b[ok])
            cosine = float(np.dot(aa, bb)) if ok.sum() >= 2 else np.nan
            # Bray-Curtis: negatives -> 0 (old convention, L162-163)
            bc_a, bc_b = np.clip(aa, 0, None), np.clip(bb, 0, None)
            bray = float(1 - braycurtis(bc_a, bc_b)) if ok.sum() >= 2 else np.nan
            score_rows.append({"species": species, "c": c, "cluster": cl, "stress": stress,
                               "cosine_similarity": cosine,
                               "braycurtis_similarity": bray,
                               "n_ions_compared": int(ok.sum()),
                               "n_ions_total": len(ions),
                               "n_window_times": n_window[stress]})

    # ---- temporal enrichment of each centroid 
    cen = centroid_df[centroid_df["species"] == species].sort_values("time").reset_index(drop=True)
    for cl in clusters:
        series = cen[cl].to_numpy(dtype=float)
        global_mean = np.nanmean(series)
        global_std = np.nanstd(series)
        for stress in stress_order:
            vals = series[window_mask(cen["time"], species, stress)]
            vals = vals[np.isfinite(vals)]
            event_mean = np.mean(vals) if len(vals) else np.nan
            diff = event_mean - global_mean
            event_rows.append({"species": species, "c": c, "cluster": cl, "stress": stress,
                               "event_mean": event_mean, "global_mean": global_mean,
                               "global_std": global_std,
                               "event_minus_global": diff,
                               "event_zscore": diff / global_std if global_std > 0 else np.nan,
                               "n_event_times": len(vals)})

    print(f"{species}: {len(ions)} ions, {len(frame)} raw times, {len(cen)} centroid times, "
          f"window sizes {n_window}")

all_scores_df = pd.DataFrame(score_rows)
all_centroid_event_checks_df = pd.DataFrame(event_rows)

# %% Margin labels per cluster (conventions carried over
discrim_rows, temporal_rows = [], []
for (species, cl), d in all_scores_df.groupby(["species", "cluster"], sort=True):
    vals = d["cosine_similarity"].to_numpy(dtype=float)
    order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
    best, second = vals[order[0]], vals[order[1]]
    margin = best - second
    quality = ("Unresolved" if not np.isfinite(margin) else
               "High" if margin >= cos_high else
               "Moderate" if margin >= cos_moderate else "Low/Mixed")
    discrim_rows.append({"species": species, "c": c, "cluster": cl,
                         "best_stress": d["stress"].iloc[order[0]],
                         "best_score": best, "second_best_score": second,
                         "best_minus_second": margin,
                         "best_minus_mean_other": best - np.nanmean(vals[order[1:]]),
                         "discrimination_quality": quality})

for (species, cl), d in all_centroid_event_checks_df.groupby(["species", "cluster"], sort=True):
    vals = d["event_zscore"].to_numpy(dtype=float)
    order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))
    best, second = vals[order[0]], vals[order[1]]
    margin = best - second
    quality = ("Unresolved" if not np.isfinite(margin) else
               "High" if margin >= z_high else
               "Moderate" if margin >= z_moderate else "Low/Mixed")
    temporal_rows.append({"species": species, "c": c, "cluster": cl,
                          "best_temporal_stress": d["stress"].iloc[order[0]],
                          "best_temporal_zscore": best, "second_temporal_zscore": second,
                          "temporal_margin": margin, "temporal_quality": quality})

discrimination_df = pd.DataFrame(discrim_rows)
temporal_summary_df = pd.DataFrame(temporal_rows)
agreement_df = discrimination_df.merge(temporal_summary_df, on=["species", "c", "cluster"], how="outer")
agreement_df["stress_agreement"] = agreement_df["best_stress"] == agreement_df["best_temporal_stress"]

print(agreement_df[["species", "cluster", "best_stress", "discrimination_quality",
                    "best_temporal_stress", "temporal_quality", "stress_agreement"]])

# %% Write the workbook
settings_df = pd.DataFrame({
    "parameter": ["c", "spectral_space", "negative_flux_before_log1p", "reference_spectrum",
                  "cluster_spectrum", "normalisation", "ions_compared", "braycurtis_negatives",
                  "window_bounds", "hour_filter", "centroid_space", "temporal_zscore",
                  "cosine_margin_High", "cosine_margin_Moderate",
                  "temporal_margin_High", "temporal_margin_Moderate", "screening"],
    "value": [c, "log1p", "set to NaN (convention carried over from the old script, L65)",
              "mean log1p over the window's measured cells",
              "membership u_k x global median log1p profile (no noise weight)",
              "L2 on the ions finite in both vectors",
              "ions finite in both vectors (all ions otherwise kept; no top-N)",
              "set to 0 (convention carried over from the old script, L162-163)",
              "include/exclude inclusive at both ends",
              "none (input already cut to 06-18 h)",
              "UWFCM centroids, L2-normalised space (script 1)",
              "(window mean - series mean) / series std (ddof 0)",
              cos_high, cos_moderate, z_high, z_moderate, "none (all ions kept)"],
    "status": ["settled", "convention carried over from the old script", "convention carried over from the old script",
               "method", "method", "method", "method", "convention carried over from the old script",
               "method", "method", "method", "method",
               "convention carried over from the old script, not validated",
               "convention carried over from the old script, not validated",
               "convention carried over from the old script, not validated",
               "convention carried over from the old script, not validated", "method"],
})

with pd.ExcelWriter(out_excel) as writer:
    all_scores_df.to_excel(writer, sheet_name="all_scores", index=False)
    discrimination_df.to_excel(writer, sheet_name="discrimination", index=False)
    all_centroid_event_checks_df.to_excel(writer, sheet_name="centroid_event_check", index=False)
    temporal_summary_df.to_excel(writer, sheet_name="centroid_temporal_sum", index=False)
    agreement_df.to_excel(writer, sheet_name="agreement_summary", index=False)
    pd.concat(spectra_tables, ignore_index=True).to_excel(writer, sheet_name="spectra", index=False)
    settings_df.to_excel(writer, sheet_name="settings", index=False)

print("Saved:", out_excel)

# %% Figure: per-species heatmap plate (cosine | Bray-Curtis | temporal z)
panels = [("cosine_similarity", all_scores_df, "Cosine similarity", "viridis", 0, 1),
          ("braycurtis_similarity", all_scores_df, "1 \u2212 Bray\u2013Curtis", "viridis", 0, 1),
          ("event_zscore", all_centroid_event_checks_df, "Temporal z-score", "coolwarm", -2, 2)]

for species in species_list:
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.2), constrained_layout=True)
    for ax, (col, df, label, cmap, vmin, vmax) in zip(axes, panels):
        pivot = (df[df["species"] == species]
                 .pivot(index="cluster", columns="stress", values=col)
                 .reindex(index=clusters, columns=stress_order))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(stress_order)))
        ax.set_xticklabels(stress_order)
        ax.set_yticks(range(c))
        ax.set_yticklabels(clusters if ax is axes[0] else [])
        ax.tick_params(length=2, width=0.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
        cb.set_label(label)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(length=2, width=0.6)
    axes[0].set_ylabel("Cluster")
    axes[1].set_xlabel("Stress window")
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"UWFCM_c5_similarity_plate_{species}.{ext}"),
                    dpi=800, bbox_inches="tight")
    plt.close(fig)

# %% Figure: Bray-Curtis vs cosine, both species
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6), sharex=True, sharey=True, constrained_layout=True)
for ax, species in zip(axes, species_list):
    d = all_scores_df[all_scores_df["species"] == species]
    for k, cl in enumerate(clusters):
        for stress in stress_order:
            dd = d[(d["cluster"] == cl) & (d["stress"] == stress)]
            ax.scatter(dd["cosine_similarity"], dd["braycurtis_similarity"],
                       marker=stress_markers[stress], s=22, color=cluster_colors[k],
                       edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_title(species, fontsize=8)
    ax.set_xlabel("Cosine similarity")
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.tick_params(length=2, width=0.6)
axes[0].set_ylabel("1 \u2212 Bray\u2013Curtis")

# clusters as colour patches (not markers, so they cannot be read as a window)
handles = ([Patch(facecolor=col, edgecolor="black", linewidth=0.3, label=cl)
            for cl, col in zip(clusters, cluster_colors)] +
           [Line2D([], [], marker=stress_markers[s], linestyle="", markersize=5,
                   markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, label=s)
            for s in stress_order])
fig.legend(handles=handles, loc="outside right center", frameon=False, handletextpad=0.3)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(fig_dir, f"UWFCM_c5_bray_vs_cosine.{ext}"), dpi=800, bbox_inches="tight")
plt.close(fig)

print("Figures saved to", fig_dir)
