# -*- coding: utf-8 -*-
r"""
@author: b.dey
email: biplobforestry@gmail.com

UWFCM_stress_assignment.py


Input  : Output\tables\UWFCM_c5_similarity.xlsx
Output : Output\tables\UWFCM_c5_stress_assignment.xlsx
         Output\figures\UWFCM_c5_assignment_dots_{species}.png/.svg
         Output\figures\UWFCM_c5_assignment_bubble_matrix.png/.svg
"""

# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    code_dir = os.getcwd()

# %% Paths and settings
# set base_dir to the folder that contains Data\
base_dir = r"\data"
table_dir = os.path.join(code_dir, "Output", "tables")
fig_dir = os.path.join(code_dir, "Output", "figures")
os.makedirs(fig_dir, exist_ok=True)

in_excel = os.path.join(table_dir, "UWFCM_c5_similarity.xlsx")
out_excel = os.path.join(table_dir, "UWFCM_c5_stress_assignment.xlsx")

species_list = ["Chestnut", "Linden"]
c = 5
clusters = [f"C{k + 1}" for k in range(c)]
stress_order = ["Ac", "Ht", "HtO3"]


weight_temporal, weight_cosine, weight_bray = 0.60, 0.25, 0.15
strong_margin, moderate_margin = 0.25, 0.10
min_temporal_z, min_cosine, min_bray = 0.25, 0.60, 0.40

stress_colors = {"Ac": "#009E73", "Ht": "#CC79A7", "HtO3": "#E69F00"}

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.6})


def minmax_scale(x):
    """Min-max over the finite values; all equal -> 0.5 (old L70-86)."""
    x = pd.Series(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return pd.Series(np.nan, index=x.index)
    xmin, xmax = x[ok].min(), x[ok].max()
    if np.isclose(xmin, xmax):
        return pd.Series(np.where(ok, 0.5, np.nan), index=x.index)
    return (x - xmin) / (xmax - xmin)


def best_and_second(values):
    vals = np.asarray(values, dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(vals))
    others = np.delete(vals, i)
    return i, vals[i], (np.nanmax(others) if np.isfinite(others).any() else np.nan)


def dot_size(score, smin, smax):
    """Fixed mapping of the combined score (0-1) to marker area."""
    return smin + np.clip(np.nan_to_num(score, nan=0.0), 0, 1) * (smax - smin)


# %% Merge spectral and temporal evidence 
all_scores_df = pd.read_excel(in_excel, sheet_name="all_scores")
all_centroid_event_checks_df = pd.read_excel(in_excel, sheet_name="centroid_event_check")

spec = all_scores_df[["species", "c", "cluster", "stress", "cosine_similarity", "braycurtis_similarity"]]
temp = all_centroid_event_checks_df[["species", "c", "cluster", "stress", "event_zscore",
                                     "event_mean", "global_mean", "event_minus_global"]]
cluster_stress_df = spec.merge(temp, on=["species", "c", "cluster", "stress"], how="outer")
cluster_stress_df["stress"] = pd.Categorical(cluster_stress_df["stress"], categories=stress_order, ordered=True)
cluster_stress_df = cluster_stress_df.sort_values(["species", "cluster", "stress"]).reset_index(drop=True)

# %% Combined score, confidence and final label per cluster 
detail_parts, summary_rows = [], []

for (species, cluster), d in cluster_stress_df.groupby(["species", "cluster"], sort=True, observed=True):
    d = d.copy().reset_index(drop=True)
    d["temporal_scaled"] = minmax_scale(d["event_zscore"])
    d["cosine_scaled"] = minmax_scale(d["cosine_similarity"])
    d["bray_scaled"] = minmax_scale(d["braycurtis_similarity"])
    d["combined_score"] = (weight_temporal * d["temporal_scaled"] +
                           weight_cosine * d["cosine_scaled"] +
                           weight_bray * d["bray_scaled"])

    t_idx, t_best, t_second = best_and_second(d["event_zscore"])
    s_idx, s_best, s_second = best_and_second(d["cosine_similarity"])
    c_idx, c_best, c_second = best_and_second(d["combined_score"])

    temporal_winner = d.loc[t_idx, "stress"] if pd.notna(t_idx) else np.nan
    spectral_winner = d.loc[s_idx, "stress"] if pd.notna(s_idx) else np.nan
    combined_winner = d.loc[c_idx, "stress"] if pd.notna(c_idx) else np.nan

    combined_margin = c_best - c_second if np.isfinite(c_second) else np.nan
    confidence = ("Unresolved" if pd.isna(combined_margin) else
                  "High" if combined_margin >= strong_margin else
                  "Moderate" if combined_margin >= moderate_margin else "Low/Mixed")

    win = d.loc[c_idx] if pd.notna(c_idx) else pd.Series(dtype=float)
    win_z = win.get("event_zscore", np.nan)
    win_cos = win.get("cosine_similarity", np.nan)
    win_bray = win.get("braycurtis_similarity", np.nan)


    final_label, note = combined_winner, "Consistent"
    if pd.isna(combined_winner) or pd.isna(win_z) or pd.isna(win_cos) or pd.isna(win_bray):
        final_label, note = "Unresolved", "Missing evidence"
    elif win_z < min_temporal_z:
        final_label, note = "Unresolved", "Weak temporal evidence"
    elif win_cos < min_cosine and win_bray < min_bray:
        final_label, note = "Unresolved", "Weak spectral evidence"
    elif confidence == "Low/Mixed":
        if temporal_winner != spectral_winner:
            final_label, note = "Mixed", "Temporal-spectral disagreement"
        else:
            final_label, note = f"{combined_winner} (weak)", "Low margin"
    elif temporal_winner != spectral_winner:
        final_label, note = f"{combined_winner} (mixed)", "Temporal-spectral disagreement"

    detail_parts.append(d)
    summary_rows.append({
        "species": species, "c": c, "cluster": cluster,
        "temporal_winner": temporal_winner, "temporal_best_zscore": t_best,
        "temporal_second_zscore": t_second,
        "spectral_winner": spectral_winner, "spectral_best_cosine": s_best,
        "spectral_second_cosine": s_second,
        "combined_winner": combined_winner, "combined_best_score": c_best,
        "combined_second_score": c_second, "combined_margin": combined_margin,
        "winning_temporal_zscore": win_z, "winning_cosine": win_cos, "winning_bray": win_bray,
        "temporal_agreement": temporal_winner == combined_winner,
        "spectral_agreement": spectral_winner == combined_winner,
        "full_agreement": temporal_winner == spectral_winner == combined_winner,
        "confidence": confidence, "final_label": final_label, "note": note,
    })

assignment_detail_df = pd.concat(detail_parts, ignore_index=True)
assignment_summary_df = pd.DataFrame(summary_rows)

print(assignment_summary_df[["species", "cluster", "temporal_winner", "spectral_winner",
                             "combined_winner", "combined_margin", "confidence",
                             "final_label", "note"]].to_string())


carried = "convention carried over from 6.FCM_Noise_solution.py, not validated"
settings_df = pd.DataFrame({
    "parameter": ["c", "weight_temporal", "weight_cosine", "weight_bray", "scaling",
                  "confidence_High_margin", "confidence_Moderate_margin",
                  "unresolved_min_temporal_z", "unresolved_min_cosine", "unresolved_min_bray",
                  "unresolved_spectral_rule", "mixed_rule", "input"],
    "value": [c, weight_temporal, weight_cosine, weight_bray,
              "min-max over the 3 windows of each cluster; all equal -> 0.5",
              strong_margin, moderate_margin, min_temporal_z, min_cosine, min_bray,
              "winning cosine < min_cosine AND winning Bray < min_bray",
              "Low/Mixed confidence + temporal != spectral winner -> Mixed; "
              "otherwise '(weak)'; higher confidence + disagreement -> '(mixed)'",
              "UWFCM_c5_similarity.xlsx (script 3)"],
    "status": ["settled"] + [carried] * 11 + ["input"],
})

with pd.ExcelWriter(out_excel) as writer:
    cluster_stress_df.to_excel(writer, sheet_name="merged_cluster_stress", index=False)
    assignment_detail_df.to_excel(writer, sheet_name="assignment_detail", index=False)
    assignment_summary_df.to_excel(writer, sheet_name="assignment_summary", index=False)
    settings_df.to_excel(writer, sheet_name="settings", index=False)

print("Saved:", out_excel)
#%%
norm = Normalize(vmin=-2, vmax=2)
cmap = plt.get_cmap("coolwarm")
smin, smax = 15, 260
legend_scores = [0.25, 0.5, 1.0]

for species in species_list:
    det = assignment_detail_df[assignment_detail_df["species"] == species]
    winners = dict(zip(assignment_summary_df.loc[assignment_summary_df["species"] == species, "cluster"],
                       assignment_summary_df.loc[assignment_summary_df["species"] == species, "combined_winner"]))

    fig, ax = plt.subplots(figsize=(3.2, 2.6), constrained_layout=True)
    for _, row in det.iterrows():
        x = stress_order.index(row["stress"])
        y = clusters.index(row["cluster"])
        is_winner = row["stress"] == winners[row["cluster"]]
        ax.scatter(x, y, s=dot_size(row["combined_score"], smin, smax),
                   color=cmap(norm(row["event_zscore"])),
                   edgecolors="black" if is_winner else "0.6",
                   linewidths=1.0 if is_winner else 0.4, zorder=3)
    ax.set_xticks(range(len(stress_order)))
    ax.set_xticklabels(stress_order)
    ax.set_yticks(range(c))
    ax.set_yticklabels(clusters)
    ax.set_xlim(-0.5, len(stress_order) - 0.5)
    ax.set_ylim(c - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, len(stress_order)), minor=True)
    ax.set_yticks(np.arange(-0.5, c), minor=True)
    ax.grid(which="minor", color="0.88", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2, width=0.6)
    ax.set_xlabel("Stress window")
    ax.set_ylabel("Cluster")

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.07, pad=0.03)
    cb.set_label("Temporal z-score")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2, width=0.6)

    handles = [plt.scatter([], [], s=dot_size(v, smin, smax), color="0.75", edgecolors="0.6",
                           linewidths=0.4, label=f"{v:g}") for v in legend_scores]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                          markeredgecolor="black", markeredgewidth=1.0, label="winner"))
    fig.legend(handles=handles, title="Combined score", title_fontsize=8, loc="outside lower center",
               ncol=4, frameon=False, handletextpad=0.6, columnspacing=1.2)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"UWFCM_c5_assignment_dots_{species}.{ext}"),
                    dpi=800, bbox_inches="tight")
    plt.close(fig)

# %% Figure
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.4), sharey=True, constrained_layout=True)
for ax, species in zip(axes, species_list):
    det = assignment_detail_df[assignment_detail_df["species"] == species]
    for _, row in det.iterrows():
        ax.scatter(stress_order.index(row["stress"]), clusters.index(row["cluster"]),
                   s=dot_size(row["combined_score"], 8, 160),
                   color=stress_colors[row["stress"]], edgecolors="none", alpha=0.9, zorder=3)
    ax.set_xticks(range(len(stress_order)))
    ax.set_xticklabels(stress_order)
    ax.set_yticks(range(c))
    ax.set_yticklabels(clusters)
    ax.set_xlim(-0.5, len(stress_order) - 0.5)
    ax.set_ylim(c - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, len(stress_order)), minor=True)
    ax.set_yticks(np.arange(-0.5, c), minor=True)
    ax.grid(which="minor", color="0.88", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2, width=0.6)
    ax.set_title(species, fontsize=8)
axes[0].set_ylabel("Cluster")

handles = [plt.scatter([], [], s=dot_size(v, 8, 160), color="0.6", edgecolors="none",
                       label=f"{v:g}") for v in legend_scores]
fig.legend(handles=handles, title="Combined score", title_fontsize=8, loc="outside right center",
           frameon=False, handletextpad=0.2, labelspacing=0.9)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(fig_dir, f"UWFCM_c5_assignment_bubble_matrix.{ext}"),
                dpi=800, bbox_inches="tight")
plt.close(fig)

print("Figures saved to", fig_dir)
