# -*- coding: utf-8 -*-
r"""
@author: b.dey
email: biplobforestry@gmail.com

"""

# %% Imports
import os
import sys
import json
import numpy as np
import pandas as pd

# folder of this script: holds uwfcm_functions.py and receives Output\
try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # Spyder cell run without __file__
    code_dir = os.getcwd()
sys.path.insert(0, code_dir)

from uwfcm_functions import Dataset, Orientation
from uwfcm_functions import FCMParameters, fit_best
from uwfcm_functions import fpc, partition_entropy
from uwfcm_functions import centroid_separation
from uwfcm_functions import working_view

# %% Paths and settings
# set base_dir to the folder that contains Data\
base_dir = r"C:\My_PC\2025_Campaign\Manuscript\npj Clim Atmos Sci"
data_dir = os.path.join(base_dir, "Data")
u_dir = os.path.join(code_dir, "Output", "uncertainty")
table_dir = os.path.join(code_dir, "Output", "tables")
os.makedirs(table_dir, exist_ok=True)

identity_path = os.path.join(data_dir, "Master_data", "Identity_masslist_Clean.xlsx")

species_list = ["Chestnut", "Linden"]
short_name = {"Chestnut": "Che", "Linden": "Lin"}

transform = "l2"
c = 5
m = 1.6
seed = 24
n_init = 10
max_iter = 300
tolerance = 1e-6
balance_objects = True

# objective of the c = 5 rebuild (c_selection.json); the fit should return it
reference_objective = {"Chestnut": 1.0074214531030319, "Linden": 2.942}

excel_name = f"UWFCM_c{c}_{transform}_m{m}_seed{seed}.xlsx"

# %% Identity map (labels only)
identity_df = pd.read_excel(identity_path)
identity_df["amu"] = pd.to_numeric(identity_df["amu"], errors="coerce")
identity_df = identity_df.dropna(subset=["amu"]).drop_duplicates("amu")
name_of = dict(zip(identity_df["amu"].round(4), identity_df["compound_name"].astype(str).str.strip()))

# %% Fit each species
membership_rows, contribution_rows, centroid_rows = [], [], []
amu_tables, diagnostics_rows, restart_rows = {}, [], []

for species in species_list:
    frame = pd.read_csv(os.path.join(data_dir, f"{species}_signal_LOD.csv"))
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)

    u_df = pd.read_csv(os.path.join(u_dir, f"{species}_u_precision_k3.csv"))
    u_df["time"] = pd.to_datetime(u_df["time"])
    ions = [col for col in u_df.columns if col != "time"]
    assert (u_df["time"].values == frame["time"].values).all(), "time axes differ"

    dataset = Dataset(frame[ions], u_df[ions], time=frame["time"], name=species)

    # ions are the objects, time points are the features; NaN u -> W = 0
    view, _ = working_view(dataset, Orientation.IONS, transform,
                           balance_objects=balance_objects,
                           missing_as_unmeasured=True)

    restarts = fit_best(view.x, view.u,
                        FCMParameters(c=c, m=m, seed=seed, max_iter=max_iter,
                                      tolerance=tolerance,
                                      balance_objects=balance_objects),
                        weights=view.weights,
                        weight_diagnostics=view.weight_diagnostics,
                        n_init=n_init,
                        object_labels=view.object_labels,
                        feature_labels=view.feature_labels)
    best = restarts.best

    U = best.memberships                      # (c, n_ions)
    V = best.centers                          # (c, n_time)  in L2 space
    obj_ions = list(best.object_labels)
    feat_time = pd.to_datetime(list(best.feature_labels))
    amu = np.array([float(i) for i in obj_ions])

    # mean raw flux of each ion over its measured cells (old "mean_intensity")
    raw = frame[obj_ions].to_numpy(dtype=float)
    u_raw = u_df[obj_ions].to_numpy(dtype=float)
    measured = np.isfinite(raw) & np.isfinite(u_raw) & (u_raw > 0)
    mean_flux = np.nanmean(np.where(measured, raw, np.nan), axis=0)

    print(f"{species}: {len(obj_ions)} ions x {len(feat_time)} times, "
          f"J = {best.objective:.6f} (rebuild {reference_objective[species]:.6f}), "
          f"distinct optima {restarts.n_distinct_optima}, converged {restarts.n_converged}/{restarts.n_attempted}")

    # ---- long tables, same layout as the old workbook ----
    for k in range(c):
        cl = f"C{k + 1}"
        names = [name_of.get(round(a, 4)) for a in amu]
        membership_rows.append(pd.DataFrame({
            "species": species, "c": c, "cluster": cl, "amu": amu,
            "membership": U[k], "compound_name": names}))
        contribution_rows.append(pd.DataFrame({
            "species": species, "c": c, "cluster": cl, "amu": amu,
            "contribution": U[k] * mean_flux, "compound_name": names}))

    cen = pd.DataFrame(V.T, columns=[f"C{k + 1}" for k in range(c)])
    cen.insert(0, "time", feat_time)
    cen.insert(0, "c", c)
    cen.insert(0, "species", species)
    centroid_rows.append(cen)

    # ---- per-ion wide table, ranked by dominant cluster then membership ----
    wide = pd.DataFrame({"AMU": amu, "compound_name": [name_of.get(round(a, 4)) for a in amu]})
    for k in range(c):
        wide[f"membership_cluster_{k + 1}"] = U[k]
    wide["dominant_cluster"] = U.argmax(axis=0) + 1
    wide["dominant_membership"] = U.max(axis=0)
    wide["measured_fraction"] = measured.mean(axis=0)
    wide["mean_flux"] = mean_flux
    wide = wide.sort_values(["dominant_cluster", "dominant_membership"],
                            ascending=[True, False]).reset_index(drop=True)
    amu_tables[f"{short_name[species]}_c{c}_amu"] = wide

    # ---- diagnostics for c = 5 only ----
    diagnostics_rows.append({
        "species": species, "c": c, "m": m, "transform": transform,
        "objective_J": best.objective,
        "rebuild_objective_J": reference_objective[species],
        "FPC": fpc(U), "PE": partition_entropy(U),
        "max_abs_centroid_r": centroid_separation(V),
        "distinct_optima": restarts.n_distinct_optima,
        "restarts_converged": restarts.n_converged,
        "restarts_attempted": restarts.n_attempted,
        "best_seed": best.parameters.seed,
        "n_iter": best.n_iter,
        "n_ions": len(obj_ions), "n_time": len(feat_time),
        "unmeasured_cells": int((~measured).sum()),
        **{f"n_dominant_C{k + 1}": int((U.argmax(axis=0) == k).sum()) for k in range(c)},
    })
    restart_rows.append(pd.DataFrame({"species": species,
                                      "restart_seed": seed + np.arange(len(restarts.objectives)),
                                      "objective": restarts.objectives}))

    np.savez(os.path.join(table_dir, f"{species}_UWFCM_c{c}.npz"),
             memberships=U, centers=V, ions=np.array(obj_ions),
             time=feat_time.values.astype("datetime64[ns]"),
             objective=best.objective)

# %% Write the workbook
settings_df = pd.DataFrame({
    "parameter": ["transform", "c", "m", "seed", "n_init", "max_iter", "tolerance",
                  "balance_objects", "uncertainty", "missing_cells", "screening"],
    "value": [transform, c, m, seed, n_init, max_iter, tolerance, balance_objects,
              "u^2 = (EF_j |x|)^2 + (LOD/3)^2, EF per ion (precision)",
              "W = 0, not imputed", "none (all ions kept)"],
})

with pd.ExcelWriter(os.path.join(table_dir, excel_name)) as writer:
    pd.concat(membership_rows).to_excel(writer, sheet_name="membership", index=False)
    pd.concat(contribution_rows).to_excel(writer, sheet_name="contribution", index=False)
    pd.concat(centroid_rows).to_excel(writer, sheet_name="centroid_timeseries", index=False)
    for sheet, df in amu_tables.items():
        df.to_excel(writer, sheet_name=sheet, index=False)
    pd.DataFrame(diagnostics_rows).to_excel(writer, sheet_name="diagnostics", index=False)
    pd.concat(restart_rows).to_excel(writer, sheet_name="restarts", index=False)
    settings_df.to_excel(writer, sheet_name="settings", index=False)

with open(os.path.join(table_dir, f"UWFCM_c{c}_settings.json"), "w") as f:
    json.dump(dict(zip(settings_df["parameter"], settings_df["value"].astype(str))), f, indent=2)

print(pd.DataFrame(diagnostics_rows).T)
