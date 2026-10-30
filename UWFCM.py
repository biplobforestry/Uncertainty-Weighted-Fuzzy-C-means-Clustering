# -*- coding: utf-8 -*-
r"""
@author: b.dey
email: biplobforestry@gmail.com

UWFCM_uncertainty.py

Builds the per-cell uncertainty matrix for Chestnut and Linden with the UWFCM
precision model, and a per-ion data-quality table that only DESCRIBES.

Replaces:  Uncertinity_matrix_data_matrix_for_fuzzy.py, FCC_with_uncertinity.py

    u^2(t, j) = (EF_j * |x(t, j)|)^2 + (LOD(t, j) / 3)^2

  - EF_j is the relative PRECISION of ion j, estimated from its own record.
  - LOD/3 because the instrument writes LOD = 3 sigma of the background.

Input  : Data\{species}_signal_LOD.csv   (time, ions, "LOD: ion" columns)
Output : Output\uncertainty\{species}_u_precision_k3.csv
         Output\uncertainty\UWFCM_uncertainty_summary.xlsx
"""

# %% Imports
import os
import sys
import numpy as np
import pandas as pd

try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:                       
    code_dir = os.getcwd()
sys.path.insert(0, code_dir)

from uwfcm_functions import uncertainty_from_limits

# %% Paths and settings
base_dir = r"C:\My_PC\2025_Campaign\Manuscript\npj Clim Atmos Sci"
data_dir = os.path.join(base_dir, "Data")
out_dir = os.path.join(code_dir, "Output", "uncertainty")
os.makedirs(out_dir, exist_ok=True)

identity_path = os.path.join(data_dir, "Master_data", "Identity_masslist_Clean.xlsx")

species_list = ["Chestnut", "Linden"]

# %% Identity map (amu -> compound name), used for labelling only
identity_df = pd.read_excel(identity_path)
identity_df["amu"] = pd.to_numeric(identity_df["amu"], errors="coerce")
identity_df = identity_df.dropna(subset=["amu"]).drop_duplicates("amu")
identity_df["amu_key"] = identity_df["amu"].round(4)
name_of = dict(zip(identity_df["amu_key"], identity_df["compound_name"].astype(str).str.strip()))

# %% Build u for each species
summary_rows = []
noise_tables = []

for species in species_list:
    frame = pd.read_csv(os.path.join(data_dir, f"{species}_signal_LOD.csv"))
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)

    # pair each ion with its "LOD: ion" column
    lod_of = {c.replace("LOD:", "").strip(): c for c in frame.columns if str(c).startswith("LOD:")}
    ions = [c for c in frame.columns if c != "time" and not str(c).startswith("LOD:") and c in lod_of]
    unpaired = [c for c in frame.columns if c != "time" and not str(c).startswith("LOD:") and c not in lod_of]

    signal = frame[["time"] + ions]
    detection = frame[["time"] + [lod_of[i] for i in ions]].copy()
    detection.columns = ["time"] + ions

    # default route = precision model, EF per ion, LOD/3, no below-LOD branch
    u_df, report = uncertainty_from_limits(signal, detection, "time")

    u_df.to_csv(os.path.join(out_dir, f"{species}_u_precision_k3.csv"), index=False)

    # ---- descriptive per-ion table (measured cells only) ----
    x = signal[ions].to_numpy(dtype=float)
    lod = detection[ions].to_numpy(dtype=float)
    u = u_df[ions].to_numpy(dtype=float)
    measured = np.isfinite(x) & np.isfinite(u) & (u > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(measured, np.abs(x) / u, np.nan)
        above = np.where(measured, x > lod, np.nan)

    ef_of = report["error_fraction_per_ion"]          # dict: ion -> EF_j
    ef = np.array([ef_of[i] for i in ions], dtype=float)

    table = pd.DataFrame({
        "species": species,
        "amu": [float(i) for i in ions],
        "EF_precision": ef,
        "median_LOD": np.nanmedian(np.where(lod > 0, lod, np.nan), axis=0),
        "measured_fraction": measured.mean(axis=0),
        "fraction_above_LOD": np.nanmean(above, axis=0),
        "median_SNR": np.nanmedian(snr, axis=0),
        "mean_signal": np.nanmean(np.where(measured, x, np.nan), axis=0),
    })
    table["compound_name"] = table["amu"].round(4).map(name_of)
    noise_tables.append(table)

    summary_rows.append({
        "species": species,
        "n_time": len(frame),
        "n_ions": len(ions),
        "unpaired_ions_without_LOD": len(unpaired),
        "cells": x.size,
        "unmeasured_cells": int((~measured).sum()),
        "unmeasured_fraction": float((~measured).mean()),
        "EF_median": float(np.nanmedian(ef)),
        "EF_min": float(np.nanmin(ef)),
        "EF_max": float(np.nanmax(ef)),
        "LOD_divisor_k": 3,
    })

    print(f"{species}: {len(frame)} times x {len(ions)} ions, "
          f"EF median {np.nanmedian(ef):.4f}, unmeasured cells {int((~measured).sum())}")

# %% Write the summary workbook
summary_df = pd.DataFrame(summary_rows)
noise_df = pd.concat(noise_tables, ignore_index=True)

with pd.ExcelWriter(os.path.join(out_dir, "UWFCM_uncertainty_summary.xlsx")) as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    noise_df.to_excel(writer, sheet_name="ion_quality", index=False)

print(summary_df.T)
