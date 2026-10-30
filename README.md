# UWFCM analysis scripts 
Uncertainty-weighted fuzzy c-means (UWFCM) for Chestnut and
Linden, c = 5. The scripts need only numpy, pandas, scipy, matplotlib and
openpyxl. The method follows the UWFCM software (under developing); the parts the scripts use are
reimplemented in `uwfcm_functions.py`, which reproduces that software's results numerically.

## Scripts 

| Script | What it does |
|---|---|
| `0_UWFCM_uncertainty.py` | Per-cell uncertainty `u² = (EF_j·|x|)² + (LOD/3)²`, EF_j per ion from the record; per-ion quality table |
| `UWFCM_fit_c5.py` | Fits the c = 5 UWFCM (ions = objects, time = features); writes memberships, centroids, diagnostics |
| `UWFCM_figures.py` | Membership heatmap, centroid/member lines and envelopes, membership and contribution stems |
| `UWFCM_similarity.py` | Cluster vs stress window: cosine and Bray-Curtis similarity, temporal z-score |
| `UWFCM_stress_assignment.py` | Combined temporal/spectral assignment of each cluster to a stress window |
| `uwfcm_functions.py` | All UWFCM functions the scripts call |


## Environment

Python 3.13.14, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, matplotlib 3.10.8,
openpyxl 3.1.5 (`pip install -r requirements.txt`). 
