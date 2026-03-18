# AGENTS.md

## Cursor Cloud specific instructions

This is a scientific research project for correcting CMIP6 climate model projections (SWH and SLA) using satellite altimeter observations and Random Forest regression. There are no web services, APIs, or databases — only standalone Python scripts processing NetCDF data.

### Data files
- All `.nc` files are stored via **Git LFS**. After cloning, run `git lfs pull` to download the actual binary data (without this, scripts will fail with "Unknown file format" errors).
- Data is already in the repo: `cowcliphs.nc`, `io-altimeter.nc` (SWH), `altimeter_sla/` (SLA observations), `sla_2/` (CMIP6 ZOS model).

### Key scripts
- `train_correction_model.py` — Trains SWH bias correction model (uses `cowcliphs.nc` + `io-altimeter.nc`)
- `train_sla_model.py` — Trains SLA bias correction model (uses `altimeter_sla/` + `sla_2/`)
- `analyze_and_report.py` — Generates SWH analysis plots and reports
- `inspect_data.py` — Quick inspection of NetCDF file structure
- `analyze_sla.py` — SLA analysis (note: has hardcoded paths to `/Users/shyam/...`, use `train_sla_model.py` instead for local runs)

### Running scripts
All scripts run with `python3 <script>.py` from the workspace root. No virtual environment is set up; dependencies are installed at user level. Use `matplotlib.use('Agg')` or ensure `MPLBACKEND=Agg` is set when running headless (no display).

### Gotchas
- The `analyze_sla.py` and `inspect_sla.py` scripts have hardcoded absolute paths to `/Users/shyam/Downloads/isro/...`. Use `train_sla_model.py` for SLA analysis with correct relative paths.
- Some xarray FutureWarnings about `Dataset.dims` and `concat` defaults are expected and non-breaking.
- sklearn warnings about feature names when using models trained with DataFrames but predicting with numpy arrays are harmless.
