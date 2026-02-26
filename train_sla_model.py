import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import glob
import joblib
from scipy.spatial import cKDTree

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

def train_sla_model():
    print("=" * 60)
    print("  SLA Correction Model: Altimeter → CMIP6 Bias Correction")
    print("=" * 60)

    sla_dir = "./altimeter_sla"
    model_path = "./sla_2/zos_Omon_MPI-ESM1-2-LR_ssp245_r1i1p1f1_gn_20150116-20251216.nc"

    lat_min, lat_max = 60, 80
    lon_min, lon_max = 50, 110

    print(f"\nROI: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")

    # --- 1. Load Observational (Altimeter) SLA ---
    print("\n[1/7] Loading satellite altimeter SLA data...")
    sla_files = sorted(glob.glob(os.path.join(sla_dir, "*.nc")))
    sla_files = [
        f for f in sla_files
        if any(y in f for y in ["2015", "2016", "2017", "2018", "2019", "2020"])
    ]
    print(f"  Found {len(sla_files)} monthly files (2015-2020)")

    obs_datasets = []
    for f in sla_files:
        try:
            with xr.open_dataset(f) as ds:
                if "latitude" in ds.coords:
                    ds = ds.rename({"latitude": "lat"})
                if "longitude" in ds.coords:
                    ds = ds.rename({"longitude": "lon"})
                ds_sliced = ds.sel(
                    lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
                )
                if "time" not in ds_sliced.dims:
                    ds_sliced = ds_sliced.expand_dims("time")
                obs_datasets.append(ds_sliced.load())
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: {e}")

    obs_ds = xr.concat(obs_datasets, dim="time")
    obs_ds = obs_ds.sortby("time")
    obs_monthly = obs_ds["sla"].resample(time="1MS").mean()
    print(f"  Obs monthly shape: {obs_monthly.shape}")
    print(f"  Obs time range: {str(obs_monthly.time.values[0])[:10]} → {str(obs_monthly.time.values[-1])[:10]}")

    # --- 2. Load CMIP6 Model (ZOS → SLA) ---
    print("\n[2/7] Loading CMIP6 model data (ZOS)...")
    model_ds = xr.open_dataset(model_path)
    model_ds = model_ds.sel(time=slice("2015-01-01", "2020-12-31"))
    model_monthly = model_ds["zos"].resample(time="1MS").mean()

    train_slice = slice("2015-01-01", "2018-12-31")
    model_mean = model_monthly.sel(time=train_slice).mean(dim="time")
    model_sla = model_monthly - model_mean
    print(f"  Model SLA shape: {model_sla.shape}")
    print(f"  Converted ZOS → SLA (anomaly relative to 2015-2018 mean)")

    # --- 3. Regrid Model (curvilinear) → Obs (rectilinear) ---
    print("\n[3/7] Regridding model to observation grid (cKDTree nearest-neighbor)...")
    mlat = model_ds["latitude"].values
    mlon = model_ds["longitude"].values
    points = np.column_stack((mlat.ravel(), mlon.ravel()))

    olat = obs_monthly.lat.values
    olon = obs_monthly.lon.values
    olat_grid, olon_grid = np.meshgrid(olat, olon, indexing="ij")
    target_points = np.column_stack((olat_grid.ravel(), olon_grid.ravel()))

    tree = cKDTree(points)
    _, indices = tree.query(target_points)

    model_times = model_sla.time.values
    obs_times = obs_monthly.time.values
    common_times = sorted(list(set(model_times) & set(obs_times)))
    print(f"  Common time steps: {len(common_times)}")

    obs_aligned = obs_monthly.sel(time=common_times)
    model_aligned = model_sla.sel(time=common_times)

    regridded_vals = []
    for t in common_times:
        m_data = model_aligned.sel(time=t).values.ravel()
        mapped_data = m_data[indices]
        regridded_vals.append(mapped_data.reshape(len(olat), len(olon)))

    model_sla_regridded = xr.DataArray(
        np.array(regridded_vals),
        coords={"time": common_times, "lat": olat, "lon": olon},
        dims=["time", "lat", "lon"],
        name="model_sla",
    )
    print(f"  Regridded model shape: {model_sla_regridded.shape}")

    # --- 4. Build Feature Table ---
    print("\n[4/7] Building feature table...")
    df_obs = obs_aligned.to_dataframe(name="obs").reset_index()
    df_model = model_sla_regridded.to_dataframe(name="model").reset_index()
    df = pd.merge(df_obs, df_model, on=["time", "lat", "lon"])
    df["month"] = df["time"].dt.month
    df = df.dropna()
    print(f"  Total valid samples: {len(df)}")

    # --- 5. Train/Test Split ---
    print("\n[5/7] Splitting data...")
    train_df = df[df["time"].dt.year <= 2018]
    test_df = df[df["time"].dt.year >= 2019]
    print(f"  Training (2015-2018): {len(train_df)} samples")
    print(f"  Testing  (2019-2020): {len(test_df)} samples")

    if len(train_df) == 0 or len(test_df) == 0:
        print("ERROR: Insufficient data for train/test split.")
        return

    # --- 6. Train Random Forest ---
    print("\n[6/7] Training Random Forest Regressor (50 trees, depth 10)...")
    features = ["model", "lat", "lon", "month"]
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )
    rf.fit(train_df[features], train_df["obs"])
    print("  Training complete.")

    test_df = test_df.copy()
    test_df["predicted"] = rf.predict(test_df[features])

    # --- 7. Evaluate ---
    print("\n[7/7] Evaluating on test set (2019-2020)...")
    rmse_raw = np.sqrt(mean_squared_error(test_df["obs"], test_df["model"]))
    rmse_ml = np.sqrt(mean_squared_error(test_df["obs"], test_df["predicted"]))
    mae_raw = mean_absolute_error(test_df["obs"], test_df["model"])
    mae_ml = mean_absolute_error(test_df["obs"], test_df["predicted"])
    bias_raw = np.mean(test_df["model"] - test_df["obs"])
    bias_ml = np.mean(test_df["predicted"] - test_df["obs"])
    r2 = r2_score(test_df["obs"], test_df["predicted"])
    improvement = (rmse_raw - rmse_ml) / rmse_raw * 100

    print("\n" + "=" * 60)
    print("  RESULTS (Test Period: 2019-2020)")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Raw CMIP6':>12} {'ML Corrected':>14} {'Improvement':>14}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*14}")
    print(f"  {'RMSE (m)':<25} {rmse_raw:>12.4f} {rmse_ml:>14.4f} {improvement:>13.1f}%")
    print(f"  {'MAE (m)':<25} {mae_raw:>12.4f} {mae_ml:>14.4f}")
    print(f"  {'Bias (m)':<25} {bias_raw:>12.4f} {bias_ml:>14.4f}")
    print(f"  {'R² Score':<25} {'N/A':>12} {r2:>14.4f}")
    print("=" * 60)

    # --- Save model ---
    os.makedirs("sla_plots", exist_ok=True)
    joblib.dump(rf, "sla_correction_model.pkl")
    print(f"\n  Model saved → sla_correction_model.pkl")

    # --- Generate Plots ---
    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test_df["obs"], test_df["model"], alpha=0.3, s=2, label="Raw CMIP6")
    lims = [test_df["obs"].min(), test_df["obs"].max()]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_xlabel("Observed SLA (m)")
    axes[0].set_ylabel("Model SLA (m)")
    axes[0].set_title(f"Raw CMIP6 vs Obs\nRMSE: {rmse_raw:.4f} m")
    axes[0].legend()

    axes[1].scatter(test_df["obs"], test_df["predicted"], alpha=0.3, s=2, color="green", label="ML Corrected")
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_xlabel("Observed SLA (m)")
    axes[1].set_ylabel("Corrected SLA (m)")
    axes[1].set_title(f"ML Corrected vs Obs\nRMSE: {rmse_ml:.4f} m, R²: {r2:.4f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("sla_plots/sla_correction_scatter.png", dpi=150)
    plt.close()

    # Time series
    df["predicted"] = rf.predict(df[features])
    ts_df = df.groupby("time")[["obs", "model", "predicted"]].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(ts_df.index, ts_df["obs"], label="Satellite Obs", color="black", linewidth=2)
    plt.plot(ts_df.index, ts_df["model"], label="Raw CMIP6", color="blue", alpha=0.7)
    plt.plot(ts_df.index, ts_df["predicted"], label="ML Corrected", color="red", linestyle="--", linewidth=2)
    plt.axvline(pd.to_datetime("2019-01-01"), color="gray", linestyle=":", label="Train/Test Split")
    plt.title("Regional Mean SLA Time Series (Altimeter vs CMIP6)")
    plt.ylabel("SLA (m)")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sla_plots/sla_time_series.png", dpi=150)
    plt.close()

    # Bias maps
    test_full = df[df["time"].dt.year >= 2019].copy()
    test_full["bias_raw"] = test_full["model"] - test_full["obs"]
    test_full["bias_ml"] = test_full["predicted"] - test_full["obs"]

    bias_raw_grid = test_full.groupby(["lat", "lon"])["bias_raw"].mean().reset_index()
    bias_ml_grid = test_full.groupby(["lat", "lon"])["bias_ml"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    vmax = max(abs(bias_raw_grid["bias_raw"]).max(), abs(bias_ml_grid["bias_ml"]).max())
    sc1 = axes[0].scatter(bias_raw_grid["lon"], bias_raw_grid["lat"],
                          c=bias_raw_grid["bias_raw"], cmap="RdBu_r", s=10, vmin=-vmax, vmax=vmax)
    plt.colorbar(sc1, ax=axes[0], label="Bias (m)")
    axes[0].set_title("Raw CMIP6 Bias (2019-2020)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    sc2 = axes[1].scatter(bias_ml_grid["lon"], bias_ml_grid["lat"],
                          c=bias_ml_grid["bias_ml"], cmap="RdBu_r", s=10, vmin=-vmax, vmax=vmax)
    plt.colorbar(sc2, ax=axes[1], label="Bias (m)")
    axes[1].set_title("ML Corrected Bias (2019-2020)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig("sla_plots/sla_bias_maps.png", dpi=150)
    plt.close()

    print(f"  Plots saved → sla_plots/")

    # --- Demonstration: predict SLA for specific CMIP6 time points ---
    print("\n" + "=" * 60)
    print("  DEMONSTRATION: CMIP6 → Corrected SLA for specific months")
    print("=" * 60)

    demo_times = ["2019-03-01", "2019-07-01", "2020-01-01", "2020-06-01"]
    for t_str in demo_times:
        t = pd.Timestamp(t_str)
        t_match = min(common_times, key=lambda x: abs(pd.Timestamp(x) - t))

        month_data = df[df["time"] == t_match].copy()
        if len(month_data) == 0:
            continue

        pred = rf.predict(month_data[features])
        obs_mean = month_data["obs"].mean()
        raw_mean = month_data["model"].mean()
        pred_mean = pred.mean()

        print(f"\n  Time point: {str(t_match)[:10]}")
        print(f"    CMIP6 raw regional mean SLA : {raw_mean:+.4f} m")
        print(f"    ML corrected regional mean  : {pred_mean:+.4f} m")
        print(f"    Satellite observed mean      : {obs_mean:+.4f} m")
        print(f"    Raw error                    : {abs(raw_mean - obs_mean):.4f} m")
        print(f"    Corrected error              : {abs(pred_mean - obs_mean):.4f} m")

    print("\n" + "=" * 60)
    print("  Model training and evaluation complete!")
    print("=" * 60)

    return rf


if __name__ == "__main__":
    train_sla_model()
