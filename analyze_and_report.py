
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib

# Publication-ish defaults (readable in IEEE columns)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def analyze():
    print("--- Starting Analysis ---")

    # 1. Load Data
    try:
        model_ds = xr.open_dataset("cowcliphs.nc")
        obs_ds = xr.open_dataset("io-altimeter.nc")
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # 2. Preprocessing & Renaming
    # Rename coords for consistency
    model_ds = model_ds.rename({
        'LATITUDE41_321': 'lat',
        'LONGITUDE101_221': 'lon',
        'TIME': 'time',
        'HS': 'swh'
    })
    
    obs_ds = obs_ds.rename({
        'LATITUDE11_80': 'lat',
        'LONGITUDE116_145': 'lon',
        'TIME': 'time',
        'VAVH_DAILY_MEAN': 'swh'
    })

    print(f"Model time range: {model_ds.time.min().values} to {model_ds.time.max().values}")
    print(f"Obs time range: {obs_ds.time.min().values} to {obs_ds.time.max().values}")

    # 3. Time Alignment
    # Find common start and end dates
    start_date = max(model_ds.time.min().values, obs_ds.time.min().values)
    end_date = min(model_ds.time.max().values, obs_ds.time.max().values)
    
    print(f"Overlapping period: {start_date} to {end_date}")

    # Slice both datasets to the overlapping period
    model_sliced = model_ds.sel(time=slice(start_date, end_date))
    obs_sliced = obs_ds.sel(time=slice(start_date, end_date))

    # Check if time lengths match
    print(f"Model steps in overlap: {len(model_sliced.time)}")
    print(f"Obs steps in overlap: {len(obs_sliced.time)}")

    # If lengths slightly differ (e.g. days of month differ), we might need to resample to monthly start/end
    # Resample both to '1MS' (Month Start) to align timestamps strictly
    model_monthly = model_sliced.resample(time='1MS').mean()
    obs_monthly = obs_sliced.resample(time='1MS').mean()
    
    # Ensure they have the exact same time coordinates now
    common_times = np.intersect1d(model_monthly.time.values, obs_monthly.time.values)
    model_aligned = model_monthly.sel(time=common_times)
    obs_aligned = obs_monthly.sel(time=common_times)

    print(f"Aligned time steps: {len(model_aligned.time)}")

    # 4. Spatial Regridding
    # Regrid Model (finer) to Obs (coarser) grid
    print("Regridding model to observation grid...")
    model_regridded = model_aligned.interp_like(obs_aligned, method='linear')

    # 5. Calculate Metrics
    # Difference
    diff = model_regridded['swh'] - obs_aligned['swh']
    
    # BIAS (Mean Error) over time
    bias = diff.mean(dim='time')
    
    # RMSE over time
    rmse = np.sqrt((diff ** 2).mean(dim='time'))
    
    # Correlation over time
    corr = xr.corr(model_regridded['swh'], obs_aligned['swh'], dim='time')

    # Spatially averaged time series
    # Weights for latitude area correction
    weights = np.cos(np.deg2rad(model_regridded.lat))
    weights.name = "weights"
    
    model_ts = model_regridded['swh'].weighted(weights).mean(dim=['lat', 'lon'])
    obs_ts = obs_aligned['swh'].weighted(weights).mean(dim=['lat', 'lon'])

    # 6. Plotting
    # Create directory for plots
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Optionally compute corrected time series (if model file exists)
    corrected_ts = None
    try:
        rf_model = joblib.load("swh_correction_model.pkl")
        lats_1d = model_regridded.lat.values
        lons_1d = model_regridded.lon.values
        lat2d, lon2d = np.meshgrid(lats_1d, lons_1d, indexing="ij")
        lat_flat = lat2d.ravel()
        lon_flat = lon2d.ravel()
        w_lat = np.cos(np.deg2rad(lats_1d))
        w2d = w_lat[:, None] * np.ones((len(lats_1d), len(lons_1d)))

        corrected_vals = []
        for t in range(len(model_regridded.time)):
            month = int(pd.to_datetime(model_regridded.time.values[t]).month)
            model_2d = model_regridded["swh"].isel(time=t).values
            model_flat = model_2d.ravel()
            valid = ~np.isnan(model_flat)

            X = np.column_stack([
                model_flat[valid],
                lat_flat[valid],
                lon_flat[valid],
                np.full(valid.sum(), month),
            ])
            pred = np.full(model_flat.shape, np.nan, dtype=float)
            pred[valid] = rf_model.predict(X)
            pred2d = pred.reshape(model_2d.shape)

            # weighted mean, ignoring NaNs
            m = ~np.isnan(pred2d)
            if np.any(m):
                corrected_vals.append(np.nansum(pred2d[m] * w2d[m]) / np.nansum(w2d[m]))
            else:
                corrected_vals.append(np.nan)

        corrected_ts = xr.DataArray(
            corrected_vals,
            coords={"time": model_regridded.time.values},
            dims=["time"],
        )
        print("Computed corrected time series using swh_correction_model.pkl")
    except Exception as e:
        print(f"Note: corrected time series not generated ({e})")

    # Plot 1: Time Series (raw vs obs vs corrected)
    plt.figure(figsize=(7.2, 3.2))
    plt.plot(model_ts.time.values, model_ts.values, label="Raw CMIP6", linewidth=1.8)
    if corrected_ts is not None:
        plt.plot(corrected_ts.time.values, corrected_ts.values, label="ML corrected", linewidth=1.8)
    plt.plot(obs_ts.time.values, obs_ts.values, label="Satellite", linewidth=1.8)
    plt.title("Regionally Averaged Significant Wave Height (2015--2020)")
    plt.ylabel("SWH (m)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("plots/time_series.png", bbox_inches="tight")
    plt.close()

    # Plot 2: Bias Map (cleaner labels)
    plt.figure(figsize=(7.2, 3.2))
    ax = plt.gca()
    bias.plot(ax=ax, cmap="RdBu_r", cbar_kwargs={"label": "Bias (m)"})
    ax.set_title("Mean SWH Bias (Model - Satellite), 2015--2020")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig("plots/bias_map.png", bbox_inches="tight")
    plt.close()

    # Plot 3: RMSE Map
    plt.figure(figsize=(7.2, 3.2))
    ax = plt.gca()
    rmse.plot(ax=ax, cmap="viridis", cbar_kwargs={"label": "RMSE (m)"})
    ax.set_title("SWH RMSE vs Satellite, 2015--2020")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig("plots/rmse_map.png", bbox_inches="tight")
    plt.close()
    
    # Plot 4: Scatter Plot (density-friendly)
    obs_flat = obs_aligned["swh"].values.flatten()
    mod_flat = model_regridded["swh"].values.flatten()
    m = ~np.isnan(obs_flat) & ~np.isnan(mod_flat)
    obs_flat = obs_flat[m]
    mod_flat = mod_flat[m]

    plt.figure(figsize=(3.8, 3.6))
    hb = plt.hexbin(obs_flat, mod_flat, gridsize=55, mincnt=1, cmap="Blues")
    min_val = float(min(obs_flat.min(), mod_flat.min()))
    max_val = float(max(obs_flat.max(), mod_flat.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.2)
    plt.xlabel("Satellite SWH (m)")
    plt.ylabel("Raw CMIP6 SWH (m)")
    plt.title("Raw CMIP6 vs Satellite (All months)")
    cb = plt.colorbar(hb)
    cb.set_label("Count")
    plt.tight_layout()
    plt.savefig("plots/scatter.png", bbox_inches="tight")
    plt.close()

    # 7. Print Summary Stats
    mean_bias = float(bias.weighted(weights).mean())
    mean_rmse = float(rmse.weighted(weights).mean())
    correlation_mean = float(corr.weighted(weights).mean())
    
    print("\n=== Summary Statistics ===")
    print(f"Mean Bias (Global): {mean_bias:.4f} m")
    print(f"Mean RMSE (Global): {mean_rmse:.4f} m")
    print(f"Mean Temporal Correlation: {correlation_mean:.4f}")
    
    print("\nComparison complete. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    analyze()

