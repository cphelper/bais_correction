
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import glob
import joblib
from scipy.interpolate import griddata

# Set plot style
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3
})

def analyze_sla():
    print("--- Starting SLA Analysis (Robust Curvilinear Support) ---")
    
    # 1. Load Data
    sla_dir = "/Users/shyam/Downloads/isro/altimeter_sla"
    sla_2_path = "/Users/shyam/Downloads/isro/sla_2/zos_Omon_MPI-ESM1-2-LR_ssp245_r1i1p1f1_gn_20150116-20251216.nc"
    
    # Define ROI based on Model extent (from inspection: 63N-79N, 50E-108E)
    lat_min, lat_max = 60, 80
    lon_min, lon_max = 50, 110
    
    print(f"ROI: Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")

    print("Loading observational data (SLA)...")
    sla_files = sorted(glob.glob(os.path.join(sla_dir, "*.nc")))
    # Filter for 2015-2020
    sla_files = [f for f in sla_files if any(y in f for y in ['2015', '2016', '2017', '2018', '2019', '2020'])]
    
    if not sla_files:
        print("No matching SLA files found!")
        return

    # Load and Slice Obs
    obs_datasets = []
    print(f"Loading {len(sla_files)} files...")
    for f in sla_files:
        try:
            with xr.open_dataset(f) as ds:
                # Rename if needed
                if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat'})
                if 'longitude' in ds.coords: ds = ds.rename({'longitude': 'lon'})
                
                # Slice
                ds_sliced = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                
                # Expand time if needed
                if 'time' not in ds_sliced.dims:
                    ds_sliced = ds_sliced.expand_dims('time')
                
                # Load into memory to avoid file issues
                obs_datasets.append(ds_sliced.load())
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not obs_datasets:
        print("No valid data loaded.")
        return

    obs_ds = xr.concat(obs_datasets, dim='time')
    obs_ds = obs_ds.sortby('time')
    # Monthly resample
    obs_monthly = obs_ds['sla'].resample(time='1MS').mean()
    print(f"Obs Monthly Shape: {obs_monthly.shape}")

    # Load Model
    print("Loading model data (ZOS)...")
    model_ds = xr.open_dataset(sla_2_path)
    model_ds = model_ds.sel(time=slice("2015-01-01", "2020-12-31"))
    model_monthly = model_ds['zos'].resample(time='1MS').mean()
    
    # Calculate Model Anomaly
    train_slice = slice("2015-01-01", "2018-12-31")
    model_mean = model_monthly.sel(time=train_slice).mean(dim='time')
    model_sla = model_monthly - model_mean
    
    print("Regridding Model (Curvilinear) to Obs (Rectilinear)...")
    # Get coordinates
    # Model Coords (2D)
    mlat = model_ds['latitude'].values
    mlon = model_ds['longitude'].values
    
    # Flatten Model Grid
    points = np.column_stack((mlat.ravel(), mlon.ravel()))
    
    # Obs Coords (1D dims -> 2D mesh)
    olat = obs_monthly.lat.values
    olon = obs_monthly.lon.values
    olat_grid, olon_grid = np.meshgrid(olat, olon, indexing='ij')
    target_points = np.column_stack((olat_grid.ravel(), olon_grid.ravel()))
    
    # Pre-compute nearest indices (since grid is static)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, indices = tree.query(target_points)
    
    # Create regridded array
    # Shape: (Time, Lat, Lon)
    n_time = len(obs_monthly.time)
    regridded_data = np.zeros((n_time, len(olat), len(olon))) * np.nan
    
    # Interpolate time step by time step
    print("Interpolating time steps...")
    model_times = model_sla.time.values
    obs_times = obs_monthly.time.values
    
    # Align times
    common_times = sorted(list(set(model_times) & set(obs_times)))
    
    # Create aligned DataArrays
    obs_aligned = obs_monthly.sel(time=common_times)
    model_aligned = model_sla.sel(time=common_times)
    
    regridded_vals = []
    
    for t_idx, t in enumerate(common_times):
        # Get model data frame
        m_data = model_aligned.sel(time=t).values.ravel()
        
        # Map to obs grid using nearest neighbor indices
        # Check for NaNs in model data
        valid_mask = ~np.isnan(m_data)
        
        # We can only map valid points. If model has NaNs, we propagate them?
        # Nearest neighbor will pick a point. If it's NaN, we get NaN.
        mapped_data = m_data[indices]
        
        # Reshape to (Lat, Lon)
        regridded_vals.append(mapped_data.reshape(len(olat), len(olon)))
        
    model_sla_regridded = xr.DataArray(
        np.array(regridded_vals),
        coords={'time': common_times, 'lat': olat, 'lon': olon},
        dims=['time', 'lat', 'lon'],
        name='model_sla'
    )
    
    # Prepare DataFrame
    print("Creating DataFrame...")
    # Flatten both
    df_obs = obs_aligned.to_dataframe(name='obs').reset_index()
    df_model = model_sla_regridded.to_dataframe(name='model').reset_index()
    
    # Merge
    df = pd.merge(df_obs, df_model, on=['time', 'lat', 'lon'])
    df['month'] = df['time'].dt.month
    df = df.dropna()
    
    print(f"Total samples: {len(df)}")
    
    # Split
    train_df = df[df['time'].dt.year <= 2018]
    test_df = df[df['time'].dt.year >= 2019]
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    if len(train_df) == 0:
        print("No training data found (check overlap).")
        return

    # Train RF
    print("Training RF...")
    features = ['model', 'lat', 'lon', 'month']
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(train_df[features], train_df['obs'])
    
    # Predict
    test_df = test_df.copy()
    test_df['predicted'] = rf.predict(test_df[features])
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(test_df['obs'], test_df['predicted']))
    r2 = r2_score(test_df['obs'], test_df['predicted'])
    raw_rmse = np.sqrt(mean_squared_error(test_df['obs'], test_df['model']))
    
    print(f"Raw RMSE: {raw_rmse:.4f}")
    print(f"ML RMSE: {rmse:.4f}")
    print(f"Improvement: {(raw_rmse-rmse)/raw_rmse*100:.1f}%")
    
    # Plotting
    os.makedirs("sla_plots", exist_ok=True)
    
    # 1. Scatter
    plt.figure()
    plt.scatter(test_df['obs'], test_df['model'], alpha=0.3, label='Raw', s=2)
    plt.scatter(test_df['obs'], test_df['predicted'], alpha=0.3, label='ML', s=2)
    plt.plot([test_df['obs'].min(), test_df['obs'].max()], [test_df['obs'].min(), test_df['obs'].max()], 'k--')
    plt.xlabel("Observed SLA (m)")
    plt.ylabel("Predicted SLA (m)")
    plt.legend()
    plt.title("SLA Correction Performance")
    plt.savefig("sla_plots/scatter_sla.png")
    plt.close()

    # 2. Time Series
    # Aggregate over space
    ts_df = df.groupby('time')[['obs', 'model']].mean()
    # Predict full period
    df['predicted'] = rf.predict(df[features])
    ts_df['corrected'] = df.groupby('time')['predicted'].mean()
    
    plt.figure(figsize=(12, 5))
    plt.plot(ts_df.index, ts_df['obs'], label='Obs', color='black')
    plt.plot(ts_df.index, ts_df['model'], label='Raw Model', color='blue', alpha=0.6)
    plt.plot(ts_df.index, ts_df['corrected'], label='ML Corrected', color='red', linestyle='--')
    plt.axvline(pd.to_datetime('2019-01-01'), color='gray', linestyle=':', label='Split')
    plt.title("Regional Mean SLA Time Series")
    plt.ylabel("SLA (m)")
    plt.legend()
    plt.savefig("sla_plots/time_series_sla.png")
    plt.close()

    # 3. Bias Map (Test Period)
    test_df_full = df[df['time'].dt.year >= 2019].copy()
    test_df_full['bias_raw'] = test_df_full['model'] - test_df_full['obs']
    test_df_full['bias_ml'] = test_df_full['predicted'] - test_df_full['obs']
    
    # Pivot to grid
    bias_raw_grid = test_df_full.groupby(['lat', 'lon'])['bias_raw'].mean().reset_index()
    bias_ml_grid = test_df_full.groupby(['lat', 'lon'])['bias_ml'].mean().reset_index()
    
    # Plot using scatter for irregular/subset grid or pcolormesh if regular
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw
    sc1 = axes[0].scatter(bias_raw_grid['lon'], bias_raw_grid['lat'], c=bias_raw_grid['bias_raw'], cmap='RdBu_r', s=10)
    plt.colorbar(sc1, ax=axes[0], label='Bias (m)')
    axes[0].set_title("Raw Model Bias (2019-2020)")
    
    # ML
    sc2 = axes[1].scatter(bias_ml_grid['lon'], bias_ml_grid['lat'], c=bias_ml_grid['bias_ml'], cmap='RdBu_r', s=10)
    plt.colorbar(sc2, ax=axes[1], label='Bias (m)')
    axes[1].set_title("ML Corrected Bias (2019-2020)")
    
    plt.savefig("sla_plots/bias_maps_sla.png")
    plt.close()
    
    print("All plots generated in 'sla_plots/'.")
    print("Done.")

if __name__ == "__main__":
    analyze_sla()
