
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_and_correct():
    print("--- Starting AI/ML Correction Model Training ---")

    # 1. Load Data
    try:
        model_ds = xr.open_dataset("cowcliphs.nc")
        obs_ds = xr.open_dataset("io-altimeter.nc")
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # 2. Preprocessing
    model_ds = model_ds.rename({'LATITUDE41_321': 'lat', 'LONGITUDE101_221': 'lon', 'TIME': 'time', 'HS': 'swh'})
    obs_ds = obs_ds.rename({'LATITUDE11_80': 'lat', 'LONGITUDE116_145': 'lon', 'TIME': 'time', 'VAVH_DAILY_MEAN': 'swh'})

    # 3. Time Alignment (Monthly)
    start_date = "2015-01-01"
    end_date = "2020-12-31"
    
    model_sliced = model_ds.sel(time=slice(start_date, end_date)).resample(time='1MS').mean()
    obs_sliced = obs_ds.sel(time=slice(start_date, end_date)).resample(time='1MS').mean()

    # Align timestamps exactly
    common_times = np.intersect1d(model_sliced.time.values, obs_sliced.time.values)
    model_aligned = model_sliced.sel(time=common_times)
    obs_aligned = obs_sliced.sel(time=common_times)

    # 4. Regrid Model to Obs Grid
    print("Regridding model to observation grid...")
    model_regridded = model_aligned.interp_like(obs_aligned, method='linear')

    # 5. Prepare Data for ML
    # We want to predict Obs (Target) from Model + Space/Time features
    
    # Flatten data
    # Create a DataFrame
    # Extract arrays
    m_swh = model_regridded['swh'].values.flatten()
    o_swh = obs_aligned['swh'].values.flatten()
    
    # Create coordinate meshes
    # Dimensions are (time, lat, lon)
    times = model_regridded.time.values
    lats = model_regridded.lat.values
    lons = model_regridded.lon.values
    
    # Create meshgrid of coordinates repeated for each time step
    # Note: meshgrid order needs to match the flattening order of xarray/numpy (C-style: time, lat, lon)
    T, Lat, Lon = np.meshgrid(times, lats, lons, indexing='ij')
    
    T_flat = T.flatten()
    Lat_flat = Lat.flatten()
    Lon_flat = Lon.flatten()
    
    # Extract Month as feature
    Months_flat = pd.to_datetime(T_flat).month

    # Create DataFrame
    df = pd.DataFrame({
        'model_swh': m_swh,
        'lat': Lat_flat,
        'lon': Lon_flat,
        'month': Months_flat,
        'obs_swh': o_swh
    })

    # Drop NaNs (land masks, etc.)
    print(f"Original samples: {len(df)}")
    df_clean = df.dropna()
    print(f"Clean samples (non-NaN): {len(df_clean)}")

    # 6. Train/Test Split
    # Train: 2015-2018, Test: 2019-2020
    # Convert 'time' back to datetime for splitting if needed, or use index
    # But we constructed T_flat, so we can use that if we kept it in DF. 
    # Let's add Year to DF for easy splitting
    df_clean['year'] = pd.to_datetime(df.loc[df_clean.index, 'model_swh'].index if False else T_flat[df_clean.index]).year

    train_mask = df_clean['year'] <= 2018
    test_mask = df_clean['year'] >= 2019

    train_df = df_clean[train_mask]
    test_df = df_clean[test_mask]

    print(f"Training samples (2015-2018): {len(train_df)}")
    print(f"Testing samples (2019-2020): {len(test_df)}")

    features = ['model_swh', 'lat', 'lon', 'month']
    target = 'obs_swh'

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # 7. Model Training (Random Forest)
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)

    # 8. Prediction & Evaluation
    print("Predicting on Test set...")
    y_pred = rf_model.predict(X_test)
    
    # Compare Original Model vs Obs in Test Set
    rmse_original = np.sqrt(mean_squared_error(y_test, X_test['model_swh']))
    bias_original = np.mean(X_test['model_swh'] - y_test)
    
    # Compare Corrected vs Obs
    rmse_corrected = np.sqrt(mean_squared_error(y_test, y_pred))
    bias_corrected = np.mean(y_pred - y_test)
    
    r2_corrected = r2_score(y_test, y_pred)

    print("\n=== Correction Results (Test Set: 2019-2020) ===")
    print(f"Original Model RMSE: {rmse_original:.4f} m")
    print(f"Corrected Model RMSE: {rmse_corrected:.4f} m")
    print(f"Improvement: {(rmse_original - rmse_corrected)/rmse_original * 100:.1f}%")
    print(f"Original Model Bias: {bias_original:.4f} m")
    print(f"Corrected Model Bias: {bias_corrected:.4f} m")
    print(f"R2 Score (Corrected): {r2_corrected:.4f}")

    # 9. Plotting Results
    # Scatter plot of Test Set
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, X_test['model_swh'], alpha=0.1, s=1, label='Original')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1')
    plt.title(f'Original CMIP6 vs Obs\nRMSE: {rmse_original:.3f}m')
    plt.xlabel('Observed SWH (m)')
    plt.ylabel('Modeled SWH (m)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.1, s=1, color='g', label='Corrected')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1')
    plt.title(f'Corrected vs Obs\nRMSE: {rmse_corrected:.3f}m')
    plt.xlabel('Observed SWH (m)')
    plt.ylabel('Corrected SWH (m)')
    
    plt.tight_layout()
    plt.savefig('plots/correction_performance.png')
    plt.close()

    # Save the model
    joblib.dump(rf_model, 'swh_correction_model.pkl')
    print("\nModel saved to 'swh_correction_model.pkl'")
    print("Plots saved to 'plots/correction_performance.png'")

if __name__ == "__main__":
    train_and_correct()

