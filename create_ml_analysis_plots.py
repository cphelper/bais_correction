#!/usr/bin/env python3
"""
Create machine learning analysis and interpretation plots.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_ml_data():
    """Load and prepare data for ML analysis"""
    print("Loading and preparing ML data...")
    
    # Load datasets
    model_ds = xr.open_dataset('cowcliphs.nc')
    model_ds = model_ds.rename({
        'TIME': 'time',
        'LATITUDE41_321': 'lat', 
        'LONGITUDE101_221': 'lon',
        'HS': 'swh'
    })
    
    obs_ds = xr.open_dataset('io-altimeter.nc')
    obs_ds = obs_ds.rename({
        'TIME': 'time',
        'LATITUDE11_80': 'lat',
        'LONGITUDE116_145': 'lon', 
        'VAVH_DAILY_MEAN': 'swh'
    })
    
    # Restrict to common time period and resample
    start_date = '2015-01-01'
    end_date = '2020-12-31'
    
    model_ds = model_ds.sel(time=slice(start_date, end_date))
    obs_ds = obs_ds.sel(time=slice(start_date, end_date))
    
    model_monthly = model_ds.resample(time='MS').mean()
    obs_monthly = obs_ds.resample(time='MS').mean()
    
    # Interpolate model to observation grid
    model_regrid = model_monthly.interp_like(obs_monthly)
    
    # Vectorized flatten into a table (much faster than nested loops)
    times = model_regrid.time.values
    lats = model_regrid.lat.values
    lons = model_regrid.lon.values

    n_t = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    n_per_t = n_lat * n_lon

    model_flat = model_regrid.swh.values.reshape(-1)
    obs_flat = obs_monthly.swh.values.reshape(-1)

    lat_flat_one = np.repeat(lats, n_lon)
    lon_flat_one = np.tile(lons, n_lat)
    lat_flat = np.tile(lat_flat_one, n_t)
    lon_flat = np.tile(lon_flat_one, n_t)

    time_idx = np.repeat(np.arange(n_t), n_per_t)
    months = pd.DatetimeIndex(times).month
    month_flat = months[time_idx]

    df = pd.DataFrame({
        'model_swh': model_flat,
        'lat': lat_flat,
        'lon': lon_flat,
        'month': month_flat.astype(int),
        'obs_swh': obs_flat,
        'time_idx': time_idx.astype(int),
    })

    df = df.dropna()
    
    # Split data (2015-2018 for training, 2019-2020 for testing)
    train_mask = df['time_idx'] < 48  # First 48 months (2015-2018)
    
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    
    return train_df, test_df

def create_feature_importance_plot():
    """Create feature importance analysis plot"""
    print("Creating feature importance plot...")
    
    train_df, test_df = load_and_prepare_ml_data()
    
    # Prepare features and target
    feature_cols = ['model_swh', 'lat', 'lon', 'month']
    X_train = train_df[feature_cols]
    y_train = train_df['obs_swh']
    X_test = test_df[feature_cols]
    y_test = test_df['obs_swh']
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Feature importance
    importance = rf.feature_importances_
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance bar plot
    ax1 = axes[0, 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(feature_cols, importance * 100, color=colors, alpha=0.8)
    ax1.set_ylabel('Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, imp in zip(bars, importance):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp*100:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Learning curve (training vs validation performance)
    ax2 = axes[0, 1]
    n_estimators_range = range(10, 101, 10)
    train_scores = []
    test_scores = []
    
    for n_est in n_estimators_range:
        rf_temp = RandomForestRegressor(n_estimators=n_est, max_depth=10, random_state=42)
        rf_temp.fit(X_train, y_train)
        
        train_pred = rf_temp.predict(X_train)
        test_pred = rf_temp.predict(X_test)
        
        train_scores.append(r2_score(y_train, train_pred))
        test_scores.append(r2_score(y_test, test_pred))
    
    ax2.plot(n_estimators_range, train_scores, 'b-o', label='Training R²', linewidth=2)
    ax2.plot(n_estimators_range, test_scores, 'r-s', label='Validation R²', linewidth=2)
    ax2.set_xlabel('Number of Estimators', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residual analysis
    ax3 = axes[1, 0]
    residuals_test = y_test - y_pred_test
    ax3.scatter(y_pred_test, residuals_test, alpha=0.5, s=1)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted SWH (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residuals (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Residual Analysis (Test Set)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Prediction accuracy by SWH range
    ax4 = axes[1, 1]
    
    # Bin predictions by SWH range
    swh_bins = np.arange(0, 6, 0.5)
    bin_centers = (swh_bins[:-1] + swh_bins[1:]) / 2
    
    rmse_by_bin = []
    bias_by_bin = []
    
    for i in range(len(swh_bins)-1):
        mask = (y_test >= swh_bins[i]) & (y_test < swh_bins[i+1])
        if np.sum(mask) > 10:  # Only if enough samples
            rmse_bin = np.sqrt(mean_squared_error(y_test[mask], y_pred_test[mask]))
            bias_bin = np.mean(y_pred_test[mask] - y_test[mask])
            rmse_by_bin.append(rmse_bin)
            bias_by_bin.append(bias_bin)
        else:
            rmse_by_bin.append(np.nan)
            bias_by_bin.append(np.nan)
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(bin_centers, rmse_by_bin, 'b-o', linewidth=2, label='RMSE')
    line2 = ax4_twin.plot(bin_centers, bias_by_bin, 'r-s', linewidth=2, label='Bias')
    
    ax4.set_xlabel('Observed SWH (m)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold', color='blue')
    ax4_twin.set_ylabel('Bias (m)', fontsize=12, fontweight='bold', color='red')
    ax4.set_title('Performance by SWH Range', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/ml_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_spatial_correction_patterns():
    """Create spatial patterns of correction effectiveness"""
    print("Creating spatial correction patterns...")
    
    train_df, test_df = load_and_prepare_ml_data()
    
    # Train model
    feature_cols = ['model_swh', 'lat', 'lon', 'month']
    X_train = train_df[feature_cols]
    y_train = train_df['obs_swh']
    X_test = test_df[feature_cols]
    y_test = test_df['obs_swh']
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)
    
    # Calculate spatial statistics
    test_df_copy = test_df.copy()
    test_df_copy['pred_swh'] = y_pred_test
    test_df_copy['raw_error'] = test_df_copy['model_swh'] - test_df_copy['obs_swh']
    test_df_copy['corrected_error'] = test_df_copy['pred_swh'] - test_df_copy['obs_swh']
    test_df_copy['improvement'] = np.abs(test_df_copy['raw_error']) - np.abs(test_df_copy['corrected_error'])
    
    # Group by spatial location
    spatial_stats = test_df_copy.groupby(['lat', 'lon']).agg({
        'raw_error': ['mean', 'std'],
        'corrected_error': ['mean', 'std'],
        'improvement': 'mean'
    }).reset_index()
    
    # Flatten column names
    spatial_stats.columns = ['lat', 'lon', 'raw_bias', 'raw_std', 'corr_bias', 'corr_std', 'improvement']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Raw bias
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(spatial_stats['lon'], spatial_stats['lat'], 
                          c=spatial_stats['raw_bias'], cmap='RdBu_r', 
                          s=50, alpha=0.7, vmin=-0.5, vmax=0.5)
    ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax1.set_title('Raw Model Bias (m)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1)
    
    # Corrected bias
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(spatial_stats['lon'], spatial_stats['lat'], 
                          c=spatial_stats['corr_bias'], cmap='RdBu_r', 
                          s=50, alpha=0.7, vmin=-0.5, vmax=0.5)
    ax2.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax2.set_title('ML Corrected Bias (m)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2)
    
    # Improvement map
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(spatial_stats['lon'], spatial_stats['lat'], 
                          c=spatial_stats['improvement'], cmap='RdYlGn', 
                          s=50, alpha=0.7, vmin=0, vmax=0.3)
    ax3.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Error Improvement (m)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3)
    
    # Improvement histogram
    ax4 = axes[1, 1]
    ax4.hist(spatial_stats['improvement'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=spatial_stats['improvement'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {spatial_stats["improvement"].mean():.3f}m')
    ax4.set_xlabel('Absolute Error Improvement (m)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Improvements', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/spatial_correction_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("Creating ML analysis plots...")
    
    create_feature_importance_plot()
    create_spatial_correction_patterns()
    
    print("ML analysis plots created successfully!")
    print("Generated plots:")
    print("- plots/ml_analysis.png")
    print("- plots/spatial_correction_patterns.png")

if __name__ == "__main__":
    main()