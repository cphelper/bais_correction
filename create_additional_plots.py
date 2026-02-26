#!/usr/bin/env python3
"""
Create additional plots and visualizations for the ISRO wave height bias correction paper.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    _HAS_CARTOPY = False
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def _load_rf_model():
    """Load trained RF model if present; return None otherwise."""
    import joblib
    try:
        return joblib.load('swh_correction_model.pkl')
    except Exception:
        return None

def _build_flat_table(model_ds, obs_ds):
    """
    Vectorized flattening into a DataFrame with time metadata.
    Returns df with columns: model_swh, obs_swh, lat, lon, month, year.
    """
    m = model_ds.swh.values.reshape(-1)
    o = obs_ds.swh.values.reshape(-1)

    times = model_ds.time.values
    lats = model_ds.lat.values
    lons = model_ds.lon.values

    T, Lat, Lon = np.meshgrid(times, lats, lons, indexing='ij')
    t_flat = T.reshape(-1)

    df = pd.DataFrame({
        'model_swh': m,
        'obs_swh': o,
        'lat': Lat.reshape(-1),
        'lon': Lon.reshape(-1),
        'month': pd.to_datetime(t_flat).month,
        'year': pd.to_datetime(t_flat).year,
    })
    df = df.dropna()
    return df

def load_and_process_data():
    """Load and process the datasets"""
    print("Loading datasets...")
    
    # Load model data (CMIP6)
    model_ds = xr.open_dataset('cowcliphs.nc')
    model_ds = model_ds.rename({
        'TIME': 'time',
        'LATITUDE41_321': 'lat', 
        'LONGITUDE101_221': 'lon',
        'HS': 'swh'
    })
    
    # Load observational data (satellite altimeter)
    obs_ds = xr.open_dataset('io-altimeter.nc')
    obs_ds = obs_ds.rename({
        'TIME': 'time',
        'LATITUDE11_80': 'lat',
        'LONGITUDE116_145': 'lon', 
        'VAVH_DAILY_MEAN': 'swh'
    })
    
    # Restrict to common time period (2015-2020)
    start_date = '2015-01-01'
    end_date = '2020-12-31'
    
    model_ds = model_ds.sel(time=slice(start_date, end_date))
    obs_ds = obs_ds.sel(time=slice(start_date, end_date))
    
    # Resample to monthly means
    model_monthly = model_ds.resample(time='MS').mean()
    obs_monthly = obs_ds.resample(time='MS').mean()
    
    # Interpolate model to observation grid
    model_regrid = model_monthly.interp_like(obs_monthly)
    
    return model_regrid, obs_monthly

def create_seasonal_analysis_plot(model_ds, obs_ds):
    """Create seasonal analysis plots"""
    print("Creating seasonal analysis plot...")
    if not _HAS_CARTOPY:
        print("Skipping seasonal_analysis.png (cartopy not available).")
        return
    
    # Calculate seasonal means
    model_seasonal = model_ds.groupby('time.season').mean()
    obs_seasonal = obs_ds.groupby('time.season').mean()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    for i, season in enumerate(seasons):
        # Model data
        ax1 = axes[0, i]
        im1 = model_seasonal.swh.sel(season=season).plot(
            ax=ax1, transform=ccrs.PlateCarree(),
            cmap='viridis', add_colorbar=False,
            vmin=0, vmax=4
        )
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.BORDERS)
        ax1.set_title(f'CMIP6 SWH - {season}', fontsize=12, fontweight='bold')
        ax1.set_global()
        
        # Observation data
        ax2 = axes[1, i]
        im2 = obs_seasonal.swh.sel(season=season).plot(
            ax=ax2, transform=ccrs.PlateCarree(),
            cmap='viridis', add_colorbar=False,
            vmin=0, vmax=4
        )
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS)
        ax2.set_title(f'Satellite SWH - {season}', fontsize=12, fontweight='bold')
        ax2.set_global()
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=axes, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=40)
    cbar.set_label('Significant Wave Height (m)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Seasonal Significant Wave Height Patterns (2015-2020)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('plots/seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_monthly_climatology_plot(model_ds, obs_ds):
    """Create monthly climatology comparison"""
    print("Creating monthly climatology plot...")
    
    # Calculate monthly climatology
    model_clim = model_ds.groupby('time.month').mean()
    obs_clim = obs_ds.groupby('time.month').mean()
    
    # Calculate regional averages
    model_regional = model_clim.swh.mean(dim=['lat', 'lon'])
    obs_regional = obs_clim.swh.mean(dim=['lat', 'lon'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Monthly climatology line plot
    months = np.arange(1, 13)
    ax1.plot(months, model_regional.values, 'b-o', linewidth=2, 
             markersize=8, label='CMIP6 Model', alpha=0.8)
    ax1.plot(months, obs_regional.values, 'r-s', linewidth=2, 
             markersize=8, label='Satellite Obs', alpha=0.8)
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean SWH (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Monthly Climatology (Regional Average)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(months)
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Bias by month
    bias_monthly = model_regional - obs_regional
    ax2.bar(months, bias_monthly.values, color='coral', alpha=0.7, edgecolor='darkred')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bias (Model - Obs) [m]', fontsize=12, fontweight='bold')
    ax2.set_title('Monthly Bias Pattern', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(months)
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout()
    plt.savefig('plots/monthly_climatology.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_distribution_plot(model_ds, obs_ds):
    """Create statistical distribution comparison plots"""
    print("Creating statistical distribution plots...")

    rf_model = _load_rf_model()
    df = _build_flat_table(model_ds, obs_ds)

    # Focus on the independent test period for corrected distributions
    df_test = df[df['year'] >= 2019].copy()

    model_clean = df_test['model_swh'].values
    obs_clean = df_test['obs_swh'].values

    corrected_clean = None
    if rf_model is not None and len(df_test) > 0:
        X = df_test[['model_swh', 'lat', 'lon', 'month']].values
        corrected_clean = rf_model.predict(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(obs_clean, bins=50, alpha=0.55, label='Satellite (test)', 
             color='#d62728', density=True)
    ax1.hist(model_clean, bins=50, alpha=0.45, label='Raw CMIP6 (test)', 
             color='#1f77b4', density=True)
    if corrected_clean is not None:
        ax1.hist(corrected_clean, bins=50, alpha=0.35, label='ML corrected (test)', 
                 color='#2ca02c', density=True)
    ax1.set_xlabel('SWH (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('SWH Distribution (2019--2020 test period)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = [obs_clean, model_clean] if corrected_clean is None else [obs_clean, model_clean, corrected_clean]
    labels = ['Satellite', 'Raw CMIP6'] if corrected_clean is None else ['Satellite', 'Raw CMIP6', 'ML corrected']
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
    ax2.set_ylabel('SWH (m)', fontsize=12, fontweight='bold')
    ax2.set_title('SWH Distribution Statistics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax3 = axes[1, 0]
    # Q-Q: compare quantiles against satellite for raw and corrected
    percentiles = np.arange(1, 100, 1)
    q_obs = np.percentile(obs_clean, percentiles)
    q_raw = np.percentile(model_clean, percentiles)
    ax3.plot(q_obs, q_raw, color='#1f77b4', linewidth=2, label='Raw CMIP6')
    if corrected_clean is not None:
        q_corr = np.percentile(corrected_clean, percentiles)
        ax3.plot(q_obs, q_corr, color='#2ca02c', linewidth=2, label='ML corrected')
    min_val = float(min(q_obs.min(), q_raw.min()))
    max_val = float(max(q_obs.max(), q_raw.max()))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1')
    ax3.set_xlabel('Satellite quantiles (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Model quantiles (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Q--Q Diagnostics (test period)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Percentile comparison
    ax4 = axes[1, 1]
    percentiles2 = np.arange(5, 100, 5)
    model_percentiles = np.percentile(model_clean, percentiles2)
    obs_percentiles = np.percentile(obs_clean, percentiles2)
    ax4.plot(percentiles2, obs_percentiles, color='#d62728', marker='s', label='Satellite', markersize=4)
    ax4.plot(percentiles2, model_percentiles, color='#1f77b4', marker='o', label='Raw CMIP6', markersize=4)
    if corrected_clean is not None:
        corr_percentiles = np.percentile(corrected_clean, percentiles2)
        ax4.plot(percentiles2, corr_percentiles, color='#2ca02c', marker='^', label='ML corrected', markersize=4)
    ax4.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax4.set_ylabel('SWH (m)', fontsize=12, fontweight='bold')
    ax4.set_title('Percentile Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/statistical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_tail_performance_plot(model_ds, obs_ds, quantile=0.95):
    """Create defensible upper-tail error comparison plot (test period only)."""
    print("Creating tail performance plot...")

    rf_model = _load_rf_model()
    if rf_model is None:
        print("Skipping tail plot: swh_correction_model.pkl not found.")
        return

    df = _build_flat_table(model_ds, obs_ds)
    df_test = df[df['year'] >= 2019].copy()
    if len(df_test) == 0:
        print("Skipping tail plot: empty test set.")
        return

    X = df_test[['model_swh', 'lat', 'lon', 'month']].values
    y_obs = df_test['obs_swh'].values
    y_raw = df_test['model_swh'].values
    y_corr = rf_model.predict(X)

    thr = np.quantile(y_obs, float(quantile))
    m = y_obs >= thr
    if m.sum() < 50:
        print("Skipping tail plot: too few tail samples.")
        return

    err_raw = y_raw[m] - y_obs[m]
    err_corr = y_corr[m] - y_obs[m]

    # Summary stats for annotation
    rmse_raw = np.sqrt(np.mean((err_raw) ** 2))
    rmse_corr = np.sqrt(np.mean((err_corr) ** 2))
    bias_raw = np.mean(err_raw)
    bias_corr = np.mean(err_corr)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.2))
    parts = ax.violinplot([err_raw, err_corr], showmeans=True, showextrema=False)
    # Color violins
    colors = ['#1f77b4', '#2ca02c']
    for i, b in enumerate(parts['bodies']):
        b.set_facecolor(colors[i])
        b.set_edgecolor('black')
        b.set_alpha(0.55)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)

    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Raw CMIP6', 'ML corrected'])
    ax.set_ylabel('Error (prediction - satellite) [m]')
    qlab = int(round(quantile * 100))
    ax.set_title(f'Upper-tail error distribution ({qlab}th percentile, 2019--2020)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.text(
        0.02, 0.98,
        f'Threshold: {thr:.2f} m\n'
        f'Raw: RMSE={rmse_raw:.3f} m, Bias={bias_raw:+.3f} m\n'
        f'Corr: RMSE={rmse_corr:.3f} m, Bias={bias_corr:+.3f} m',
        transform=ax.transAxes,
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig('plots/tail_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_analysis_plot(model_ds, obs_ds):
    """Create detailed error analysis plots"""
    print("Creating error analysis plots...")
    if not _HAS_CARTOPY:
        print("Skipping error_analysis.png (cartopy not available).")
        return
    
    # Calculate errors
    bias = model_ds.swh - obs_ds.swh
    abs_error = np.abs(bias)
    rel_error = bias / obs_ds.swh * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Mean absolute error
    ax1 = axes[0, 0]
    mae_map = abs_error.mean(dim='time')
    im1 = mae_map.plot(ax=ax1, transform=ccrs.PlateCarree(),
                       cmap='Reds', add_colorbar=False)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_title('Mean Absolute Error (m)', fontsize=14, fontweight='bold')
    ax1.set_global()
    
    # Standard deviation of errors
    ax2 = axes[0, 1]
    std_map = bias.std(dim='time')
    im2 = std_map.plot(ax=ax2, transform=ccrs.PlateCarree(),
                       cmap='Oranges', add_colorbar=False)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_title('Error Standard Deviation (m)', fontsize=14, fontweight='bold')
    ax2.set_global()
    
    # Relative error
    ax3 = axes[1, 0]
    rel_error_mean = rel_error.mean(dim='time')
    im3 = rel_error_mean.plot(ax=ax3, transform=ccrs.PlateCarree(),
                              cmap='RdBu_r', add_colorbar=False,
                              vmin=-50, vmax=50)
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS)
    ax3.set_title('Mean Relative Error (%)', fontsize=14, fontweight='bold')
    ax3.set_global()
    
    # Correlation map
    ax4 = axes[1, 1]
    correlation = xr.corr(model_ds.swh, obs_ds.swh, dim='time')
    im4 = correlation.plot(ax=ax4, transform=ccrs.PlateCarree(),
                           cmap='RdYlBu_r', add_colorbar=False,
                           vmin=0, vmax=1)
    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS)
    ax4.set_title('Temporal Correlation', fontsize=14, fontweight='bold')
    ax4.set_global()
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0, 0], orientation='horizontal', pad=0.05, shrink=0.8)
    plt.colorbar(im2, ax=axes[0, 1], orientation='horizontal', pad=0.05, shrink=0.8)
    plt.colorbar(im3, ax=axes[1, 0], orientation='horizontal', pad=0.05, shrink=0.8)
    plt.colorbar(im4, ax=axes[1, 1], orientation='horizontal', pad=0.05, shrink=0.8)
    
    plt.suptitle('Comprehensive Error Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('plots/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_regional_analysis_plot(model_ds, obs_ds):
    """Create regional analysis for different ocean basins"""
    print("Creating regional analysis plot...")
    
    # Define regions
    regions = {
        'Arabian Sea': {'lat': slice(10, 25), 'lon': slice(60, 75)},
        'Bay of Bengal': {'lat': slice(5, 22), 'lon': slice(80, 95)},
        'Southern Ocean': {'lat': slice(-60, -40), 'lon': slice(50, 110)},
        'Equatorial Indian': {'lat': slice(-10, 10), 'lon': slice(50, 110)}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (region_name, bounds) in enumerate(regions.items()):
        ax = axes[i]
        
        # Extract regional data
        model_region = model_ds.swh.sel(lat=bounds['lat'], lon=bounds['lon'])
        obs_region = obs_ds.swh.sel(lat=bounds['lat'], lon=bounds['lon'])
        
        # Calculate regional means
        model_ts = model_region.mean(dim=['lat', 'lon'])
        obs_ts = obs_region.mean(dim=['lat', 'lon'])
        
        # Plot time series
        ax.plot(model_ts.time, model_ts.values, 'b-', linewidth=2, 
                label='CMIP6', alpha=0.8)
        ax.plot(obs_ts.time, obs_ts.values, 'r-', linewidth=2, 
                label='Satellite', alpha=0.8)
        
        ax.set_title(f'{region_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('SWH (m)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display statistics
        bias = float(model_ts.mean() - obs_ts.mean())
        rmse = float(np.sqrt(((model_ts - obs_ts)**2).mean()))
        corr = float(xr.corr(model_ts, obs_ts))
        
        stats_text = f'Bias: {bias:.3f}m\nRMSE: {rmse:.3f}m\nCorr: {corr:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.suptitle('Regional Time Series Analysis (2015-2020)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_performance_summary():
    """Create a comprehensive model performance summary plot"""
    print("Creating model performance summary...")
    
    # Performance metrics (from the analysis)
    metrics_data = {
        'Metric': ['RMSE (m)', 'Bias (m)', 'R² Score', 'MAE (m)', 'Correlation'],
        'Raw CMIP6': [0.610, 0.179, 0.692, 0.485, 0.832],
        'ML Corrected': [0.408, -0.013, 0.880, 0.312, 0.938],
        'Improvement (%)': [33.2, 92.7, 27.2, 35.7, 12.7]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance comparison bar plot
    x = np.arange(len(metrics_data['Metric']))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, metrics_data['Raw CMIP6'], width, 
                    label='Raw CMIP6', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, metrics_data['ML Corrected'], width,
                    label='ML Corrected', color='lightblue', alpha=0.8)
    
    ax1.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_data['Metric'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Improvement percentage plot
    colors = ['green' if x > 0 else 'red' for x in metrics_data['Improvement (%)']]
    bars3 = ax2.bar(metrics_data['Metric'], metrics_data['Improvement (%)'], 
                    color=colors, alpha=0.7)
    ax2.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(metrics_data['Metric'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to create all plots"""
    print("Starting plot generation...")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    model_ds, obs_ds = load_and_process_data()
    
    # Create plots used in the paper (cartopy-only plots are optional)
    create_statistical_distribution_plot(model_ds, obs_ds)
    create_tail_performance_plot(model_ds, obs_ds)
    create_model_performance_summary()

    # Optional extra plots (only if cartopy is available)
    create_seasonal_analysis_plot(model_ds, obs_ds)
    create_monthly_climatology_plot(model_ds, obs_ds)
    create_error_analysis_plot(model_ds, obs_ds)
    create_regional_analysis_plot(model_ds, obs_ds)
    
    print("All plots created successfully!")
    print("Generated plots:")
    print("- plots/seasonal_analysis.png")
    print("- plots/monthly_climatology.png") 
    print("- plots/statistical_distributions.png")
    print("- plots/error_analysis.png")
    print("- plots/regional_analysis.png")
    print("- plots/performance_summary.png")

if __name__ == "__main__":
    main()