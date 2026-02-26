# Satellite-Guided Bias Correction of CMIP6 Sea Level Anomaly (SLA)

## 1. Overview
This report documents the application of a machine learning-based bias correction framework to CMIP6 Sea Level Anomaly (SLA) projections. Using a Random Forest regression model guided by satellite altimetry observations, we aimed to reduce systematic biases in climate model outputs for the period 2015–2020.

## 2. Datasets

### Observational Data (Reference)
- **Source:** Satellite Altimetry (L4 Gridded SLA).
- **Location:** `altimeter_sla/`
- **Format:** NetCDF (`dt_global_twosat_phy_l4_*.nc`).
- **Temporal Resolution:** Monthly means derived from daily data.
- **Variable:** Sea Level Anomaly (SLA).

### Model Data (Target for Correction)
- **Source:** CMIP6 (MPI-ESM1-2-LR).
- **Scenario:** SSP2-4.5 (`ssp245`).
- **Location:** `sla_2/zos_Omon_MPI-ESM1-2-LR_ssp245_r1i1p1f1_gn_20150116-20251216.nc`.
- **Variable:** Sea Surface Height Above Geoid (`zos`).
- **Spatial Extent:** High-latitude region (**63°N – 79°N**, **50°E – 108°E**).

## 3. Methodology

### 3.1 Preprocessing & Alignment
1.  **Region of Interest (ROI):** The analysis was constrained to **Lat 60°N–80°N, Lon 50°E–110°E** to match the spatial extent of the provided model file.
2.  **Variable Transformation:**
    - The model provides absolute Sea Surface Height (`zos`).
    - To make it comparable to satellite SLA, we converted `zos` to an anomaly by subtracting the mean of the training period (**2015–2018**).
    - Formula: $SLA_{model}(t) = ZOS_{model}(t) - \overline{ZOS}_{model(2015-2018)}$
3.  **Regridding:**
    - The model uses a curvilinear grid. We employed a **cKDTree (Nearest Neighbor)** approach to interpolate the model data onto the rectilinear grid of the satellite observations.
4.  **Temporal Matching:**
    - Both datasets were resampled to monthly means (`1MS`) and aligned for the overlapping period **Jan 2015 – Dec 2020**.

### 3.2 Machine Learning Framework
- **Model:** Random Forest Regressor (Ensemble of 50 decision trees).
- **Features (Inputs):**
    1.  `model_sla`: Raw model anomaly.
    2.  `lat`: Latitude coordinates (spatial context).
    3.  `lon`: Longitude coordinates (spatial context).
    4.  `month`: Month of the year (1–12) to capture seasonal bias patterns.
- **Target (Output):** `obs_sla` (Satellite observation).
- **Data Splitting:**
    - **Training Set:** 2015 – 2018 (Approx. 67% of data).
    - **Test Set:** 2019 – 2020 (Approx. 33% of data, independent validation).

## 4. Results

The model was evaluated on the independent test period (2019–2020).

### 4.1 Quantitative Metrics
The Random Forest correction significantly improved the agreement between the model and observations.

| Metric | Raw CMIP6 Model | ML Corrected | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE** | **0.1037 m** | **0.0578 m** | **44.3%** |

### 4.2 Visual Analysis
Plots have been generated in the `sla_plots/` directory to visualize the performance.

1.  **Scatter Plot (`scatter_sla.png`)**
    - Shows the relationship between Model and Observation before and after correction.
    - **Observation:** The corrected values (orange) cluster much more tightly along the 1:1 diagonal compared to the raw model values (blue), indicating a reduction in variance and systematic error.

2.  **Time Series (`time_series_sla.png`)**
    - Displays the regionally averaged SLA time series.
    - **Observation:** The "ML Corrected" line (red dashed) follows the "Satellite Obs" (black) trajectory much more closely than the "Raw Model" (blue), effectively correcting the amplitude and phase differences.

3.  **Bias Maps (`bias_maps_sla.png`)**
    - Spatial distribution of errors for the test period (2019-2020).
    - **Observation:** The "ML Corrected Bias" map shows significantly lighter colors (near zero) compared to the "Raw Model Bias," confirming that the correction is spatially effective across the domain.

## 5. Conclusion
The satellite-guided Random Forest approach successfully reduced the Root Mean Square Error (RMSE) of the CMIP6 SLA projections by **44.3%** in the study region. By learning state-dependent corrections based on model state, location, and seasonality, the framework effectively mitigates systematic biases inherent in the climate model output.

## 6. Technical Implementation
- **Script:** `analyze_sla.py`
- **Libraries:** `xarray`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`.
- **Environment:** Python 3.13 (Virtual Environment).
