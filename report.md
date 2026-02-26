# Research Report: Utilizing Satellite-Based Observations to Correct CMIP6 Climate Projections

**Date:** November 27, 2025

## 1. Introduction
This report addresses the objectives of the research proposal:
1.  Quantifying uncertainties in CMIP6 projections of significant wave height (SWH).
2.  Developing an AI/ML-based model to correct these uncertainties.
3.  Demonstrating the correction on future/unseen data.

## 2. Methodology

### 2.1 Datasets
Two datasets were analyzed:
1.  **Model Data (CMIP6 Proxy):** `cowcliphs.nc` (SWH, ~0.5° grid).
2.  **Observational Data (Satellite Altimeter):** `io-altimeter.nc` (Daily Mean SWH, 2.0° grid).

**Period Analyzed:** January 2015 – December 2020.

### 2.2 Uncertainty Quantification (Objective A)
*   **Preprocessing:** Model data was regridded to the 2.0° observational grid. Both datasets were aligned monthly.
*   **Metrics:** Bias, RMSE, and Temporal Correlation.

### 2.3 AI/ML Correction Model (Objective B)
*   **Model Architecture:** Random Forest Regressor (Ensemble Learning).
    *   *Rationale:* Selected for its ability to handle non-linear relationships and spatial variability without complex stationarity assumptions.
*   **Features (Inputs):** 
    *   CMIP6 Model SWH
    *   Latitude & Longitude (to capture spatial bias patterns)
    *   Month (to capture seasonal bias patterns)
*   **Target (Output):** Satellite Observed SWH.
*   **Training/Testing Strategy:**
    *   **Training Set:** 2015 – 2018 (4 years).
    *   **Testing/Validation Set:** 2019 – 2020 (2 years, simulating "future" unseen data).

## 3. Results

### 3.1 Uncertainty in Raw Projections (2015-2020)
The overall comparison over the Indian Ocean region yielded:
*   **Mean Bias:** +0.1349 m (Model overestimates wave height).
*   **Mean RMSE:** 0.5142 m.
*   **Correlation:** 0.6916 (Moderate agreement in trends).

### 3.2 AI/ML Model Performance (Correction Capability)
The AI/ML model was trained on 2015-2018 data and applied to the 2019-2020 "future" simulation period.

| Metric | Original CMIP6 (Test Period) | **Corrected (AI/ML Output)** | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE** | 0.6103 m | **0.4080 m** | **33.2% Reduction** |
| **Bias** | +0.1788 m | **-0.0130 m** | **~93% Reduction** |
| **R² Score** | N/A | **0.8798** | High fidelity |

*   **Interpretation:** The AI/ML model successfully learned the bias patterns. It reduced the systematic error (bias) to nearly zero and significantly improved the accuracy (RMSE) of the projections on unseen data.

## 4. Conclusion & Deliverables
1.  **Quantification:** CMIP6 models in this region exhibit a systematic positive bias (~13-18cm) and moderate random error.
2.  **Correction Model:** A Random Forest-based correction model has been developed and validated. It effectively removes the bias and sharpens the projections.
3.  **Future Application:** This trained model (`swh_correction_model.pkl`) can now be applied to any future CMIP6 scenario (e.g., SSP2-4.5, SSP5-8.5) up to 2100. By inputting the raw future projection along with location and month, the model will output the "bias-corrected" wave height, providing more reliable data for coastal planning.

## 5. Visualizations
Generated plots are available in the `plots/` directory:
*   `time_series.png`: Visual comparison of raw model vs observations.
*   `bias_map.png`: Spatial distribution of errors.
*   `correction_performance.png`: Scatter plots showing the tightening of predictions after AI/ML correction.

## 6. DeepAR Rolling Forecast Runs

### 6.1 IO Altimeter SWH (io-altimeter.nc)
*   **Dataset cadence:** Monthly series of daily-mean SWH (TIME spacing mostly 29–31 days).
*   **Time range:** 2002-01-23 to 2020-12-16 (228 monthly points).
*   **Spatial Coverage:** 51.0°E to 109.0°E, -69.0°S to 69.0°N.
*   **Duration:** ~19 years (6902 days).
*   **Target series:** Area-weighted mean of `VAVH_DAILY_MEAN` over lat/lon.
*   **Model:** DeepAR-style LSTM with Gaussian likelihood.
*   **Hyperparameters:** context_length=24, epochs=40, batch_size=32, lr=1e-3, hidden_size=40, num_layers=2, dropout=0.1.
*   **Train/Test split:** 182 train (79.82%), 46 test (20.18%).
*   **Rolling forecast horizon:** 1-step ahead across full series after context window (204 forecasts).
*   **Outputs:** `forecast_mean`, `forecast_p10`, `forecast_p50`, `forecast_p90`, `forecast_sigma`, `observed`.
*   **Metrics (test period):** RMSE=0.13524, MAE=0.10128, P10–P90 coverage=0.69565, interval width=0.29359.
*   **Output file:** `io_altimeter_deepar_rolling_forecast.nc`.
*   **Run command:**
    ```
    .venv/bin/python /Users/shyam/Downloads/isro/deepar_rolling_forecast.py
    ```

### 6.2 IO SLA Altimeter (altimeter_sla/*.nc)
*   **Dataset cadence:** Monthly mean SLA, one time step per file.
*   **Time range:** 2015-01-15 to 2023-12-15 (108 monthly points).
*   **Target series:** Area-weighted mean of `sla` over lat/lon.
*   **Model:** DeepAR-style LSTM with Gaussian likelihood.
*   **Hyperparameters:** context_length=24, epochs=40, batch_size=32, lr=1e-3, hidden_size=40, num_layers=2, dropout=0.1.
*   **Train/Test split:** 86 train (79.63%), 22 test (20.37%).
*   **Rolling forecast horizon:** 1-step ahead across full series after context window (84 forecasts).
*   **Outputs:** `forecast_mean`, `forecast_p10`, `forecast_p50`, `forecast_p90`, `forecast_sigma`, `observed`.
*   **Metrics (test period):** RMSE=0.01123, MAE=0.00889, P10–P90 coverage=0.45455, interval width=0.01302.
*   **Output file:** `io_sla_altimeter_deepar_rolling_forecast.nc`.
*   **Run command:**
    ```
    .venv/bin/python /Users/shyam/Downloads/isro/deepar_rolling_forecast.py --sla-glob "/Users/shyam/Downloads/isro/altimeter_sla/*.nc" --context-length 24 --output /Users/shyam/Downloads/isro/io_sla_altimeter_deepar_rolling_forecast.nc
    ```
