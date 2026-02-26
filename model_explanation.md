# AI/ML Correction Model Details

**Date:** November 27, 2025
**Model Type:** Random Forest Regressor (Ensemble Learning)

## Data Flow / Workflow (SWH Bias Correction)
- Inputs: CMIP6 proxy SWH (`cowcliphs.nc`), satellite altimeter SWH (`io-altimeter.nc`).
- Preprocess: rename coords, slice to common 2015–2020 window, monthly resample, align timestamps.
- Regrid: interpolate model grid to observation grid.
- Feature build: flatten to table with `model_swh`, `lat`, `lon`, `month`; target `obs_swh`; drop NaNs.
- Split: Train on 2015–2018; test on 2019–2020.
- Train: RandomForestRegressor (50 trees, depth 10) to predict corrected SWH.
- Evaluate: compare original vs corrected (RMSE, bias, R²); generate plots.
- Persist/apply: save `swh_correction_model.pkl`; use for future CMIP6 scenarios with raw SWH + lat/lon/month → corrected SWH.

```mermaid
flowchart TD
    A[Inputs\nCMIP6 SWH (cowcliphs.nc)\nAltimeter SWH (io-altimeter.nc)]
    B[Preprocess & Align\nRename coords\n2015-2020 overlap\nMonthly resample]
    C[Regrid\nInterp model -> obs grid]
    D[Diagnostics\nBias / RMSE / Corr\nMaps & time series]
    E[Feature Prep\nFlatten to table\nmodel_swh, lat, lon, month\nTarget: obs_swh\nDrop NaNs]
    F[Split\nTrain: 2015-2018\nTest: 2019-2020]
    G[Train RF Regressor\n50 trees\nDepth 10]
    H[Evaluate\nRMSE, Bias, R²\nScatter/performance plots]
    I[Persist Model\nswh_correction_model.pkl]
    J[Apply to Future CMIP6\nRaw SWH + lat/lon/month\n→ corrected SWH]

    A --> B --> C --> D
    C --> E --> F --> G --> H --> I --> J
```

## 1. Data Overview
The model was trained to correct biases in CMIP6 Significant Wave Height (SWH) projections using high-fidelity satellite altimeter observations.

*   **Total Data Points Processed:** 151,200 (Initial Grid Points)
*   **Valid Data Points (Non-NaN):** 85,796 (After removing land masks and missing values)

### Data Split Strategy
The data was split temporally to simulate a "future" correction scenario:
*   **Training Set (Historic):** January 2015 – December 2018
    *   **Count:** 57,228 data points (67% of valid data)
    *   *Purpose:* Used to learn the bias patterns.
*   **Testing Set (Future/Unseen):** January 2019 – December 2020
    *   **Count:** 28,568 data points (33% of valid data)
    *   *Purpose:* Used to evaluate how well the model corrects "future" data it hasn't seen before.

## 2. Model Features (Inputs)
The model uses **4 features** to predict the corrected wave height. These features were chosen to capture both the systematic model errors and spatiotemporal variability.

| Feature Name | Description | Rationale |
| :--- | :--- | :--- |
| **`model_swh`** | Raw SWH from CMIP6 | The primary baseline prediction that needs correction. |
| **`lat`** | Latitude | Captures spatial biases (e.g., model errors varying by region). |
| **`lon`** | Longitude | Captures spatial biases (e.g., coastal vs. open ocean differences). |
| **`month`** | Month (1-12) | Captures seasonal biases (e.g., model errors specific to monsoon seasons). |

**Target Variable (Output):**
*   **`obs_swh`**: The observed Significant Wave Height from Satellite Altimeters (Ground Truth).

## 3. Model Architecture & Hyperparameters
We utilized a **Random Forest Regressor** from the `scikit-learn` library.

*   **Algorithm:** Random Forest (an ensemble of Decision Trees).
*   **Number of Trees (`n_estimators`):** 50
    *   *Effect:* Averages the predictions of 50 independent trees to reduce overfitting and variance.
*   **Maximum Depth (`max_depth`):** 10
    *   *Effect:* Limits how deep each tree can grow. A depth of 10 allows capturing complex non-linear patterns without memorizing the noise in the training data.
*   **Random State:** 42 (Ensures reproducibility).

## 4. Performance & Results (on Test Set)
The model was evaluated on the unseen data from 2019-2020.

| Metric | Original CMIP6 Data | Corrected (AI/ML) Data | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE (Root Mean Square Error)** | 0.6103 m | **0.4080 m** | **33.2% Reduction** |
| **Bias (Mean Error)** | +0.1788 m | **-0.0130 m** | **~93% Reduction** |
| **R² Score** | N/A | **0.8798** | High predictive accuracy |

### Interpretation
*   **Bias Correction:** The original model consistently overestimated wave height by ~18cm. The AI/ML model removed this almost entirely (bias $\approx$ -1cm).
*   **Accuracy:** The spread of errors (RMSE) was reduced by one-third, meaning the corrected projections are significantly closer to reality.

