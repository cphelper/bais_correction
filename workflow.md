# SWH Correction Workflow

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


