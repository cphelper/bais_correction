"""
Quantile Prediction Model for SLA using Tube Loss.

Tube Loss from: https://github.com/ltpritamanand/tube_loss
Paper: "Tube Loss: A Novel Approach for Prediction Interval Estimation
        and Probabilistic Forecasting" (arXiv:2412.06853)

Trains a neural network on satellite altimeter SLA data with tube loss
to produce prediction intervals (upper/lower quantile bounds) for
CMIP6 Sea Level Anomaly projections.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import joblib
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# Tube Loss (PyTorch) — ported from ltpritamanand/tube_loss
# ──────────────────────────────────────────────────────────────

class TubeLoss(nn.Module):
    """
    Tube Loss for prediction interval estimation.

    The network outputs two values per sample: f1 (upper bound) and
    f2 (lower bound).  The loss encourages the true value to fall
    inside [f2, f1] with coverage q, while penalising interval width
    via delta.  Parameter r controls vertical positioning of the tube
    to capture denser regions of skewed distributions.

    Reference: https://github.com/ltpritamanand/tube_loss
    """

    def __init__(self, q=0.95, r=0.5, delta=0.001):
        super().__init__()
        self.q = q
        self.r = r
        self.delta = delta

    def forward(self, y_pred, y_true):
        f1 = y_pred[:, 0]  # upper bound
        f2 = y_pred[:, 1]  # lower bound

        c1 = (1 - self.q) * (y_true - f2)
        c2 = (1 - self.q) * (f1 - y_true)
        c3 = self.q * (f2 - y_true)
        c4 = self.q * (y_true - f1)

        inside = (y_true <= f1) & (y_true >= f2)
        above_mid = y_true > (self.r * f1 + (1 - self.r) * f2)

        loss_inside = torch.where(above_mid, c1, c2)
        loss_outside = torch.where(f2 > y_true, c3, c4)
        loss = torch.where(inside, loss_inside, loss_outside)

        width_penalty = self.delta * torch.abs(f1 - f2)
        return (loss + width_penalty).mean()


# ──────────────────────────────────────────────────────────────
# Neural Network
# ──────────────────────────────────────────────────────────────

class QuantileNet(nn.Module):
    """
    Network that outputs upper/lower bounds via base + positive offsets.

    Architecture: shared trunk → base prediction (center)
                                → log_half_width (always > 0 via softplus)
                                → asymmetry    (shift within interval)

    f1 = base + softplus(log_hw) * (1 + sigmoid(asym))
    f2 = base - softplus(log_hw) * (2 - sigmoid(asym) - 1)

    This guarantees f1 >= f2 and provides smooth gradients for
    interval widening.
    """

    def __init__(self, n_features, hidden_sizes=(128, 64, 32)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head_base = nn.Linear(in_dim, 1)
        self.head_hw = nn.Linear(in_dim, 1)
        self.head_asym = nn.Linear(in_dim, 1)

        nn.init.constant_(self.head_hw.bias, 2.0)

    def forward(self, x):
        h = self.trunk(x)
        base = self.head_base(h).squeeze(-1)
        half_w = nn.functional.softplus(self.head_hw(h).squeeze(-1))
        asym = torch.sigmoid(self.head_asym(h).squeeze(-1))

        upper = base + half_w * (1.0 + asym)
        lower = base - half_w * (2.0 - asym)

        return torch.stack([upper, lower], dim=1)


# ──────────────────────────────────────────────────────────────
# Data loading (same pipeline as train_sla_model.py)
# ──────────────────────────────────────────────────────────────

def load_sla_data():
    sla_dir = "./altimeter_sla"
    model_path = "./sla_2/zos_Omon_MPI-ESM1-2-LR_ssp245_r1i1p1f1_gn_20150116-20251216.nc"
    lat_min, lat_max = 60, 80
    lon_min, lon_max = 50, 110

    print("[1/4] Loading satellite altimeter SLA...")
    sla_files = sorted(glob.glob(os.path.join(sla_dir, "*.nc")))
    sla_files = [
        f for f in sla_files
        if any(y in f for y in ["2015", "2016", "2017", "2018", "2019", "2020"])
    ]

    obs_datasets = []
    for f in sla_files:
        try:
            with xr.open_dataset(f) as ds:
                if "latitude" in ds.coords:
                    ds = ds.rename({"latitude": "lat"})
                if "longitude" in ds.coords:
                    ds = ds.rename({"longitude": "lon"})
                ds_sliced = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                if "time" not in ds_sliced.dims:
                    ds_sliced = ds_sliced.expand_dims("time")
                obs_datasets.append(ds_sliced.load())
        except Exception:
            pass

    obs_ds = xr.concat(obs_datasets, dim="time").sortby("time")
    obs_monthly = obs_ds["sla"].resample(time="1MS").mean()

    print("[2/4] Loading CMIP6 model (ZOS → SLA)...")
    model_ds = xr.open_dataset(model_path)
    model_ds = model_ds.sel(time=slice("2015-01-01", "2020-12-31"))
    model_monthly = model_ds["zos"].resample(time="1MS").mean()
    model_mean = model_monthly.sel(time=slice("2015-01-01", "2018-12-31")).mean(dim="time")
    model_sla = model_monthly - model_mean

    print("[3/4] Regridding model → obs grid...")
    mlat = model_ds["latitude"].values
    mlon = model_ds["longitude"].values
    points = np.column_stack((mlat.ravel(), mlon.ravel()))
    olat = obs_monthly.lat.values
    olon = obs_monthly.lon.values
    olat_grid, olon_grid = np.meshgrid(olat, olon, indexing="ij")
    target_points = np.column_stack((olat_grid.ravel(), olon_grid.ravel()))
    tree = cKDTree(points)
    _, indices = tree.query(target_points)

    common_times = sorted(set(model_sla.time.values) & set(obs_monthly.time.values))
    obs_aligned = obs_monthly.sel(time=common_times)
    model_aligned = model_sla.sel(time=common_times)

    regridded_vals = []
    for t in common_times:
        m_data = model_aligned.sel(time=t).values.ravel()
        regridded_vals.append(m_data[indices].reshape(len(olat), len(olon)))

    model_sla_regridded = xr.DataArray(
        np.array(regridded_vals),
        coords={"time": common_times, "lat": olat, "lon": olon},
        dims=["time", "lat", "lon"],
    )

    print("[4/4] Building feature table...")
    df_obs = obs_aligned.to_dataframe(name="obs").reset_index()
    df_model = model_sla_regridded.to_dataframe(name="model").reset_index()
    df = pd.merge(df_obs, df_model, on=["time", "lat", "lon"])
    df["month"] = df["time"].dt.month
    df = df.dropna()

    return df, common_times


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Quantile SLA Model with Tube Loss (ltpritamanand/tube_loss)")
    print("=" * 65)

    df, common_times = load_sla_data()
    features = ["model", "lat", "lon", "month"]

    train_df = df[df["time"].dt.year <= 2017].copy()
    cal_df = df[df["time"].dt.year == 2018].copy()
    test_df = df[df["time"].dt.year >= 2019].copy()
    print(f"\nTrain: {len(train_df)}  |  Cal: {len(cal_df)}  |  Test: {len(test_df)}")

    scaler_x = StandardScaler()

    X_train = scaler_x.fit_transform(train_df[features].values)
    y_train = train_df["obs"].values.astype(np.float32)
    X_cal = scaler_x.transform(cal_df[features].values)
    y_cal = cal_df["obs"].values.astype(np.float32)
    X_test = scaler_x.transform(test_df[features].values)
    y_test_raw = test_df["obs"].values

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_cal_t = torch.tensor(X_cal, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # ── Train for multiple quantile levels ──
    quantile_configs = [
        {"q": 0.90, "label": "90%", "r": 0.5, "delta": 0.0005},
        {"q": 0.95, "label": "95%", "r": 0.5, "delta": 0.0002},
    ]

    results = {}
    for cfg in quantile_configs:
        q, r, delta = cfg["q"], cfg["r"], cfg["delta"]
        label = cfg["label"]
        print(f"\n{'─' * 65}")
        print(f"  Training Tube Loss model — {label} PI  (q={q}, r={r}, δ={delta})")
        print(f"{'─' * 65}")

        model = QuantileNet(n_features=len(features))
        criterion = TubeLoss(q=q, r=r, delta=delta)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=15, factor=0.5
        )

        n_epochs = 800
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimiser.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)
            scheduler.step(epoch_loss)
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 60:
                print(f"    Early stop at epoch {epoch}")
                break
            if epoch % 100 == 0 or epoch == 1:
                print(f"    Epoch {epoch:>3d}/{n_epochs}  loss={epoch_loss:.6f}")

        model.eval()
        with torch.no_grad():
            preds_cal = model(X_cal_t).numpy()
            preds_test = model(X_test_t).numpy()

        # --- Raw test predictions ---
        upper_raw = preds_test[:, 0]
        lower_raw = preds_test[:, 1]
        upper_raw_final = np.maximum(upper_raw, lower_raw)
        lower_raw_final = np.minimum(upper_raw, lower_raw)
        median_pred = (upper_raw_final + lower_raw_final) / 2.0
        raw_cov = np.mean((y_test_raw >= lower_raw_final) & (y_test_raw <= upper_raw_final))

        # --- Conformal calibration on held-out 2018 data ---
        cal_upper = np.maximum(preds_cal[:, 0], preds_cal[:, 1])
        cal_lower = np.minimum(preds_cal[:, 0], preds_cal[:, 1])

        # Nonconformity: max(lower - y, y - upper). Positive = outside interval.
        scores = np.maximum(cal_lower - y_cal, y_cal - cal_upper)

        n_cal = len(scores)
        cal_idx = int(np.ceil((n_cal + 1) * q)) - 1
        cal_idx = min(cal_idx, n_cal - 1)
        sorted_scores = np.sort(scores)
        margin = sorted_scores[cal_idx]
        print(f"    Conformal margin: {margin:+.4f} m  (additive expansion)")

        upper_final = upper_raw_final + margin
        lower_final = lower_raw_final - margin

        coverage = np.mean((y_test_raw >= lower_final) & (y_test_raw <= upper_final))
        avg_width = np.mean(upper_final - lower_final)
        rmse_median = np.sqrt(mean_squared_error(y_test_raw, median_pred))

        results[label] = {
            "upper": upper_final,
            "lower": lower_final,
            "median": median_pred,
            "coverage": coverage,
            "width": avg_width,
            "rmse": rmse_median,
            "model": model,
            "q": q,
            "raw_coverage": raw_cov,
            "margin": margin,
        }

        print(f"    Raw coverage  : {raw_cov * 100:.1f}%")
        print(f"    Calibrated cov: {coverage * 100:.1f}%  (target {q * 100:.0f}%)")
        print(f"    Avg width     : {avg_width:.4f} m")
        print(f"    RMSE(mid)     : {rmse_median:.4f} m")

    # ── Save models ──
    os.makedirs("sla_plots", exist_ok=True)
    for label, res in results.items():
        tag = label.replace("%", "")
        torch.save(res["model"].state_dict(), f"sla_quantile_tube_{tag}.pt")
    joblib.dump({"scaler_x": scaler_x}, "sla_quantile_scalers.pkl")
    print("\nModels saved.")

    # ── Results Table ──
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY (Test Period 2019-2020)")
    print("=" * 65)
    print(f"  {'PI Level':<10} {'Raw Cov':>10} {'Cal Cov':>10} {'Target':>10} {'Avg Width':>12} {'RMSE(mid)':>12}")
    print(f"  {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 12}")
    for label, res in results.items():
        print(
            f"  {label:<10} {res['raw_coverage'] * 100:>9.1f}%"
            f" {res['coverage'] * 100:>9.1f}% {res['q'] * 100:>9.0f}%"
            f" {res['width']:>11.4f}m {res['rmse']:>11.4f}m"
        )
    print("=" * 65)

    # ── Raw CMIP6 baseline ──
    raw_rmse = np.sqrt(mean_squared_error(y_test_raw, test_df["model"].values))
    print(f"\n  Raw CMIP6 RMSE (no correction): {raw_rmse:.4f} m")

    # ──────────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────────

    # 1. Time-series with prediction intervals
    test_df = test_df.copy()
    for label, res in results.items():
        test_df[f"upper_{label}"] = res["upper"]
        test_df[f"lower_{label}"] = res["lower"]
        test_df[f"median_{label}"] = res["median"]

    ts = test_df.groupby("time").agg(
        obs_mean=("obs", "mean"),
        model_mean=("model", "mean"),
        **{
            f"upper_{lbl}": (f"upper_{lbl}", "mean")
            for lbl in results
        },
        **{
            f"lower_{lbl}": (f"lower_{lbl}", "mean")
            for lbl in results
        },
        **{
            f"median_{lbl}": (f"median_{lbl}", "mean")
            for lbl in results
        },
    )

    plt.figure(figsize=(14, 6))
    plt.plot(ts.index, ts["obs_mean"], "k-", lw=2, label="Satellite Obs")
    plt.plot(ts.index, ts["model_mean"], "b-", alpha=0.5, lw=1.5, label="Raw CMIP6")

    colors = {"90%": "green", "95%": "orange"}
    alphas = {"90%": 0.35, "95%": 0.2}
    for label in results:
        c = colors.get(label, "gray")
        a = alphas.get(label, 0.2)
        plt.fill_between(
            ts.index,
            ts[f"lower_{label}"],
            ts[f"upper_{label}"],
            alpha=a,
            color=c,
            label=f"Tube Loss {label} PI",
        )
        plt.plot(ts.index, ts[f"median_{label}"], "--", color=c, lw=1.5,
                 label=f"Tube Loss {label} median")

    plt.title("SLA Quantile Prediction with Tube Loss (Test: 2019-2020)")
    plt.ylabel("SLA (m)")
    plt.xlabel("Time")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig("sla_plots/sla_tube_loss_timeseries.png", dpi=150)
    plt.close()
    print("\n  Plot → sla_plots/sla_tube_loss_timeseries.png")

    # 2. Scatter: observed vs quantile bounds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, res) in zip(axes, results.items()):
        ax.scatter(y_test_raw, res["median"], s=1, alpha=0.3, color="blue", label="Median pred")
        ax.scatter(y_test_raw, res["upper"], s=1, alpha=0.15, color="red", label="Upper")
        ax.scatter(y_test_raw, res["lower"], s=1, alpha=0.15, color="green", label="Lower")
        lim = [min(y_test_raw.min(), res["lower"].min()), max(y_test_raw.max(), res["upper"].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Observed SLA (m)")
        ax.set_ylabel("Predicted SLA (m)")
        ax.set_title(f"Tube Loss {label} PI\nCoverage={res['coverage'] * 100:.1f}%, Width={res['width']:.4f}m")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("sla_plots/sla_tube_loss_scatter.png", dpi=150)
    plt.close()
    print("  Plot → sla_plots/sla_tube_loss_scatter.png")

    # 3. Coverage histogram by month
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, results.items()):
        monthly_cov = []
        for m in range(1, 13):
            mask = test_df["month"] == m
            if mask.sum() > 0:
                cov = np.mean(
                    (y_test_raw[mask] >= res["lower"][mask])
                    & (y_test_raw[mask] <= res["upper"][mask])
                )
                monthly_cov.append(cov * 100)
            else:
                monthly_cov.append(0)
        ax.bar(range(1, 13), monthly_cov, color=colors.get(label, "gray"), alpha=0.7)
        ax.axhline(res["q"] * 100, color="red", ls="--", lw=1.5, label=f"Target {res['q'] * 100:.0f}%")
        ax.set_xlabel("Month")
        ax.set_ylabel("Coverage (%)")
        ax.set_title(f"Monthly Coverage — Tube Loss {label}")
        ax.set_xticks(range(1, 13))
        ax.set_ylim(0, 105)
        ax.legend()
    plt.tight_layout()
    plt.savefig("sla_plots/sla_tube_loss_monthly_coverage.png", dpi=150)
    plt.close()
    print("  Plot → sla_plots/sla_tube_loss_monthly_coverage.png")

    # ── Per-time-point demonstration ──
    print("\n" + "=" * 65)
    print("  DEMO: Quantile predictions for specific CMIP6 time points")
    print("=" * 65)

    demo_times = ["2019-03-01", "2019-07-01", "2020-01-01", "2020-06-01"]
    for t_str in demo_times:
        t = pd.Timestamp(t_str)
        t_match = min(common_times, key=lambda x: abs(pd.Timestamp(x) - t))
        month_data = test_df[test_df["time"] == t_match]
        if len(month_data) == 0:
            continue

        obs_mean = month_data["obs"].mean()
        raw_mean = month_data["model"].mean()

        print(f"\n  {str(t_match)[:10]}:")
        print(f"    Satellite observed mean : {obs_mean:+.4f} m")
        print(f"    CMIP6 raw mean          : {raw_mean:+.4f} m")
        for label, res in results.items():
            mask = test_df["time"] == t_match
            u = res["upper"][mask.values].mean()
            l = res["lower"][mask.values].mean()
            md = res["median"][mask.values].mean()
            inside = "YES" if l <= obs_mean <= u else "NO"
            print(
                f"    Tube {label:>3s}  →  [{l:+.4f}, {u:+.4f}]  "
                f"median={md:+.4f}  obs inside={inside}"
            )

    print("\n" + "=" * 65)
    print("  Quantile model training complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
