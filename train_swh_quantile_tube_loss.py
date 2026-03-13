"""
Quantile Prediction + PICP/MPIW for SWH using Tube Loss & QD Loss.

Mirrors the SLA pipeline (train_sla_quantile_tube_loss.py / plot_picp_mpiw.py)
but applied to the Significant Wave Height dataset:
  - cowcliphs.nc   (CMIP6 SWH)
  - io-altimeter.nc (satellite altimeter SWH)
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.ndimage import uniform_filter1d

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ─── Loss functions ──────────────────────────────────────────

class TubeLoss(nn.Module):
    """Tube Loss — ported from ltpritamanand/tube_loss (arXiv:2412.06853)."""
    def __init__(self, q=0.95, r=0.5, delta=0.0005):
        super().__init__()
        self.q, self.r, self.delta = q, r, delta

    def forward(self, y_pred, y_true):
        f1, f2 = y_pred[:, 0], y_pred[:, 1]
        c1 = (1 - self.q) * (y_true - f2)
        c2 = (1 - self.q) * (f1 - y_true)
        c3 = self.q * (f2 - y_true)
        c4 = self.q * (y_true - f1)
        inside = (y_true <= f1) & (y_true >= f2)
        above_mid = y_true > (self.r * f1 + (1 - self.r) * f2)
        loss = torch.where(inside,
                           torch.where(above_mid, c1, c2),
                           torch.where(f2 > y_true, c3, c4))
        return (loss + self.delta * torch.abs(f1 - f2)).mean()


class QDLoss(nn.Module):
    """Quality-Driven (dual pinball) loss for upper & lower quantiles."""
    def __init__(self, q=0.95):
        super().__init__()
        self.tau_upper = (1 + q) / 2
        self.tau_lower = (1 - q) / 2

    def forward(self, y_pred, y_true):
        upper, lower = y_pred[:, 0], y_pred[:, 1]
        e_u = y_true - upper
        e_l = y_true - lower
        loss_u = torch.where(e_u >= 0, self.tau_upper * e_u, (self.tau_upper - 1) * e_u)
        loss_l = torch.where(e_l >= 0, self.tau_lower * e_l, (self.tau_lower - 1) * e_l)
        return (loss_u + loss_l).mean()


# ─── Network ─────────────────────────────────────────────────

class PINet(nn.Module):
    def __init__(self, n_features, hidden=(128, 64, 32)):
        super().__init__()
        layers = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_base = nn.Linear(d, 1)
        self.head_hw = nn.Linear(d, 1)
        self.head_asym = nn.Linear(d, 1)
        nn.init.constant_(self.head_hw.bias, 2.0)

    def forward(self, x):
        h = self.trunk(x)
        base = self.head_base(h).squeeze(-1)
        hw = nn.functional.softplus(self.head_hw(h).squeeze(-1))
        a = torch.sigmoid(self.head_asym(h).squeeze(-1))
        upper = base + hw * (1.0 + a)
        lower = base - hw * (2.0 - a)
        return torch.stack([upper, lower], dim=1)


# ─── Data loading ────────────────────────────────────────────

def load_swh_data():
    print("[1/3] Loading SWH data...")
    model_ds = xr.open_dataset("cowcliphs.nc")
    obs_ds = xr.open_dataset("io-altimeter.nc")

    model_ds = model_ds.rename({
        'LATITUDE41_321': 'lat', 'LONGITUDE101_221': 'lon',
        'TIME': 'time', 'HS': 'swh'
    })
    obs_ds = obs_ds.rename({
        'LATITUDE11_80': 'lat', 'LONGITUDE116_145': 'lon',
        'TIME': 'time', 'VAVH_DAILY_MEAN': 'swh'
    })

    start, end = "2015-01-01", "2020-12-31"
    model_sliced = model_ds.sel(time=slice(start, end)).resample(time='1MS').mean()
    obs_sliced = obs_ds.sel(time=slice(start, end)).resample(time='1MS').mean()

    common_times = np.intersect1d(model_sliced.time.values, obs_sliced.time.values)
    model_aligned = model_sliced.sel(time=common_times)
    obs_aligned = obs_sliced.sel(time=common_times)

    print("[2/3] Regridding model → obs grid...")
    model_regridded = model_aligned.interp_like(obs_aligned, method='linear')

    print("[3/3] Building feature table...")
    m_swh = model_regridded['swh'].values.flatten()
    o_swh = obs_aligned['swh'].values.flatten()

    times = model_regridded.time.values
    lats = model_regridded.lat.values
    lons = model_regridded.lon.values
    T, Lat, Lon = np.meshgrid(times, lats, lons, indexing='ij')

    df = pd.DataFrame({
        'model': m_swh,
        'lat': Lat.flatten(),
        'lon': Lon.flatten(),
        'month': pd.to_datetime(T.flatten()).month,
        'obs': o_swh,
        'time': T.flatten(),
    })
    df = df.dropna()
    print(f"  Total valid samples: {len(df)}")
    return df


# ─── Training helper ─────────────────────────────────────────

def train_model(X_tr, y_tr, criterion, n_features, epochs=800):
    model = PINet(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    best, pat = float("inf"), 0
    for ep in range(1, epochs + 1):
        model.train()
        eloss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            eloss += loss.item() * len(xb)
        eloss /= len(ds)
        sched.step(eloss)
        if eloss < best - 1e-6:
            best, pat = eloss, 0
        else:
            pat += 1
        if pat >= 60:
            print(f"      early stop ep {ep}")
            break
        if ep % 200 == 0 or ep == 1:
            print(f"      ep {ep:>4d}  loss={eloss:.6f}")
    return model


def calibrate(model, X_cal, y_cal, q):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X_cal, dtype=torch.float32)).numpy()
    upper = np.maximum(p[:, 0], p[:, 1])
    lower = np.minimum(p[:, 0], p[:, 1])
    scores = np.maximum(lower - y_cal, y_cal - upper)
    n = len(scores)
    ci = min(int(np.ceil((n + 1) * q)) - 1, n - 1)
    return np.sort(scores)[ci]


def picp(y, lower, upper):
    return np.mean((y >= lower) & (y <= upper)) * 100

def mpiw(lower, upper):
    return np.mean(upper - lower)

def empirical_pi(x_sort, y, q, n_bins=60):
    tau_u, tau_l = (1 + q) / 2, (1 - q) / 2
    edges = np.linspace(x_sort.min(), x_sort.max(), n_bins + 1)
    centres, ups, lows = [], [], []
    for i in range(n_bins):
        mask = (x_sort >= edges[i]) & (x_sort < edges[i + 1])
        if mask.sum() < 5:
            continue
        centres.append((edges[i] + edges[i + 1]) / 2)
        ups.append(np.quantile(y[mask], tau_u))
        lows.append(np.quantile(y[mask], tau_l))
    return np.array(centres), np.array(ups), np.array(lows)


# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SWH: RF Correction + Tube Loss / QD Loss Quantile Prediction")
    print("=" * 70)

    df = load_swh_data()
    features = ["model", "lat", "lon", "month"]
    q = 0.95

    train_df = df[df["time"].dt.year <= 2017].copy()
    cal_df = df[df["time"].dt.year == 2018].copy()
    test_df = df[df["time"].dt.year >= 2019].copy()
    print(f"\n  Train: {len(train_df)}  |  Cal: {len(cal_df)}  |  Test: {len(test_df)}")

    # ════════════════════════════════════════════════════════
    # PART 1: Random Forest Deterministic Correction
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1: Random Forest Deterministic Correction")
    print("=" * 70)

    train_full = df[df["time"].dt.year <= 2018].copy()
    test_full = df[df["time"].dt.year >= 2019].copy()

    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(train_full[features], train_full["obs"])
    test_full = test_full.copy()
    test_full["predicted"] = rf.predict(test_full[features])

    rmse_raw = np.sqrt(mean_squared_error(test_full["obs"], test_full["model"]))
    rmse_ml = np.sqrt(mean_squared_error(test_full["obs"], test_full["predicted"]))
    mae_raw = mean_absolute_error(test_full["obs"], test_full["model"])
    mae_ml = mean_absolute_error(test_full["obs"], test_full["predicted"])
    bias_raw = np.mean(test_full["model"] - test_full["obs"])
    bias_ml = np.mean(test_full["predicted"] - test_full["obs"])
    r2 = r2_score(test_full["obs"], test_full["predicted"])
    improv = (rmse_raw - rmse_ml) / rmse_raw * 100

    print(f"\n  {'Metric':<25} {'Raw CMIP6':>12} {'ML Corrected':>14} {'Improvement':>14}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*14}")
    print(f"  {'RMSE (m)':<25} {rmse_raw:>12.4f} {rmse_ml:>14.4f} {improv:>13.1f}%")
    print(f"  {'MAE (m)':<25} {mae_raw:>12.4f} {mae_ml:>14.4f}")
    print(f"  {'Bias (m)':<25} {bias_raw:>12.4f} {bias_ml:>14.4f}")
    print(f"  {'R² Score':<25} {'N/A':>12} {r2:>14.4f}")

    os.makedirs("swh_plots", exist_ok=True)

    # RF scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test_full["obs"], test_full["model"], s=1, alpha=0.2)
    lims = [test_full["obs"].min(), test_full["obs"].max()]
    axes[0].plot(lims, lims, "k--")
    axes[0].set_title(f"Raw CMIP6 vs Obs\nRMSE: {rmse_raw:.3f}m")
    axes[0].set_xlabel("Observed SWH (m)"); axes[0].set_ylabel("Model SWH (m)")
    axes[1].scatter(test_full["obs"], test_full["predicted"], s=1, alpha=0.2, c="green")
    axes[1].plot(lims, lims, "k--")
    axes[1].set_title(f"ML Corrected vs Obs\nRMSE: {rmse_ml:.3f}m, R²: {r2:.4f}")
    axes[1].set_xlabel("Observed SWH (m)"); axes[1].set_ylabel("Corrected SWH (m)")
    plt.tight_layout()
    plt.savefig("swh_plots/swh_rf_scatter.png", dpi=150); plt.close()

    # RF time series
    df["rf_pred"] = rf.predict(df[features])
    ts = df.groupby("time")[["obs", "model", "rf_pred"]].mean()
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts["obs"], "k-", lw=2, label="Satellite Obs")
    plt.plot(ts.index, ts["model"], "b-", alpha=0.6, label="Raw CMIP6")
    plt.plot(ts.index, ts["rf_pred"], "r--", lw=2, label="ML Corrected")
    plt.axvline(pd.Timestamp("2019-01-01"), color="gray", ls=":", label="Train/Test Split")
    plt.title("SWH: Regional Mean Time Series"); plt.ylabel("SWH (m)"); plt.legend()
    plt.tight_layout(); plt.savefig("swh_plots/swh_rf_timeseries.png", dpi=150); plt.close()

    # Bias maps
    tf = test_full.copy()
    tf["bias_raw"] = tf["model"] - tf["obs"]
    tf["bias_ml"] = tf["predicted"] - tf["obs"]
    br = tf.groupby(["lat","lon"])["bias_raw"].mean().reset_index()
    bm = tf.groupby(["lat","lon"])["bias_ml"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vmax = max(br["bias_raw"].abs().max(), bm["bias_ml"].abs().max())
    sc1 = axes[0].scatter(br["lon"], br["lat"], c=br["bias_raw"], cmap="RdBu_r", s=15, vmin=-vmax, vmax=vmax)
    plt.colorbar(sc1, ax=axes[0], label="Bias (m)"); axes[0].set_title("Raw CMIP6 Bias (2019-2020)")
    sc2 = axes[1].scatter(bm["lon"], bm["lat"], c=bm["bias_ml"], cmap="RdBu_r", s=15, vmin=-vmax, vmax=vmax)
    plt.colorbar(sc2, ax=axes[1], label="Bias (m)"); axes[1].set_title("ML Corrected Bias (2019-2020)")
    plt.tight_layout(); plt.savefig("swh_plots/swh_bias_maps.png", dpi=150); plt.close()

    # ════════════════════════════════════════════════════════
    # PART 2: Tube Loss + QD Loss Quantile Prediction
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 2: Tube Loss / QD Loss — PICP & MPIW Comparison")
    print("=" * 70)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[features].values)
    y_tr = train_df["obs"].values.astype(np.float32)
    X_cal = scaler.transform(cal_df[features].values)
    y_cal = cal_df["obs"].values.astype(np.float32)
    X_te = scaler.transform(test_df[features].values)
    y_te = test_df["obs"].values.astype(np.float32)

    results = {}
    for name, criterion in [("QD Loss", QDLoss(q=q)),
                             ("Tube Loss", TubeLoss(q=q, r=0.5, delta=0.0005))]:
        print(f"\n  ── Training {name} NN ──")
        model = train_model(X_tr, y_tr, criterion, len(features))
        margin = calibrate(model, X_cal, y_cal, q)
        print(f"      conformal margin = {margin:+.4f} m")

        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
        raw_u = np.maximum(p[:, 0], p[:, 1])
        raw_l = np.minimum(p[:, 0], p[:, 1])
        mid = (raw_u + raw_l) / 2
        cal_u = raw_u + margin
        cal_l = raw_l - margin

        results[name] = {
            "upper": cal_u, "lower": cal_l, "mid": mid,
            "raw_upper": raw_u, "raw_lower": raw_l,
            "picp": picp(y_te, cal_l, cal_u),
            "mpiw": mpiw(cal_l, cal_u),
            "picp_raw": picp(y_te, raw_l, raw_u),
            "mpiw_raw": mpiw(raw_l, raw_u),
        }

    # ── Metrics table ──
    print("\n" + "=" * 70)
    print(f"  {'Method':<14} {'PICP':>8} {'MPIW':>10}  {'PICP(raw)':>10} {'MPIW(raw)':>10}")
    print(f"  {'─'*14} {'─'*8} {'─'*10}  {'─'*10} {'─'*10}")
    for name, r in results.items():
        print(f"  {name:<14} {r['picp']:>7.1f}% {r['mpiw']:>9.4f}m"
              f"  {r['picp_raw']:>9.1f}% {r['mpiw_raw']:>9.4f}m")
    print(f"\n  Raw CMIP6 RMSE: {rmse_raw:.4f} m")
    print(f"  RF Corrected RMSE: {rmse_ml:.4f} m")
    print("=" * 70)

    # ════════════════════════════════════════════════════════
    # PART 3: Tube Loss 90% + 95% PI
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 3: Tube Loss 90% & 95% Prediction Intervals")
    print("=" * 70)

    tube_results = {}
    for q_level, delta in [(0.90, 0.0005), (0.95, 0.0002)]:
        label = f"{int(q_level*100)}%"
        print(f"\n  ── Tube Loss {label} PI ──")
        mdl = train_model(X_tr, y_tr, TubeLoss(q=q_level, r=0.5, delta=delta), len(features))
        margin = calibrate(mdl, X_cal, y_cal, q_level)
        print(f"      conformal margin = {margin:+.4f} m")
        mdl.eval()
        with torch.no_grad():
            p = mdl(torch.tensor(X_te, dtype=torch.float32)).numpy()
        raw_u = np.maximum(p[:, 0], p[:, 1])
        raw_l = np.minimum(p[:, 0], p[:, 1])
        mid = (raw_u + raw_l) / 2
        cal_u, cal_l = raw_u + margin, raw_l - margin

        cov = picp(y_te, cal_l, cal_u)
        wid = mpiw(cal_l, cal_u)
        rmse_mid = np.sqrt(mean_squared_error(y_te, mid))
        raw_cov = picp(y_te, raw_l, raw_u)

        tube_results[label] = {
            "upper": cal_u, "lower": cal_l, "mid": mid,
            "picp": cov, "mpiw": wid, "rmse": rmse_mid,
            "picp_raw": raw_cov, "q": q_level,
        }
        print(f"      Raw coverage  : {raw_cov:.1f}%")
        print(f"      Calibrated cov: {cov:.1f}%  (target {q_level*100:.0f}%)")
        print(f"      Avg width     : {wid:.4f} m")
        print(f"      RMSE(mid)     : {rmse_mid:.4f} m")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print(f"  {'PI Level':<10} {'Raw Cov':>10} {'Cal Cov':>10} {'Target':>10} {'Width':>10} {'RMSE(mid)':>12}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
    for lbl, r in tube_results.items():
        print(f"  {lbl:<10} {r['picp_raw']:>9.1f}% {r['picp']:>9.1f}% {r['q']*100:>9.0f}%"
              f" {r['mpiw']:>9.4f}m {r['rmse']:>11.4f}m")
    print("=" * 70)

    # ════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════

    # 1. Tube Loss time series with PI bands
    test_df = test_df.copy()
    for lbl, r in tube_results.items():
        test_df[f"upper_{lbl}"] = r["upper"]
        test_df[f"lower_{lbl}"] = r["lower"]
        test_df[f"mid_{lbl}"] = r["mid"]

    ts2 = test_df.groupby("time").agg(
        obs_mean=("obs", "mean"), model_mean=("model", "mean"),
        **{f"upper_{l}": (f"upper_{l}", "mean") for l in tube_results},
        **{f"lower_{l}": (f"lower_{l}", "mean") for l in tube_results},
        **{f"mid_{l}": (f"mid_{l}", "mean") for l in tube_results},
    )

    plt.figure(figsize=(14, 6))
    plt.plot(ts2.index, ts2["obs_mean"], "k-", lw=2, label="Satellite Obs")
    plt.plot(ts2.index, ts2["model_mean"], "b-", alpha=0.5, lw=1.5, label="Raw CMIP6")
    colors = {"90%": "green", "95%": "orange"}
    alphas = {"90%": 0.35, "95%": 0.2}
    for lbl in tube_results:
        c, a = colors.get(lbl, "gray"), alphas.get(lbl, 0.2)
        plt.fill_between(ts2.index, ts2[f"lower_{lbl}"], ts2[f"upper_{lbl}"],
                         alpha=a, color=c, label=f"Tube Loss {lbl} PI")
        plt.plot(ts2.index, ts2[f"mid_{lbl}"], "--", color=c, lw=1.5, label=f"Tube {lbl} median")
    plt.title("SWH: Quantile Prediction with Tube Loss (Test 2019-2020)")
    plt.ylabel("SWH (m)"); plt.xlabel("Time"); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig("swh_plots/swh_tube_loss_timeseries.png", dpi=150); plt.close()

    # 2. Monthly coverage
    fig, axes = plt.subplots(1, len(tube_results), figsize=(7*len(tube_results), 5))
    if len(tube_results) == 1: axes = [axes]
    for ax, (lbl, r) in zip(axes, tube_results.items()):
        mc = []
        for m in range(1, 13):
            mask = test_df["month"] == m
            if mask.sum() > 0:
                mc.append(picp(y_te[mask.values], r["lower"][mask.values], r["upper"][mask.values]))
            else:
                mc.append(0)
        ax.bar(range(1, 13), mc, color=colors.get(lbl, "gray"), alpha=0.7)
        ax.axhline(r["q"]*100, color="red", ls="--", lw=1.5, label=f"Target {r['q']*100:.0f}%")
        ax.set_xlabel("Month"); ax.set_ylabel("Coverage (%)")
        ax.set_title(f"Monthly Coverage — Tube Loss {lbl}")
        ax.set_xticks(range(1, 13)); ax.set_ylim(0, 105); ax.legend()
    plt.tight_layout(); plt.savefig("swh_plots/swh_tube_loss_monthly_coverage.png", dpi=150); plt.close()

    # 3. QD vs Tube comparison plot (PICP style)
    x_axis = test_df["model"].values
    sort_idx = np.argsort(x_axis)
    x_sorted, y_sorted = x_axis[sort_idx], y_te[sort_idx]
    centres, true_up, true_lo = empirical_pi(x_sorted, y_sorted, q, n_bins=50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    titles = {"QD Loss": "(a) QD Loss based NN", "Tube Loss": "(b) Tube Loss based NN"}
    for ax, (name, r) in zip(axes, results.items()):
        upper_s, lower_s = r["upper"][sort_idx], r["lower"][sort_idx]
        k = max(len(x_sorted) // 80, 15)
        upper_sm = uniform_filter1d(upper_s, size=k)
        lower_sm = uniform_filter1d(lower_s, size=k)
        ax.scatter(x_sorted, y_sorted, c="red", s=3, alpha=0.25, rasterized=True, label="Data")
        ax.plot(centres, true_up, "k--", lw=2.2, label="True PI")
        ax.plot(centres, true_lo, "k--", lw=2.2)
        ax.plot(x_sorted, upper_sm, "b-", lw=2, label="Estimated PI")
        ax.plot(x_sorted, lower_sm, "b-", lw=2)
        ax.set_xlabel("CMIP6 Model SWH (m)", fontsize=12)
        ax.set_title(f"{titles[name]}\nPICP = {r['picp']:.1f}%   MPIW = {r['mpiw']:.4f} m", fontsize=12)
        ax.legend(loc="upper left", fontsize=10); ax.grid(True, alpha=0.15)
    axes[0].set_ylabel("Observed SWH (m)", fontsize=12)
    plt.tight_layout(); plt.savefig("swh_plots/swh_picp_mpiw_qd_vs_tube.png", dpi=200); plt.close()

    # 4. PICP/MPIW bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    names = list(results.keys())
    picps = [results[n]["picp"] for n in names]
    mpiws = [results[n]["mpiw"] for n in names]
    x = np.arange(len(names))
    ax = axes[0]
    bars = ax.bar(x, picps, 0.5, color=["#4C72B0","#55A868"], edgecolor="black", lw=0.6)
    ax.axhline(q*100, color="red", ls="--", lw=1.5, label=f"Target {q*100:.0f}%")
    for b, v in zip(bars, picps):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("PICP (%)")
    ax.set_title("Prediction Interval Coverage Probability"); ax.set_ylim(0, 110); ax.legend(); ax.grid(axis="y", alpha=0.2)
    ax = axes[1]
    bars = ax.bar(x, mpiws, 0.5, color=["#4C72B0","#55A868"], edgecolor="black", lw=0.6)
    for b, v in zip(bars, mpiws):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.4f} m", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("MPIW (m)")
    ax.set_title("Mean Prediction Interval Width"); ax.set_ylim(0, max(mpiws)*1.35); ax.grid(axis="y", alpha=0.2)
    plt.tight_layout(); plt.savefig("swh_plots/swh_picp_mpiw_bars.png", dpi=200); plt.close()

    # 5. Quantile summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SWH Tube Loss Quantile Model — Performance Summary", fontsize=15, fontweight="bold", y=1.02)
    levels = list(tube_results.keys())
    x = np.arange(len(levels))
    bw = 0.25
    ax = axes[0]
    raw_covs = [tube_results[l]["picp_raw"] for l in levels]
    cal_covs = [tube_results[l]["picp"] for l in levels]
    targets = [tube_results[l]["q"]*100 for l in levels]
    b1 = ax.bar(x-bw, raw_covs, bw, label="Raw Tube Loss", color="#4C72B0", edgecolor="black", lw=0.6)
    b2 = ax.bar(x, cal_covs, bw, label="Calibrated", color="#55A868", edgecolor="black", lw=0.6)
    b3 = ax.bar(x+bw, targets, bw, label="Target", color="#C44E52", edgecolor="black", lw=0.6, alpha=0.5)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels); ax.set_ylabel("Coverage (%)"); ax.set_ylim(0, 110)
    ax.set_title("Coverage: Raw vs Calibrated vs Target"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)
    ax = axes[1]
    widths = [tube_results[l]["mpiw"] for l in levels]
    bars = ax.bar(x, widths, 0.45, color=["#4C72B0","#55A868"], edgecolor="black", lw=0.6)
    for b, w in zip(bars, widths):
        ax.text(b.get_x()+b.get_width()/2, w+0.02, f"{w:.3f} m", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels); ax.set_ylabel("Width (m)")
    ax.set_title("Average Prediction Interval Width"); ax.set_ylim(0, max(widths)*1.4); ax.grid(axis="y", alpha=0.2)
    ax = axes[2]
    rmses_mid = [tube_results[l]["rmse"] for l in levels]
    b1 = ax.bar(x-0.15, [rmse_raw]*len(levels), 0.28, label="Raw CMIP6", color="#C44E52", edgecolor="black", lw=0.6)
    b2 = ax.bar(x+0.15, rmses_mid, 0.28, label="Tube Loss Median", color="#55A868", edgecolor="black", lw=0.6)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.3f} m", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(levels); ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE: Raw CMIP6 vs Tube Loss Median"); ax.set_ylim(0, rmse_raw*1.4); ax.legend(); ax.grid(axis="y", alpha=0.2)
    plt.tight_layout(); plt.savefig("swh_plots/swh_quantile_summary.png", dpi=200); plt.close()

    # ── Demo for specific time points ──
    print("\n" + "=" * 70)
    print("  DEMO: Predictions for specific CMIP6 time points")
    print("=" * 70)
    demo_times = ["2019-03-01", "2019-07-01", "2020-01-01", "2020-06-01"]
    for t_str in demo_times:
        t = pd.Timestamp(t_str)
        t_match = min(test_df["time"].unique(), key=lambda x: abs(pd.Timestamp(x) - t))
        md = test_df[test_df["time"] == t_match]
        if len(md) == 0: continue
        obs_m = md["obs"].mean()
        raw_m = md["model"].mean()
        rf_m = rf.predict(md[features]).mean()
        print(f"\n  {str(t_match)[:10]}:")
        print(f"    Satellite obs mean : {obs_m:.4f} m")
        print(f"    CMIP6 raw mean     : {raw_m:.4f} m  (error: {abs(raw_m-obs_m):.4f})")
        print(f"    RF corrected       : {rf_m:.4f} m  (error: {abs(rf_m-obs_m):.4f})")
        for lbl, r in tube_results.items():
            mask = (test_df["time"] == t_match).values
            u, l, md_v = r["upper"][mask].mean(), r["lower"][mask].mean(), r["mid"][mask].mean()
            inside = "YES" if l <= obs_m <= u else "NO"
            print(f"    Tube {lbl:>3s} → [{l:.3f}, {u:.3f}] median={md_v:.3f} obs_inside={inside}")

    print("\n" + "=" * 70)
    print("  All SWH results complete! Plots saved → swh_plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
