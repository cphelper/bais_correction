"""
Compute PICP & MPIW and produce a side-by-side plot
(a) QD (Quantile Decomposition / pinball) loss NN
(b) Tube Loss NN
matching the style of the reference figure from the Tube Loss paper.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, glob, joblib
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ─── Loss functions ──────────────────────────────────────────

class TubeLoss(nn.Module):
    """Tube Loss — ported from ltpritamanand/tube_loss."""
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
    """
    Quality-Driven (QD) loss — dual pinball loss for upper & lower quantiles.
    Equivalent to training two separate quantile regressors jointly.
    """
    def __init__(self, q=0.95):
        super().__init__()
        self.tau_upper = (1 + q) / 2   # e.g. 0.975
        self.tau_lower = (1 - q) / 2   # e.g. 0.025

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


# ─── Data loader (reused from train_sla_quantile_tube_loss.py) ──

def load_data():
    sla_dir, model_path = "./altimeter_sla", \
        "./sla_2/zos_Omon_MPI-ESM1-2-LR_ssp245_r1i1p1f1_gn_20150116-20251216.nc"
    lat_min, lat_max, lon_min, lon_max = 60, 80, 50, 110

    sla_files = sorted(glob.glob(os.path.join(sla_dir, "*.nc")))
    sla_files = [f for f in sla_files
                 if any(y in f for y in ["2015","2016","2017","2018","2019","2020"])]
    obs_ds_list = []
    for f in sla_files:
        try:
            with xr.open_dataset(f) as ds:
                if "latitude" in ds.coords: ds = ds.rename({"latitude": "lat"})
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
                sl = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                if "time" not in sl.dims: sl = sl.expand_dims("time")
                obs_ds_list.append(sl.load())
        except Exception:
            pass
    obs_ds = xr.concat(obs_ds_list, dim="time").sortby("time")
    obs_monthly = obs_ds["sla"].resample(time="1MS").mean()

    mds = xr.open_dataset(model_path).sel(time=slice("2015-01-01","2020-12-31"))
    mm = mds["zos"].resample(time="1MS").mean()
    mm_mean = mm.sel(time=slice("2015-01-01","2018-12-31")).mean("time")
    model_sla = mm - mm_mean

    pts = np.column_stack((mds["latitude"].values.ravel(), mds["longitude"].values.ravel()))
    olat, olon = obs_monthly.lat.values, obs_monthly.lon.values
    og, olg = np.meshgrid(olat, olon, indexing="ij")
    tp = np.column_stack((og.ravel(), olg.ravel()))
    _, idx = cKDTree(pts).query(tp)

    ct = sorted(set(model_sla.time.values) & set(obs_monthly.time.values))
    oa, ma = obs_monthly.sel(time=ct), model_sla.sel(time=ct)
    rv = [ma.sel(time=t).values.ravel()[idx].reshape(len(olat), len(olon)) for t in ct]
    mr = xr.DataArray(np.array(rv), coords={"time": ct, "lat": olat, "lon": olon},
                      dims=["time","lat","lon"])

    dfo = oa.to_dataframe(name="obs").reset_index()
    dfm = mr.to_dataframe(name="model").reset_index()
    df = pd.merge(dfo, dfm, on=["time","lat","lon"])
    df["month"] = df["time"].dt.month
    df = df.dropna()
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


# ─── Conformal calibration ───────────────────────────────────

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


# ─── Metrics ─────────────────────────────────────────────────

def picp(y, lower, upper):
    return np.mean((y >= lower) & (y <= upper)) * 100

def mpiw(lower, upper):
    return np.mean(upper - lower)


# ─── Empirical "True PI" from binned data ────────────────────

def empirical_pi(x_sort, y, q, n_bins=60):
    """Compute empirical upper/lower quantile bounds by binning x."""
    tau_u, tau_l = (1 + q) / 2, (1 - q) / 2
    bin_edges = np.linspace(x_sort.min(), x_sort.max(), n_bins + 1)
    centres, ups, lows = [], [], []
    for i in range(n_bins):
        mask = (x_sort >= bin_edges[i]) & (x_sort < bin_edges[i + 1])
        if mask.sum() < 5:
            continue
        centres.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        ups.append(np.quantile(y[mask], tau_u))
        lows.append(np.quantile(y[mask], tau_l))
    return np.array(centres), np.array(ups), np.array(lows)


# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PICP / MPIW Comparison — QD Loss vs Tube Loss")
    print("=" * 65)

    df = load_data()
    features = ["model", "lat", "lon", "month"]
    q = 0.95

    train_df = df[df["time"].dt.year <= 2017]
    cal_df   = df[df["time"].dt.year == 2018]
    test_df  = df[df["time"].dt.year >= 2019]
    print(f"  Train {len(train_df)} | Cal {len(cal_df)} | Test {len(test_df)}")

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

    # ── Print metrics table ──
    print("\n" + "=" * 65)
    print(f"  {'Method':<14} {'PICP':>8} {'MPIW':>10}  {'PICP(raw)':>10} {'MPIW(raw)':>10}")
    print(f"  {'─'*14} {'─'*8} {'─'*10}  {'─'*10} {'─'*10}")
    for name, r in results.items():
        print(f"  {name:<14} {r['picp']:>7.1f}% {r['mpiw']:>9.4f}m"
              f"  {r['picp_raw']:>9.1f}% {r['mpiw_raw']:>9.4f}m")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════
    # PLOT — matching reference figure style
    # ══════════════════════════════════════════════════════════

    x_axis = test_df["model"].values          # CMIP6 model SLA as x-axis
    sort_idx = np.argsort(x_axis)
    x_sorted = x_axis[sort_idx]
    y_sorted = y_te[sort_idx]

    # Empirical "True PI"
    centres, true_up, true_lo = empirical_pi(x_sorted, y_sorted, q, n_bins=50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    titles = {"QD Loss": "(a) QD Loss based NN", "Tube Loss": "(b) Tube Loss based NN"}

    for ax, (name, r) in zip(axes, results.items()):
        upper_s = r["upper"][sort_idx]
        lower_s = r["lower"][sort_idx]

        # smooth the estimated bounds for plotting
        from scipy.ndimage import uniform_filter1d
        k = max(len(x_sorted) // 80, 15)
        upper_smooth = uniform_filter1d(upper_s, size=k)
        lower_smooth = uniform_filter1d(lower_s, size=k)

        # scatter data
        ax.scatter(x_sorted, y_sorted, c="red", s=3, alpha=0.35, zorder=1,
                   label="Data", rasterized=True)

        # true PI
        ax.plot(centres, true_up, "k--", lw=2.2, label="True PI", zorder=3)
        ax.plot(centres, true_lo, "k--", lw=2.2, zorder=3)

        # estimated PI
        ax.plot(x_sorted, upper_smooth, "b-", lw=2, label="Estimated PI", zorder=4)
        ax.plot(x_sorted, lower_smooth, "b-", lw=2, zorder=4)

        ax.set_xlabel("CMIP6 Model SLA (m)", fontsize=12)
        ax.set_title(f"{titles[name]}\nPICP = {r['picp']:.1f}%   MPIW = {r['mpiw']:.4f} m",
                     fontsize=12)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("Observed SLA (m)", fontsize=12)

    plt.tight_layout()
    out = "sla_plots/picp_mpiw_qd_vs_tube.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {out}")

    # ── Metrics bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    names = list(results.keys())
    picps = [results[n]["picp"] for n in names]
    mpiws = [results[n]["mpiw"] for n in names]
    x = np.arange(len(names))

    ax = axes[0]
    bars = ax.bar(x, picps, 0.5, color=["#4C72B0", "#55A868"], edgecolor="black", lw=0.6)
    ax.axhline(q * 100, color="red", ls="--", lw=1.5, label=f"Target {q*100:.0f}%")
    for b, v in zip(bars, picps):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("PICP (%)")
    ax.set_title("Prediction Interval Coverage Probability")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    bars = ax.bar(x, mpiws, 0.5, color=["#4C72B0", "#55A868"], edgecolor="black", lw=0.6)
    for b, v in zip(bars, mpiws):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f} m",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("MPIW (m)")
    ax.set_title("Mean Prediction Interval Width")
    ax.set_ylim(0, max(mpiws) * 1.35)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out2 = "sla_plots/picp_mpiw_bars.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
