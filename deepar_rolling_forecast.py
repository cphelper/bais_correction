import argparse
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DeepAR(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        params = self.proj(out)
        mu = params[..., 0]
        sigma = F.softplus(params[..., 1]) + 1e-3
        return mu, sigma


def gaussian_nll(mu, sigma, y):
    return 0.5 * np.log(2 * np.pi) + torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


def load_series(path):
    ds = xr.open_dataset(path)
    ds = ds.rename(
        {
            "TIME": "time",
            "LATITUDE11_80": "lat",
            "LONGITUDE116_145": "lon",
            "VAVH_DAILY_MEAN": "swh",
        }
    )
    weights = np.cos(np.deg2rad(ds["lat"].values))
    weights = xr.DataArray(weights, dims=["lat"])
    ts = ds["swh"].weighted(weights).mean(dim=["lat", "lon"])
    return pd.DatetimeIndex(ds["time"].values), ts.values.astype(np.float32)


def load_sla_series(path_glob):
    files = sorted(glob.glob(path_glob))
    datasets = [xr.open_dataset(f) for f in files]
    ds = xr.concat(datasets, dim="time")
    for d in datasets:
        d.close()
    weights = np.cos(np.deg2rad(ds["latitude"].values))
    weights = xr.DataArray(weights, dims=["latitude"])
    ts = ds["sla"].weighted(weights).mean(dim=["latitude", "longitude"])
    return pd.DatetimeIndex(ds["time"].values), ts.values.astype(np.float32)


def build_windows(series, window_length):
    x_list = []
    y_list = []
    for i in range(window_length, len(series)):
        window = series[i - window_length : i]
        x_list.append(window[:-1])
        y_list.append(window[1:])
    x = np.stack(x_list)
    y = np.stack(y_list)
    return x, y


def train_model(series, context_length, epochs, batch_size, lr, hidden_size, num_layers, dropout, device):
    window_length = context_length + 1
    x, y = build_windows(series, window_length)
    x_t = torch.tensor(x[:, :, None], dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DeepAR(hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, sigma = model(xb)
            loss = gaussian_nll(mu, sigma, yb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def rolling_forecast(model, series, context_length, device):
    preds = []
    sigmas = []
    model.eval()
    with torch.no_grad():
        for i in range(context_length, len(series)):
            context = series[i - context_length : i]
            xb = torch.tensor(context[None, :, None], dtype=torch.float32, device=device)
            mu, sigma = model(xb)
            preds.append(mu[0, -1].item())
            sigmas.append(sigma[0, -1].item())
    return np.array(preds), np.array(sigmas)


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    from scipy.stats import norm

    return norm.cdf(x)


def normal_ppf(p):
    p = np.asarray(p, dtype=float)
    from scipy.stats import norm

    return norm.ppf(p)


def make_pi_plots(times, observed, mu, sigma, test_slice, out_dir, tag, last_n):
    os.makedirs(out_dir, exist_ok=True)

    z10 = float(normal_ppf(0.10))
    z90 = float(normal_ppf(0.90))
    p10 = mu + z10 * sigma
    p50 = mu
    p90 = mu + z90 * sigma

    if last_n is not None and last_n > 0:
        start = max(0, len(times) - last_n)
        t_plot = times[start:]
        y_plot = observed[start:]
        p10_plot = p10[start:]
        p50_plot = p50[start:]
        p90_plot = p90[start:]
    else:
        t_plot = times
        y_plot = observed
        p10_plot = p10
        p50_plot = p50
        p90_plot = p90

    plt.figure(figsize=(12, 4))
    plt.fill_between(t_plot, p10_plot, p90_plot, alpha=0.25, label="Predicted PI (P10–P90)")
    plt.plot(t_plot, p50_plot, linewidth=1.5, label="Predicted median (P50)")
    plt.plot(t_plot, y_plot, linewidth=1.2, color="black", alpha=0.8, label="Observed")
    plt.title(f"DeepAR Rolling Forecast: Observed vs Predicted PI ({tag})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"deepar_{tag}_timeseries_pi.png"), dpi=200)
    plt.close()

    y_test = observed[test_slice]
    mu_test = mu[test_slice]
    sigma_test = sigma[test_slice]

    central_levels = np.linspace(0.1, 0.95, 18)
    empirical = []
    for c in central_levels:
        alpha = 0.5 * (1.0 - c)
        z_lo = float(normal_ppf(alpha))
        z_hi = float(normal_ppf(1.0 - alpha))
        lo = mu_test + z_lo * sigma_test
        hi = mu_test + z_hi * sigma_test
        empirical.append(float(np.mean((y_test >= lo) & (y_test <= hi))))

    plt.figure(figsize=(5.5, 5.5))
    plt.plot(central_levels, empirical, marker="o", linewidth=1.5, label="Empirical coverage")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Ideal")
    plt.title(f"PI Calibration (test) ({tag})")
    plt.xlabel("Nominal central PI coverage")
    plt.ylabel("Empirical coverage")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"deepar_{tag}_pi_calibration.png"), dpi=200)
    plt.close()

    pit = normal_cdf((y_test - mu_test) / sigma_test)
    plt.figure(figsize=(6.5, 4))
    plt.hist(pit, bins=12, range=(0, 1), edgecolor="black", alpha=0.8)
    plt.title(f"PIT Histogram (test) ({tag})")
    plt.xlabel("PIT = CDF(observed)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"deepar_{tag}_pit_hist.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/Users/shyam/Downloads/isro/io-altimeter.nc")
    parser.add_argument("--sla-glob", default="")
    parser.add_argument("--context-length", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=40)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output", default="/Users/shyam/Downloads/isro/io_altimeter_deepar_rolling_forecast.nc")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-dir", default="/Users/shyam/Downloads/isro/plots")
    parser.add_argument("--plot-last", type=int, default=120)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.sla_glob:
        time, series = load_sla_series(args.sla_glob)
    else:
        time, series = load_series(args.path)
    n = len(series)
    train_len = int(n * args.train_ratio)
    train_series = series[:train_len]
    mean = train_series.mean()
    std = train_series.std() if train_series.std() > 0 else 1.0
    series_norm = (series - mean) / std

    device = torch.device("cpu")
    model = train_model(
        series_norm[:train_len],
        context_length=args.context_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )

    preds_norm, sigmas_norm = rolling_forecast(model, series_norm, args.context_length, device)
    preds = preds_norm * std + mean
    sigmas = sigmas_norm * std

    forecast_start = args.context_length
    forecast_times = time[forecast_start:]
    observed = series[forecast_start:]

    z10 = -1.281551565545
    z90 = 1.281551565545
    p10 = preds + z10 * sigmas
    p50 = preds
    p90 = preds + z90 * sigmas

    test_start = max(train_len, forecast_start)
    test_slice = slice(test_start - forecast_start, None)
    test_obs = observed[test_slice]
    test_pred = preds[test_slice]
    rmse = float(np.sqrt(np.mean((test_pred - test_obs) ** 2)))
    mae = float(np.mean(np.abs(test_pred - test_obs)))
    coverage = float(np.mean((test_obs >= p10[test_slice]) & (test_obs <= p90[test_slice])))
    interval_width = float(np.mean(p90[test_slice] - p10[test_slice]))

    ds_out = xr.Dataset(
        data_vars={
            "forecast_mean": ("time", preds),
            "forecast_p10": ("time", p10),
            "forecast_p50": ("time", p50),
            "forecast_p90": ("time", p90),
            "forecast_sigma": ("time", sigmas),
            "observed": ("time", observed),
        },
        coords={"time": forecast_times},
        attrs={
            "model": "DeepAR",
            "context_length": args.context_length,
            "train_ratio": args.train_ratio,
            "rmse_test": rmse,
            "mae_test": mae,
            "p10_p90_coverage_test": coverage,
            "p10_p90_interval_width_test": interval_width,
        },
    )
    ds_out.to_netcdf(args.output)

    if args.make_plots:
        tag = "sla" if args.sla_glob else "swh"
        make_pi_plots(
            times=forecast_times,
            observed=observed,
            mu=preds,
            sigma=sigmas,
            test_slice=test_slice,
            out_dir=args.plot_dir,
            tag=tag,
            last_n=args.plot_last,
        )

    print("Forecast saved:", args.output)
    print("Test RMSE:", rmse)
    print("Test MAE:", mae)
    print("Test P10-P90 coverage:", coverage)
    print("Test P10-P90 interval width:", interval_width)


if __name__ == "__main__":
    main()
