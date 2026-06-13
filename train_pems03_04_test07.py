"""Train LightGBM on PEMS03+04, evaluate on PEMS07 (cross-dataset). v3

Fixes vs v2:
1. MEMORY: samples rows *per sensor while building* instead of materializing
   the full feature matrix first.  The real PEMS07 (28224, 883, 1) would
   otherwise create ~25M rows x 27 features (~3 GB+) before sampling, which
   caused the ArrayMemoryError.
2. Sensible non-zero defaults for --max-test-samples / --max-val-samples.
3. PEMS07 is now the original (un-normalized) dataset, so test metrics are
   also reported in original units (denormalized), fully comparable to train.
4. Frees intermediate arrays aggressively (del + gc).
"""

from pathlib import Path
import argparse
import gc
import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILES = [
    DATA_DIR / "PEMS03.npz",
    DATA_DIR / "PEMS04.npz",
]
TEST_FILE = DATA_DIR / "PEMS07.npz"

MODEL_FILE = BASE_DIR / "app" / "traffic_model.txt"
METRICS_FILE = DATA_DIR / "model_metrics.json"
DETAIL_METRICS_FILE = DATA_DIR / "model_metrics_pems03_04_to_pems07.json"

RANDOM_SEED = 42
STEPS_PER_DAY = 288  # 5-minute intervals


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_npz_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    loaded = np.load(path)
    array = loaded["data"] if "data" in loaded.files else loaded[loaded.files[0]]
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected (time, sensors, features), got {array.shape} from {path}")
    return array


def clean_series(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    median_value = np.nanmedian(series)
    if not np.isfinite(median_value):
        median_value = 0.0
    return np.nan_to_num(series, nan=median_value, posinf=median_value, neginf=median_value)


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def build_features_for_sensor(series: np.ndarray, window: int):
    """Features from a per-sensor z-scored series.

    Layout: [lag_W..lag_1, diff_(W-1)..diff_1, roll_mean, roll_std, sin_tod, cos_tod]
    Returns (x, y, t_index) or None.
    """
    n = len(series)
    if n <= window + 1:
        return None

    win = np.lib.stride_tricks.sliding_window_view(series, window + 1)
    lags = win[:, :window]
    y = win[:, window]

    diffs = np.diff(lags, axis=1)
    roll_mean = lags.mean(axis=1, keepdims=True)
    roll_std = lags.std(axis=1, keepdims=True)

    t_index = np.arange(window, n)
    tod = (t_index % STEPS_PER_DAY) / STEPS_PER_DAY
    sin_tod = np.sin(2 * np.pi * tod, dtype=np.float32)[:, None]
    cos_tod = np.cos(2 * np.pi * tod, dtype=np.float32)[:, None]

    x = np.hstack([lags, diffs, roll_mean, roll_std, sin_tod, cos_tod]).astype(np.float32)
    return x, y.astype(np.float32), t_index


def feature_names(window: int):
    names = [f"lag_{i}" for i in range(window, 0, -1)]
    names += [f"diff_{i}" for i in range(window - 1, 0, -1)]
    names += ["roll_mean", "roll_std", "sin_tod", "cos_tod"]
    return names


def build_dataset(path: Path, name: str, window: int, max_sensors: int | None,
                  val_ratio: float, max_train_samples: int | None,
                  max_val_samples: int | None, seed: int):
    """Per-sensor z-score + feature building, with PER-SENSOR subsampling.

    Sampling happens inside the sensor loop so the full feature matrix is
    never materialized -> memory stays bounded regardless of dataset size.
    """
    array = load_npz_array(path)
    print("=" * 80)
    print(f"Loading {name}: {path}")
    print(f"Raw shape: {array.shape}")

    time_steps, sensor_count, _ = array.shape
    if max_sensors is not None:
        sensor_count = min(sensor_count, max_sensors)

    split_t = int(time_steps * (1 - val_ratio)) if val_ratio > 0 else time_steps

    # keep-ratios so that total kept rows ~= the caps
    rows_per_sensor = max(time_steps - window, 1)
    total_train_rows = sensor_count * int(rows_per_sensor * (split_t / time_steps))
    total_val_rows = sensor_count * (rows_per_sensor - int(rows_per_sensor * (split_t / time_steps)))

    train_ratio = 1.0 if not max_train_samples else min(1.0, max_train_samples / max(total_train_rows, 1))
    val_ratio_keep = 1.0 if not max_val_samples else min(1.0, max_val_samples / max(total_val_rows, 1))

    rng = np.random.default_rng(seed)
    parts = {"train": {"x": [], "y": [], "mu": [], "sd": []},
             "val": {"x": [], "y": [], "mu": [], "sd": []}}

    for s in range(sensor_count):
        flow = clean_series(array[:, s, 0])

        mu = float(np.mean(flow))
        sd = float(np.std(flow))
        if sd < 1e-6:  # dead/constant sensor
            continue
        z = ((flow - mu) / sd).astype(np.float32)

        built = build_features_for_sensor(z, window)
        if built is None:
            continue
        x, y, t_index = built

        train_mask = t_index < split_t
        for key, mask, ratio in (("train", train_mask, train_ratio),
                                 ("val", ~train_mask, val_ratio_keep)):
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            if ratio < 1.0:
                k = max(1, int(round(len(idx) * ratio)))
                idx = rng.choice(idx, size=k, replace=False)
            parts[key]["x"].append(x[idx])
            parts[key]["y"].append(y[idx])
            k = len(idx)
            parts[key]["mu"].append(np.full(k, mu, dtype=np.float32))
            parts[key]["sd"].append(np.full(k, sd, dtype=np.float32))

        del x, y, built

    del array
    gc.collect()

    out = {}
    for key in ("train", "val"):
        if parts[key]["x"]:
            out[key] = {
                "x": np.vstack(parts[key]["x"]),
                "y": np.concatenate(parts[key]["y"]),
                "mu": np.concatenate(parts[key]["mu"]),
                "sd": np.concatenate(parts[key]["sd"]),
            }
            parts[key]["x"].clear(); parts[key]["y"].clear()
            parts[key]["mu"].clear(); parts[key]["sd"].clear()
        else:
            out[key] = None
    gc.collect()

    n_train = len(out["train"]["y"]) if out["train"] is not None else 0
    n_val = len(out["val"]["y"]) if out["val"] is not None else 0
    print(f"Used sensors: {sensor_count}")
    print(f"Samples kept: train={n_train}, val={n_val}")
    return out


def concat_blocks(blocks):
    blocks = [b for b in blocks if b is not None]
    out = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    for b in blocks:
        b.clear()
    gc.collect()
    return out


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def safe_mape(y_true, y_pred, min_abs: float = 10.0):
    mask = np.abs(y_true) > min_abs
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(model, block, name: str):
    pred_z = model.predict(block["x"]).astype(np.float32)
    y_z = block["y"]

    result = {
        "mae_normalized": float(mean_absolute_error(y_z, pred_z)),
        "rmse_normalized": float(np.sqrt(mean_squared_error(y_z, pred_z))),
    }

    y_raw = y_z * block["sd"] + block["mu"]
    pred_raw = pred_z * block["sd"] + block["mu"]
    result["mae"] = float(mean_absolute_error(y_raw, pred_raw))
    result["rmse"] = float(np.sqrt(mean_squared_error(y_raw, pred_raw)))
    result["mape_percent"] = safe_mape(y_raw, pred_raw)

    print()
    print(f"[{name}]")
    print(f"MAE  (z-space):        {result['mae_normalized']:.4f}")
    print(f"RMSE (z-space):        {result['rmse_normalized']:.4f}")
    print(f"MAE  (original units): {result['mae']:.4f}")
    print(f"RMSE (original units): {result['rmse']:.4f}")
    if result["mape_percent"] is not None:
        print(f"MAPE (|y|>10 only):    {result['mape_percent']:.2f}%")

    del pred_z, y_raw, pred_raw
    gc.collect()
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=12,
                        help="Lag steps (12 = 1 hour). model_service.py must match.")
    parser.add_argument("--max-sensors", type=int, default=0,
                        help="Max sensors per dataset; 0 = all.")
    parser.add_argument("--max-train-samples", type=int, default=2_000_000,
                        help="Total cap across all training datasets.")
    parser.add_argument("--max-val-samples", type=int, default=500_000)
    parser.add_argument("--max-test-samples", type=int, default=1_000_000,
                        help="Cap for PEMS07 evaluation (sampled per sensor).")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    window = args.window
    max_sensors = None if args.max_sensors <= 0 else args.max_sensors
    n_train_files = len(TRAIN_FILES)
    cap_train = None if args.max_train_samples <= 0 else args.max_train_samples // n_train_files
    cap_val = None if args.max_val_samples <= 0 else args.max_val_samples // n_train_files
    cap_test = None if args.max_test_samples <= 0 else args.max_test_samples

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("Training LightGBM (per-sensor normalized, cross-dataset) v3")
    print("=" * 80)
    print(f"Train: PEMS03 + PEMS04 | Test: PEMS07 | Window: {window}")

    train_blocks, val_blocks = [], []
    for i, f in enumerate(TRAIN_FILES):
        out = build_dataset(f, f.stem, window, max_sensors,
                            val_ratio=args.val_ratio,
                            max_train_samples=cap_train,
                            max_val_samples=cap_val,
                            seed=RANDOM_SEED + i)
        train_blocks.append(out["train"])
        val_blocks.append(out["val"])

    train = concat_blocks(train_blocks)
    val = concat_blocks(val_blocks)

    test_out = build_dataset(TEST_FILE, "PEMS07", window, max_sensors,
                             val_ratio=0.0,
                             max_train_samples=cap_test,
                             max_val_samples=None,
                             seed=RANDOM_SEED + 99)
    test = test_out["train"]

    print()
    print("=" * 80)
    print(f"Train samples: {len(train['y'])}")
    print(f"Val samples:   {len(val['y'])}")
    print(f"Test samples:  {len(test['y'])}")

    names = feature_names(window)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print()
    print("Training model with early stopping...")
    model.fit(
        train["x"], train["y"],
        feature_name=names,
        eval_set=[(val["x"], val["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    print(f"Best iteration: {model.best_iteration_}")

    train_metrics = evaluate(model, train, "Train: PEMS03 + PEMS04")
    val_metrics = evaluate(model, val, "Validation (held-out time)")
    test_metrics = evaluate(model, test, "Final Test: PEMS07")

    model.booster_.save_model(str(MODEL_FILE))

    metrics = {
        "model_name": "LightGBMRegressor",
        "task": "short_term_traffic_flow_prediction",
        "train_datasets": ["PEMS03", "PEMS04"],
        "final_test_dataset": "PEMS07",
        "window_size": window,
        "normalization": "per-sensor z-score",
        "features": names,
        "train_samples": int(len(train["y"])),
        "val_samples": int(len(val["y"])),
        "test_samples": int(len(test["y"])),
        "best_iteration": int(model.best_iteration_ or 0),
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "model_file": str(MODEL_FILE),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(DETAIL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 80)
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")


if __name__ == "__main__":
    main()

