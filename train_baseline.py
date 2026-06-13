import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "PEMS03.npz"
MODEL_FILE = BASE_DIR / "app" / "traffic_model.txt"
METRICS_FILE = DATA_DIR / "model_metrics.json"

WINDOW_SIZE = 3
TEST_RATIO = 0.2
RANDOM_SEED = 42


def load_pems_flow_series(data_file: Path, sensor_index: int = 0) -> np.ndarray:
    """
    Load one sensor's traffic flow series from PEMS03.

    Expected PEMS03 shape is usually:
    (time_steps, sensors, features)

    This project uses feature index 0 as traffic flow.
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    data = np.load(data_file)

    if "data" in data:
        traffic_matrix = data["data"]
    else:
        first_key = list(data.keys())[0]
        traffic_matrix = data[first_key]

    if traffic_matrix.ndim != 3:
        raise ValueError(
            f"Expected PEMS data shape to be 3D, got shape: {traffic_matrix.shape}"
        )

    if sensor_index >= traffic_matrix.shape[1]:
        raise ValueError(
            f"sensor_index={sensor_index} is out of range. "
            f"Dataset has {traffic_matrix.shape[1]} sensors."
        )

    flow_series = traffic_matrix[:, sensor_index, 0].astype(float)

    return flow_series


def create_sliding_window_dataset(series: np.ndarray, window_size: int):
    """
    Convert a 1D time series into supervised learning samples.

    X[t] = previous window_size flow values
    y[t] = next flow value
    """
    X = []
    y = []

    for index in range(window_size, len(series)):
        X.append(series[index - window_size:index])
        y.append(series[index])

    return np.array(X), np.array(y)


def train_test_split_time_series(X, y, test_ratio: float):
    """
    Time-series split: use earlier data for training and later data for testing.
    No shuffling is used because this is temporal data.
    """
    split_index = int(len(X) * (1 - test_ratio))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    Zero true values are ignored to avoid division by zero.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        return None

    return float(
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        )
        * 100
    )


def train_model(X_train, y_train):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    return model


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    flow_series = load_pems_flow_series(DATA_FILE, sensor_index=0)

    X, y = create_sliding_window_dataset(flow_series, WINDOW_SIZE)

    X_train, X_test, y_train, y_test = train_test_split_time_series(
        X,
        y,
        TEST_RATIO,
    )

    model = train_model(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

    train_mape = calculate_mape(y_train, y_train_pred)
    test_mape = calculate_mape(y_test, y_test_pred)

    booster = model.booster_
    booster.save_model(str(MODEL_FILE))

    metrics = {
        "model": "LightGBMRegressor",
        "dataset": "PEMS03",
        "sensor_index": 0,
        "feature": "flow",
        "window_size": WINDOW_SIZE,
        "test_ratio": TEST_RATIO,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_mape_percent": train_mape,
        "test_mape_percent": test_mape,
        "model_file": str(MODEL_FILE),
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== Local Model Training Completed ===")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print("\n[Evaluation Metrics]")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
