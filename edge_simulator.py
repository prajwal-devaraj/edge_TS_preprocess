import os
import json
import numpy as np
import requests

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "testing_cycles.npz")
EDGE_PARAMS_PATH = os.path.join(ROOT_DIR, "data", "processed", "edge_params.json")

API_URL = "http://127.0.0.1:5000/predict"


def load_edge_params():
    if not os.path.exists(EDGE_PARAMS_PATH):
        print("edge_params.json not found. Skipping edge parameter load.")
        return None

    with open(EDGE_PARAMS_PATH, "r") as f:
        return json.load(f)


def simulate_streaming(cycle_name, cycle_windows, batch_size=5):
    """
    cycle_windows shape: (num_windows, 500, 3)
    Sends small batches to backend to simulate edge streaming.
    """
    n = cycle_windows.shape[0]
    print(f"\nStreaming cycle: {cycle_name} | total windows: {n}")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = cycle_windows[start:end]

        payload = {
            "cycle_name": cycle_name,
            "data": batch.tolist()
        }

        response = requests.post(API_URL, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(
                f"[{cycle_name}] windows {start}:{end} | "
                f"mean_mae={result['mean_mae']:.6f} | "
                f"alarm_3sigma={result['alarm_3sigma']} | "
                f"alarm_4sigma={result['alarm_4sigma']}"
            )
        else:
            print(f"Request failed for {cycle_name} windows {start}:{end}")
            print(response.text)


def main():
    edge_params = load_edge_params()
    if edge_params is not None:
        print("Loaded edge params successfully.")
        if "global_median" in edge_params:
            print("Found global_median in edge params.")
        if "global_iqr" in edge_params:
            print("Found global_iqr in edge params.")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Testing archive not found: {TEST_DATA_PATH}")

    archive = np.load(TEST_DATA_PATH)

    print("Available cycles in testing_cycles.npz:")
    for name in archive.files:
        print(" -", name)

    for cycle_name in archive.files:
        cycle_windows = archive[cycle_name]

        if cycle_windows.ndim != 3 or cycle_windows.shape[1:] != (500, 3):
            print(f"Skipping {cycle_name}: unexpected shape {cycle_windows.shape}")
            continue

        simulate_streaming(cycle_name, cycle_windows, batch_size=5)


if __name__ == "__main__":
    main()