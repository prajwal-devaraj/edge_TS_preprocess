# Cloud-Edge Alarm System & Dashboard Logic Design

## 1. Overview
This document outlines the real-time anomaly detection logic and visual monitoring interface for the CNC machining Edge-Cloud collaborative framework. The core predictive model is a 1D CNN + LSTM Autoencoder that evaluates incoming sensor data in discrete time-windows. To ensure operational reliability and prevent false positives caused by transient noise or normal machine state changes, the Cloud inference pipeline implements state masking, statistical dynamic thresholding, and signal debouncing.

## 2. Edge Device Responsibilities (Protocol Extensions)
Beyond performing local decentralized normalization and transmitting the data windows, the Edge device MUST append state metadata to each payload. This context is critical for the Cloud to apply the correct filtering logic.

* **`operation_id`:** Identifies the current machining operation (e.g., OP07, OP03). This allows the Cloud to dynamically load operation-specific model weights if necessary.
* **`machine_state`:** A status flag (`STARTING`, `RUNNING`, `STOPPING`, `IDLE`). The edge infers this based on overall spindle load or explicit G-code triggers.

**Payload Format Example:**
{
  "machine_id": "M01",
  "operation_id": "OP07",
  "state": "RUNNING",
  "window_data": [[...], [...], ...]
}

## 3. Cloud Inference Pipeline (Flask API)
The Flask API acts as the centralized inference engine. Its primary role is to run the heavy computation and return the raw results, leaving the complex thresholding and alerting logic to the front-end dashboard.

**API Endpoint: `/predict` (POST)**

1. **Data Reception:** Receives the JSON payload from the Edge device.
2. **Inference:**
    * Converts `window_data` back into a 3D tensor suitable for the 1D CNN + LSTM Autoencoder.
    * Performs the forward pass to reconstruct the input.
    * Calculates the Mean Absolute Error (MAE) or Mean Squared Error (MSE) between the input and the reconstruction.
3. **State Masking (Filter 1):**
    * **Logic:** Mechanical vibrations are highly unstable during machine start-up and shutdown, leading to naturally high, non-anomalous reconstruction errors.
    * **Action:** If `machine_state` is `STARTING` or `STOPPING`, the API appends a flag `is_masked: true` to the response. If it is `RUNNING`, the flag is `is_masked: false`.
4. **Response Construction:** Returns the calculated error and the masking flag to the client.

**API Response Example:**
{
  "machine_id": "M01",
  "mae": 0.0452,
  "is_masked": false,
  "timestamp": "2026-04-04T10:37:21Z"
}

## 4. Streamlit Dashboard Integration (Operational Visibility)
Streamlit serves as the interactive control center for factory operators. It acts as an HTTP client, continuously fetching the data from the Flask API (or intercepting the stream) and applying the complex business logic (thresholds, debouncing, health status) to determine the final machine state.

### A. Core Logic Implementation (Front-End Filtering)

1. **Dynamic Thresholding (Statistical Control):**
    * The dashboard maintains the historical baseline parameters ($\mu$ and $\sigma$) calculated from a known "Good" validation set.
    * The user sets a multiplier ($k$) to define the upper control limit: $Threshold = \mu + k\sigma$.
2. **Debounce Mechanism (Filter 2):**
    * **Logic:** A single window with high MAE could be random electromagnetic noise. True mechanical failures (e.g., tool breakage) persist.
    * **Action:** The dashboard maintains a rolling `violation_counter`.
        * If the incoming `mae > Threshold` AND `is_masked == false`: `violation_counter += 1`.
        * If the incoming `mae <= Threshold`: `violation_counter = 0`.
    * An **🔴 Anomaly Alarm** is triggered ONLY if `violation_counter >= Debounce_N`.
3. **Sub-Health Monitoring (Early Warning):**
    * **Logic:** Machine wear and tear (concept drift) causes the overall distribution of errors to shift upwards, increasing the frequency of minor threshold violations even if they don't trigger the continuous debounce alarm.
    * **Action:** The dashboard analyzes a larger rolling window (e.g., the last 50 windows). If the percentage of windows where `mae > Threshold` exceeds a predefined ratio (e.g., 20%), the system triggers a **🟡 Sub-Health Warning**.

### B. Interactive UI Modules

1. **Live MAE Plot:** A real-time line chart plotting the incoming reconstruction errors, overlaying the dynamic dynamic threshold line ($\mu + k\sigma$).
2. **Interactive Parameters Panel (Sidebar):**
    * **Threshold Multiplier ($k$):** Slider to adjust from 3.0 to 4.0.
    * **Debounce Limit ($N$):** Number input to set the required consecutive violations (e.g., 1 to 10).
    * **Sub-Health Settings:** Controls for the evaluation window size and the trigger ratio.
    * *Visual Effect:* Adjusting these parameters instantly recalculates the machine's state on the dashboard.
3. **Model Version Switcher (Demonstrating Concept Drift):**
    * A dropdown to toggle the underlying inference model used by the Flask API between the "Base Model (v1.0)" and a "Fine-tuned Model (v1.1)".
    * *Demonstration Scenario:* When testing data from an aged machine, selecting the Base Model will show a rising baseline MAE (concept drift), frequently triggering sub-health or anomaly alarms. Switching to the Fine-tuned Model will visually drop the MAE back below the threshold, effectively demonstrating the value of continuous learning adaptation.