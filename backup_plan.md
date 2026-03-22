# Project Proposal (Backup): Edge-Cloud Collaborative Human Activity Recognition (HAR) with Cross-Device Heterogeneity

## Project Objective
This project aims to implement an Edge-Cloud collaborative framework for [Human Activity Recognition (HAR)](https://www.kaggle.com/datasets/chumajin/heterogeneity-human-activity-recognition-dataset). Using streaming data from smartphone and smartwatch sensors (accelerometer and gyroscope), the system classifies user activities into six categories (e.g., walking, sitting, standing). The core focus is on handling Device Heterogeneity—addressing signal variations across hardware—and optimizing the deployment of 1D-Convolutional Neural Networks (1D-CNN) for real-time inference.

## Dataset & Heterogeneity Challenge
The project utilizes the Heterogeneity Human Activity Recognition (HHAR) Dataset. The technical challenge shifts from "temporal drift" to System Heterogeneity:

- Sensor Diversity: Data originates from multiple smartphone and watch models, each with distinct noise profiles and dynamic ranges.

- Sampling Rate Mismatch: Devices operate at varying frequencies (e.g., 50Hz vs. 100Hz), requiring edge-side synchronization.

- Supervised Learning: The dataset provides high-fidelity labels, making it ideal for training robust multiclass classifiers.

## System Architecture (Edge-Cloud Strategy)
Phase 1: Cloud-Side Model Development
- Model Selection:
  1. Baseline (Logistic Regression): Utilizing hand-crafted features (Mean, Variance, FFT coefficients) to establish a performance floor.
  2. Champion (1D-CNN): A deep learning approach that treats sensor time-series as "1D images." The CNN will automatically extract spatial-temporal features (e.g., the specific "curve" of a walking step) without manual feature engineering.

- Computation: The 1D-CNN is designed to be lightweight, ensuring it can be trained on standard consumer-grade laptops while remaining efficient enough for eventual edge deployment.

Phase 2: Edge-Side Preprocessing & Resampling

To address the severe system heterogeneity and minimize the computational load on edge devices, a strict, low-latency $O(N)$ preprocessing pipeline will be executed locally before data transmission. The execution order is mathematically optimized to handle hardware jitter and unsynchronized clocks:

- **Step 1: Absolute Time Synchronization (Offset Calibration):** Hardware clocks across different brands are often unsynchronized (e.g., some record Unix Epoch, others record system uptime). The edge device will first calculate a fixed temporal offset using the median difference between the local `Creation_Time` and the globally synced `Arrival_Time`. This calibration maps all heterogeneous data streams onto a unified absolute timeline.
- **Step 2: Threshold Clipping:** Before any mathematical transformations, extreme physical outliers and spikes (e.g., caused by dropping a device or hardware collisions) are clipped to a predefined physical threshold. This prevents severe anomalies from skewing subsequent interpolation steps.
- **Step 3: Linear Interpolation (Frequency Alignment):** Because sensor sampling rates vary (e.g., 50Hz vs. 100Hz) and suffer from CPU scheduling jitter, the calibrated time series is linearly interpolated to a strict, uniform frequency (e.g., 50Hz / 20ms intervals). This crucial step ensures that the temporal distances between all data points are perfectly equal, which is a prerequisite for accurate moving averages and 1D-CNN temporal feature extraction.
- **Step 4: Exponential Moving Average (EMA) Denoising:** With the timeline now perfectly uniform, a lightweight EMA filter is applied to smooth out high-frequency sensor noise. Performing EMA after interpolation ensures the smoothing weights remain physically accurate.
- **Step 5: Decentralized Normalization:** Finally, the edge device applies global normalization parameters ($\mu$ and $\sigma$) broadcasted via OTA (Over-The-Air) updates from the cloud. This zero-centers the data, mitigating individual hardware biases before transmitting the standardized, lightweight data windows to the cloud for inference.

Phase 3: Cloud-Side Real-Time Inference
- Standardized data windows are streamed to the cloud, where the 1D-CNN performs rapid classification.

- The system outputs a live "Activity Status" with a confidence probability (Softmax output).

## Adaptation & Personalization (The MLOps Loop)
Instead of mechanical wear-and-tear, this project simulates User Adaptation:

- User-Specific Fine-Tuning: If the global model underperforms for a specific user’s unique movement pattern, the cloud triggers an Incremental Learning session.

- Transfer Learning: Demonstrating how a model trained on "Phone Data" can be adapted to "Watch Data" with minimal retraining, showcasing the flexibility of the CNN features.

## Interactive Dashboard (Streamlit)
1. Real-Time Motion Visualizer: Displaying the raw XYZ accelerometer waves alongside the 1D-CNN's internal "feature maps" (simplified).
2. Live Classification: A dynamic UI showing the current activity (e.g., "Running," "Stairs") and the model's confidence levels.
3. Heterogeneity Stress Test: A feature allowing users to simulate data from different hardware brands (Samsung vs. LG) to observe how the model handles varying sensor sensitivities.

## Reference
This project idea is inspired by *Tawakuli, A., Kaiser, D., and Engel, T. Modern data preprocessing is holistic, normalized and distributed.2022*.

The HHAR dataset originates from *A. Stisen et al., "Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition," in Proc. 13th ACM Conf. Embedded Networked Sensor Syst. (SenSys '15), 2015, pp. 127–140. doi: 10.1145/2809695.2809718*.