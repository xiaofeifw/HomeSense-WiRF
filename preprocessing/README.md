# Preprocessing Scripts

This folder contains preprocessing scripts for the **HomeSense-WiRF** dataset.  
The scripts implement signal cleaning, alignment, and formatting procedures for **Wi-Fi CSI** and **RFID phase** data, strictly following the preprocessing protocols described in the paper.

The preprocessing stage converts raw sensing data into standardized numpy arrays that can be directly used for benchmarking and model training.

---

## 📂 Files

### `preprocess_wifi.py`

Preprocessing pipeline for **Wi-Fi CSI** data.

Main steps include:
- Parsing raw CSI frames from `.dat` files
- Extracting CSI amplitude matrices with shape `(3, 30)`
- Temporal alignment and frame truncation
- **Wavelet-based denoising** along the temporal axis
- Multi-receiver fusion by concatenation
- Output formatting as fixed-length feature vectors

**Output format:**
- `.npy` files with shape `(270, T)`, where:
  - `270 = 3 receivers × 3 antennas × 30 subcarriers`
  - `T` is the number of aligned frames

This script corresponds to the Wi-Fi preprocessing described in the time–frequency modeling section of the paper.

---

### `preprocess_rfid.py`

Preprocessing pipeline for **RFID phase** data.

Main steps include:
- Loading raw RFID phase measurements from `.csv` files
- Grouping measurements by EPC (tag ID)
- **Outlier detection and correction** using per-EPC statistical thresholds
- Linear interpolation for temporal alignment
- Resampling to a fixed number of frames
- Padding or truncation to a fixed number of RFID tags

**Output format:**
- `.npy` files with shape `(N, T)`, where:
  - `N` is the maximum number of RFID tags (default: 24)
  - `T` is the resampled temporal length (default: 148)

This script implements the RFID phase preprocessing and normalization procedures described in the dataset construction section.

---

## ▶️ Usage

Each script can be executed independently.  
Example usage (Windows paths):

```bash
python preprocess_wifi.py
python preprocess_rfid.py
