# HomeSense-WiRF: A Multimodal Wireless Sensing Dataset for Household Abnormal Activity Recognition

HomeSense-WiRF is a multimodal wireless sensing dataset and benchmark designed for **household abnormal activity recognition**. The dataset synchronously captures **Wi-Fi Channel State Information (CSI)** and **RFID phase measurements**, enabling **privacy-preserving, contact-free human behavior sensing** without the use of cameras or wearable devices.

This repository provides:

* The HomeSense-WiRF dataset (raw and processed)
* Preprocessing scripts for Wi-Fi CSI and RFID phase signals
* Baseline models and multimodal fusion implementations
* Experimental configurations and evaluation utilities for reproducible benchmarking

---

## 📌 Key Features

* **Multimodal wireless sensing:** Wi-Fi CSI + RFID phase
* **Privacy-preserving:** No vision sensors or wearable devices
* **Risk-oriented activities:** Focus on safety-critical household abnormal behaviors
* **Benchmark-ready:** Unified preprocessing and evaluation protocols
* **Cross-subject evaluation:** Explicit support for generalization studies

---

## 🔄 Preprocessing

All preprocessing steps, including **signal denoising**, **temporal alignment**, **interpolation**, and **dataset splitting**, are implemented through the preprocessing-related scripts and directories in this repository.

The processed dataset is **fully aligned with the experimental protocols described in the paper** and is recommended for **direct benchmarking and result reproduction**.

---

## 📥 Dataset Access

### Processed Dataset (Benchmark Dataset)

The processed dataset used in our experiments is publicly available and can be directly used for training and evaluation.

* Download link: [https://www.kaggle.com/datasets/xiaofeifw/homesense-wirf](https://www.kaggle.com/datasets/xiaofeifw/homesense-wirf)
* Directory: `./dataset/raw_dataset/`

### Raw Dataset

The raw Wi-Fi CSI and RFID phase measurements will be **released after the acceptance of the associated paper**.

* Status: Not publicly available at this stage
* Release plan: To be made available upon paper acceptance

> **Note:** The processed dataset is fully aligned with the experimental protocols described in the paper and is recommended for direct benchmarking and result reproduction.

---

## ✅ Reproducing Experiments

### Prerequisites

* Linux (recommended)
* Python 3.8 or later
* CPU or NVIDIA GPU with CUDA/cuDNN support

### Installation

Clone the repository:

```bash
git clone <REPO_URL>
cd HomeSense-WiRF
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Using Conda:

```bash
conda env create -f environment.yaml
conda activate homesense-wirf
```

---

## 🧪 Training and Evaluation

The following commands reflect the default experimental workflow. Specific arguments can be adjusted according to your configuration.

1. Split training and testing data:

```bash
python split_train_test.py
```

2. Generate label files:

```bash
python generate_txt.py
```

3. Train models:

```bash
python STR-Net_WiFi.py
python STR-Net_RFID.py
python WiRF_Fusion.py
```

4. Results:
   Training logs, model checkpoints, and evaluation outputs are saved under the `./results/` directory (depending on script settings).

---

## 📂 Repository Structure

```
HomeSense-WiRF/
│  README.md
│  environment.yaml
│  opts.py
│  generate_txt.py
│  split_train_test.py
│  XRFDataset.py
│  STR-Net_RFID.py
│  STR-Net_WiFi.py
│  WiRF_Fusion.py
│
├─dataset/
│  │  README.md
│  ├─HS-WIRF_dataset/      # processed / benchmark-ready data (recommended for training)
│  └─Raw_dataset/          
│
├─preprocessing/
│  │  README.md
│  │  preprocess_wifi.py
│  └─preprocess_rfid.py
│
└─results/                 # logs, checkpoints, metrics, confusion matrices, etc.


---

## 📌 Benchmark Protocol

* **Evaluation setting:** Cross-subject evaluation protocol
* **Metrics:** Accuracy, F1-score, etc. (as specified in the paper)
* **Reproducibility:** Fixed data splits and unified preprocessing are provided

## 📜 License

The processed dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

You are free to: use, share, and adapt the dataset for non-commercial research and educational purposes, with appropriate attribution.

You may not: use the dataset for commercial purposes without prior permission.

The source code in this repository is provided for research use. (Optionally specify a code license if applicable.)
