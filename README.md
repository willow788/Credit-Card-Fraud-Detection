# Credit-Card-Fraud-Detection

A machine learning project that explores **credit card fraud detection** using **Random Forest** classifiers and **class-imbalance handling** (including **Adaptive Synthetic Sampling / ADASYN**) in Jupyter notebooks.

> **Note:** This repository is notebook-based (no Python package/module structure yet). The workflow is meant to be run interactively in Jupyter/Colab.

---

## Project Goals

- Understand and explore a credit card transactions dataset (highly imbalanced classification).
- Build and evaluate baseline models using **Random Forest**.
- Improve fraud-class recall/precision tradeoffs using **oversampling techniques** (e.g., **ADASYN** via `imbalanced-learn`).
- Compare experiments across notebook versions.

---

## Repository Structure

```
.
├── Added Adaptive Synthetic Sampling/
│   └── v2.ipynb
├── Random Forest model 1/
│   └── fraud_v1.ipynb
├── Random Forest model 2/
│   └── v2.ipynb
├── Dataset/
│   └── text.txt
├── LICENSE
└── README.md
```

### What each folder contains

- **Random Forest model 1/fraud_v1.ipynb**
  - Loads common data science libraries (Pandas / NumPy / Matplotlib / Seaborn).
  - Performs exploratory steps and trains a Random Forest-based fraud detector.

- **Random Forest model 2/v2.ipynb**
  - A later Random Forest iteration.
  - Includes installing and using `imblearn` / `imbalanced-learn` (useful for imbalance strategies).

- **Added Adaptive Synthetic Sampling/v2.ipynb**
  - Builds on the modeling pipeline by adding **Adaptive Synthetic Sampling (ADASYN)** to address class imbalance.

- **Dataset/text.txt**
  - A short note about dataset sourcing.
  - **The dataset file itself is not currently stored in this repository.**

---

## Dataset

This project expects a credit card fraud dataset similar to the commonly used transactions dataset where:

- Features often include `Time`, `Amount`, and PCA-like components (`V1` … `V28`)
- Target label typically appears as `Class` (e.g., `1` = fraud, `0` = normal)

### Where to get the dataset
The repo note says the dataset was obtained from **GeeksforGeeks**, but the dataset file is not included here.  
To run the notebooks, download the dataset you used and place it locally (or update the notebook paths to point to it).

**Recommendation:** Add the dataset filename (e.g., `creditcard.csv`) to `Dataset/` and update the notebooks to load from `Dataset/creditcard.csv`.

---

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/willow788/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies

This repo does not include a `requirements.txt` yet. Based on the notebooks, you’ll typically need:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

If you run into missing packages while executing cells, install them as needed.

### 4) Launch Jupyter

```bash
jupyter notebook
```

Then open one of the notebooks:

- `Random Forest model 1/fraud_v1.ipynb`
- `Random Forest model 2/v2.ipynb`
- `Added Adaptive Synthetic Sampling/v2.ipynb`

---

## Approach (High Level)

### Baseline Modeling: Random Forest
Random Forest is used as a strong baseline classifier for tabular data. The notebooks generally follow this structure:

1. Load dataset  
2. Quick exploration / summary statistics  
3. Train/test split  
4. Model training (RandomForestClassifier)  
5. Evaluation (confusion matrix, classification report, etc.)

### Handling Class Imbalance: ADASYN (and `imbalanced-learn`)
Fraud detection datasets are usually highly imbalanced (fraud cases are rare). To address this, the project explores synthetic oversampling:

- **ADASYN** generates synthetic minority samples adaptively, focusing on harder-to-learn regions.

This is done via the `imbalanced-learn` ecosystem (installed through `imblearn` / `imbalanced-learn`).

---

## Results / Metrics

This repository currently stores the experiment notebooks, but it does not yet summarize results in the README.

Suggested metrics to report (common for fraud detection):

- Precision / Recall / F1 (especially for the fraud class)
- ROC-AUC / PR-AUC
- Confusion matrix
- Cost-sensitive evaluation (optional)
---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Author

- GitHub: `@willow788`
