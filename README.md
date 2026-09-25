# 🏗️ Heavy Equipment Price Prediction

**Predicting the resale price of heavy industrial equipment using classical machine learning**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-red)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-gradient%20boosting-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)

**Best Model: XGBoost (tuned) | Validation RMSLE: 0.2066 | Leaderboard RMSLE: 0.20491 | Validation R²: 0.9048**

---

## 📌 Overview

This project predicts `TargetValue` — the transaction price of a piece of heavy equipment — from transactional, operational, and technical specification data. It's framed as a regression problem evaluated on **RMSLE (Root Mean Squared Logarithmic Error)**, submitted as part of the MLP Diploma Project (Kaggle competition format).

The pipeline combines careful exploratory analysis of target skew, missingness, and categorical cardinality with engineered age/usage/specification features, then compares five classical ML regressors before tuning and finalizing the strongest one.

---

## 🗂️ Repository Structure

```
heavy-equipment-price-prediction/
├── Notebook/
│   ├── EDA_and_Visualisation.ipynb                  # Exploratory data analysis
│   ├── Feature_eng_and_preprocessing_data.ipynb     # Feature engineering & preprocessing pipeline
│   └── complete_notebook_code.ipynb                 # End-to-end annotated notebook (EDA → submission)
├── Models/
│   └── Models.ipynb                                 # Candidate model training (Ridge, RF, XGBoost, LightGBM, CatBoost)
├── Scripts/
│   └── model_comaprison_code.ipynb                  # Validation metrics & model comparison
├── src/
│   └── hyperparameter_tunning_optimization.ipynb    # RandomizedSearchCV tuning of the winning model
├── report/
│   └── Heavy_Equipment_Price_Prediction_Report.pdf  # Full written project report
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

The dataset describes individual heavy-equipment transactions: **138,701** training rows and **15,000** test rows, spanning transactional metadata, operational indicators, and a large block of technical specification fields.

| Field | Description |
|---|---|
| `TargetValue` | Transaction price in USD — the regression target |
| `ManufactureYear` | Year the equipment was produced |
| `OperationalHoursMeter` | Lifetime logged run-time hours |
| `TransactionDate` | Date of the transaction |
| `Spec_FullDescriptor` | Free-text string of technical parameters |
| `Spec_BaseClass` / `SubClass` | Hierarchical equipment category |
| `RegionCode` / `VendorPartnerID` | Geographic and vendor identifiers |
| `col1`–`col30` | Anonymized technical spec fields; several missing in 80–99.95% of rows |

> ⚠️ Raw competition data (`train.csv`, `test.csv`) is **not included** in this repo, per Kaggle competition data-sharing rules. See the notebooks for the expected input path.

---

## 🔧 Pipeline

| Step | Description |
|---|---|
| **1. Data Loading** | Load train/test/metadata; sanity-check shapes, dtypes, and target skew |
| **2. EDA** | Target distribution (raw vs. log1p), missing-value profiling, numeric/categorical split, uni/bivariate plots vs. price, correlation heatmap, cardinality analysis |
| **3. Feature Engineering** | `EquipmentAge`, `HoursPerYear`, date-part decomposition, regex-parsed specification tokens (`Spec_TokenCount`, `Spec_FirstNumber`), `MissingSpecCount` |
| **4. Preprocessing** | Shared `ColumnTransformer` — median imputation + scaling for numerics, constant imputation + ordinal encoding for categoricals |
| **5. Model Training** | 5 candidate regressors trained on an 80/20 split: Ridge, Random Forest, XGBoost, LightGBM, CatBoost |
| **6. Model Comparison** | RMSLE, MAE, and R² compared across all five on a common validation set |
| **7. Hyperparameter Tuning** | `RandomizedSearchCV` (15 candidates, 3-fold CV) tuning the winning model — XGBoost |
| **8. Feature Importance** | Top-20 importances from the tuned model, used to guide further iteration |
| **9. Final Fit & Submission** | Refit on full training data; predictions written to `submission.csv` |

---

## 🧠 Models & Results

| Model | RMSLE | MAE (USD) | R² |
|---|---|---|---|
| Ridge (baseline) | 0.5280 | 15,986.7 | 0.3779 |
| Random Forest | 0.2249 | 6,159.0 | 0.8872 |
| CatBoost | 0.2278 | 6,243.3 | 0.8842 |
| LightGBM | 0.2169 | 5,918.9 | 0.8950 |
| XGBoost | 0.2137 | 5,789.1 | 0.8981 |
| **XGBoost (tuned) ⭐** | **0.2066** | **5,546.3** | **0.9048** |

**Best model:** XGBoost, tuned via `RandomizedSearchCV` — 500 trees, max depth 8, learning rate 0.08.
**Leaderboard RMSLE:** 0.20491 (closely tracking the validation estimate).

---

## ✨ Key Features Engineered

- **Age & usage:** `EquipmentAge` (transaction year − manufacture year), `HoursPerYear` (meter hours ÷ age)
- **Date parts:** transaction year, month, day-of-week
- **Specification parsing:** `Spec_TokenCount`, `Spec_FirstNumber` — extracted via regex from the free-text `Spec_FullDescriptor` field
- **Missingness as signal:** `MissingSpecCount` — count of missing anonymized spec fields per row, turning a structural missingness pattern into a usable feature

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/23f3001514/heavy-equipment-price-prediction.git
cd heavy-equipment-price-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost

# Launch the notebooks
jupyter notebook Notebook/
```

---

## 🛠️ Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · XGBoost · LightGBM · CatBoost · regex · SciPy

---

## 📁 Outputs

- `submission.csv` — final `TargetValue` predictions in competition format
- EDA visualizations — target distribution, missing-value ranking, correlation heatmap, category-wise price boxplots
- Feature-importance chart from the tuned model
- `report/Heavy_Equipment_Price_Prediction_Report.pdf` — full written report

---

## 📈 Future Improvements

- Blend XGBoost, LightGBM, and CatBoost predictions (inverse-RMSLE weighted) instead of a single final model
- Drop or better-engineer the two columns missing in 99.95% of rows (`col18`, `col19`)
- Parse numeric values out of the remaining anonymized `colN` text fields
- Pass CatBoost raw (unencoded) categorical columns via its native `cat_features` handling instead of pre-encoding
- Expand the randomized hyperparameter search given additional compute budget

---

## 📄 License

This project is available under the [MIT License](LICENSE).

---

*Built for the Heavy Equipment Selling Price Prediction Challenge — MLP Diploma Project, 2026 T2.*
