# Week 2 —  Boston Housing Dataset

**Assignment 2 | Introduction to Machine Learning**


---

## Objective

Predict house prices (MEDV) from the Boston Housing Dataset using Linear Regression implemented **from scratch** (Batch GD and SGD) and via **scikit-learn**, with a full EDA and correlation-based feature selection pipeline.

---

## Dataset

- **Source:** Boston Housing Dataset (`HousingData.csv`)
- **Size:** 506 samples, 13 features + 1 target (MEDV)
- **Target:** Median value of owner-occupied homes (in $1000s)
- **Preprocessing:** Mean imputation for missing values, Z-score normalization

---

## Pipeline

```
Load Data → Missing Value Imputation (mean) → Z-score Normalization
→ EDA (pairplot, correlation heatmap) → Feature Selection (threshold = 0.75)
→ Train/Test Split (75/25) → Model Training → Evaluation
```

---

## Models Implemented

### 1. Linear Regression from Scratch

**Batch Gradient Descent (BGD)**
- Updates weights using gradient computed over the entire training set
- Converges stably; suitable for small datasets
- `lr = 0.01`, `epochs = 1500`

**Stochastic Gradient Descent (SGD)**
- Updates weights one sample at a time per epoch
- Noisier convergence; faster per-epoch but less stable
- `lr = 0.001`, `epochs = 1500`

### 2. Scikit-learn `LinearRegression`
- Closed-form OLS as baseline comparison

---

## Experiments: With vs Without Correlated Feature Removal

| Experiment | Description |
|------------|-------------|
| Full features | All 13 features used |
| Reduced features | Highly correlated pairs (ρ > 0.75) removed |

---

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Key Finding

> Batch Gradient Descent is the most reliable approach for linear regression on small datasets like Boston Housing. Feature correlation handling improves model stability and interpretability, though it may slightly reduce raw accuracy. Linear Regression provides a strong baseline.

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python 3 | Core language |
| NumPy | Matrix ops, gradient computation |
| Pandas | Data loading and EDA |
| Matplotlib / Seaborn | Visualization (pairplot, heatmap, scatter) |
| Scikit-learn | Baseline LinearRegression, train_test_split |

---

## Files

```
Bosting_Housing_Dataset_Assignment2_PoonamGupta.ipynb
HousingData.csv  (required to run)
```

---

## Learning Outcomes

- Implemented BGD and SGD from scratch without ML libraries
- Understood impact of feature correlation on regression stability
- Compared from-scratch vs library-based model performance
