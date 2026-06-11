# Introduction to Machine Learning 


---

## Repository Structure

```
intro-to-ml-assignments/
│
├── Week1_FunctionAnalyzer/
│   └── Assignment1_240751_PoonamGupta_Intro_to_ML.ipynb
│
├── Week2_BostonHousing/
│   └── Bosting_Housing_Dataset_Assignment2_PoonamGupta.ipynb
│
├── Week3_ClassificationClustering/
│   ├── Assignment3_240751_Purchase_logistic.ipynb
│   ├── Assignment3_240751_iris_dataset.ipynb
│   ├── Assignment3_Titanic_dataset_240751.ipynb
│   └── Assignment3_Report_240751.pdf
│
└── Week4_MLFundamentals/
    └── Assignment4__1_.pdf
```

---

## Assignment Summary

| Week | Assignment | Topics | Tools |
|------|------------|--------|-------|
| 1 | Mathematical Function Analyzer | Numerical methods, differentiation, integration, boundary handling | NumPy, Matplotlib |
| 2 | Boston Housing Price Prediction | Linear regression from scratch, BGD, SGD, feature correlation | NumPy, Pandas, Scikit-learn |
| 3 | Multi-Dataset Classification & Clustering | Logistic Regression, KNN, SVM, Random Forest, K-Means | Scikit-learn, Seaborn |
| 4 | ML Theory & Derivations | Least squares, gradient descent, neural networks, backpropagation | LaTeX (PDF report) |

---

## Week 1 — Mathematical Function Analyzer

**Notebook:** `Assignment1_240751_PoonamGupta_Intro_to_ML.ipynb`

Builds an interactive calculator for two user-defined mathematical functions evaluated over a continuous domain.

**Implemented:**
- Dynamic function parsing with `eval()` and safe math wrappers
- Arithmetic operations: addition, subtraction, multiplication, division, power, modulo, log-ratio
- Numerical differentiation (`numpy.gradient`) and integration (`numpy.trapz`)
- Boundary condition detection: division-by-zero, negative base with fractional exponent, overflow (`> 1e10`), log of non-positives
- NaN masking and `clean()` utility for overflow/infinity replacement
- Multi-panel visualization with NaN regions highlighted

**Key Takeaway:** Numerical robustness requires explicit boundary handling; `np.nan` propagation is the correct approach over silent errors.

---

## Week 2 — Linear Regression on Boston Housing Dataset

**Notebook:** `Bosting_Housing_Dataset_Assignment2_PoonamGupta.ipynb`

Predicts median house prices (MEDV) using Linear Regression implemented fully from scratch, with scikit-learn as a validation baseline.

**Implemented:**
- EDA: distribution histograms, pairplot, correlation heatmap
- Missing value imputation (column mean)
- Z-score normalization
- Correlation-based feature selection (threshold ρ > 0.75)
- **Batch Gradient Descent from scratch** (`lr=0.01`, `epochs=1500`)
- **Stochastic Gradient Descent from scratch** (`lr=0.001`, `epochs=1500`)
- Custom `evaluate()` returning MSE, RMSE, R²
- Experiments: full features vs. reduced (correlated features removed)
- Baseline: scikit-learn `LinearRegression`

**Key Finding:** BGD converges more stably than SGD on the Boston dataset. Removing highly correlated features improves model interpretability and reduces multicollinearity without significantly hurting R².

---

## Week 3 — Classification & Clustering (3 Experiments)

### Q1 — Customer Purchase Prediction

**Notebook:** `Assignment3_240751_Purchase_logistic.ipynb`

**Dataset:** Customer demographics (Gender, Age, EstimatedSalary) → Purchased (binary)

| Model | Accuracy | Recall (Buyers) | F1-Score |
|-------|----------|-----------------|----------|
| Logistic Regression (Unbalanced) | 0.81 | 0.61 | 0.70 |
| Logistic Regression (Balanced) | 0.83 | 0.81 | 0.77 |
| KNN (Distance-Weighted) | **0.91** | **0.89** | **0.88** |

K-Means clustering (Age + EstimatedSalary, k=2) produced interpretable customer segments:
- **High-Value Customer:** Older, higher salary, high purchase probability
- **Low-Value Customer:** Younger, lower salary, low purchase probability

**Key Finding:** `class_weight='balanced'` boosts buyer recall from 0.61 → 0.81. KNN captures non-linear purchase boundaries that logistic regression cannot.

---

### Q2 — Iris Sepal Length Prediction (Regression)

**Notebook:** `Assignment3_240751_iris_dataset.ipynb`

**Dataset:** IRIS (Kaggle) — predict sepal_length from sepal_width, petal_length, petal_width

| Model | MSE | R² | Correct Predictions (±0.35 cm) |
|-------|-----|----|-------------------------------|
| KNN (Euclidean) | 0.1069 | 0.854 | 27/30 |
| KNN (Manhattan) | 0.1058 | 0.854 | 26/30 |
| **SVM (RBF)** | **0.1003** | **0.862** | **29/30** |

Hyperparameters tuned via GridSearchCV + 5-fold CV. Custom tolerance metric (±0.35 cm) used alongside standard R².

**Key Finding:** SVM with RBF kernel is the best model. Euclidean distance outperforms Manhattan for KNN on this dataset.

---

### Q3 — Titanic Survival Prediction

**Notebook:** `Assignment3_Titanic_dataset_240751.ipynb`

**Dataset:** Titanic (train + test + gender_submission)

EDA insights: female survival significantly higher; 1st class passengers survived more; Age imputed per Pclass median (1→37, 2→29, 3→24); Cabin dropped due to high missingness.

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **0.93** |
| Random Forest | ~0.89 |
| SVM | ~0.85 |
| KNN | ~0.74 |

**Key Finding:** Logistic Regression outperforms all complex models, indicating strong linear separability in the feature space after preprocessing. Feature engineering is the primary driver of performance.

**Report:** `Assignment3_Report_240751.pdf`

---

## Week 4 — ML Fundamentals: Theory & Derivations

**Report:** `Assignment4__1_.pdf` | **Group 3:** Aditya Singh, Krishnapriya Reddy, Poonam Gupta

34-question theoretical assignment covering the complete ML curriculum:

- **Regression:** Least squares derivation, normal equation, Moore-Penrose pseudoinverse, SVD-based solution
- **Optimization:** Gradient descent (BGD, SGD, Mini-batch), learning rate effects, convergence
- **Classification:** Logistic regression, sigmoid, binary cross-entropy, gradient derivation
- **Regularization:** L1 (Lasso) — feature selection via sparsity; L2 (Ridge) — weight shrinkage
- **Clustering:** K-Means steps and math, elbow method (WCSS), silhouette score
- **Evaluation:** Confusion matrix, precision, recall, F1-score, K-fold cross-validation
- **Neural Networks:** Perceptron, feedforward networks, activation functions (ReLU vs Sigmoid), backpropagation via chain rule, overfitting prevention

---

## Tech Stack

```
Languages    : Python 3
Notebooks    : Jupyter / Google Colab
ML Library   : Scikit-learn (LogisticRegression, KNN, SVM, RandomForest, SVR, Pipeline, GridSearchCV)
Data          : Pandas, NumPy
Visualization: Matplotlib, Seaborn
Reports      : LaTeX
```

---

## Setup & Run

```bash
git clone https://github.com/<your-username>/intro-to-ml-assignments.git
cd intro-to-ml-assignments
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Open any notebook
jupyter notebook Week2_BostonHousing/Bosting_Housing_Dataset_Assignment2_PoonamGupta.ipynb
```

> **Note:** Notebooks were originally developed on Google Colab. The `from google.colab import files` cell can be removed when running locally — replace with `pd.read_csv('filename.csv')` directly.

---

