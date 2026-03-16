# 🏠 Housing Price Prediction — Ridge, Lasso & Elastic Net

A beginner-to-intermediate machine learning project that builds and compares three regularized linear regression models to predict housing prices. The notebook is heavily commented and includes output analysis, making it a useful learning reference for anyone starting out in ML.

---

## 📌 Project Overview

This project explores how **regularization** affects linear regression models. Instead of just training a model and reporting accuracy, the goal is to understand *why* each model behaves the way it does — how they differ in their approach to coefficients, feature selection, and generalization.

**Three models are compared:**
| Model | Regularization | Key Behaviour |
|---|---|---|
| Ridge | L2 (sum of squared coefficients) | Shrinks all coefficients — keeps every feature |
| Lasso | L1 (sum of absolute coefficients) | Can zero-out coefficients → automatic feature selection |
| Elastic Net | L1 + L2 combined | Balance between Ridge and Lasso |

---

## 📁 Project Structure

```
├── main.ipynb              # Main notebook — models, analysis, visualizations
├── Datasets/
│   └── Housing.csv         # Dataset (545 houses, 12 features)
└── README.md
```

---

## 📊 Dataset

The **Housing dataset** contains 545 records of residential properties with the following features:

**Numeric features (5):** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`

**Categorical features (7):** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

**Target:** `price` (log-transformed during training for better model behaviour)

| Split | Samples | Features |
|---|---|---|
| Training | 381 | 13 (after encoding) |
| Test | 164 | 13 (after encoding) |

---

## ⚙️ Pipeline

```
Raw Data
   │
   ├── Numeric Features  → RobustScaler
   └── Categorical Features → OneHotEncoder (drop='first')
                │
        np.concatenate()
                │
        Processed Matrix (381 × 13)
                │
   ┌────────────┼────────────┐
 Ridge        Lasso     Elastic Net
   └────────────┴────────────┘
                │
        GridSearchCV (5-fold CV)
                │
        Tuned Best Models
                │
        Evaluation & Visualization
```

---

## 📈 Results

### Baseline (manual alpha values)

| Model | Alpha | R² | RMSE | MAE |
|---|---|---|---|---|
| Ridge | 10 | 0.6828 | 1,136,255 | 816,449 |
| Lasso | 0.01 | 0.6419 | 1,207,308 | 850,015 |
| Elastic Net | 0.1 | 0.6668 | 1,164,527 | 829,706 |

### After Hyperparameter Tuning (GridSearchCV)

| Model | Best Alpha | R² | RMSE | MAE |
|---|---|---|---|---|
| Ridge | 1 | **0.6902** | 1,122,911 | 812,754 |
| Lasso | 0.0001 | **0.6906** | 1,122,140 | 813,209 |
| Elastic Net | 0.001 (l1=0.3) | **0.6899** | 1,123,497 | 812,896 |

> **Key finding:** After tuning, all three models converge to nearly identical performance. The best Lasso alpha (0.0001 ≈ no regularization) reveals that all features in this dataset carry meaningful signal — Lasso's feature selection offers no benefit here. The dataset is small and clean, which limits how much regularization can help any of these linear models.

---

## 🛠️ Tech Stack

- Python 3
- pandas 3.0.0
- numpy 2.3.2
- scikit-learn
- matplotlib
- seaborn

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Launch the notebook:
```bash
jupyter notebook main.ipynb
```

> Make sure the `Datasets/Housing.csv` file is in the correct relative path (`../Datasets/Housing.csv` from the notebook location), or update the path in Cell 2.

---

## 📚 What I Learned

- How Ridge, Lasso, and Elastic Net differ mathematically and in practice
- Why Lasso's feature selection is most useful in high-dimensional datasets (many features, few informative ones) — not necessarily on small clean datasets
- The importance of fitting preprocessors only on training data to avoid data leakage
- How GridSearchCV with cross-validation finds better hyperparameters than manual guessing
- Why log-transforming a skewed target variable helps linear models

---

## 🔮 Next Steps

- [ ] Refactor preprocessing with `ColumnTransformer` + `Pipeline` for cleaner, safer code
- [ ] Try **Random Forest** and **Gradient Boosting** — tree-based models capture non-linear relationships that linear models miss, expected R² improvement to ~0.80–0.85
- [ ] Add **feature engineering** (e.g., `area_per_bedroom`, `bathroom_ratio`) to help linear models approximate non-linear patterns
- [ ] Expand the hyperparameter grid using `RandomizedSearchCV` over a continuous range
- [ ] Explore **SHAP values** for model explainability

---

## 📖 References

- [Scikit-learn: Ridge, Lasso, Elastic Net](https://scikit-learn.org/stable/modules/linear_model.html)
- [Implementation of Lasso, Ridge and Elastic Net — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/implementation-of-lasso-ridge-and-elastic-net/)
- [Using ColumnTransformer in Scikit-learn — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/using-columntransformer-in-scikit-learn-for-data-preprocessing/)
