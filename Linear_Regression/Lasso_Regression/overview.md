# 📌 Lasso Regression on Insurance Dataset

## 📖 Project Overview

This project uses **Lasso Regression (L1 Regularization)** to predict medical insurance charges based on personal and health-related features.

The goal is to build a regression model that:

* Predicts insurance cost (`charges`)
* Reduces overfitting using regularization
* Performs feature selection

---

## 📊 Dataset Information

The dataset contains the following features:

* `age` → Age of the individual
* `sex` → Gender (male/female)
* `bmi` → Body Mass Index
* `children` → Number of children
* `smoker` → Smoking status
* `region` → Residential area
* `charges` → Medical insurance cost (target variable)

---

## ⚙️ Data Preprocessing

### 1. Feature & Target Separation

* Features (X): All columns except `charges`
* Target (Y): `charges`

### 2. Encoding

* `sex`: male → 1, female → 0
* `smoker`: yes → 1, no → 0
* `region`: One-hot encoding using `get_dummies()`

### 3. Feature Engineering

New features created:

* `age_smoker = age × smoker`
* `bmi_smoker = bmi × smoker`

👉 These capture interaction effects between smoking and health factors.

---

## 🔀 Train-Test Split

* 80% Training Data
* 20% Testing Data
* `random_state = 42` for reproducibility

---

## 🧠 Model Used

### Lasso Regression

* Uses **L1 Regularization**
* Helps in:

  * Reducing overfitting
  * Feature selection (some coefficients become zero)

```python
lasso_model = Lasso(alpha=0.5)
```

---

## 📉 Model Evaluation

### Without Cross-Validation

* MSE: **20918648.88**

---

## 🔄 Lasso with Cross-Validation

Used `LassoCV` to find optimal alpha value.

```python
lasso_cv_model = LassoCV(alphas=a, cv=5)
```

### Results:

* MSE: **20922607.93**
* R² Score: **0.8652**

---

## 📈 Interpretation of Results

* **R² = 0.8652**

  * Model explains ~86.5% of variance in insurance charges
* Slight difference in MSE shows model stability
* Lasso helps in simplifying the model by penalizing coefficients

---

## 🚀 Key Learnings

* Lasso Regression improves generalization
* Feature engineering improves performance
* Cross-validation helps in selecting optimal hyperparameters

---

## ▶️ How to Run

1. Install required libraries:

```bash
pip install pandas scikit-learn
```

2. Run the notebook:

```bash
insurance.ipynb
```

---

## 📌 Conclusion

Lasso Regression successfully predicts insurance costs while controlling model complexity and selecting important features automatically.
