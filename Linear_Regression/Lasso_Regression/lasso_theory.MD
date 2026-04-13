# 📌 Lasso Regression (L1 Regularization)

## 📖 Overview

Lasso Regression is a type of linear regression that adds a regularization penalty to reduce overfitting and improve model performance.

It uses L1 regularization (absolute values of coefficients), which helps in feature selection.

---

## 🎯 Objective

* Minimize prediction error
* Reduce overfitting
* Perform automatic feature selection

---

## 🧠 Cost Function

```
J(θ) = (1 / 2m) * Σ(i=1 to m) (hθ(x(i)) - y(i))^2 + λ * Σ(j=1 to n) |θj|
```

### 🔍 Explanation

* First term → Mean Squared Error (MSE)
* Second term → L1 penalty
* λ (lambda) → Regularization strength
* θⱼ → Model coefficients
* j = 1 to n → number of features

---

## ⚙️ Key Concepts

### 1. Regularization Strength (λ)

* Controls how strong the penalty is
* Higher λ → more shrinkage → simpler model

### 2. Penalty Term

* Uses absolute values of coefficients
* Forces some coefficients to become zero

---

## 📉 Effect of Lasso

* Increase in features → penalty increases
* Increase in coefficient values → cost increases
* Some coefficients become exactly zero

👉 This results in automatic feature selection

---

## 📊 Observations

* Features ↑ → Penalty ↑
* Coefficient values ↑ → Cost ↑
* Features ↓ → Model becomes simpler

---

## ✅ Advantages

* Prevents overfitting
* Performs feature selection
* Creates simple and interpretable models

---

## ❌ Disadvantages

* May remove important features if λ is too high
* Not stable with highly correlated features

---

## 🔑 Key Insight

Lasso Regression reduces overfitting and automatically removes irrelevant features by making their coefficients zero.

---

## 📌 Summary

* Cost = Error + λ × Penalty
* Uses L1 Regularization
* Performs feature selection
* Simplifies the model
