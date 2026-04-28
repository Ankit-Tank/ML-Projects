# 🌸 Iris Flower Species Prediction
### Supervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Problem Statement

A botanical research centre is studying different species of Iris flowers to **automate the plant identification process**. Currently, botanists manually identify flower species by measuring sepal and petal dimensions, which is:

- ⏱️ Time-consuming
- ❌ Error-prone
- 📉 Not scalable

The goal is to build a **Machine Learning based classification system** that can automatically predict the **species of an Iris flower** based on its physical measurements.

---

## 🎯 Objective

Train and compare **3 classification models**:

| Model | Algorithm Type |
|---|---|
| Logistic Regression | Linear Classifier |
| K-Nearest Neighbours (KNN) | Distance-based Classifier |
| Naive Bayes | Probabilistic Classifier |

---

## 📊 Dataset Description

**Dataset:** Iris.csv — 150 samples, 3 classes (50 each)

| Feature | Description |
|---|---|
| `SepalLengthCm` | Length of sepal (cm) |
| `SepalWidthCm` | Width of sepal (cm) |
| `PetalLengthCm` | Length of petal (cm) |
| `PetalWidthCm` | Width of petal (cm) |
| `Species` *(Target)* | Iris-setosa / Iris-versicolor / Iris-virginica |

---

## 🗂️ Project Structure

```
📦 Iris-flower-prediction
 ┣ 📓 predict_flower_species_combined.ipynb   ← All 3 models in one notebook
 ┣ 📄 Iris.csv                                ← Dataset
 ┗ 📄 overview.md
```

---

## ⚙️ Steps Followed

1. **Load Data** — Read `Iris.csv` using pandas
2. **Explore Data** — `.info()`, `.columns()`, `.sample()`
3. **Label Encoding** — Convert species names → 0, 1, 2
4. **Train-Test Split** — 30% train / 70% test (`test_size=0.7`)
5. **Feature Scaling** — StandardScaler for LR & KNN
6. **Train Models** — Logistic Regression, KNN, Naive Bayes
7. **Evaluate** — Accuracy, Precision, Confusion Matrix, Classification Report
8. **Compare** — All 3 models side by side

---

## 📈 Results

| Model | Accuracy | Precision |
|---|---|---|
| Logistic Regression | 1.000000 | 1.000000 |
| KNN | 0.990476 | 0.990756 |
| Naive Bayes | 0.980952 | 0.980952 |

> **Note:** Iris is a small, well-balanced & clean dataset with only 150 samples split across 3 categories (50 each). So the models perform extremely well. This should **not** be a realistic expectation from real-life data, which is not always balanced, has noise & needs a lot of pre-processing.

---

## 🛠️ Technologies Used

- Python 3.13
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Ankit-Tank/Iris-flower-prediction.git

# 2. Install dependencies
pip install pandas scikit-learn jupyter

# 3. Launch Jupyter Notebook
jupyter notebook

# 4. Open predict_flower_species_combined.ipynb and run all cells
```

---

## 📚 Key Learnings

- How to apply **Label Encoding** on target variable
- When to use **StandardScaler** (KNN & LR need it, Naive Bayes doesn't)
- Difference between **Accuracy** and **Precision**
- How to use `classification_report` and `confusion_matrix`
- Comparing multiple ML models on the same dataset

---

## 👨‍💻 Author

Made with ❤️ as part of Supervised Machine Learning
