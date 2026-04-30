# 🤖 ML-Projects

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A collection of Machine Learning projects organized by algorithm type — each with a dataset, trained model, and detailed overview.

---

## 📚 Table of Contents

- [About](#-about)
- [Projects](#-projects)
- [Repo Structure](#-repo-structure)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [What I Learned](#-what-i-learned)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📌 About

This repository contains hands-on ML projects built to learn and demonstrate core machine learning algorithms. Each project is self-contained with its own dataset, Jupyter notebook, and `overview.md` explaining the approach, methodology, and results.

---

## 📂 Projects

### 📈 Linear Regression

| # | Project | Dataset | Notebook | Overview |
|---|---|---|---|---|
| 1 | 🏠 House Price Prediction | `HousePricePrediction.csv` | [price_prediction.ipynb](./Linear_Regression/House_Price_Prediction/price_prediction.ipynb) | [overview.md](./Linear_Regression/House_Price_Prediction/overview.md) |
| 2 | 🏥 Insurance Charges Prediction | `insurance.csv` | [insurance_model.ipynb](./Linear_Regression/Insurance_Charges_predict/insurance_model.ipynb) | [overview.md](./Linear_Regression/Insurance_Charges_predict/overview.md) |
| 3 | 🔵 Lasso Regression | `insurance.csv` | [insurance.ipynb](./Linear_Regression/Lasso_Regression/insurance.ipynb) | [overview.md](./Linear_Regression/Lasso_Regression/overview.md) |

---

### 🔀 Logistic Regression

| # | Project | Dataset | Notebook | Overview |
|---|---|---|---|---|
| 4 | 👥 Employee Turnover Prediction | `employee_turnover.csv` | [employee_prediction.ipynb](./Logistic_Regression/Employee_turnover/employee_prediction.ipynb) | [overview.md](./Logistic_Regression/Employee_turnover/overview.md) |
| 5 | ❤️ Heart Attack Prediction | `heart.csv` | [heart_attack_possibility.ipynb](./Logistic_Regression/Heart_attack_prediction/heart_attack_possibility.ipynb) | [overview.md](./Logistic_Regression/Heart_attack_prediction/overview.md) |
| 6 | 🌸 Flower Species Classification | `iris.csv` | [predict_flower_species.ipynb](./Logistic_Regression/Flower_Species/predict_flower_species.ipynb) | — |

---

### 📊 Logistic Regression — Evaluation Metrics

| # | Metric | Notebook |
|---|---|---|
| 7 | ✅ Accuracy | [accuracy.ipynb](./Logistic_Regression/Evaluation_Metrices/accuracy.ipynb) |
| 8 | 🎯 Precision | [precision.ipynb](./Logistic_Regression/Evaluation_Metrices/precision.ipynb) |
| 9 | 🔁 Recall | [recall.ipynb](./Logistic_Regression/Evaluation_Metrices/recall.ipynb) |
| 10 | 📐 F1 Score | [f1_score.ipynb](./Logistic_Regression/Evaluation_Metrices/f1%20score.ipynb) |
| 11 | 🟦 Confusion Matrix | [confusion_metrix.ipynb](./Logistic_Regression/Evaluation_Metrices/confusion%20metrix.ipynb) |

---

### 🌿 KNN — K-Nearest Neighbors

| # | Project | Dataset | Notebook |
|---|---|---|---|
| 12 | 🫀 KNN Classification | `heart.csv` | [knn.ipynb](./KNN/knn.ipynb) |
| 13 | 🔄 Cross Validation | — | [cross_validation.ipynb](./KNN/cross%20validation.ipynb) |

---

### 🌺 Iris Flower Prediction

| # | Project | Dataset | Notebook | Overview |
|---|---|---|---|---|
| 14 | 🌸 Iris Flower Species | `Iris.csv` | [predict_flower_species_combined.ipynb](./Iris-flower-prediction/predict_flower_species_combined.ipynb) | [overview.md](./Iris-flower-prediction/overview.md) |

---

### 🧮 Naive Bayes

| # | Project | Dataset | Notebook |
|---|---|---|---|
| 15 | 📊 Naive Bayes Classifier | `heart.csv` | [naive_bayes.ipynb](./Naive_Bayes/naive%20bayes.ipynb) |

---

## 📁 Repo Structure

```
ML-Projects/
│
├── 📁 Linear_Regression/
│   ├── 📁 House_Price_Prediction/
│   ├── 📁 Insurance_Charges_predict/
│   └── 📁 Lasso_Regression/
│
├── 📁 Logistic_Regression/
│   ├── 📁 Employee_turnover/
│   ├── 📁 Heart_attack_prediction/
│   ├── 📁 Flower_Species/
│   └── 📁 Evaluation_Metrices/
│       ├── accuracy.ipynb
│       ├── precision.ipynb
│       ├── recall.ipynb
│       ├── f1 score.ipynb
│       └── confusion metrix.ipynb
│
├── 📁 KNN/
│   ├── knn.ipynb
│   ├── cross validation.ipynb
│   └── heart.csv
│
├── 📁 Iris-flower-prediction/
│   ├── predict_flower_species_combined.ipynb
│   ├── Iris.csv
│   └── overview.md
│
├── 📁 Naive_Bayes/
│   ├── naive bayes.ipynb
│   └── heart.csv
│
├── .gitignore
├── LICENSE
├── requirement.txt
└── README.md
```

Each project folder contains:
- 📓 `.ipynb` — Jupyter notebook with full code, EDA, and model training
- 📊 `.csv` — Dataset used for training and testing
- 📄 `overview.md` — Project summary, methodology, and results (where available)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Jupyter Notebook | Interactive development environment |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computation |
| Scikit-Learn | ML models, preprocessing, and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |

---

## ▶️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Ankit-Tank/ML-Projects.git
cd ML-Projects
```

### 2. Install dependencies
```bash
pip install -r requirement.txt
```

### 3. Open any notebook
```bash
jupyter notebook
```

Navigate to any project folder and open the `.ipynb` file.

---

## 🧠 What I Learned

### Linear Regression
- Feature scaling significantly affects model performance
- Lasso regularization reduces overfitting by penalizing weak features
- Correlation heatmaps are essential for identifying multicollinearity

### Logistic Regression
- Class imbalance directly impacts accuracy and needs to be handled
- Confusion matrix reveals model weaknesses beyond just accuracy score
- Precision, Recall, and F1 Score give a fuller picture than accuracy alone

### KNN
- Choosing the right value of K heavily impacts performance
- Cross-validation helps prevent overfitting and gives reliable accuracy estimates
- KNN is sensitive to feature scaling — normalization is essential

### Naive Bayes
- Works surprisingly well even with the "naive" independence assumption
- Very fast to train and effective for classification problems
- Performs well even with small datasets

---

## 🗺️ Roadmap

Upcoming algorithm implementations:

- [x] Linear Regression
- [x] Lasso Regression
- [x] Logistic Regression
- [x] KNN
- [x] Naive Bayes
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Support Vector Machine (SVM)
- [ ] K-Means Clustering
- [ ] Principal Component Analysis (PCA)
- [ ] Neural Networks (basic)

---

## 🤝 Contributing

Found a bug or want to suggest an improvement?

1. Fork this repo
2. Create a branch: `git checkout -b improve-model`
3. Commit your changes
4. Submit a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>⭐ Star this repo if you find it useful! | Made with ❤️ by <a href="https://github.com/Ankit-Tank">Ankit-Tank</a></sub>
</div>
