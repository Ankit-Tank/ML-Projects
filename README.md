# 🤖 ML-Projects

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A collection of Machine Learning projects organized by algorithm type — each with a dataset, trained model, and detailed overview.

---

## 📚 Table of Contents

- [About](#-about)
- [Projects](#-projects)
- [Tech Stack](#-tech-stack)
- [Repo Structure](#-repo-structure)
- [How to Run](#-how-to-run)
- [License](#-license)

---

## 📌 About

This repository contains hands-on ML projects built to learn and demonstrate core machine learning algorithms. Each project is self-contained with its own dataset, Jupyter notebook, and `overview.md` explaining the approach and results.

---

## 📂 Projects

### 📈 Linear Regression

| Project | Dataset | Notebook | Overview |
|---|---|---|---|
| 🏠 House Price Prediction | `HousePricePrediction.csv` | [price_prediction.ipynb](./Linear_Regression/House_Price_Prediction/price_prediction.ipynb) | [overview.md](./Linear_Regression/House_Price_Prediction/overview.md) |
| 🏥 Insurance Charges Prediction | `insurance.csv` | [insurance_model.ipynb](./Linear_Regression/Insurance_Charges_Predict/insurance_model.ipynb) | [overview.md](./Linear_Regression/Insurance_Charges_Predict/overview.md) |
| 🔵 Lasso Regression | `insurance.csv` | [insurance.ipynb](./Linear_Regression/Lasso_Regression/insurance.ipynb) | [overview.md](./Linear_Regression/Lasso_Regression/overview.md) |

---

### 🔀 Logistic Regression

| Project | Dataset | Notebook | Overview |
|---|---|---|---|
| 👥 Employee Turnover | `employee_turnover.csv` | [employee_prediction.ipynb](./Logistic_Regression/Employee_Turnover/employee_prediction.ipynb) | [overview.md](./Logistic_Regression/Employee_Turnover/overview.md) |
| ❤️ Heart Attack Prediction | `heart.csv` | [heart_attack_possibility.ipynb](./Logistic_Regression/Heart_Attack_Prediction/heart_attack_possibility.ipynb) | [overview.md](./Logistic_Regression/Heart_Attack_Prediction/overview.md) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Jupyter Notebook | Interactive development |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-Learn | ML models and evaluation |
| Matplotlib / Seaborn | Data visualization |

---

## 📁 Repo Structure

```
ML-Projects/
│
├── 📁 Linear_Regression/
│   ├── 📁 House_Price_Prediction/
│   ├── 📁 Insurance_Charges_Predict/
│   └── 📁 Lasso_Regression/
│
├── 📁 Logistic_Regression/
│   ├── 📁 Employee_Turnover/
│   └── 📁 Heart_Attack_Prediction/
│
├── LICENSE
└── README.md
```

Each project folder contains:
- 📓 `.ipynb` — Jupyter notebook with full code
- 📊 `.csv` — Dataset used for training/testing
- 📄 `overview.md` — Project summary, approach, and results

---

## ▶️ How to Run

1. **Clone the repo**
```bash
git clone https://github.com/Ankit-Tank/ML-Projects.git
cd ML-Projects
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. **Open any notebook**
```bash
jupyter notebook
```

Then navigate to any project folder and open the `.ipynb` file.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>⭐ Star this repo if you find it useful! | Made with ❤️ by <a href="https://github.com/Ankit-Tank">Ankit-Tank</a></sub>
</div>
