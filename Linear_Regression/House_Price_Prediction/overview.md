# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

HomeVista Properties is a real estate company operating across multiple cities, handling thousands of residential property transactions every year. The company aims to automate its house pricing process by leveraging Machine Learning.

This project focuses on building an intelligent regression model that can accurately predict the market price of a house based on its physical characteristics, location, and condition.

---

## 🎯 Objective

The primary objective of this project is to:

* Analyze historical housing data
* Perform data cleaning and preprocessing
* Apply feature engineering techniques
* Train and evaluate regression models
* Predict house prices with high accuracy

---

## 📊 Dataset Description

The dataset contains detailed information about residential properties. Each row represents a house along with its features.

### 🔹 Features:

| Feature Name | Description                               |
| ------------ | ----------------------------------------- |
| Id           | Unique identification number              |
| MSSubClass   | Type of dwelling (e.g., 1-Story, 2-Story) |
| MSZoning     | Zoning classification (Residential types) |
| LotArea      | Lot size (in square feet)                 |
| LotConfig    | Lot configuration (Corner, Inside, etc.)  |
| BldgType     | Type of building (1Fam, Duplex, etc.)     |
| OverallCond  | Overall condition rating (1–10)           |
| YearBuilt    | Year of construction                      |
| YearRemodAdd | Year of remodeling                        |
| Exterior1st  | Exterior covering type                    |
| BsmtFinSF2   | Finished basement area (Type 2)           |
| TotalBsmtSF  | Total basement area                       |
| SalePrice    | Final house price (Target Variable)       |

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

## 🔧 Project Workflow

1. **Data Collection**

   * Load dataset using Pandas

2. **Data Preprocessing**

   * Handle missing values
   * Remove irrelevant features
   * Encode categorical variables

3. **Feature Engineering**

   * Create new features like house age and total area
   * Generate interaction features

4. **Model Training**

   * Linear Regression
   * Lasso Regression
   * Ridge Regression
   * Random Forest Regressor
   * Gradient Boosting Regressor

5. **Model Evaluation**

   * R² Score
   * RMSE (Root Mean Squared Error)

---

## 📈 Results

* Linear models showed limited performance due to linear assumptions
* Tree-based models significantly improved prediction accuracy
* Gradient Boosting provided the best performance among all models

---

## 🚀 Key Insights

* Feature engineering plays a crucial role in improving model performance
* Real-world datasets often require handling non-linear relationships
* Ensemble models outperform simple linear models in complex problems

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be applied to automate real estate price prediction. By combining proper preprocessing, feature engineering, and advanced models, we can build accurate and reliable predictive systems.

---

## 📂 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Adding more features (location-based, amenities, etc.)
* Deploying the model using Flask or Streamlit

---
