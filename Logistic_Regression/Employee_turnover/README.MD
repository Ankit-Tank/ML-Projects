📊 Dataset Overview

The dataset used in this project contains employee-related information collected from a multinational organization. It is designed to analyze workforce patterns and predict employee attrition.

Total Records: 900
Total Features: 15 (including engineered features)
Problem Type: Binary Classification

The dataset captures key factors such as job satisfaction, compensation, experience, and work-life balance to model employee turnover.

🎯 Problem Statement

Employee attrition leads to increased recruitment costs, project delays, and loss of skilled talent.

This dataset is used to build a machine learning system that can:

Predict whether an employee is likely to leave the company
Help HR teams take proactive retention measures
Improve workforce stability
📁 Feature Description
🔹 Input Features
Feature Name	Description
Job_Satisfaction	Employee's satisfaction level with their job
Performance_Rating	Performance evaluation score
Years_At_Company	Number of years the employee has worked
Work_Life_Balance	Balance between professional and personal life
Distance_From_Home	Distance from home to workplace
Monthly_Income	Monthly salary
Education_Level	Education qualification level
Age	Age of the employee
Num_Companies_Worked	Number of previous companies worked
Employee_Role	Encoded job role
Annual_Bonus	Annual bonus received
Training_Hours	Training hours completed
Department	Encoded department
🔹 Engineered Features
Feature Name	Description
Annual_Bonus_Squared	Square of annual bonus (captures non-linear impact)
Annual_Bonus_Training_Hours_Interaction	Interaction between bonus and training hours
🎯 Target Variable
Feature	Description
Employee_Turnover	Indicates whether an employee leaves the company
	1 → Employee Left
	0 → Employee Stayed
⚙️ Dataset Characteristics
Contains numerical and encoded categorical features
Includes feature engineering to improve model performance
Suitable for:
Logistic Regression (baseline model)
Regularized models (L1 / L2)
General classification tasks
📌 Usage in This Project

This dataset is used to:

Build a baseline Logistic Regression model
Apply Regularization techniques (L1 & L2)
Compare model performance using evaluation metrics
Recommend the best-performing model
💡 Notes
No raw categorical data — already encoded for ML use
Engineered features help capture complex relationships
Ideal for practicing:
Feature importance
Model comparison
Overfitting vs Regularization concepts