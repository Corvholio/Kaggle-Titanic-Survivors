# Predicting Titanic Survival: A Complete Machine Learning Walkthrough

## 🧠 Objective
Use machine learning to predict which passengers survived the Titanic disaster using data from Kaggle.

## 📊 Summary
- Cleaned and preprocessed all data (no leakage)
- Performed advanced feature engineering (`Title`, `TicketPrefix`, `FamilySize`, `IsAlone`, `Age*Class`)
- Built multiple models: Logistic Regression, Random Forest, KNN, and XGBoost
- Tuned models with GridSearchCV and evaluated using 5-fold cross-validation
- Final public Kaggle score: **0.76**
- Learned how leaderboard accuracy does not always reflect real model quality

## 🛠 Tools Used
- Python (pandas, sklearn, xgboost)
- Jupyter Notebook
- Kaggle's Titanic dataset

## 🔍 Highlights
- Performed leak-free Age imputation using RandomForestRegressor
- Created new features from messy columns like `Ticket` and `Name`
- Compared model performance using accuracy, confusion matrix, and classification report
- Discussed overfitting, generalization, and leaderboard traps

## 📁 Files
- `titanic_notebook.ipynb`: Full workflow and code
- `submission.csv`: Final Kaggle submission
- `data/`: Raw CSVs
