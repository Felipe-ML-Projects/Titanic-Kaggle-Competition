# Titanic Survival Prediction — End-to-End Machine Learning Pipeline
This project builds a complete end-to-end machine learning pipeline to predict passenger survival on the Titanic. The workflow includes data cleaning, exploratory analysis, feature engineering, model experimentation, hyperparameter tuning, and ensemble learning — following a real data science production mindset.

1. Data Cleaning & Preprocessing
Imputed missing values for Age, Fare, and Embarked.
Removed duplicates and standardized numerical features.
Used ColumnTransformer and Pipeline to keep preprocessing reproducible.

2. Exploratory Data Analysis (EDA)
Analyzed survival patterns across sex, ticket class, family size, fare, and age.
Visualized distributions, correlations, and group-level survival rates.
Identified Sex and Pclass as primary drivers of survival.

3. Feature Engineering
Engineered predictive features that boosted model performance:
FamilySize, IsAlone, Title (extracted from Name)
One-hot encoding for all categorical variables
These features allowed models to capture nonlinear relationships and social structure.

4. Model Training
Implemented and compared multiple ML algorithms:
Logistic Regression
Decision Tree
Random Forest
SVM
KNN
XGBoost
Voting Classifier (ensemble of best models)
Used GridSearchCV for hyperparameter tuning and cross-validation.

5. Model Evaluation
Evaluated models using:
Accuracy
Precision, Recall
F1 Score
Cross-validated performance
Random Forest and XGBoost consistently performed best, with VotingClassifier improving robustness.

6. Reproducible Pipeline (separate file)
End-to-end pipeline using scikit-learn
Clean separation of preprocessing, modeling, and evaluation
Exportable predictions & ready for deployment or Kaggle submission

7. Results
Achieved an accuracy of 79%.
