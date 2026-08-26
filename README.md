📊 Customer Churn Prediction Using Machine Learning
📌 Project Overview

Customer Churn Prediction is a supervised machine learning project that predicts whether a customer is likely to churn (leave the service) or stay with the company.

The project uses a telecom customer churn dataset and applies multiple classification algorithms to compare their performance.

The following machine learning models are implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

The best-performing model is selected using the F1 Score and saved using Joblib. A Streamlit web application is then used to make predictions for new customers.

---

🔗 Project Link 
https://github.com/SujataShere

---

🎯 Project Objectives

The main objectives of this project are:

- Load and understand the customer churn dataset.
- Perform data cleaning and preprocessing.
- Handle missing values.
- Remove duplicate records.
- Convert categorical variables into numerical features.
- Scale numerical features.
- Train multiple supervised machine learning classification models.
- Compare model performance.
- Evaluate models using multiple classification metrics.
- Select the best model based on F1 Score.
- Save the trained model using Joblib.
- Create an interactive Streamlit web application.
- Predict customer churn for new customer information.

---

🛠️ Technologies Used

# Programming Language

- Python 3.x

# Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

# Machine Learning

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

# Development Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub
- Streamlit

---

🔄 Complete Machine Learning Workflow
                 Customer Churn Dataset
                         │
                         ▼
                  Data Collection
                         │
                         ▼
                   Data Cleaning
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Numerical Data        Categorical Data
              │                     │
              ▼                     ▼
         Imputation             Imputation
              │                     │
              ▼                     ▼
       Standard Scaling       One-Hot Encoding
              │                     │
              └──────────┬──────────┘
                         ▼
                  Train/Test Split
                         │
                         ▼
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
        Logistic     Decision    Random Forest
       Regression      Tree
             │           │           │
             └───────────┼───────────┘
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              AdaBoost     Gradient Boosting
                  │             │
                  └──────┬──────┘
                         ▼
                 Model Evaluation
                         │
                         ▼
                 F1 Score Comparison
                         │
                         ▼
                   Best Model
                         │
                         ▼
                  churn_model.pkl
                         │
                         ▼
                    Streamlit App
                         │
                         ▼
                 Customer Prediction
                         │
                  ┌──────┴──────┐
                  ▼             ▼
                Churn        No Churn

---

📌 Key Features of the Project

- Complete data preprocessing pipeline
- Missing value handling
- Duplicate removal
- Numerical feature scaling
- Categorical feature encoding
- Multiple classification algorithms
- Ensemble learning
- Model comparison
- Accuracy evaluation
- Precision evaluation
- Recall evaluation
- F1 Score evaluation
- ROC-AUC evaluation
- Confusion matrix
- ROC curve
- Automatic best-model selection
- Model serialization using Joblib
- Interactive Streamlit application

---

💼 Business Use Case

Customer churn prediction can help a telecom company identify customers who are likely to leave.

The company can use the prediction to:

- Identify high-risk customers
- Offer special discounts
- Improve customer service
- Provide retention offers
- Understand customer behavior
- Reduce customer loss
- Improve customer satisfaction
- Increase customer lifetime value

---

🌐 Deployment

The Streamlit application can be deployed after the project is tested locally.

Deployment options include:

- Streamlit hosting

Before deployment, make sure the repository contains:

- app.py
- churn_model.pkl
- feature_columns.pkl
- requirements.txt
- README.md
