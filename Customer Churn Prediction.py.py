#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import joblib
import warnings

warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("Customer Churn.csv")
print("Dataset loaded successfully!")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe()


# In[9]:


missing_values = df.isnull().sum()
print(missing_values)


# In[10]:


missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)


# In[11]:


print("Number of duplicate rows:", df.duplicated().sum())


# In[12]:


df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)


# In[13]:


print(df["Churn Label"].value_counts())


# In[14]:


print(df["Churn Label"].value_counts(normalize=True) * 100)


# In[15]:


plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Churn Label")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()


# In[16]:


print(df["Total Charges"].dtype)


# In[17]:


df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)


# In[18]:


print(df["Total Charges"].dtype)


# In[19]:


print(df.isnull().sum())


# In[20]:


drop_columns = [
    "CustomerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason"
]
df_model = df.drop(columns=drop_columns)
print(df_model.shape)


# In[21]:


X = df_model
y = df["Churn Label"]
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[22]:


y = y.map({
    "No": 0,
    "Yes": 1
})


# In[23]:


print(y.value_counts())


# In[24]:


numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# In[26]:


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# In[27]:


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# In[28]:


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ))
    ]
)


# In[29]:


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# In[30]:


## Logistic Regression
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)


# In[31]:


logistic_model.fit(X_train, y_train)


# In[32]:


y_pred_lr = logistic_model.predict(X_test)


# In[33]:


print("Logistic Regression")

print(
    classification_report(
        y_test,
        y_pred_lr
    )
)


# In[34]:


lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("Accuracy:", lr_accuracy)


# In[35]:


decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        ))
    ]
)


# In[36]:


decision_tree_model.fit(X_train, y_train)


# In[37]:


y_pred_dt = decision_tree_model.predict(X_test)


# In[38]:


print("Decision Tree")

print(
    classification_report(
        y_test,
        y_pred_dt
    )
)


# In[39]:


dt_accuracy = accuracy_score(
    y_test,
    y_pred_dt
)

print("Accuracy:", dt_accuracy)


# In[40]:


random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)


# In[41]:


random_forest_model.fit(
    X_train,
    y_train
)


# In[42]:


y_pred_rf = random_forest_model.predict(X_test)


# In[43]:


print("Random Forest")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# In[44]:


rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("Accuracy:", rf_accuracy)


# In[45]:


adaboost_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=42
        ))
    ]
)


# In[46]:


adaboost_model.fit(
    X_train,
    y_train
)


# In[47]:


y_pred_ab = adaboost_model.predict(X_test)


# In[48]:


print("AdaBoost")
print(
    classification_report(
        y_test,
        y_pred_ab
    )
)


# In[49]:


ab_accuracy = accuracy_score(
    y_test,
    y_pred_ab
)
print("Accuracy:", ab_accuracy)


# In[50]:


gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ]
)


# In[51]:


gradient_boosting_model.fit(
    X_train,
    y_train
)


# In[52]:


y_pred_gb = gradient_boosting_model.predict(X_test)


# In[53]:


print("Gradient Boosting")
print(
    classification_report(
        y_test,
        y_pred_gb
    )
)


# In[54]:


gb_accuracy = accuracy_score(
    y_test,
    y_pred_gb
)
print("Accuracy:", gb_accuracy)


# In[55]:


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Gradient Boosting"
    ],
    
    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy,
        ab_accuracy,
        gb_accuracy
    ]
})
results


# In[56]:


results = results.sort_values(
    by="Accuracy",
    ascending=False
)
results


# In[57]:


plt.figure(figsize=(10, 5))
sns.barplot(
    data=results,
    x="Accuracy",
    y="Model"
)
plt.title("Model Accuracy Comparison")
plt.xlim(0, 1)
plt.show()


# In[58]:


def evaluate_model(model_name, y_true, y_pred, y_probability):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probability) 
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    }


# In[59]:


lr_prob = logistic_model.predict_proba(X_test)[:, 1]

dt_prob = decision_tree_model.predict_proba(X_test)[:, 1]

rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

ab_prob = adaboost_model.predict_proba(X_test)[:, 1]

gb_prob = gradient_boosting_model.predict_proba(X_test)[:, 1]


# In[60]:


all_results = []

all_results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        y_pred_lr,
        lr_prob
    )
)

all_results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        y_pred_dt,
        dt_prob
    )
)

all_results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        y_pred_rf,
        rf_prob
    )
)

all_results.append(
    evaluate_model(
        "AdaBoost",
        y_test,
        y_pred_ab,
        ab_prob
    )
)

all_results.append(
    evaluate_model(
        "Gradient Boosting",
        y_test,
        y_pred_gb,
        gb_prob
    )
)

comparison_df = pd.DataFrame(all_results)

comparison_df


# In[61]:


comparison_df.sort_values(
    by="F1 Score",
    ascending=False
)


# In[62]:


cm = confusion_matrix(
    y_test,
    y_pred_rf
)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[63]:


fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_ab, tpr_ab, _ = roc_curve(y_test, ab_prob)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_lr,
    tpr_lr,
    label="Logistic Regression"
)
plt.plot(
    fpr_dt,
    tpr_dt,
    label="Decision Tree"
)
plt.plot(
    fpr_rf,
    tpr_rf,
    label="Random Forest"
)
plt.plot(
    fpr_ab,
    tpr_ab,
    label="AdaBoost"
)
plt.plot(
    fpr_gb,
    tpr_gb,
    label="Gradient Boosting"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# In[64]:


best_model_name = comparison_df.loc[
    comparison_df["F1 Score"].idxmax(),
    "Model"
]
print("Best Model:", best_model_name)


# In[65]:


models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model,
    "Gradient Boosting": gradient_boosting_model
}
best_model = models[best_model_name]
print(best_model)


# In[66]:


joblib.dump(
    best_model,
    "churn_model.pkl"
)
print("Model saved successfully!")


# In[67]:


joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)
print("Feature columns saved successfully!")


# In[68]:


loaded_model = joblib.load(
    "churn_model.pkl"
)
test_prediction = loaded_model.predict(
    X_test.iloc[:5]
)
print(test_prediction)


# In[69]:


predicted_labels = np.where(
    test_prediction == 1,
    "Yes",
    "No"
)
print(predicted_labels)


# In[70]:


get_ipython().system('pip install streamlit')


# In[ ]:


get_ipython().system('python -m streamlit run app.py')


# In[ ]:


import os

print(os.getcwd())


# In[ ]:


import os

os.chdir(r"C:\Users\admin\Projects\Customer Churn Prediction")

print(os.getcwd())


# In[ ]:


import os

print(os.listdir())


# In[ ]:


get_ipython().system('python -m streamlit run app.py')


# In[ ]:


get_ipython().system('python -m streamlit --version')


# In[ ]:




