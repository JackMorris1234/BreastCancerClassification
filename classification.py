import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


df = pd.read_csv("Data/Data.csv")

#drop ID column
df = df.drop(columns=["ID"], errors="ignore")

#encode labels: M=1, B=0
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

#feature selection based on correlation with target, if correlation < 0.3, drop feature

corr_matrix = df.corr(numeric_only=True)
corr_with_target = corr_matrix["Diagnosis"].abs()

selected_features = corr_with_target[corr_with_target >= 0.3].index.tolist()
selected_features = [c for c in selected_features if c != "Diagnosis"]

print("Initial Selected Features:", selected_features)

#remove redundant features based on inter-feature correlation, if correlation > 0.9, drop one with lower correlation to target

def remove_redundant_features(df, features, threshold=0.90):
    """
    Removes features that are highly correlated with each other.
    Keeps the feature with stronger correlation to the target.
    """
    corr = df[features].corr().abs()

    #features to remove
    to_remove = set()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]

            #if two features are highly correlated
            if corr.loc[f1, f2] >= threshold:
                #remove the one less correlated with target
                if corr_with_target[f1] >= corr_with_target[f2]:
                    to_remove.add(f2)
                else:
                    to_remove.add(f1)

    # Final feature list
    cleaned = [f for f in features if f not in to_remove]
    return cleaned, list(to_remove)

cleaned_features, removed = remove_redundant_features(df, selected_features)

print("\nRemoved Redundant Features:", removed)
print("Final Feature Set:", cleaned_features)

#seperate X and y, x being features, y being target

X = df[cleaned_features]
y = df["Diagnosis"]

#split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#scale features so that models perform better

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#define models to evaluate

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM (RBF Kernel)": SVC(probability=True, kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
}

#cross-validate models and print results

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#iterate through models and evaluate
for name, model in models.items():
    print("\n==============================")
    print(f"Model: {name}")
    print("==============================")

    scores = cross_validate(
        model,
        X_train_scaled,
        y_train,
        cv=kf,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    print(f"Accuracy:       {scores['test_accuracy'].mean():.4f}")
    print(f"Precision:      {scores['test_precision'].mean():.4f}")
    print(f"Recall:         {scores['test_recall'].mean():.4f}")
    print(f"F1 Score:       {scores['test_f1'].mean():.4f}")
    print(f"ROC-AUC:        {scores['test_roc_auc'].mean():.4f}")


#fit the final model
final_model = SVC(probability=True, kernel="rbf")
final_model.fit(X_train_scaled, y_train)

y_pred = final_model.predict(X_test_scaled)
y_prob = final_model.predict_proba(X_test_scaled)[:, 1]

print("\n\n==============================")
print("FINAL MODEL TEST SET RESULTS")
print("==============================")
print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_prob):.4f}")

# ----------------------------------------------------------
# 10. FUNCTION FOR FINAL PRODUCT
# ----------------------------------------------------------

def predict_malignancy(input_values):
    """
    input_values must follow the order of cleaned_features
    """
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = final_model.predict_proba(arr_scaled)[0][1]
    predicted = final_model.predict(arr_scaled)[0]

    label = "Malignant" if predicted == 1 else "Benign"
    return label, round(prob * 100, 2)
