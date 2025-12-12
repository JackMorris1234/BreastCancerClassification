
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

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------

df = pd.read_csv("Data/Data.csv")   # <-- replace with your CSV filename

# Drop ID column
df = df.drop(columns=["id"], errors="ignore")

# Encode labels: M=1, B=0
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# ----------------------------------------------------------
# 2. OPTIONAL: FEATURE SELECTION (CORR >= 0.30 WITH TARGET)
# ----------------------------------------------------------

corr_matrix = df.corr(numeric_only=True)
corr_with_target = corr_matrix["Diagnosis"].abs()

selected_features = corr_with_target[corr_with_target >= 0.3].index.tolist()

# Remove "Diagnosis" from feature list
selected_features = [c for c in selected_features if c != "Diagnosis"]

X = df[selected_features]
y = df["Diagnosis"]

print("Selected Features:", selected_features)

# ----------------------------------------------------------
# 3. TRAIN–TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------
# 4. SCALING
# ----------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 5. DEFINE MODELS
# ----------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM (RBF Kernel)": SVC(probability=True, kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
}

# ----------------------------------------------------------
# 6. K-FOLD CROSS VALIDATION
# ----------------------------------------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

# ----------------------------------------------------------
# 7. FIT FINAL MODEL (CHOOSE BEST PERFORMER)
# ----------------------------------------------------------

final_model = SVC(probability=True, kernel="rbf")  # or logistic / RF
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
# 8. HOW TO USE MODEL IN FINAL PRODUCT
# ----------------------------------------------------------

def predict_malignancy(input_values):
    """
    input_values = list of features in SAME ORDER as selected_features
    Example: [mean_radius, mean_texture, mean_smoothness, ...]
    """
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob = final_model.predict_proba(arr_scaled)[0][1]
    predicted = final_model.predict(arr_scaled)[0]

    label = "Malignant" if predicted == 1 else "Benign"
    return label, round(prob * 100, 2)