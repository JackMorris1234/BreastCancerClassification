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
    #
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

model_scores = {}  # stores model -> ROC-AUC

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

    # Store mean ROC-AUC score
    model_scores[name] = scores["test_roc_auc"].mean()

#select best model based on ROC-AUC

best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

print("\n===============================================")
print(f"BEST MODEL SELECTED (via K-Fold ROC-AUC): {best_model_name}")
print("===============================================\n")

# Fit the best model on full training split
best_model.fit(X_train_scaled, y_train)

# Replace final_model so GUI uses best model
final_model = best_model

# Evaluate on test set
y_pred = final_model.predict(X_test_scaled)
y_prob = final_model.predict_proba(X_test_scaled)[:, 1]

print("=========== TEST SET PERFORMANCE ===========")
print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_prob):.4f}")
print("=============================================")

#function to predict malignancy based on user input
def predict_malignancy(input_values):
    #convert input to numpy array and reshape for single sample
    arr = np.array(input_values).reshape(1, -1)

    #scales entered values and predicts malignancy
    arr_scaled = scaler.transform(arr)
    prob = final_model.predict_proba(arr_scaled)[0][1]
    predicted = final_model.predict(arr_scaled)[0]

    label = "Malignant" if predicted == 1 else "Benign"
    return label, round(prob * 100, 2)


#shows the features required for manual entry

print("\n====================================================")
print("FEATURES REQUIRED FOR MANUAL ENTRY (IN THIS ORDER):")
print("====================================================")

for i, feat in enumerate(cleaned_features):
    print(f"{i+1}. {feat}")

print("\nThese are the EXACT attributes users must provide.\n")


#creates a manual prompt for user input

def prompt_user_for_prediction():
    
    print("\nEnter feature values for a new sample:")
    print("(Press Enter to use the same order shown above.)\n")

    user_values = []

    for feat in cleaned_features:
        while True:
            try:
                val = float(input(f"Enter value for {feat}: "))
                user_values.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    label, prob = predict_malignancy(user_values)

    print("\n================ PREDICTION RESULT ================")
    print(f"Diagnosis: {label}")
    print(f"Probability of Malignancy: {prob}%")
    print("====================================================\n")


#actually runs the prompt
prompt_user_for_prediction()
