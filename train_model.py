import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

dataset_path = "dataset/Insurance.csv"

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("insuranceclaim", axis=1)
y = df["insuranceclaim"]


# ==========================================
# 3. DEFINE NUMERICAL FEATURES
# ==========================================

numeric_features = [
    "age",
    "bmi",
    "children",
    "annual_medical_expenses"
]


# ==========================================
# 4. DEFINE CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "sex",
    "smoker",
    "medical_condition",
    "chronic_disease",
    "alcohol_consumption",
    "family_history"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. CREATE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Prediction probability
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================
# 9. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["ROC-AUC", "F1 Score"],
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================
# 10. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Selected Model:", best_model_name)


# ==========================================
# 11. CREATE MODELS FOLDER
# ==========================================

import os

os.makedirs("models", exist_ok=True)


# ==========================================
# 12. SAVE BEST MODEL
# ==========================================

model_path = "models/insurance_claim_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\nModel saved successfully!")
print("Location:", model_path)


# ==========================================
# 13. SAVE MODEL RESULTS
# ==========================================

results_path = "models/model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("Model results saved successfully!")
print("Location:", results_path)


# ==========================================
# 14. FINISHED
# ==========================================

print("\n")
print("=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)