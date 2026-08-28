import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    roc_auc_score,
    confusion_matrix
)


# ==============================
# 1. LOAD DATASET
# ==============================

print("\nLoading dataset...")

df = pd.read_csv("dataset/Insurance.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==============================
# 2. PREPARE FEATURES AND TARGET
# ==============================

X = df.drop("insuranceclaim", axis=1)
y = df["insuranceclaim"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("insuranceclaim")


# ==============================
# 3. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==============================
# 4. DEFINE MODELS
# ==============================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ])
}


# ==============================
# 5. TRAIN AND EVALUATE MODELS
# ==============================

results = []
best_model = None
best_model_name = ""
best_accuracy = 0


print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

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

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # Save best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# ==============================
# 6. DISPLAY RESULTS
# ==============================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ==============================
# 7. CONFUSION MATRIX
# ==============================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\nBest Model:", best_model_name)
print(f"Best Accuracy: {best_accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ==============================
# 8. SAVE MODEL
# ==============================

# ==============================
# 8. SAVE FINAL MODEL
# ==============================

os.makedirs("models", exist_ok=True)

# Use Random Forest as the final model
final_model = models["Random Forest"]

model_path = "models/insurance_claim_model.pkl"

joblib.dump(final_model, model_path)

print("\n" + "=" * 60)
print("FINAL MODEL SAVED SUCCESSFULLY!")
print("=" * 60)

print("Final Model: Random Forest")
print("Model saved at:", model_path)


# ==============================
# 9. SAVE MODEL RESULTS
# ==============================

results_df.to_csv(
    "models/model_results.csv",
    index=False
)

print("Results saved at: models/model_results.csv")