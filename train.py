import pickle
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

os.makedirs("/app/models", exist_ok=True)

# ── Model 1: Iris Classifier ──────────────────────────────
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

iris_model = RandomForestClassifier(n_estimators=100, random_state=42)
iris_model.fit(X_train, y_train)
iris_acc = accuracy_score(y_test, iris_model.predict(X_test))

with open("/app/models/iris_model.pkl", "wb") as f:
    pickle.dump({"model": iris_model, "accuracy": iris_acc, "classes": list(iris.target_names)}, f)

print(f"✅ Iris model saved | Accuracy: {iris_acc:.4f}")

# ── Model 2: Cancer Classifier ────────────────────────────
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cancer_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
cancer_model.fit(X_train_scaled, y_train)
cancer_acc = accuracy_score(y_test, cancer_model.predict(X_test_scaled))

with open("/app/models/cancer_model.pkl", "wb") as f:
    pickle.dump({"model": cancer_model, "scaler": scaler, "accuracy": cancer_acc, "classes": list(cancer.target_names), "features": list(cancer.feature_names)}, f)

print(f"✅ Cancer model saved | Accuracy: {cancer_acc:.4f}")
print("All models trained and saved!")
