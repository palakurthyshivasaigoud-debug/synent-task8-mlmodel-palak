"""
============================================================
  SYNENT TECHNOLOGIES - DATA SCIENCE INTERNSHIP
  Task 8: Machine Learning Model
  Name   : Palakurthy Shiva Sai Goud
  Dataset: Student Performance Prediction Dataset
============================================================

Task Objectives:
  - Data preprocessing
  - Feature selection
  - Train/test split
  - Model training
  - Model evaluation
Output:
  - Working ML model with evaluation
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

# ============================================================
#  CONFIGURATION - UPDATE THIS PATH IF NEEDED
# ============================================================

DATA_PATH = os.path.join(_BASE, "data", "student_performance_prediction.csv")
TARGET_COLUMN = "Passed"

# ============================================================

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

os.makedirs("data/charts", exist_ok=True)

print("=" * 60)
print("   TASK 8 - MACHINE LEARNING MODEL")
print("   Student Performance Prediction")
print("   Synent Technologies Data Science Internship")
print("=" * 60)


def add_takeaway(fig, lines):
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5,
        0.03,
        "\n".join(lines),
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )


# ============================================================
#  PHASE 1: DATA CLEANING & PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 1: DATA CLEANING & PREPROCESSING")
print("=" * 60)

print(f"\n>> Loading dataset from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"\n   ERROR: File not found -> {DATA_PATH}")
    print("   Please update DATA_PATH at the top of this script.")
    sys.exit(1)

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"   ERROR reading file: {e}")
    sys.exit(1)

print(f"   Raw shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Columns   : {list(df.columns)}")

if TARGET_COLUMN not in df.columns:
    print(f"\n   ERROR: Target column '{TARGET_COLUMN}' not found.")
    sys.exit(1)

print("\n>> Cleaning dataset...")
before_rows = len(df)
df = df.drop_duplicates()

id_columns = [col for col in df.columns if "id" in col.lower()]
if id_columns:
    print(f"   Dropping ID columns: {id_columns}")
    df = df.drop(columns=id_columns)

df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
df = df.dropna(subset=[TARGET_COLUMN])
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

for col in ["Study Hours per Week", "Attendance Rate", "Previous Grades"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if {"Study Hours per Week", "Attendance Rate"}.issubset(df.columns):
    df["Study_Attendance_Index"] = df["Study Hours per Week"] * df["Attendance Rate"] / 100

if {"Study Hours per Week", "Previous Grades"}.issubset(df.columns):
    df["Study_Grade_Index"] = df["Study Hours per Week"] * df["Previous Grades"] / 100

if "Previous Grades" in df.columns:
    df["High_Grades"] = np.where(df["Previous Grades"].isna(), np.nan, (df["Previous Grades"] >= 75).astype(int))

print(f"   Duplicates removed: {before_rows - len(df):,}")
print(f"   Clean shape       : {df.shape[0]:,} rows x {df.shape[1]} columns")

missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("   No missing feature values found.")
else:
    print("   Missing feature values will be handled by imputation:")
    for col, count in missing.items():
        print(f"   - {col}: {count:,}")

print("   [Data Cleaning Complete]")


# ============================================================
#  PHASE 2: FEATURE SELECTION
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 2: FEATURE SELECTION")
print("=" * 60)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print(f"\n   Target column       : {TARGET_COLUMN}")
print(f"   Numeric features    : {numeric_features}")
print(f"   Categorical features: {categorical_features}")
print(f"   Class distribution  : Passed={y.sum():,}, Not Passed={(y == 0).sum():,}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ============================================================
#  PHASE 3: TRAIN/TEST SPLIT & MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3: TRAIN/TEST SPLIT & MODEL TRAINING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\n   Training rows: {len(X_train):,}")
print(f"   Testing rows : {len(X_test):,}")

baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"   Baseline accuracy: {baseline_accuracy:.3f} (always predicting the majority class)")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=3,
        min_samples_leaf=100,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=4,
        min_samples_leaf=80,
        max_features="sqrt",
        class_weight="balanced",
    ),
}

results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\n>> Training {model_name}...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }
    results.append(metrics)
    trained_models[model_name] = pipeline

    print(f"   Accuracy : {metrics['Accuracy']:.3f}")
    print(f"   Precision: {metrics['Precision']:.3f}")
    print(f"   Recall   : {metrics['Recall']:.3f}")
    print(f"   F1 Score : {metrics['F1 Score']:.3f}")

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

print("\n   [Model Training Complete]")


# ============================================================
#  PHASE 4: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 4: MODEL EVALUATION")
print("=" * 60)

print("\n   Model Comparison:")
for _, row in results_df.iterrows():
    print(
        f"   {row['Model']:<20} "
        f"Accuracy={row['Accuracy']:.3f}  "
        f"Precision={row['Precision']:.3f}  "
        f"Recall={row['Recall']:.3f}  "
        f"F1={row['F1 Score']:.3f}"
    )

print(f"\n   Best Model: {best_model_name}")
print(f"   Baseline Accuracy: {baseline_accuracy:.3f}")
print(f"   Improvement Over Baseline: {results_df.iloc[0]['Accuracy'] - baseline_accuracy:.3f}")
print("\n   Classification Report:")
print(classification_report(y_test, best_predictions, target_names=["Not Passed", "Passed"]))


# ============================================================
#  PHASE 5: VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 5: VISUALIZATION")
print("=" * 60)
print("\n   Generating charts...")
print("   Each graph will open on screen. Close the graph window to see the next one.\n")

for file_name in os.listdir("data/charts"):
    if file_name.lower().endswith(".png"):
        os.remove(os.path.join("data/charts", file_name))


# Chart 1: Class distribution
print(">> Chart 1/4: Pass vs Not Pass Distribution")
fig, ax = plt.subplots(figsize=(7, 5))
class_counts = y.map({1: "Passed", 0: "Not Passed"}).value_counts()
bars = ax.bar(class_counts.index, class_counts.values, color=["#4caf7d", "#e05c5c"], edgecolor="white")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h):,}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Student Outcome Distribution")
ax.set_ylabel("Number of Students")
ax.grid(axis="y", alpha=0.35)
add_takeaway(fig, [
    f"Passed students: {int(y.sum()):,}",
    f"Not passed students: {int((y == 0).sum()):,}",
])
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig("data/charts/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("   Saved: 01_class_distribution.png")


# Chart 2: Model comparison
print("\n>> Chart 2/4: Model Accuracy Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(results_df["Model"], results_df["Accuracy"], color=sns.color_palette("Blues", len(results_df)), edgecolor="white")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.35)
add_takeaway(fig, [
    f"Best model: {best_model_name}",
    f"Best accuracy: {results_df.iloc[0]['Accuracy']:.3f}",
])
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig("data/charts/02_model_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("   Saved: 02_model_accuracy_comparison.png")


# Chart 3: Confusion matrix
print("\n>> Chart 3/4: Confusion Matrix")
cm = confusion_matrix(y_test, best_predictions)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Passed", "Passed"],
    yticklabels=["Not Passed", "Passed"],
    ax=ax,
)
ax.set_title(f"Confusion Matrix - {best_model_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
add_takeaway(fig, [
    f"Correct predictions: {cm.trace():,}",
    f"Wrong predictions: {cm.sum() - cm.trace():,}",
])
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig("data/charts/03_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("   Saved: 03_confusion_matrix.png")


# Chart 4: Feature importance for tree-based best model, or coefficients for logistic regression
print("\n>> Chart 4/4: Feature Importance")
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
model_step = best_model.named_steps["model"]

if hasattr(model_step, "feature_importances_"):
    importance_values = model_step.feature_importances_
    importance_title = f"Feature Importance - {best_model_name}"
else:
    importance_values = np.abs(model_step.coef_[0])
    importance_title = f"Feature Impact - {best_model_name}"

importance_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
    .sort_values("Importance", ascending=False)
    .head(10)
)
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#4C72B0", edgecolor="white")
for bar in bars:
    w = bar.get_width()
    ax.text(w, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=8)
ax.set_title(importance_title)
ax.set_xlabel("Importance")
ax.grid(axis="x", alpha=0.35)
top_feature = importance_df.iloc[0]["Feature"]
add_takeaway(fig, [
    f"Most important feature: {top_feature}",
    f"Used to predict whether a student passed",
])
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig("data/charts/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("   Saved: 04_feature_importance.png")

print("\n   [Visualization Complete]")


# ============================================================
#  FINAL: ML MODEL REPORT
# ============================================================
print("\n" + "=" * 60)
print("  MACHINE LEARNING MODEL REPORT")
print("=" * 60)

print(f"""
  PROBLEM TYPE
  ------------
  Classification: Predict whether a student Passed or Not Passed

  DATASET
  -------
  Rows after cleaning : {len(df):,}
  Features used       : {len(X.columns)}
  Target              : {TARGET_COLUMN}

  BEST MODEL
  ----------
  Model               : {best_model_name}
  Accuracy            : {results_df.iloc[0]['Accuracy']:.3f}
  Baseline Accuracy   : {baseline_accuracy:.3f}
  Baseline Improvement: {results_df.iloc[0]['Accuracy'] - baseline_accuracy:.3f}
  Precision           : {results_df.iloc[0]['Precision']:.3f}
  Recall              : {results_df.iloc[0]['Recall']:.3f}
  F1 Score            : {results_df.iloc[0]['F1 Score']:.3f}

  KEY INSIGHT
  -----------
  The model predicts student pass status using study hours, attendance,
  previous grades, extracurricular participation, and parent education level.
  Accuracy is low because the available features have very weak relationship
  with the pass/fail target in this dataset.
""")

print("=" * 60)
print("  4 charts saved in: data/charts/")
print("  Task 8 - Machine Learning Model Complete!")
print("=" * 60)
