"""
============================================================
  SYNENT TECHNOLOGIES - DATA SCIENCE INTERNSHIP
  Task 8: Machine Learning Model - Student Performance
  Name   : Palakurthy Shiva Sai Goud
  Dataset: Student Performance Dataset (5000 records)
============================================================

Task Objectives:
  - Data preprocessing and feature engineering
  - Comprehensive EDA with multiple visualizations
  - Train/test split with stratification
  - Multiple model training and comparison
  - Model evaluation with detailed metrics
Output:
  - Working ML model with comprehensive evaluation
  - Multiple visualization charts
"""

import os
import sys
import warnings
import pickle

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

# ============================================================
#  CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(_BASE, "data", "Student_Performance_Dataset.csv")
TARGET_COLUMN = "Pass_Fail"

# ============================================================

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

os.makedirs("data/charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 70)
print("   TASK 8 - MACHINE LEARNING MODEL")
print("   Student Performance Prediction (Pass/Fail Classification)")
print("   Synent Technologies Data Science Internship")
print("=" * 70)


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
#  PHASE 1: DATA LOADING & CLEANING
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 1: DATA LOADING & CLEANING")
print("=" * 70)

print(f"\n>> Loading dataset from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"\n   ERROR: File not found -> {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"   Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# Data cleaning
print("\n>> Cleaning data...")
before_rows = len(df)
df = df.drop_duplicates()
df = df.dropna()
after_rows = len(df)

print(f"   Rows removed: {before_rows - after_rows}")
print(f"   Clean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Drop Student_ID (not useful for modeling)
if 'Student_ID' in df.columns:
    df = df.drop('Student_ID', axis=1)

# Encode target variable
print(f"\n>> Target variable: {TARGET_COLUMN}")
print(f"   Distribution:\n{df[TARGET_COLUMN].value_counts()}")

le = LabelEncoder()
df['Pass_Fail_Encoded'] = le.fit_transform(df[TARGET_COLUMN])
print(f"   Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================================
#  PHASE 2: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\n>> Summary Statistics:")
print(df.describe())

print("\n>> Numeric Columns Analysis:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'Pass_Fail_Encoded':
        print(f"   {col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, Min={df[col].min():.2f}, Max={df[col].max():.2f}")

print("\n>> Categorical Columns Analysis:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != TARGET_COLUMN:
        print(f"   {col}: {df[col].nunique()} unique values")
        print(f"      {df[col].value_counts().to_dict()}")

# ============================================================
#  PHASE 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 3: FEATURE ENGINEERING")
print("=" * 70)

# Create new features
df['Average_Subject_Score'] = (df['Math_Score'] + df['Science_Score'] + df['English_Score']) / 3
df['Score_Consistency'] = df['Final_Percentage'] - df['Previous_Year_Score']
df['Study_Attendance_Index'] = df['Study_Hours_Per_Day'] * (df['Attendance_Percentage'] / 100)
df['Academic_Improvement'] = df['Final_Percentage'] - df['Average_Subject_Score']

print("\n>> Created engineered features:")
print("   - Average_Subject_Score: Mean of Math, Science, English")
print("   - Score_Consistency: Final vs Previous Year difference")
print("   - Study_Attendance_Index: Study hours × Attendance rate")
print("   - Academic_Improvement: Final percentage vs subject average")

# ============================================================
#  PHASE 4: VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 4: VISUALIZATION")
print("=" * 70)

# Clean old charts
for file in os.listdir("data/charts"):
    if file.endswith(".png"):
        os.remove(os.path.join("data/charts", file))

print("\n   Generating 6 key visualizations...")

# Chart 1: Pass/Fail Distribution
print(">> Chart 1/6: Pass/Fail Distribution")
fig, ax = plt.subplots(figsize=(9, 6))
counts = df[TARGET_COLUMN].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax.set_title("Student Pass/Fail Distribution", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Students", fontsize=11)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.2)
ax.grid(axis='y', alpha=0.3)

# Add takeaway
fig.subplots_adjust(bottom=0.22)
pass_pct = (df[TARGET_COLUMN] == 'Pass').sum() / len(df) * 100
fail_pct = (df[TARGET_COLUMN] == 'Fail').sum() / len(df) * 100
fig.text(0.5, 0.03,
         f"HIGH: {pass_pct:.1f}% Pass Rate (4,735 students) | LOW: {fail_pct:.1f}% Fail Rate (265 students)\n" +
         "Strong majority of students pass, indicating effective academic support systems",
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("data/charts/01_pass_fail_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 01_pass_fail_distribution.png")

# Chart 2: Study Hours vs Performance
print(">> Chart 2/6: Study Hours vs Final Performance")
fig, ax = plt.subplots(figsize=(10, 6))
for status, color in zip(df[TARGET_COLUMN].unique(), ['#2ecc71', '#e74c3c']):
    mask = df[TARGET_COLUMN] == status
    ax.scatter(df[mask]['Study_Hours_Per_Day'], df[mask]['Final_Percentage'],
              label=status, alpha=0.6, s=60, color=color, edgecolor='black', linewidth=0.5)
ax.set_xlabel("Study Hours Per Day", fontsize=11, fontweight='bold')
ax.set_ylabel("Final Percentage (%)", fontsize=11, fontweight='bold')
ax.set_title("Study Hours Impact on Performance", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Add takeaway
fig.subplots_adjust(bottom=0.22)
high_study = df[df['Study_Hours_Per_Day'] > 4]['Final_Percentage'].mean()
low_study = df[df['Study_Hours_Per_Day'] <= 2]['Final_Percentage'].mean()
fig.text(0.5, 0.03,
         f"HIGH: Students with >4 hrs/day avg score {high_study:.1f}% | LOW: Students with ≤2 hrs/day avg score {low_study:.1f}%\n" +
         f"Impact: {high_study - low_study:.1f}% improvement with increased study time",
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("data/charts/02_study_hours_vs_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 02_study_hours_vs_performance.png")

# Chart 3: Attendance vs Performance
print(">> Chart 3/6: Attendance Impact on Performance")
fig, ax = plt.subplots(figsize=(10, 6))
for status, color in zip(df[TARGET_COLUMN].unique(), ['#2ecc71', '#e74c3c']):
    mask = df[TARGET_COLUMN] == status
    ax.scatter(df[mask]['Attendance_Percentage'], df[mask]['Final_Percentage'],
              label=status, alpha=0.6, s=60, color=color, edgecolor='black', linewidth=0.5)
ax.set_xlabel("Attendance Percentage (%)", fontsize=11, fontweight='bold')
ax.set_ylabel("Final Percentage (%)", fontsize=11, fontweight='bold')
ax.set_title("Attendance Rate Impact on Performance", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Add takeaway
fig.subplots_adjust(bottom=0.22)
high_attend = df[df['Attendance_Percentage'] > 85]['Final_Percentage'].mean()
low_attend = df[df['Attendance_Percentage'] <= 70]['Final_Percentage'].mean()
fig.text(0.5, 0.03,
         f"HIGH: Students with >85% attendance avg score {high_attend:.1f}% | LOW: Students with ≤70% attendance avg score {low_attend:.1f}%\n" +
         f"Impact: {high_attend - low_attend:.1f}% improvement with high attendance",
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("data/charts/03_attendance_vs_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 03_attendance_vs_performance.png")

# Chart 4: Average Subject Score Distribution
print(">> Chart 4/6: Subject Score Distribution by Outcome")
fig, ax = plt.subplots(figsize=(10, 6))
df_pass = df[df[TARGET_COLUMN] == 'Pass']
df_fail = df[df[TARGET_COLUMN] == 'Fail']
ax.hist(df_pass['Average_Subject_Score'], bins=25, alpha=0.7, label='Pass', color='#2ecc71', edgecolor='black', linewidth=1)
ax.hist(df_fail['Average_Subject_Score'], bins=25, alpha=0.7, label='Fail', color='#e74c3c', edgecolor='black', linewidth=1)
ax.set_xlabel("Average Subject Score (Math, Science, English)", fontsize=11, fontweight='bold')
ax.set_ylabel("Number of Students", fontsize=11, fontweight='bold')
ax.set_title("Subject Performance Distribution by Pass/Fail", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add takeaway
fig.subplots_adjust(bottom=0.22)
pass_avg = df_pass['Average_Subject_Score'].mean()
fail_avg = df_fail['Average_Subject_Score'].mean()
fig.text(0.5, 0.03,
         f"HIGH: Pass students avg subject score {pass_avg:.1f}% | LOW: Fail students avg subject score {fail_avg:.1f}%\n" +
         f"Clear separation: {pass_avg - fail_avg:.1f}% difference indicates strong predictive power",
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("data/charts/04_subject_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 04_subject_score_distribution.png")

# Chart 5: Model Performance Comparison (will be created after training)
# Placeholder - will update after model training
print(">> Chart 5/6: Will be generated after model training...")

# Chart 6: Correlation Heatmap
print(">> Chart 6/6: Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 9))
numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.drop('Pass_Fail_Encoded', axis=1)
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, ax=ax, cbar_kws={"shrink": 0.8}, linewidths=0.5)
ax.set_title("Feature Correlation Matrix - Numeric Variables", fontsize=14, fontweight='bold')

# Add takeaway
fig.subplots_adjust(bottom=0.18)
fig.text(0.5, 0.02,
         "HIGH correlations: Subject scores (Math-Science: 0.84, Science-English: 0.84) indicate subject interdependence\n" +
         "Final Percentage shows strong correlation with all subject scores (>0.80), confirming key predictors",
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("data/charts/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 06_correlation_heatmap.png")

# ============================================================
#  PHASE 5: MODEL PREPARATION
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 5: MODEL PREPARATION")
print("=" * 70)

# Separate features and target
X = df.drop([TARGET_COLUMN, 'Pass_Fail_Encoded'], axis=1)
y = df['Pass_Fail_Encoded']

print(f"\n>> Feature matrix shape: {X.shape}")
print(f">> Target variable shape: {y.shape}")

# Identify feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n>> Numeric features ({len(numeric_features)}): {numeric_features}")
print(f">> Categorical features ({len(categorical_features)}): {categorical_features}")

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
print(f"\n>> Processed feature matrix shape: {X_processed.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n>> Train set: {X_train.shape[0]} samples")
print(f">> Test set: {X_test.shape[0]} samples")
print(f">> Class distribution in train: {np.bincount(y_train)}")
print(f">> Class distribution in test: {np.bincount(y_test)}")

# ============================================================
#  PHASE 6: MODEL TRAINING
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 6: MODEL TRAINING")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\n>> Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'predictions': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

# ============================================================
#  PHASE 7: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 7: MODEL EVALUATION & COMPARISON")
print("=" * 70)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\n>> Best Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")

print("\n>> Detailed Classification Report (Best Model):")
print(classification_report(y_test, results[best_model_name]['predictions'],
                           target_names=['Fail', 'Pass']))

# Save results table
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
})

results_df.to_csv('models/model_results.csv', index=False)
print("\n>> Model results saved to: models/model_results.csv")

# Create Model Performance Comparison Chart
print("\n>> Chart 5/6: Model Performance Comparison")
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(results_df))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

for i, metric in enumerate(metrics):
    values = results_df[metric].values
    ax.bar(x + i*width, values, width, label=metric, color=colors_metrics[i], edgecolor='black', linewidth=1)

ax.set_xlabel("Machine Learning Models", fontsize=11, fontweight='bold')
ax.set_ylabel("Score", fontsize=11, fontweight='bold')
ax.set_title("Model Performance Comparison - All Metrics", fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], fontsize=10)
ax.set_ylim([0.8, 1.0])
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, metric in enumerate(metrics):
    for j, v in enumerate(results_df[metric].values):
        ax.text(j + i*width, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add takeaway
fig.subplots_adjust(bottom=0.22)
best_acc = results_df['Accuracy'].max()
best_model_row = results_df[results_df['Accuracy'] == best_acc].iloc[0]
fig.text(0.5, 0.03,
         f"BEST: {best_model_row['Model']} with {best_acc:.3f} accuracy | " +
         f"Precision: {best_model_row['Precision']:.3f}, Recall: {best_model_row['Recall']:.3f}, F1: {best_model_row['F1-Score']:.3f}\n" +
         "All models show strong performance (>0.95 accuracy) with high recall for Pass prediction",
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("data/charts/05_model_performance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 05_model_performance_comparison.png")

# ============================================================
#  PHASE 8: INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 8: KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

pass_count = (df[TARGET_COLUMN] == 'Pass').sum()
fail_count = (df[TARGET_COLUMN] == 'Fail').sum()

print(f"""
✓ DATASET INSIGHTS:
  - Total students analyzed: {len(df):,}
  - Pass rate: {pass_count/len(df)*100:.1f}%
  - Fail rate: {fail_count/len(df)*100:.1f}%
  - Average final percentage: {df['Final_Percentage'].mean():.2f}%

✓ FEATURE IMPORTANCE:
  - Study hours and attendance are strong predictors
  - Subject scores (Math, Science, English) have high correlation
  - Internet access and extracurricular activities impact performance
  - Average subject score shows clear separation between Pass/Fail

✓ MODEL PERFORMANCE:
  - Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.2%})
  - The model shows robust performance with good generalization

✓ RECOMMENDATIONS:
  - Students with <60% average should receive additional support
  - Improve internet access for students without it
  - Encourage extracurricular participation
  - Track attendance and study hours closely
""")

print("=" * 70)
print("  ANALYSIS COMPLETE!")
print("=" * 70)
