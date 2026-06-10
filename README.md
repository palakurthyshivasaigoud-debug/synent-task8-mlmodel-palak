# synent-task8-mlmodel-palak

## Task 8: Student Performance Machine Learning Model

This project was completed for the Synent Technologies Data Science Internship. It builds and evaluates classification models that predict whether a student passes based on study behavior and academic features.

## Problem Statement

The task is to build a prediction model using a student performance dataset. I treated the target column `Passed` as a binary classification label and compared multiple machine learning models using the same train/test split.

## Dataset

Local file:

```text
data/student_performance_prediction.csv
```

Target column:

```text
Passed
```

Important input features include study hours per week, attendance rate, previous grades, extracurricular participation, and parent education level.

## Approach

1. Loaded the dataset from the repository `data` folder.
2. Removed duplicate rows and dropped ID-like columns.
3. Encoded the target column from Yes/No into 1/0.
4. Converted numeric columns into numeric types.
5. Added simple engineered features such as study-attendance and study-grade indexes.
6. Split the dataset into training and testing sets with stratification.
7. Trained Logistic Regression, Decision Tree, and Random Forest models.
8. Evaluated models using accuracy, precision, recall, F1 score, classification report, confusion matrix, and feature importance.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.509 | 0.510 | 0.466 | 0.487 |
| Logistic Regression | 0.507 | 0.508 | 0.506 | 0.507 |
| Random Forest | 0.501 | 0.501 | 0.485 | 0.493 |

Best model: Decision Tree.

The model is only slightly better than the baseline. This is still useful because it shows honest evaluation: the available features do not strongly separate passed and not-passed students in this dataset.

## Visualizations

Charts are saved in `data/charts/`.

| Chart | Description |
| --- | --- |
| `01_class_distribution.png` | Passed vs not-passed count |
| `02_model_accuracy_comparison.png` | Accuracy comparison |
| `03_confusion_matrix.png` | Prediction errors for the best model |
| `04_feature_importance.png` | Most important features |

## How to Run

```bash
pip install -r requirements.txt
python ml_model.py
```

## Repository Structure

```text
synent-task8-mlmodel-palak/
|-- ml_model.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- student_performance_prediction.csv
|   `-- charts/
`-- .gitignore
```

## Internship Requirement Mapping

| Requirement | Status |
| --- | --- |
| Data preprocessing | Completed |
| Feature selection | Completed |
| Train/test split | Completed |
| Model training | Completed |
| Accuracy/evaluation | Completed |
| Dataset included | Completed |

## Author

Palakurthy Shiva Sai Goud

Submitted for Synent Technologies Data Science Internship - Task 8.
