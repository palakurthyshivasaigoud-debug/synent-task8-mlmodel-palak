# synent-task8-mlmodel-palak

## Task 8: Student Performance Machine Learning Model

This project was completed for the Synent Technologies Data Science Internship. It builds and evaluates classification models that predict student pass/fail status based on comprehensive academic and behavioral features from 5,000 students.

## Problem Statement

The task is to build a robust prediction model using a detailed student performance dataset. I performed comprehensive EDA, feature engineering, and compared three machine learning models to predict whether a student will Pass or Fail based on their academic performance, study habits, attendance, and socioeconomic factors.

## Dataset

**Source:** [Student Performance Dataset](https://www.kaggle.com/datasets/suvidyasonawane/student-performance-dataset) - Kaggle (5,000 students)

Local file:

```text
data/Student_Performance_Dataset.csv
```

**Target Column:** `Pass_Fail` (Pass/Fail binary classification)

**Key Features:**
- **Academic:** Math_Score, Science_Score, English_Score, Previous_Year_Score, Final_Percentage
- **Behavioral:** Study_Hours_Per_Day, Attendance_Percentage, Extracurricular_Activities
- **Demographic:** Age, Gender, Class, Parental_Education, Internet_Access
- **Engineered:** Average_Subject_Score, Score_Consistency, Study_Attendance_Index, Academic_Improvement

## Approach

1. Loaded the comprehensive student performance dataset (5,000 records, 16 columns).
2. Performed data cleaning and removed duplicates.
3. Conducted exploratory data analysis (EDA) with summary statistics and correlations.
4. Created engineered features: Average Subject Score, Score Consistency, Study-Attendance Index, Academic Improvement.
5. Analyzed relationships between predictors and pass/fail status.
6. Identified key factors: study hours, attendance, subject scores, internet access, extracurricular activities.
7. Split data into training (80%) and testing (20%) sets with stratification.
8. Trained three classification models: Logistic Regression, Decision Tree, and Random Forest.
9. Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices.
10. Visualized insights with 9 comprehensive charts.

## Key Results & Insights

**Dataset Characteristics:**
- Total students: 5,000
- Pass rate: 94.7% (4,735 students)
- Fail rate: 5.3% (265 students)
- Average final percentage: 67.48%

**Model Performance:**
- All three models achieved strong accuracy on this dataset
- Models successfully identify the key factors driving academic success
- The strong pass/fail separation in features enables effective prediction

**Key Findings:**
1. **Study Impact:** Students with higher study hours show significantly better performance
2. **Attendance Correlation:** Attendance percentage strongly correlates with final scores
3. **Subject Performance:** Math, Science, and English scores are highly correlated with each other
4. **Internet Access:** Students with internet access show improved performance trends
5. **Extracurricular Benefits:** Participation in extracurricular activities correlates with higher pass rates
6. **Performance Levels:** Clear separation between Excellent/Good and Average/Poor performers
7. **Parental Education:** Shows modest correlation with student outcomes

**Model Comparison:**
| Metric | Logistic Regression | Decision Tree | Random Forest |
| --- | ---: | ---: | ---: |
| Accuracy | ~0.95 | ~0.96 | ~0.96 |
| Precision | ~0.95 | ~0.96 | ~0.96 |
| Recall | ~0.99 | ~0.99 | ~0.99 |
| F1-Score | ~0.97 | ~0.97 | ~0.97 |

The models show robust performance with high recall on the Pass class, indicating reliable identification of passing students.

## Visualizations

6 key visualizations with detailed explanations of high/low performance factors:

| Chart | Description |
| --- | --- |
| `01_pass_fail_distribution.png` | **Pass/Fail Distribution**: HIGH 94.7% pass rate, LOW 5.3% fail rate showing strong academic support systems |
| `02_study_hours_vs_performance.png` | **Study Hours Impact**: HIGH students (>4 hrs/day) score ~8% better than LOW performers (≤2 hrs/day) |
| `03_attendance_vs_performance.png` | **Attendance Impact**: HIGH attendance (>85%) correlates with ~6% higher final scores vs LOW attendance (≤70%) |
| `04_subject_score_distribution.png` | **Subject Performance**: HIGH clear separation between Pass/Fail groups; 15%+ difference in average subject scores |
| `05_model_performance_comparison.png` | **Model Metrics**: Comparison of Accuracy, Precision, Recall, F1-Score across 3 ML models; all >0.95 accuracy |
| `06_correlation_heatmap.png` | **Feature Correlation**: HIGH correlations between subject scores (0.84+) and strong link to final performance |

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
|   |-- Student_Performance_Dataset.csv
|   `-- charts/
|-- models/
|   `-- model_results.csv
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
