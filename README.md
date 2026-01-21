# 🎓 Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

> **Course:** CIE 317/417 Machine Learning  
> **Institution:** Zewail City of Science, Technology and Innovation

---

## 📌 Project Overview

A comprehensive Machine Learning system that analyzes educational data and predicts student performance. Leveraging a dataset of **20,000+ student records**, the system identifies key factors influencing academic success and provides predictive insights through three core tasks:

1. **Regression:** Predicting exact `final_score` (0-100)
2. **Binary Classification:** Determining `pass_fail` status
3. **Multiclass Classification:** Predicting specific `final_grade` (A, B, C, D, F)

An interactive **Streamlit Dashboard** visualizes model performance and enables real-time predictions.

---

## 👥 Team Members

| Name | Student ID |
|------|------------|
| Mohammed Ali Sadek | 202200594 |
| Ahmed Amgad | 202200393 |
| Abdulrahman Madgy | 202200341 |
| SalahDin Ahmed Rezk | 202201079 |

---

## 📂 Dataset Details

**Source:** `Term_Project_Dataset_20K.csv`

- **Size:** 20,000+ samples
- **Features:** 40 input variables across 4 categories
- **Target Variables:** `final_score`, `final_grade`, `pass_fail`

### Feature Categories

| Category | Examples |
|----------|----------|
| **Demographic** | Age, Gender, Parent Income, Sibling Count |
| **Academic History** | Previous GPA, High School Grade, Attendance Rate |
| **Behavioral** | Study Hours, Participation, Alcohol Consumption |
| **Psychological** | Stress Level, Motivation, Anxiety, Sleep Hours |

---

## ⚙️ Methodology & Pipeline

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of grades
- Correlation matrices identifying relationships (e.g., Study Time vs. Score)
- Outlier detection and visualization

### 2. Data Preprocessing
- **Imputation:** Handling missing values in numerical and categorical columns
- **Encoding:** One-Hot Encoding for nominal features (e.g., Gender)
- **Balancing:** SMOTE (Synthetic Minority Over-sampling Technique) for class imbalance

### 3. Model Training
- Training multiple classical ML models
- Hyperparameter tuning via GridSearchCV/RandomizedSearchCV
- Cross-validation for robust performance estimation

### 4. Evaluation
- **Regression Metrics:** RMSE, MAE, R² Score
- **Classification Metrics:** Accuracy, Precision, Recall, F1-Score
- **Visualizations:** Confusion Matrices, ROC Curves, Feature Importance

---

## 🧠 Models Implemented

### 📉 Regression Models (Score Prediction)
- Linear Regression
- Ridge & Lasso Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)
- Gradient Boosting Regressor

### 📊 Classification Models (Grade/Pass-Fail Prediction)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/aboalis/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
To view the complete analysis and training process:
```bash
jupyter notebook notebooks/Final_Project.ipynb
```

### 4. Launch the Streamlit Dashboard (Optional)
For interactive predictions and visualizations:
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📁 Repository Structure

```
student-performance-prediction/
│
├── data/
│   └── Term_Project_Dataset_20K.csv   # Primary dataset
│
├── notebooks/
│   └── Final_Project.ipynb            # Main analysis & training notebook
│
├── models/                            # Saved trained models (generated)
│
├── app.py                             # Streamlit dashboard application
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore file
```

---

## 📊 Key Findings

### Top Predictive Features
1. **Previous GPA** - Strongest predictor of academic success
2. **Study Hours per Week** - Strong positive correlation with final scores
3. **Attendance Rate** - Critical factor in pass/fail outcomes
4. **Sleep Hours** - Significant impact on cognitive performance

### Model Performance Highlights
- **Best Regression Model:** Random Forest Regressor
- **Best Binary Classifier:** XGBoost
- **Best Multiclass Classifier:** Gradient Boosting 

### Insights
- **Non-linear relationships** between features favor ensemble methods (Random Forest, XGBoost)
- **SMOTE balancing** significantly improved minority class predictions (F grades)
- **Behavioral factors** (study time, participation) outweigh demographic factors in importance

---

## 🚀 Future Enhancements

- [ ] Deep Learning models (Neural Networks) for comparison
- [ ] Feature engineering with polynomial features
- [ ] Real-time data integration with student information systems
- [ ] Mobile application deployment
- [ ] Explainable AI (SHAP values) for model interpretability

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Course:** CIE 317/417 Machine Learning  
**Instructor:** Dr. Ahmed Tolba  
**Institution:** Zewail City of Science, Technology and Innovation

**Tools & Libraries:**
- Python, Scikit-Learn, XGBoost
- Pandas, NumPy, Matplotlib, Seaborn
- Streamlit, Jupyter Notebook
- Google Colab

---

## 👤 Contact

**Mohammed Ali Sadek**  
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohammed-ali-456101255)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/aboalis)

---

**Project Date:** Fall 2024  
**Last Updated:** January 2025
