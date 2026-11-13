# Student Dropout Early Warning System (SDEWS)

An explainable Machine Learning web application that predicts student dropout risk and provides actionable insights for mentors and Heads of Department.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.0-red)
![SHAP](https://img.shields.io/badge/SHAP-0.46.0-purple)

## 📌 Project Overview
Student dropout is a major challenge in higher education. Most institutions detect at-risk students only after poor results or long absences. SDEWS solves this by using Machine Learning to identify at-risk students **early** using demographic, socio-economic, and academic data.

The system classifies each student as **Dropout**, **Enrolled**, or **Graduate**, provides a probability-based risk score, and explains the main reasons behind each prediction.

## 🚀 Features
- **Two-Stage Prediction:** Uses an Admission-Stage model (early warning) and a Semester-Stage model (after 1st-semester grades).
- **Machine Learning Models:** Trains Logistic Regression, Random Forest, and XGBoost to find the best model for each stage.
- **Explainable AI (XAI):** Uses SHAP values to explain *why* a student is at risk (e.g., unpaid fees, low grades), not just flagging them as a black box.
- **Role-Based Dashboard:** Secure login for Mentors and HoDs to view a ranked risk list and individual student profiles.
- **Indian Validation:** Validates model risk factors against Indian student data (UCI #467).

## 🛠️ Tech Stack
- **Backend:** Flask, SQLAlchemy, SQLite
- **Machine Learning:** scikit-learn, XGBoost, imbalanced-learn
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, Bootstrap 5, JavaScript, Plotly
- **Testing:** pytest

## 📂 Project Structure