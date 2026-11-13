from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from .models import User
import traceback
import pandas as pd
from pathlib import Path
from ml.predict import predict_student

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    return render_template("index.html")

@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("main.dashboard"))
        else:
            flash("Invalid username or password.", "danger")
            
    return render_template("auth/login.html")

@main_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.index"))

@main_bp.route("/dashboard")
@login_required
def dashboard():
    BASE_DIR = Path(__file__).resolve().parent.parent
    X_test = pd.read_csv(BASE_DIR / "data" / "processed" / "X_test_sem.csv")
    y_test = pd.read_csv(BASE_DIR / "data" / "processed" / "y_test_sem.csv").values.ravel()
    
    students = []
    for i in range(10):
        actual_outcome = y_test[i]
        features_dict = X_test.iloc[i].to_dict()
        result = predict_student("sem", features_dict)
        
        students.append({
            "id": f"S-{1000 + i}",
            "course": int(features_dict.get("Course", 0) * 100),
            "prediction": result["prediction"],
            "risk_score": result["probabilities"]["Dropout"], 
            "top_factor": result["top_factors"][0]["feature"],
            "factors": result["top_factors"],
            "actual": int(actual_outcome)
        })
    
    students = sorted(students, key=lambda x: x["risk_score"], reverse=True)
    return render_template("dashboard/dashboard.html", user=current_user, students=students)

@main_bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        data = request.get_json()
        stage = data.get("stage", "sem")
        features = data.get("features")
        
        if not features:
            return {"error": "No features provided"}, 400
            
        result = predict_student(stage, features)
        return result
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}, 500