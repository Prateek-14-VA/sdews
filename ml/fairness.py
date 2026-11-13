import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import recall_score

BASE_DIR=Path(__file__).resolve().parent.parent
processed_dir=BASE_DIR/"data"/"processed"
models_dir=BASE_DIR/"ml"/"models"
raw_dir=BASE_DIR/"data"/"raw"

def check_fairness():
    print("1. FAIRNESS AUDIT (GENDER)")

    x_test=pd.read_csv(processed_dir/"X_test_sem.csv")
    y_test=pd.read_csv(processed_dir/"y_test_sem.csv").values.ravel()
    model=joblib.load(models_dir/"best_model_sem.pkl")
    y_pred=model.predict(x_test)

    #Gender: 1=Male, 0=Female
    gender=x_test["Gender"]

    for g,label in [(0,"Female"),(1,"Male")]:
        mask=(gender==g)
        if mask.sum()>0:
            #Calculate recall specifically for the 'Dropout' class (which is 0)
            recall=recall_score(y_test[mask],y_pred[mask],labels=[0],average=None)[0]
            print(f"{label:6}students: Dropout Recall={recall:,4f}")

def validate_indian_data():
    print("\n 2. INDIAN DATA VALIDATION")
    #Load the Indian dataset
    indian_df=pd.read_csv(raw_dir/"student-mat.csv",sep=";")
    print(f"Loaded Indian dataset: {indian_df.shape[0]}students,{indian_df.shape[1]}columns.")

    #In the Indian dataset, G3 is the final grade(0-20)
    #We treat G3<10 as "At Risk" (Dropout proxy)
    indian_df["At_Risk"]=(indian_df["G3"]<10).astype(int)
    print("\n At-Risk Rate (G3<10) based on key factors:")

    #1 School Support (Proxy for financial/educational support)
    if "schoolsup" in indian_df.columns:
        print("\nSchool Support:")
        print(indian_df.groupby("schoolsup")["At_Risk"].mean())
        
    # 2. Past Failures (Proxy for academic struggle)
    if "failures" in indian_df.columns:
        print("\nPast Failures:")
        print(indian_df.groupby("failures")["At_Risk"].mean())

if __name__ == "__main__":
    print("Running Week 7: Fairness & Validation...")
    check_fairness()
    validate_indian_data()
    print("\n✅ Week 7 Complete!")