import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "data" / "processed"
models_dir = BASE_DIR / "ml" / "models"

def explain_student(stage, student_index):
    """
    Explains the model's prediction for a single student from the test set.
    Stage: 'adm' or 'sem'
    """
    print(f"\n--- Explaining Student {student_index} ({stage.upper()} Stage) ---")
    
    # 1. Load data and model
    X_test = pd.read_csv(processed_dir / f"X_test_{stage}.csv")
    y_test = pd.read_csv(processed_dir / f"y_test_{stage}.csv").values.ravel()
    model = joblib.load(models_dir / f"best_model_{stage}.pkl")
    le = joblib.load(models_dir / "label_encoder.pkl")
    
    student_data = X_test.iloc[[student_index]]
    actual_label = le.inverse_transform([y_test[student_index]])[0]
    prediction = model.predict(student_data)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    
    print(f"Actual Outcome:    {actual_label}")
    print(f"Model Prediction:  {predicted_label}")
    
    # 2. Calculate SHAP values using KernelExplainer
    print("Calculating SHAP values (this may take a few seconds)...")
    
    def predict_fn(x):
        return model.predict_proba(x)
    
    background = shap.sample(X_test, 10, random_state=42)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    shap_values = explainer.shap_values(student_data)
    
    # 3. Safely extract the SHAP values for the predicted class
    class_idx = prediction
    
    # SHAP KernelExplainer outputs different shapes depending on version and binary/multiclass
    if isinstance(shap_values, list):
        # List of arrays (one per class)
        student_shap = shap_values[class_idx][0]
    elif len(shap_values.shape) == 3:
        # 3D array: (samples, features, classes)
        student_shap = shap_values[0, :, class_idx]
    elif len(shap_values.shape) == 2:
        # 2D array: (samples, features) - fallback
        student_shap = shap_values[0]
    else:
        # Any other weird format
        student_shap = np.array(shap_values).flatten()
    
    # 4. Map SHAP values to feature names
    feature_names = X_test.columns
    contributions = pd.DataFrame({
        "Feature": feature_names,
        "Impact": student_shap,
        "Value": student_data.values[0]
    })
    
    contributions["AbsImpact"] = contributions["Impact"].abs()
    top_factors = contributions.sort_values(by="AbsImpact", ascending=False).head(5)
    
    print(f"\nTop 5 Factors driving the prediction for class '{predicted_label}':")
    for _, row in top_factors.iterrows():
        direction = "INCREASES" if row["Impact"] > 0 else "DECREASES"
        print(f"  - {row['Feature']} (Value: {row['Value']:.2f}) -> {direction} risk (Impact: {row['Impact']:+.3f})")

if __name__ == "__main__":
    print("Testing SHAP Explainability...")
    explain_student("adm", 0)
    explain_student("sem", 0)