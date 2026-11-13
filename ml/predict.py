import pandas as pd
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
models_dir = BASE_DIR / "ml" / "models"

_models = {}
_le = None
_feature_names = {"adm": [], "sem": []}

def load_assets():
    """Load models and get exact feature names from training data."""
    global _le, _models, _feature_names
    if _le is None:
        print("Loading models...")
        _le = joblib.load(models_dir / "label_encoder.pkl")
        
        # Get the exact feature names directly from the processed training data
        X_adm = pd.read_csv(BASE_DIR / "data" / "processed" / "X_train_adm.csv")
        X_sem = pd.read_csv(BASE_DIR / "data" / "processed" / "X_train_sem.csv")
        _feature_names["adm"] = X_adm.columns.tolist()
        _feature_names["sem"] = X_sem.columns.tolist()
        
        for stage in ["adm", "sem"]:
            _models[stage] = joblib.load(models_dir / f"best_model_{stage}.pkl")

def predict_student(stage, features_dict):
    load_assets()
    model = _models[stage]
    
    # 1. Force the features into the exact order the model expects
    feature_order = _feature_names[stage]
    X_array = np.array([[features_dict.get(f, 0) for f in feature_order]])
    
    # 2. Predict
    pred_idx = int(model.predict(X_array)[0])
    pred_label = _le.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(X_array)[0]
    
    # 3. Explain using the model's built-in feature importances
    importances = model.feature_importances_
    
    contributions = pd.DataFrame({
        "Feature": feature_order,
        "Impact": importances
    })
    # Sort by highest importance
    top_factors = contributions.sort_values(by="Impact", ascending=False).head(5)
    
    reasons = []
    for _, row in top_factors.iterrows():
        reasons.append({
            "feature": row["Feature"],
            "impact": float(row["Impact"]),
            "direction": "increases risk"
        })
        
    return {
        "prediction": pred_label,
        "probabilities": {
            _le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))
        },
        "top_factors": reasons
    }