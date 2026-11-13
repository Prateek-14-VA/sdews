import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "data" / "processed"
models_dir = BASE_DIR / "ml" / "models"

def load_data(stage):
    """Load the processed train/test splits for a given stage (adm or sem)."""
    X_train = pd.read_csv(processed_dir / f"X_train_{stage}.csv")
    X_test = pd.read_csv(processed_dir / f"X_test_{stage}.csv")
    y_train = pd.read_csv(processed_dir / f"y_train_{stage}.csv").values.ravel()
    y_test = pd.read_csv(processed_dir / f"y_test_{stage}.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_evaluate(stage):
    print(f"\n================ {stage.upper()} STAGE ================")
    X_train, X_test, y_train, y_test = load_data(stage)
    
    # Define the 3 models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate (using macro-F1 as required by your synopsis)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        
        print(f"{name:20} | Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        
        # Keep track of the best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    print(f"🏆 Best model for {stage}: {best_name} (F1: {best_f1:.4f})")
    
    # Save the best model
    model_path = models_dir / f"best_model_{stage}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved to {model_path.name}")

if __name__ == "__main__":
    print("Starting Model Training...")
    train_and_evaluate("adm")
    train_and_evaluate("sem")
    print("\n✅ Week 5 Complete! Models trained and saved.")    