import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "dropout.csv"
processed_dir = BASE_DIR / "data" / "processed"
models_dir = BASE_DIR / "ml" / "models"

processed_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print("1. Loading raw data...")
df = pd.read_csv(data_path, sep=";")

# Clean up any accidental spaces in column names
df.columns = df.columns.str.strip()

print("2. Encoding target variable...")
le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])
joblib.dump(le, models_dir / "label_encoder.pkl")

# ---------------------------------------------------------
# WEEK 4: DEFINE THE TWO FEATURE SETS USING INDICES
# ---------------------------------------------------------
# Index 0 to 20 are admission features (including 'Admission grade' at 12)
admission_indices = list(range(0, 21))

# Semester features = Admission features + 1st Semester grades (indices 21 to 26)
semester_indices = admission_indices + list(range(21, 27))

# ---------------------------------------------------------
# MODEL 1: ADMISSION STAGE
# ---------------------------------------------------------
print("\n--- Processing Admission Stage Data ---")
X_adm = df.iloc[:, admission_indices]
y_adm = df["Target"]

scaler_adm = StandardScaler()
X_adm_scaled = pd.DataFrame(scaler_adm.fit_transform(X_adm), columns=X_adm.columns)

X_train_adm, X_test_adm, y_train_adm, y_test_adm = train_test_split(
    X_adm_scaled, y_adm, test_size=0.2, random_state=42, stratify=y_adm
)

X_train_adm.to_csv(processed_dir / "X_train_adm.csv", index=False)
X_test_adm.to_csv(processed_dir / "X_test_adm.csv", index=False)
y_train_adm.to_csv(processed_dir / "y_train_adm.csv", index=False)
y_test_adm.to_csv(processed_dir / "y_test_adm.csv", index=False)
joblib.dump(scaler_adm, models_dir / "scaler_adm.pkl")

# ---------------------------------------------------------
# MODEL 2: SEMESTER STAGE
# ---------------------------------------------------------
print("\n--- Processing Semester Stage Data ---")
X_sem = df.iloc[:, semester_indices]
y_sem = df["Target"]

scaler_sem = StandardScaler()
X_sem_scaled = pd.DataFrame(scaler_sem.fit_transform(X_sem), columns=X_sem.columns)

X_train_sem, X_test_sem, y_train_sem, y_test_sem = train_test_split(
    X_sem_scaled, y_sem, test_size=0.2, random_state=42, stratify=y_sem
)

X_train_sem.to_csv(processed_dir / "X_train_sem.csv", index=False)
X_test_sem.to_csv(processed_dir / "X_test_sem.csv", index=False)
y_train_sem.to_csv(processed_dir / "y_train_sem.csv", index=False)
y_test_sem.to_csv(processed_dir / "y_test_sem.csv", index=False)
joblib.dump(scaler_sem, models_dir / "scaler_sem.pkl")

print("\n✅ Week 4 Complete! Two distinct feature sets created.")
print(f"Admission features: {X_adm.shape[1]} columns")
print(f"Semester features:  {X_sem.shape[1]} columns")