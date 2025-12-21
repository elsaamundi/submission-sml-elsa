import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- BAGIAN TOKEN (EDIT DISINI) ---
# Untuk dijalankan di laptop sekarang, masukkan token aslimu di dalam tanda kutip.
# Nanti kalau mau push ke GitHub, ganti jadi os.environ["DAGSHUB_TOKEN"] biar aman.
try:
    # GANTI "TOKEN_PANJANG..." DENGAN TOKEN ASLI DAGSHUB KAMU
    dagshub.auth.add_app_token("TOKEN_DAGSHUB_KAMU_YANG_PANJANG_DISINI")
except:
    print("Token error, lanjut manual.")

# --- Setup DagsHub & MLflow ---
dagshub.init(repo_owner='elsaamundi', repo_name='submission-sml-elsa', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/elsaamundi/submission-sml-elsa.mlflow")
mlflow.set_experiment("Eksperimen Churn Prediction")

def main():
    # Path dataset hasil preprocessing
    # Pastikan path ini benar sesuai struktur folder di laptopmu
    data_path = 'telco_churn_preprocessing/train_processed.csv'
    
    # Cek file datanya ada apa ngga
    if not os.path.exists(data_path):
        print(f"File dataset gak ketemu di: {data_path}")
        return

    # Load data
    print("Lagi loading data...")
    df = pd.read_csv(data_path)
    
    # Pisahin fitur sama target (Churn)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data train/test 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mulai tracking experiment
    with mlflow.start_run():
        print("Proses training model...")
        
        # Setting parameter
        n_estimators = 100
        random_state = 42
        
        # Log parameter ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "Random Forest Classifier")

        # Training model Random Forest
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Prediksi & Evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metric akurasi
        mlflow.log_metric("accuracy", accuracy)
        print(f"Akurasi dapet: {accuracy:.4f}")

        # Simpan modelnya (PENTING: Ini yang bikin folder 'model_random_forest' muncul)
        mlflow.sklearn.log_model(model, "model_random_forest")

        # --- ARTEFAK 1: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan gambar cm sementara
        cm_filename = "confusion_matrix.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename) # Upload ke DagsHub
        print("Gambar Confusion Matrix udah ke-upload.")

        # --- ARTEFAK 2: Feature Importance ---
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.tight_layout()
        
        # Simpan gambar feature importance sementara
        fi_filename = "feature_importance.png"
        plt.savefig(fi_filename)
        mlflow.log_artifact(fi_filename) # Upload ke DagsHub
        print("Gambar Feature Importance udah ke-upload.")
        
        # Bersihin file gambar sementara di laptop biar gak numpuk
        for f in [cm_filename, fi_filename]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()