import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ["MLFLOW_TRACKING_USERNAME"] = "claramarsya"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "db03a6bb43195570a02d704d72f14dcb8c97871d"

mlflow.set_tracking_uri("https://dagshub.com/claramarsya/Membangun_model.mlflow")

# Load Dataset
DATA_PATH = "FakeNewsNet_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

print("Jumlah data:", df.shape)
print(df.head())

text_col = "clean_title"
label_col = "real"

# Perbaikan missing value
df[text_col] = df[text_col].astype(str).fillna("")

X = df[text_col]
y = df[label_col]

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter Tuning
param_grid = {
    "C": [0.1, 1, 5, 10],
    "solver": ["liblinear", "lbfgs"],
    "penalty": ["l2"]
}

model = LogisticRegression(max_iter=300)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="accuracy",
    verbose=2
)

# Run MLflow manual logging
with mlflow.start_run(run_name="LogReg_FakeNews_Tuning"):

    print("\nTraining model dengan GridSearchCV...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Prediksi
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n========================")
    print("Best Params :", grid.best_params_)
    print("Best CV Acc :", grid.best_score_)
    print("Test Acc    :", acc)
    print(classification_report(y_test, y_pred))

    # Manual Logging
    mlflow.log_param("best_C", grid.best_params_["C"])
    mlflow.log_param("best_solver", grid.best_params_["solver"])
    mlflow.log_param("penalty", grid.best_params_["penalty"])
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("cv_best_score", grid.best_score_)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Simpan model
    mlflow.sklearn.log_model(best_model, "best_model")
    # Simpan TF-IDF Vectorizer
    mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")

print("\nModel tuning selesai dicatat di MLflow.")
print("Untuk membuka MLflow UI:")
print("   mlflow ui")
