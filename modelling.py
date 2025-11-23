import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Dataset Preprocessing
DATA_PATH = "FakeNewsNet_preprocessing.csv"

df = pd.read_csv(DATA_PATH)

print("Jumlah data:", df.shape)
print(df.head())

# Kolom
text_col = "clean_title"
label_col = "real"

# Fix missing clean title
df[text_col] = df[text_col].astype(str)
df['clean_title'] = df['clean_title'].astype(str)

# Ganti NaN menjadi string kosong
df['clean_title'] = df['clean_title'].fillna("")

print("Missing clean_title:", df['clean_title'].isna().sum())

X = df[text_col]
y = df[label_col]

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Model Training dengan MLflow Autolog 

with mlflow.start_run(run_name="LogisticRegression_FakeNews"):
    mlflow.autolog()

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", acc)
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

print("\nModel selesai dilatih dan dicatat di MLflow.")
print("Untuk membuka MLflow UI:")
print("   mlflow ui")
