import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from app.ml.preprocessing import preprocess_dataframe
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix 



# ============================================================
# TODO 1: Load data & cek hasil preprocessing
# ============================================================
#   - baca dataset (pd.read_csv), panggil preprocess_dataframe
#   - print: shape sebelum vs sesudah, df.head(), df["label"].value_counts()
#   - pertanyaan: apakah kelasnya seimbang? apa dampaknya ke model?
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_preprocess = preprocess_dataframe(df)
    print("Sebelum:", df.shape)
    print("Setelah:", df_preprocess.shape)
    print(df_preprocess.head())
    print(df_preprocess["label"].value_counts())
    print(df_preprocess["label"].value_counts(normalize=True))

    return df_preprocess


# ============================================================
# TODO 2: Split train / test
# ============================================================
#   - train_test_split, test_size ~0.2, random_state tetap
#   - kenapa perlu stratify=label?
#   - kenapa split harus SEBELUM TF-IDF di-fit? (cari: data leakage)
def split_data(df: pd.DataFrame):
    X = df.drop(columns=['label'])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print('X_train : ')
    print(X_train.head())
    print('')
    print('X_test : ')
    print(X_test.head())
    print('')
    print('y_train : ')
    print(y_train.head())
    print('')
    print('y_test : ')
    print(y_test.head())

    return X_train, X_test, y_train, y_test


# ============================================================
# TODO 3: Pipeline fitur
# ============================================================
#   - kolom "text"  -> TfidfVectorizer (coba ngram_range, max_features, min_df)
#   - kolom numerik -> StandardScaler (num_urls, has_ip_url, has_shortener, max_urls_len)
#   - gabungkan dengan ColumnTransformer
#   - bungkus semuanya dalam sklearn Pipeline bersama model
#   - pertanyaan: apakah has_ip_url / has_shortener perlu di-scale?
def build_pipeline(model=None):
    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")

    text_pipeline = Pipeline([
        (
            "tf-idf", 
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df = 2,
            ),
        ),
    ])

    num_transformer = Pipeline([('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, "text"),
            ('num', num_transformer, ["num_urls", "max_urls_len"]),
            ('bin', "passthrough", ["has_ip_url", "has_shortener"]),
        ]
    )

    # Full model pipeline
    full_pipeline = Pipeline(
        [('preprocessor', preprocessor), ('clf', model)]
    )

    return full_pipeline

# ============================================================
# TODO 4: Training baseline
# ============================================================
#   - mulai dari LogisticRegression (coba class_weight="balanced")
#   - fit di data train saja
def train(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


# ============================================================
# TODO 5: Evaluasi
# ============================================================
#   - classification_report, confusion_matrix
#   - metrik apa yang paling penting untuk deteksi phishing? accuracy? recall? precision? kenapa?
#   - cek: email phishing yang lolos (false negative) itu kayak apa? print beberapa contoh
def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    fn_mask = (y_test.values == 1) & (y_pred == 0)
    fn_samples = X_test[fn_mask]
    
    print(f"\n=== Contoh False Negatives (Total: {len(fn_samples)}) ===")
    print(fn_samples["text"].head())


# ============================================================
# TODO 6: Bandingkan model
# ============================================================
#   - coba LinearSVC, RandomForest, (opsional) XGBoost
#   - bandingkan pakai cross_val_score (cv=5, scoring="f1" atau "recall")
#   - opsional: GridSearchCV untuk tuning TF-IDF + model
def compare_models(X_train, y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "LinearSVC": LinearSVC(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1)
    }


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, clf in models.items():
        pipe = build_pipeline(clf)

        scores = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1, scoring=["f1", "precision", "recall"])
    
        results.append({
            "Model": name,
            "F1 Mean": scores["test_f1"].mean(),
            "F1 Std": scores["test_f1"].std(),  
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean()
        })

    df_comp = pd.DataFrame(results).sort_values(by="F1 Mean", ascending=False)
    return df_comp
    


# ============================================================
# TODO 7: Simpan model
# ============================================================
#   - joblib.dump(pipeline, "models/phishing_model.joblib")
#   - kenapa yang disimpan harus seluruh pipeline, bukan modelnya saja?
def save_model(pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(pipeline, path)
    print(f"Model berhasil tersimpan di {path}")


# ============================================================
# TODO 9: Integrasi ke API
# ============================================================
#   - app/schemas : request (subject, body, sender) & response (label, score)
#   - app/services: bungkus predict()
#   - app/api     : POST /predict
#   - daftarkan router di app/main.py


if __name__ == "__main__":
    df = load_data("training/dataset/CEAS_08.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    print(compare_models(X_train, y_train))
    pipe = build_pipeline()
    model = train(pipe, X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model, "models/phishing_model.joblib")
    

