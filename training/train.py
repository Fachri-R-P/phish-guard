import pandas as pd
from app.ml.preprocessing import preprocess_dataframe

# ============================================================
# TODO 1: Load data & cek hasil preprocessing
# ============================================================
#   - baca dataset (pd.read_csv), panggil preprocess_dataframe
#   - print: shape sebelum vs sesudah, df.head(), df["label"].value_counts()
#   - pertanyaan: apakah kelasnya seimbang? apa dampaknya ke model?
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("training/dataset/CEAS_08.csv")
    df_preprocess = preprocess_dataframe(df)
    print("Sebelum:", df.shape)
    print("Setelah:", df_preprocess.shape)
    print(df_preprocess.head())
    print(df_preprocess["label"].value_counts())
    print(df_preprocess["label"].value_counts(normalize=True))

    return df


# ============================================================
# TODO 2: Split train / test
# ============================================================
#   - train_test_split, test_size ~0.2, random_state tetap
#   - kenapa perlu stratify=label?
#   - kenapa split harus SEBELUM TF-IDF di-fit? (cari: data leakage)
def split_data(df: pd.DataFrame):
    pass


# ============================================================
# TODO 3: Pipeline fitur
# ============================================================
#   - kolom "text"  -> TfidfVectorizer (coba ngram_range, max_features, min_df)
#   - kolom numerik -> StandardScaler (num_urls, has_ip_url, has_shortener, max_urls_len)
#   - gabungkan dengan ColumnTransformer
#   - bungkus semuanya dalam sklearn Pipeline bersama model
#   - pertanyaan: apakah has_ip_url / has_shortener perlu di-scale?
def build_pipeline():
    pass


# ============================================================
# TODO 4: Training baseline
# ============================================================
#   - mulai dari LogisticRegression (coba class_weight="balanced")
#   - fit di data train saja
def train(pipeline, X_train, y_train):
    pass


# ============================================================
# TODO 5: Evaluasi
# ============================================================
#   - classification_report, confusion_matrix
#   - metrik apa yang paling penting untuk deteksi phishing? accuracy? recall? precision? kenapa?
#   - cek: email phishing yang lolos (false negative) itu kayak apa? print beberapa contoh
def evaluate(pipeline, X_test, y_test):
    pass


# ============================================================
# TODO 6: Bandingkan model
# ============================================================
#   - coba LinearSVC, RandomForest, (opsional) XGBoost
#   - bandingkan pakai cross_val_score (cv=5, scoring="f1" atau "recall")
#   - opsional: GridSearchCV untuk tuning TF-IDF + model


# ============================================================
# TODO 7: Simpan model
# ============================================================
#   - joblib.dump(pipeline, "models/phishing_model.joblib")
#   - kenapa yang disimpan harus seluruh pipeline, bukan modelnya saja?
def save_model(pipeline, path: str):
    pass


# ============================================================
# TODO 8: Inference helper (nanti dipakai FastAPI)
# ============================================================
#   - load model sekali (jangan tiap request)
#   - input: subject, body -> bentuk DataFrame 1 baris dengan kolom yang sama seperti saat training
#     (hint: fitur URL & text harus dibuat dengan fungsi preprocessing yang SAMA)
#   - output: label + probabilitas (predict_proba)
def predict(subject: str, body: str) -> dict:
    pass


# ============================================================
# TODO 9: Integrasi ke API
# ============================================================
#   - app/schemas : request (subject, body, sender) & response (label, score)
#   - app/services: bungkus predict()
#   - app/api     : POST /predict
#   - daftarkan router di app/main.py


if __name__ == "__main__":
    df = load_data("training\dataset\CEAS_08.csv")
