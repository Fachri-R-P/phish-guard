import joblib
import pandas as pd
from pathlib import Path
from app.ml.preprocessing import build_text_field, extract_url_features 

# ============================================================
# TODO 8: Inference helper (nanti dipakai FastAPI)
# ============================================================
#   - load model sekali (jangan tiap request)
#   - input: subject, body -> bentuk DataFrame 1 baris dengan kolom yang sama seperti saat training
#     (hint: fitur URL & text harus dibuat dengan fungsi preprocessing yang SAMA)
#   - output: label + probabilitas (predict_proba)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "phishing_model.joblib"
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(subject: str, body: str) -> dict:
    model = load_model()

    row = {
        "subject": subject or "",
        "body": body or "",
    }

    text_clean = build_text_field(row)

    features = {
        "text": text_clean,
        **extract_url_features(row["body"]),

    }
    df_input = pd.DataFrame([features])

    proba = model.predict_proba(df_input)
    score = float(proba[0][1])          # probabilitas kelas 1 (phishing)
    label = int(score >= 0.5)

    return {
        "label": label,
        "label_name": "phishing" if label == 1 else "aman",
        "score": round(score, 4),
    }