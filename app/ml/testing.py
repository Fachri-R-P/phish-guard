import pandas as pd

# >>> Ganti baris ini saja untuk menguji modul lain <<<
from app.ml import preprocessing as pp

df = pd.read_csv("training/dataset/CEAS_08.csv")
sampel = df.sample(5, random_state=42)


def jalankan(nama, fn, *args):
    """Panggil fn(*args). Tampilkan hasil, atau errornya, tanpa menghentikan skrip."""
    try:
        return fn(*args)
    except Exception as e:
        return f"!! ERROR {type(e).__name__}: {e}"


def uji(nama, ambil_input):
    print("=" * 60)
    print(f"{nama}")
    print("=" * 60)
    fn = getattr(pp, nama, None)
    if fn is None:
        print("  (belum ada, dilewati)\n")
        return
    for _, row in sampel.iterrows():
        args = ambil_input(row)
        hasil = jalankan(nama, fn, *args)
        print("  input :", repr(str(args[0])[:80]))
        print("  hasil :", repr(hasil)[:200])
        print()


uji("clean_text",           lambda r: (r["body"],))
uji("extract_sender_domain", lambda r: (r["sender"],))
uji("extract_url_features",  lambda r: (r["body"],))
uji("build_text_field",      lambda r: (r,))

# preprocess_dataframe menerima DataFrame utuh
print("=" * 60)
print("preprocess_dataframe")
print("=" * 60)
if hasattr(pp, "preprocess_dataframe"):
    out = jalankan("preprocess_dataframe", pp.preprocess_dataframe, df.copy())
    if isinstance(out, pd.DataFrame):
        print("  baris sebelum/sesudah:", len(df), "->", len(out))
        print("  kolom :", list(out.columns))
        print("  label unik:", sorted(out["label"].unique()) if "label" in out else "-")
        print(out.head(3))
    else:
        print("  hasil :", out)
else:
    print("  (belum ada, dilewati)")
