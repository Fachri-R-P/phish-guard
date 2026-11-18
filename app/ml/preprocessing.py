import re, html
import pandas as pd

URL_PATTERN = r'(?:https?://|www\.)[^\s/$.?#].[^\s]*'
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
HTML_TAG_PATTERN = "<.*?>"
IP_URL_PATTERN = r"https?://\d{1,3}(?:\.\d{1,3}){3}(?:[:/]|$)"
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"}

def clean_text(text: str) -> str:
#   - handle None / NaN -> kembalikan ""
    if text is None or text != text:
        return ""
#   - html.unescape
    text = html.unescape(text)
#   - hapus tag HTML
    text = re.sub(HTML_TAG_PATTERN, " ", text)    
#   - ganti URL dengan token
    text = re.sub(URL_PATTERN, " <URL> ", text)
#   - ganti email dengan token " <EMAIL> "
    text = re.sub(EMAIL_PATTERN, " <EMAIL> ", text)
#   - lowercase
    text = text.lower()
#   - rapikan whitespace berlebih (spasi, \n, \t)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# TODO 4: 
def extract_sender_domain(sender: str) -> str:
#   - ambil alamat di dalam <...>, lalu ambil bagian setelah "@"
    if sender is None or sender != sender:
        return ""
    if "@" in sender:
        return sender.rsplit("@", 1)[-1].strip(" >").lower()
#   - kembalikan "" kalau format tidak valid
    return ""

# TODO 5: 
def extract_url_features(body: str) -> dict:
    if body is None or body != body:
        return {
            "num_urls": 0, 
            "has_ip_url": 0,
            "has_shortener": 0,
            "max_urls_len" : 0,
            }
    body = html.unescape(str(body))
#   - jumlah URL
    urls = re.findall(URL_PATTERN, body)
    num_urls = len(urls)
#   - ada URL berbasis IP atau tidak
    has_ip_url = int(re.search(IP_URL_PATTERN, body) is not None)
#   - ada link yang memakai shortener (bit.ly, dll) atau tidak
    shortener_regex = (
        r"(?:https?://|www\.)(?:www\.)?(?:"
        + "|".join(re.escape(s) for s in SHORTENERS)
        + r")(?:[/:?#]|$)"
        )
    has_shortener = int(any(re.match(shortener_regex, u, re.IGNORECASE) for u in urls))
#   - panjang URL terpanjang
    max_urls_len = max((len(u) for u in urls), default=0)
    return {
            "num_urls" : num_urls,
            "has_ip_url" : has_ip_url,
            "has_shortener" : has_shortener,
            "max_urls_len" : max_urls_len,
        }

# TODO 6: 
def build_text_field(row) -> str:
#   - gabungkan subject + body yang sudah dibersihkan jadi satu kolom teks
    subject = clean_text(row["subject"])
    body = clean_text(row["body"])

    return f"{subject} {body}".strip()

# TODO 7: 
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
#   - drop baris dengan body/subject kosong DAN duplikat
    df = df.dropna(subset=["subject","body"], how="all").reset_index(drop=True)
    df = df.drop_duplicates(subset=["subject", "body"], keep="first").reset_index(drop=True)
#   - pastikan label bertipe int dan hanya berisi 0/1
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)
#   - terapkan TODO 3-6 ke kolom yang sesuai
    # apply function extract_sender_domain
    if "sender" in df.columns:
        df["sender_domain"] = df["sender"].apply(extract_sender_domain)
    # apply function extract_url_features
    url_feats = pd.DataFrame(
        df["body"].apply(extract_url_features).to_list(),
        index=df.index
    )
    # apply build_text_field
    df["text"] = df.apply(build_text_field, axis=1)
#   - kembalikan DataFrame dengan kolom: text, fitur numerik, label\
    df = pd.concat([df, url_feats], axis=1)
    df = df[["text", *url_feats.columns, "label"]].reset_index(drop=True)

    return df