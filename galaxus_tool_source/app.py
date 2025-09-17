# app.py — Galaxus Sellout Analyse (immer Name+Farbe, verbessertes Matching: Albert/Theo/Finn/Finn mobile, Woche-Snapping)

import os
import io
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import Mapping
import warnings

# =========================
# Globale Settings
# =========================
pd.options.mode.use_inf_as_na = True
warnings.filterwarnings("ignore", message="overflow encountered in multiply")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered")
np.seterr(all="ignore")

st.set_page_config(page_title="Galaxus Sellout Analyse", layout="wide")
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

PERSIST_UPLOADS = True  # Uploads in /data persistieren

# =========================
# 🔐 Authentication (Passcode)
# =========================
def _to_plain_mapping(obj) -> dict:
    if obj is None: return {}
    if isinstance(obj, Mapping):
        try: return dict(obj)
        except Exception: pass
    try: return {k: obj[k] for k in obj.keys()}
    except Exception: return {}

def _auth_cfg() -> dict:
    try:
        raw = st.secrets.get("auth", {})
    except Exception:
        raw = {}
    return _to_plain_mapping(raw)

def auth_enabled() -> bool:
    return bool(_auth_cfg().get("require_login", True))

def _get_passcode():
    # 1) Query-Param
    try:
        qp = st.query_params
        if "code" in qp and str(qp["code"]).strip():
            return str(qp["code"]).strip()
    except Exception:
        pass
    # 2) Secrets [auth]
    auth = _auth_cfg()
    aliases = ("code","password","passcode","pw","passwort","secret")
    for k in aliases:
        v = auth.get(k)
        if isinstance(v, (str,int)) and str(v).strip():
            return str(v).strip()
    # 3) Root-Secrets
    try:
        root = _to_plain_mapping(st.secrets)
        for k in aliases:
            v = root.get(k)
            if isinstance(v, (str,int)) and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    # 4) Environment
    for k in ("AUTH_CODE","AUTH_PASSWORD","AUTH_PASSCODE","STREAMLIT_AUTH_CODE"):
        v = os.environ.get(k)
        if isinstance(v, (str,int)) and str(v).strip():
            return str(v).strip()
    return None

def _login_view():
    st.title("🔐 Zugang")
    with st.form("login-passcode", clear_on_submit=False):
        code = st.text_input("Code / Passwort", type="password")
        ok = st.form_submit_button("Anmelden")
    expected = _get_passcode()
    if expected is None:
        st.error("Kein Passcode in den Secrets/Env gefunden.")
        return
    if ok:
        if code.strip() == expected:
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = "passcode"
            st.session_state["auth_ts"] = datetime.utcnow().isoformat()
            st.success("Erfolgreich angemeldet.")
            st.rerun()
        else:
            st.error("Ungültiger Code.")

def ensure_auth() -> bool:
    if not auth_enabled(): return True
    if not st.session_state.get("auth_ok"):
        _login_view()
        return False
    return True

def logout_button():
    with st.sidebar:
        if st.button("Logout"):
            for k in ("auth_ok","auth_user","auth_ts"):
                st.session_state.pop(k, None)
            st.rerun()

if not ensure_auth():
    st.stop()
logout_button()

# =========================
# Anzeige-Helfer
# =========================
THOUSANDS_SEP = "'"
NUM_COLS_DEFAULT = [
    "Einkaufsmenge","Einkaufswert (CHF)",
    "Verkaufsmenge","Verkaufswert (CHF)",
    "Lagermenge","Lagerwert (CHF)"
]

def _fmt_thousands(x, sep=THOUSANDS_SEP):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}".replace(",", sep)
    except Exception: return str(x)

def style_numeric(df: pd.DataFrame, num_cols=NUM_COLS_DEFAULT, sep=THOUSANDS_SEP):
    out = df.copy()
    present = [c for c in num_cols if c in out.columns]
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    fmt = {c: (lambda v, s=sep: _fmt_thousands(v, s)) for c in present}
    return out, out.style.format(fmt)

def append_total_row_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    cols = list(df.columns)
    num_targets = ["Einkaufsmenge","Einkaufswert (CHF)","Verkaufsmenge","Verkaufswert (CHF)","Lagermenge","Lagerwert (CHF)"]
    num_cols = [c for c in num_targets if c in cols]
    label_col = next((c for c in ["Bezeichnung_anzeige","Bezeichnung","ArtikelNr","Kategorie"] if c in cols), cols[0])
    total_row = {c: "" for c in cols}
    total_row[label_col] = "Gesamt"
    for c in num_cols:
        total_row[c] = pd.to_numeric(df[c], errors="coerce").sum()
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# =========================
# Excel robust einlesen
# =========================
def read_excel_flat(upload) -> pd.DataFrame:
    raw = pd.read_excel(upload, header=None, dtype=object)
    if raw.empty: return pd.DataFrame()
    header_idx = int(raw.notna().mean(axis=1).idxmax())
    headers = raw.iloc[header_idx].fillna("").astype(str).tolist()
    headers = [re.sub(r"\s+"," ", h).strip() for h in headers]
    n = raw.shape[1]
    headers = headers[:n] + [f"col_{i}" for i in range(len(headers), n)]
    df = raw.iloc[header_idx+1:].reset_index(drop=True)
    df.columns = headers[:n]
    # doppelte Spalten entschärfen
    seen, newcols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1; newcols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0; newcols.append(c)
    df.columns = newcols
    return df

# =========================
# Utilities
# =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .map(lambda c: unicodedata.normalize("NFKC", str(c)))
        .map(lambda c: re.sub(r"\s+"," ", c).strip())
    )
    return df

def normalize_key(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    return re.sub(r"[^a-z0-9]+","", s)

def _normalize_artnr_sell(x: str) -> str:
    """Hersteller-Nr. aus Sell-Out auf 'ArtikelNr_key' trimmen (ST-, Leer/Trenner etc. weg)."""
    s = str(x or "")
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"^\s*st[-_/ ]+", "", s, flags=re.I)          # ST- Präfix entfernen
    s = re.sub(r"[^A-Za-z0-9]+", "", s).upper()               # nur A-Z0-9, upper
    return s

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True):
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols: return cand
    canon = {re.sub(r"[\s\-_/\.]+","", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-_/\.]+","", cand).lower()
        if key in canon: return canon[key]
    if required:
        raise KeyError(f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}")
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i","u","f"): return s
    def _clean(x):
        if pd.isna(x): return np.nan
        x = str(x).strip().replace("’","").replace("'","").replace(" ","").replace(",",".")
        if x.count(".") > 1:
            parts = x.split("."); x = "".join(parts[:-1]) + "." + parts[-1]
        try: return float(x)
        except Exception: return np.nan
    return s.map(_clean)

def parse_date_series_us(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64): return s
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    nums = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(nums, origin="1899-12-30", unit="d", errors="coerce")
    return dt1.combine_first(dt2)

MAX_QTY, MAX_PRICE = 1_000_000, 1_000_000

def sanitize_numbers(qty: pd.Series, price: pd.Series):
    q = pd.to_numeric(qty, errors="coerce").astype("float64").clip(lower=0, upper=MAX_QTY)
    p = pd.to_numeric(price, errors="coerce").astype("float64").clip(lower=0, upper=MAX_PRICE)
    return q, p

def safe_mul(a: pd.Series, b: pd.Series, max_a=MAX_QTY, max_b=MAX_PRICE) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce").astype("float64")
    b = pd.to_numeric(b, errors="coerce").astype("float64")
    a_vals = np.nan_to_num(a.to_numpy(), nan=0.0, posinf=max_a, neginf=0.0)
    b_vals = np.nan_to_num(b.to_numpy(), nan=0.0, posinf=max_b, neginf=0.0)
    a_vals = np.clip(a_vals, 0.0, max_a)
    b_vals = np.clip(b_vals, 0.0, max_b)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        out = a_vals * b_vals
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float64")
    return pd.Series(out, index=a.index)

# =========================
# Farben & Familie & Varianten
# =========================
_COLOR_MAP = {
    "weiss":"Weiss","weiß":"Weiss","white":"White","offwhite":"Off-White","cream":"Cream","ivory":"Ivory",
    "schwarz":"Schwarz","black":"Black","grau":"Grau","gray":"Grau","anthrazit":"Anthrazit","charcoal":"Anthrazit","graphite":"Graphit","silver":"Silber",
    "blau":"Blau","blue":"Blau","navy":"Dunkelblau","light blue":"Hellblau","dark blue":"Dunkelblau","sky blue":"Hellblau",
    "rot":"Rot","red":"Rot","bordeaux":"Bordeaux","burgundy":"Bordeaux","pink":"Pink","magenta":"Magenta",
    "lila":"Lila","violett":"Violett","purple":"Violett","fuchsia":"Fuchsia",
    "grün":"Grün","gruen":"Grün","green":"Grün","mint":"Mint","türkis":"Türkis","tuerkis":"Türkis","turquoise":"Türkis",
    "petrol":"Petrol","olive":"Olivgrün","gelb":"Gelb","yellow":"Gelb","orange":"Orange","braun":"Braun","brown":"Braun","beige":"Beige","sand":"Sand",
    "gold":"Gold","rose gold":"Roségold","rosegold":"Roségold","kupfer":"Kupfer","copper":"Kupfer","bronze":"Bronze","transparent":"Transparent","clear":"Transparent",
}
_COLOR_WORDS = set(_COLOR_MAP.keys()) | set(map(str.lower, _COLOR_MAP.values()))
_STOP_TOKENS = {"eu","ch","us","uk","mobile","little","bundle","set","kit","pro","big","mini"}

def _looks_like_not_a_color(token: str) -> bool:
    t = (token or "").strip().lower()
    return (not t) or (t in {"eu","ch","us","uk"}) or any(x in t for x in ["ml","db","m²","m2"]) or bool(re.search(r"\d", t))

def _strip_parens_units(name: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", name)
    s = re.sub(r"\b\d+([.,]\d+)?\s*(ml|db|m²|m2)\b", " ", s, flags=re.I)
    return s

def make_family_key(name: str) -> str:
    if not isinstance(name, str): return ""
    s = _strip_parens_units(name.lower())
    s = re.sub(r"\b[o0]-\d+\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t and (t not in _STOP_TOKENS) and (t not in _COLOR_WORDS)]
    return "".join(toks[:2]) if toks else ""

def extract_color_from_name(name: str) -> str:
    if not isinstance(name, str): return ""
    m = re.search(r"\(([^)]+)\)$", name.strip())
    if not m: m = re.search(r"[-–—]\s*([A-Za-z äöüÄÖÜß]+)$", name.strip())
    if not m: m = re.search(r"/\s*([A-Za-z äöüÄÖÜß]+)$", name.strip())
    if m:
        cand = m.group(1).strip().lower()
        if not _looks_like_not_a_color(cand):
            return _COLOR_MAP.get(cand, cand.title())
    for w in sorted(_COLOR_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", name, flags=re.I):
            if not _looks_like_not_a_color(w):
                return _COLOR_MAP.get(w, w.title())
    return ""

# Anzeige-Basis: Name ohne Kategorie-/EU-Suffixe, Varianten (little/big/pro/mobile) links bleiben erhalten
_CATEGORY_TOKENS = {
    "hygrometer","aroma diffuser","diffuser","ventilator","tischventilator","luftreiniger",
    "luftbefeuchter","verdunster","vernebler","luftentfeuchter","reiniger","aroma","tisch ventilator",
    "humidifier","dehumidifier","air purifier","purifier","aroma diffuser"
}
_EU_TOKENS = {"eu","ch/eu","ch","us","uk"}

def to_base_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name).strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—").strip()
    parts = re.split(r"\s+[–—-]\s+", s)
    if len(parts) >= 2:
        left, right = parts[0].strip(), " ".join(parts[1:]).strip().lower()
        if right.split() and right.split()[0] in _EU_TOKENS:
            return left.strip()
        clean_right = re.sub(r"[^\w\s/]+", " ", right).strip()
        words = [w for w in clean_right.split() if w]
        if words and all((w in _EU_TOKENS) or (w in _CATEGORY_TOKENS) for w in words):
            return left.strip()
        s = (left + " – " + " ".join(parts[1:])).strip()
    else:
        s = parts[0].strip()
    s = re.sub(rf"\b({'|'.join(map(re.escape,_EU_TOKENS))})\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" -–—").strip()
    return s

def _as_color_or_empty(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    t = text.strip()
    if not t:
        return ""
    low = t.lower()
    if _looks_like_not_a_color(low):
        return ""
    if low in _COLOR_MAP:
        return _COLOR_MAP[low]
    for w in sorted(_COLOR_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", low):
            return _COLOR_MAP.get(w, w.title())
    return ""

# =========================
# Parsing – Preislisten
# =========================
PRICE_COL_CANDIDATES = ["Preis","VK","Netto","NETTO","Einkaufspreis","Verkaufspreis","NETTO NETTO","Einkauf"]
BUY_PRICE_CANDIDATES  = ["Einkaufspreis","Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis","VK","Preis"]

ARTNR_CANDIDATES = ["Artikelnummer","Artikelnr","ArtikelNr","Artikel-Nr.","Hersteller-Nr.",
                    "Produkt ID","ProdNr","ArtNr","ArtikelNr.","Artikel"]
EAN_CANDIDATES  = ["EAN","GTIN","BarCode","Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung","Produktname","Name","Titel","Artikelname"]

CAT_CANDIDATES  = ["Kategorie","Warengruppe"]
VARIANT_CANDIDATES = ["Zusatz","Variante","Variant","Scent","Duft","Flavor","Flavour","Subname","Sub-Name"]

STOCK_CANDIDATES= ["Bestand","Verfügbar","verfügbar","Verfuegbar","Lagerbestand","Lagermenge","Available"]
COLOR_CANDIDATES= ["Farbe","Color","Colour","Farbvariante","Farbname"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES,  "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat   = find_column(df, CAT_CANDIDATES, "Kategorie", required=False)

    col_stock = find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_color = find_column(df, COLOR_CANDIDATES, "Farbe/Variante", required=False)
    col_variant = find_column(df, VARIANT_CANDIDATES, "Variante/Zusatz", required=False)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str)
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(lambda s: re.sub(r"[^A-Za-z0-9]+","", str(s)).upper())
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","",str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Familie"]         = out["Bezeichnung"].map(make_family_key)
    out["Kategorie"]       = (df[col_cat].astype(str).str.strip() if col_cat else "")

    # Farbe
    if col_color:
        out["Farbe"] = df[col_color].astype(str).map(lambda v: _COLOR_MAP.get(str(v).lower(), str(v)))
    else:
        out["Farbe"] = out["Bezeichnung"].map(extract_color_from_name)

    # Variante ggf. als Farbhilfe
    if col_variant:
        var_series = df[col_variant].astype(str)
        var_color = var_series.map(_as_color_or_empty)
        out["Farbe"] = np.where(out["Farbe"].fillna("").str.strip()=="", var_color, out["Farbe"])

    out["Farbe"] = out["Farbe"].fillna("").astype(str).str.strip()

    out["Lagermenge"] = parse_number_series(df[col_stock]).fillna(0).astype("Int64") if col_stock else pd.Series([0]*len(out), dtype="Int64")
    if col_buy:  out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell: out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if "Einkaufspreis" not in out: out["Einkaufspreis"]=out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out: out["Verkaufspreis"]=out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))

    # Dedupliziere nach ArtikelNr_key (bevorzuge Zeilen mit Preis)
    out = out.assign(_have=out["Verkaufspreis"].notna()).sort_values(["ArtikelNr_key","_have"], ascending=[True,False])
    out = out.drop_duplicates(subset=["ArtikelNr_key"], keep="first").drop(columns=["_have"])
    return out

# =========================
# Parsing – Sell-out (+ Hints)
# =========================
NAME_CANDIDATES_SO   = ["Bezeichnung","Name","Artikelname","Bezeichnung_Sales","Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty","Verkauf","Verkaufte Menge","Menge verkauft","Absatz","Stück","Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf","Einkaufsmenge","Menge Einkauf"]
DATE_START_CANDS     = ["Start","Startdatum","Start Date","Anfangs datum","Anfangsdatum","Von","Period Start"]
DATE_END_CANDS       = ["Ende","Enddatum","End Date","Bis","Period End"]
STOCK_SO_CANDIDATES  = ["Lagermenge","Lagerbestand","Bestand","Verfügbar","verfügbar","Verfuegbar","Available"]
ARTNR_SELL_CANDS     = ["Hersteller-Nr.","Hersteller Nr","HerstellerNr","ArtNr","ArtikelNr","Artikelnummer"]

def _apply_hints_to_row(name_raw: str) -> dict:
    s = (name_raw or "").lower()
    h = {"hint_family":"","hint_color":"","hint_variant":"","hint_art_exact":"","hint_art_prefix":""}
    # Familien
    for fam in ["finn mobile","finn","theo","albert","albert little","robert","peter","julia","roger","mia","simon","otto","oskar","eva little","eva","tim","charly","duftöl","duftoel","duft oil"]:
        if fam in s:
            h["hint_family"] = ("finn" if "finn" in fam else fam).replace("  "," ").strip()
    # Varianten
    if "little" in s: h["hint_variant"]="little"
    if "mobile" in s: h["hint_variant"]="mobile"
    if "pro" in s and not h["hint_variant"]: h["hint_variant"]="pro"
    if "big" in s and not h["hint_variant"]: h["hint_variant"]="big"
    # Farb-/Artikel-Hints
    if "simon" in s: h["hint_art_exact"]="s054"
    if "otto"  in s: h["hint_art_prefix"]="o013"
    if "oskar" in s and "little" in s: h["hint_art_prefix"]="o061"
    return h

def _fallback_col_by_index(df: pd.DataFrame, idx0: int):
    try: return df.columns[idx0]
    except Exception: return None

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art_sell = find_column(df, ARTNR_SELL_CANDS,   "Hersteller-/Artikelnummer (Sell-Out)", required=False)
    col_art   = find_column(df, ARTNR_CANDIDATES,   "Artikelnummer", required=False)  # falls vorhanden
    col_ean   = find_column(df, EAN_CANDIDATES,     "EAN/GTIN",      required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung",   required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)
    col_stock_so = find_column(df, STOCK_SO_CANDIDATES, "Lagermenge (Sell-out)", required=False)

    col_start = find_column(df, DATE_START_CANDS, "Startdatum", required=False)
    col_end   = find_column(df, DATE_END_CANDS,   "Enddatum",   required=False)
    if not col_start and df.shape[1]>=9:  col_start=_fallback_col_by_index(df,8)
    if not col_end   and df.shape[1]>=10: col_end  =_fallback_col_by_index(df,9)

    out = pd.DataFrame()
    # Hersteller-Nr. aus Sell-Out bevorzugt normalisieren für Key
    if col_art_sell:
        sell_art_norm = df[col_art_sell].map(_normalize_artnr_sell)
        out["ArtikelNr"] = df[col_art_sell].astype(str)
        out["ArtikelNr_key"] = sell_art_norm
    elif col_art:
        out["ArtikelNr"] = df[col_art].astype(str)
        out["ArtikelNr_key"] = out["ArtikelNr"].map(lambda s: re.sub(r"[^A-Za-z0-9]+","", str(s)).upper())
    else:
        out["ArtikelNr"] = ""
        out["ArtikelNr_key"] = ""

    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","",str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Familie"]         = out["Bezeichnung"].map(make_family_key)

    hints = out["Bezeichnung"].map(_apply_hints_to_row)
    out["Hint_Family"]   = hints.map(lambda h: h["hint_family"])
    out["Hint_Color"]    = hints.map(lambda h: h["hint_color"])
    out["Hint_Variant"]  = hints.map(lambda h: h["hint_variant"])
    out["Hint_ArtExact"] = hints.map(lambda h: h["hint_art_exact"])
    out["Hint_ArtPref"]  = hints.map(lambda h: h["hint_art_prefix"])

    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"] = parse_number_series(df[col_buy]).fillna(0).astype("Int64") if col_buy else pd.Series([0]*len(df), dtype="Int64")

    if col_stock_so:
        out["SellLagermenge"] = pd.to_numeric(df[col_stock_so], errors="coerce")

    if col_start: out["StartDatum"] = parse_date_series_us(df[col_start])
    if col_end:   out["EndDatum"]   = parse_date_series_us(df[col_end])
    if "StartDatum" in out and "EndDatum" in out:
        out.loc[out["EndDatum"].isna(),"EndDatum"] = out.loc[out["EndDatum"].isna(),"StartDatum"]
    return out

# =========================
# Matching-Backstops (familien-/variantenbewusst)
# =========================
def _assign_from_price_row(merged: pd.DataFrame, i, row: pd.Series):
    for col in ["Einkaufspreis","Verkaufspreis","Lagermenge","Kategorie","ArtikelNr","ArtikelNr_key","Bezeichnung","Farbe"]:
        if col in row:
            merged.at[i, col] = row.get(col, merged.at[i, col])

def _token_set(s: str) -> set:
    s = _strip_parens_units(s.lower())
    s = re.sub(r"[^a-z0-9]+"," ", s)
    toks = [t for t in s.split() if t and (t not in _STOP_TOKENS) and (t not in _COLOR_WORDS)]
    return set(toks)

def _best_fuzzy_in_candidates(name: str, cand_series: pd.Series):
    base = _token_set(name)
    if not len(base): return None
    best_idx, best_score = None, 0.0
    for idx, val in cand_series.items():
        cand = _token_set(str(val))
        if not cand: continue
        inter = len(base & cand); union = len(base | cand)
        score = inter/union if union else 0.0
        # Bonus, wenn "ch" vorkommt
        if re.search(r"\bch\b", str(val), flags=re.I):
            score += 0.05
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx if best_score >= 0.5 else None

def _family_variant_filter(price_df: pd.DataFrame, family: str, variant: str):
    """Filtert Preislisten-Kandidaten nach Familie + Variante-Präferenz (little/mobile/...)."""
    if not family:
        return price_df

    grp = price_df.loc[price_df["Familie"] == family]
    if grp.empty:
        grp = price_df.loc[price_df["Familie"].str.contains(re.escape(family), na=False)]
    if grp.empty:
        return price_df.iloc[0:0]

    # Variante: 'little' vs. 'mobile' vs. andere – sicherstellen, dass es passt
    if variant:
        mask_has = grp["Bezeichnung"].str.contains(variant, case=False, na=False)
        grp_var = grp.loc[mask_has]
        if not grp_var.empty:
            grp = grp_var
        else:
            # wenn klare Negativ-Variante existiert, entferne sie
            neg = grp.loc[~grp["Bezeichnung"].str.contains(variant, case=False, na=False)]
            if not neg.empty:
                grp = neg

    # Spezifische Regeln
    # - Finn mobile → nur mobile
    if re.search(r"\bfinn\b", family, flags=re.I) and variant == "mobile":
        grp2 = grp.loc[grp["Bezeichnung"].str.contains("mobile", case=False, na=False)]
        if not grp2.empty:
            grp = grp2
    # - Finn (ohne mobile) → vermeide mobile
    if re.search(r"\bfinn\b", family, flags=re.I) and (variant == "" or variant is None):
        grp2 = grp.loc[~grp["Bezeichnung"].str.contains("mobile", case=False, na=False)]
        if not grp2.empty:
            grp = grp2
    # - Albert little → nur little
    if re.search(r"\balbert\b", family, flags=re.I) and variant == "little":
        grp2 = grp.loc[grp["Bezeichnung"].str.contains("little", case=False, na=False)]
        if not grp2.empty:
            grp = grp2
    # - Albert (ohne little) → vermeide little
    if re.search(r"\balbert\b", family, flags=re.I) and (variant == "" or variant is None):
        grp2 = grp.loc[~grp["Bezeichnung"].str.contains("little", case=False, na=False)]
        if not grp2.empty:
            grp = grp2

    # - Theo: bevorzugt ArtikelNr die mit 'T' beginnen
    grp_sorted = grp.copy()
    if "ArtikelNr" in grp.columns:
        grp_sorted["_tprio"] = grp["ArtikelNr"].astype(str).str.upper().str.startswith("T").astype(int)
        grp_sorted = grp_sorted.sort_values(["_tprio"], ascending=False).drop(columns=["_tprio"], errors="ignore")

    return grp_sorted


def _apply_equivalences(hint_art_exact: str, hint_art_pref: str):
    if hint_art_exact:
        return hint_art_exact.lower()
    if hint_art_pref:
        return hint_art_pref.lower()
    return None

def _final_backstops(merged: pd.DataFrame, price_df: pd.DataFrame):
    need = merged["Verkaufspreis"].isna()
    if not need.any(): return
    for i in merged.index[need]:
        fam = str(merged.at[i,"Hint_Family"] or merged.at[i,"Familie"] or "").strip()
        var = str(merged.at[i,"Hint_Variant"] or "").strip()
        pref_color = str(merged.at[i,"Hint_Color"] or "")
        # 1) Familie + Variante eng filtern
        grp = _family_variant_filter(price_df, fam, var)
        if not grp.empty:
            # Farbe bevorzugen
            if pref_color:
                g2 = grp.loc[grp["Farbe"].str.lower()==pref_color.lower()]
                if not g2.empty: grp = g2
            _assign_from_price_row(merged, i, grp.iloc[0])
            continue
        # 2) Fuzzy auf Bezeichnung
        idx = _best_fuzzy_in_candidates(str(merged.at[i,"Bezeichnung"]), price_df["Bezeichnung"])
        if idx is not None:
            _assign_from_price_row(merged,i, price_df.loc[idx]); continue

# =========================
# Merge & Werte (+ Chart-Quelle)
# =========================
@st.cache_data(show_spinner=False)
def enrich_and_merge(filtered_sell_df: pd.DataFrame, price_df: pd.DataFrame, latest_stock_baseline_df: pd.DataFrame|None=None):
    if filtered_sell_df is None or price_df is None or filtered_sell_df.empty or price_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sell_for_stock = latest_stock_baseline_df if latest_stock_baseline_df is not None else filtered_sell_df

    # Merge via ArtikelNr_key (Sell-Out ist bereits normalisiert), rechtsseitig PL-Spalten suffixed
    merged = filtered_sell_df.merge(price_df, on=["ArtikelNr_key"], how="left", suffixes=("", "_pl"))
    stock_merged = sell_for_stock.merge(price_df, on=["ArtikelNr_key"], how="left", suffixes=("", "_pl"))

    # Hilfsdatum
    def _row_date(df):
        if ("EndDatum" in df.columns) and ("StartDatum" in df.columns):
            d = df["EndDatum"].fillna(df["StartDatum"])
        elif "StartDatum" in df.columns:
            d = df["StartDatum"]
        elif "EndDatum" in df.columns:
            d = df["EndDatum"]
        else:
            d = pd.to_datetime(pd.NaT)
        return pd.to_datetime(d, errors="coerce")

    merged["_rowdate"] = _row_date(merged)
    stock_merged["_rowdate"] = _row_date(stock_merged)

    # Fallback: EAN
    need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if need.any():
        tmp = merged.loc[need, ["EAN_key"]].merge(
            price_df[["EAN_key","Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung",
                      "Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]],
            on="EAN_key", how="left"
        )
        idx = merged.index[need]; tmp.index = idx
        for c in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]:
            merged.loc[idx,c] = merged.loc[idx,c].fillna(tmp[c])

    # Fallback: Bezeichnung_key
    need = merged["Verkaufspreis"].isna()
    if need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        for i,k in zip(merged.index[need], merged.loc[need,"Bezeichnung_key"]):
            if k in name_map.index:
                _assign_from_price_row(merged,i, name_map.loc[k])

    # Familien-/Variantenbewusste Backstops
    _final_backstops(merged, price_df)

    # Interne Artikelnummer/Kategorie/Bezeichnung/Farbe aus Preisliste bevorzugen (Anzeige/Key)
    for col_pl in ["ArtikelNr_pl","Kategorie_pl","Bezeichnung_pl","Farbe_pl","ArtikelNr_key_pl"]:
        if col_pl in merged.columns:
            pass
    merged["ArtikelNr"] = merged["ArtikelNr"].combine_first(merged.get("ArtikelNr_pl"))
    merged["ArtikelNr_key"] = merged["ArtikelNr_key"].combine_first(merged.get("ArtikelNr_key_pl"))
    merged["Kategorie"] = merged.get("Kategorie_pl", merged.get("Kategorie","")).fillna(merged.get("Kategorie",""))
    merged["Farbe"] = merged.get("Farbe_pl", merged.get("Farbe","")).fillna(merged.get("Farbe",""))
    merged["Bezeichnung"] = merged.get("Bezeichnung_pl", merged.get("Bezeichnung","")).fillna(merged.get("Bezeichnung",""))

    # Strings
    for df in (merged, stock_merged):
        df["Kategorie"] = df.get("Kategorie","").fillna("").astype(str).str.strip()
        df["Bezeichnung"] = df.get("Bezeichnung","").fillna("").astype(str)
        df["Farbe"] = df.get("Farbe","").fillna("").astype(str).str.strip()

    # Farbe ergänzen, falls leer -> aus Name extrahieren
    mask_empty_color = merged["Farbe"].eq("")
    if mask_empty_color.any():
        merged.loc[mask_empty_color, "Farbe"] = (
            merged.loc[mask_empty_color, "Bezeichnung"]
                  .map(extract_color_from_name)
                  .fillna("")
                  .astype(str)
                  .str.strip()
        )
    merged["Farbe_std"] = merged["Farbe"].map(_as_color_or_empty)

    # Anzeige-Bezeichnung: IMMER BaseName + ( – Farbe ) wenn Farbe erkannt (auch bei Einzelnennungen)
    merged["BaseName"] = merged["Bezeichnung"].map(to_base_name)
    merged["Bezeichnung_anzeige"] = merged["BaseName"]
    has_color = merged["Farbe_std"].ne("")
    merged.loc[has_color, "Bezeichnung_anzeige"] = merged.loc[has_color, "BaseName"] + " – " + merged.loc[has_color, "Farbe_std"]

    # Werte
    q_buy,p_buy   = sanitize_numbers(merged["Einkaufsmenge"], merged["Einkaufspreis"])
    q_sell,p_sell = sanitize_numbers(merged["Verkaufsmenge"], merged["Verkaufspreis"])
    merged["Einkaufswert"] = safe_mul(q_buy.fillna(0.0),  p_buy.fillna(0.0))
    merged["Verkaufswert"] = safe_mul(q_sell.fillna(0.0), p_sell.fillna(0.0))

    # Lager (jüngster Stand aus Sell-Out-Basis)
    stock_merged = stock_merged.copy()
    if "SellLagermenge" in stock_merged.columns:
        stock_valid = stock_merged.loc[stock_merged["SellLagermenge"].notna()].copy()
        stock_valid["SellLagermenge"] = (
            pd.to_numeric(stock_valid["SellLagermenge"], errors="coerce")
              .clip(lower=0, upper=MAX_QTY)
              .astype("float64")
        )
    else:
        stock_valid = stock_merged.iloc[0:0].copy()
        stock_valid["SellLagermenge"] = np.nan

    def _mk_grpkey(df):
        a = df.get("ArtikelNr_key","").astype(str).fillna("")
        e = df.get("EAN_key","").astype(str).fillna("")
        use_art = a.str.len() > 0
        return np.where(use_art, "A:" + a, "E:" + e)

    stock_valid["_grpkey"] = _mk_grpkey(stock_valid)
    merged["_grpkey"]      = _mk_grpkey(merged)

    period_min = pd.to_datetime(merged["_rowdate"]).min()
    period_max = pd.to_datetime(merged["_rowdate"]).max()

    sv_in = stock_valid
    if pd.notna(period_min) and pd.notna(period_max):
        sv_in = stock_valid.loc[(stock_valid["_rowdate"]>=period_min) & (stock_valid["_rowdate"]<=period_max)]
    if sv_in.empty and pd.notna(period_max):
        sv_in = stock_valid.loc[(stock_valid["_rowdate"]<=period_max)]

    if sv_in.empty:
        latest_qty_map = {}
    else:
        sv_in = sv_in.sort_values(["_grpkey","_rowdate"], ascending=[True, True])
        last_rows = sv_in.groupby("_grpkey", as_index=False).tail(1)
        latest_qty_map = last_rows.set_index("_grpkey")["SellLagermenge"].to_dict()

    price_map = (
        pd.to_numeric(price_df.drop_duplicates("ArtikelNr_key").set_index("ArtikelNr_key")["Verkaufspreis"], errors="coerce")
          .fillna(0.0).clip(lower=0, upper=MAX_PRICE).to_dict()
    )

    merged["Lagermenge_latest"] = (
        pd.to_numeric(merged["_grpkey"].map(latest_qty_map), errors="coerce")
          .fillna(0.0).clip(lower=0, upper=MAX_QTY).astype("float64")
    )
    merged["Verkaufspreis_latest"] = (
        pd.to_numeric(merged["ArtikelNr_key"].map(price_map), errors="coerce")
          .fillna(pd.to_numeric(merged.get("Verkaufspreis", np.nan), errors="coerce"))
          .fillna(0.0).clip(lower=0, upper=MAX_PRICE).astype("float64")
    )
    merged["Lagerwert_latest"] = safe_mul(merged["Lagermenge_latest"], merged["Verkaufspreis_latest"])

    # Detailtabelle
    detail = merged[[
        "ArtikelNr_key","ArtikelNr","Bezeichnung_anzeige","Kategorie",
        "Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert"
    ]].copy()
    detail["Lagermenge"] = merged["Lagermenge_latest"]
    detail["Lagerwert"]  = merged["Lagerwert_latest"]

    def _mode_nonempty(s: pd.Series) -> str:
        s = s.dropna().astype(str).str.strip()
        if s.empty: return ""
        try: return s.mode().iloc[0]
        except Exception: return s.iloc[0] if len(s) else ""

    totals = (
        detail.groupby("ArtikelNr_key", as_index=False)
              .agg({
                  "ArtikelNr": _mode_nonempty,
                  "Bezeichnung_anzeige": _mode_nonempty,
                  "Kategorie": _mode_nonempty,
                  "Einkaufsmenge":"sum",
                  "Einkaufswert":"sum",
                  "Verkaufsmenge":"sum",
                  "Verkaufswert":"sum",
                  "Lagermenge":"max",
                  "Lagerwert":"max"
              })
    )
    totals = totals.drop(columns=["ArtikelNr_key"])

    # Daten fürs Wochen-Chart
    ts_source = pd.DataFrame()
    if "StartDatum" in merged.columns:
        ts_source = merged[["StartDatum","Kategorie","Verkaufswert"]].copy()
        ts_source["Kategorie"] = ts_source["Kategorie"].fillna("").astype(str).str.strip()
        ts_source = ts_source[ts_source["Kategorie"] != ""]
        ts_source.rename(columns={"Verkaufswert":"Verkaufswert (CHF)"}, inplace=True)

    merged.drop(columns=["_rowdate","_grpkey"], errors="ignore", inplace=True)
    return detail, totals, ts_source

# =========================
# Daten / Persistenz / Auto-Erkennung
# =========================
BASE_DIR = Path(__file__).resolve().parent

def _find_data_dir() -> Path:
    candidates = [BASE_DIR / "data", BASE_DIR.parent / "data"]
    for p in candidates:
        if p.exists(): return p
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]

DATA_DIR = _find_data_dir()
DEFAULT_SELL_PATH  = DATA_DIR / "sellout.xlsx"
DEFAULT_PRICE_PATH = DATA_DIR / "preisliste.xlsx"
st.caption(f"📁 Datenordner: {DATA_DIR}")

def _persist_upload(uploaded_file, target_path: Path):
    if not PERSIST_UPLOADS or uploaded_file is None: return
    try:
        content = uploaded_file.getvalue()
    except Exception:
        content = uploaded_file.read()
    with open(target_path, "wb") as f:
        f.write(content)

def _guess_role_from_name(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ["sell-out","sellout","sell","sales","report"]): return "sell"
    if any(k in n for k in ["preisliste","preis","price","vk","pl ", "pl_", "pl-","rohdaten"]): return "price"
    return None

def _canon(c: str) -> str:
    return re.sub(r"[\s\-_/\.]+","", str(c)).lower()

def _classify_df(df: pd.DataFrame) -> str | None:
    if df is None or df.empty: return None
    canon = {_canon(c) for c in df.columns}
    sell_cands  = {"salesqty","verkauf","verkauftemenge","mengeverkauft","absatz","stück","stuck","menge"}
    price_cands = {"nettonetto","verkaufspreis","einkaufspreis","preis","vk","netto","bestand","kategorie"}
    if canon & sell_cands: return "sell"
    if canon & price_cands: return "price"
    if any(k in canon for k in {"nettonetto","preis","verkaufspreis"}) and not (canon & sell_cands):
        return "price"
    return None

def _maybe_swap_roles(rs: pd.DataFrame | None, rp: pd.DataFrame | None, rs_name: str | None, rp_name: str | None):
    role_s = _classify_df(rs) if rs is not None else None
    role_p = _classify_df(rp) if rp is not None else None
    if rs is not None and rp is None and role_s == "price": return None, rs, None, rs_name
    if rp is not None and rs is None and role_p == "sell": return rp, None, rp_name, None
    if rs is not None and rp is not None and role_s == "price" and role_p == "sell":
        return rp, rs, rp_name, rs_name
    return rs, rp, rs_name, rp_name

def _pick_default_files_from_dir(folder: Path):
    sell_bytes = price_bytes = None
    sell_name = price_name = None
    xlsx_files = sorted([p for p in folder.glob("*.xlsx") if p.is_file()])
    if not xlsx_files:
        return None, None, None, None
    for p in xlsx_files:
        role = _guess_role_from_name(p.name)
        if role == "sell" and sell_bytes is None:
            sell_bytes = io.BytesIO(p.read_bytes()); sell_name = p.name
        elif role == "price" and price_bytes is None:
            price_bytes = io.BytesIO(p.read_bytes()); price_name = p.name
    leftovers = [p for p in xlsx_files if p.name not in {sell_name, price_name}]
    for p in leftovers:
        if sell_bytes is None:
            sell_bytes = io.BytesIO(p.read_bytes()); sell_name = p.name
        elif price_bytes is None:
            price_bytes = io.BytesIO(p.read_bytes()); price_name = p.name
        if sell_bytes is not None and price_bytes is not None:
            break
    return sell_bytes, price_bytes, sell_name, price_name

# =========================
# UI
# =========================
st.title("Galaxus Sellout Analyse")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop or Datei wählen", type=["xlsx"], key="sell")
with c2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop or Datei wählen", type=["xlsx"], key="price")

# Auto-Load + Fallback
raw_sell = None
raw_price = None
used_sell_name = None
used_price_name = None

# 0) Uploads
if sell_file is not None:
    raw_sell = read_excel_flat(sell_file); used_sell_name = sell_file.name
    _persist_upload(sell_file, DEFAULT_SELL_PATH)
if price_file is not None:
    raw_price = read_excel_flat(price_file); used_price_name = price_file.name
    _persist_upload(price_file, DEFAULT_PRICE_PATH)

raw_sell, raw_price, used_sell_name, used_price_name = _maybe_swap_roles(raw_sell, raw_price, used_sell_name, used_price_name)

# 1) Defaults
if raw_sell is None and DEFAULT_SELL_PATH.exists():
    raw_sell = read_excel_flat(io.BytesIO(DEFAULT_SELL_PATH.read_bytes())); used_sell_name = DEFAULT_SELL_PATH.name
if raw_price is None and DEFAULT_PRICE_PATH.exists():
    raw_price = read_excel_flat(io.BytesIO(DEFAULT_PRICE_PATH.read_bytes())); used_price_name = DEFAULT_PRICE_PATH.name

raw_sell, raw_price, used_sell_name, used_price_name = _maybe_swap_roles(raw_sell, raw_price, used_sell_name, used_price_name)

# 2) Heuristische Auto-Erkennung in /data
if raw_sell is None or raw_price is None:
    sbytes, pbytes, sname, pname = _pick_default_files_from_dir(DATA_DIR)
    if raw_sell is None and sbytes is not None:
        tmp = read_excel_flat(sbytes)
        if _classify_df(tmp) == "price" and raw_price is None:
            raw_price, used_price_name = tmp, sname
        else:
            raw_sell, used_sell_name = tmp, sname
    if raw_price is None and pbytes is not None:
        tmp = read_excel_flat(pbytes)
        if _classify_df(tmp) == "sell" and raw_sell is None:
            raw_sell, used_sell_name = tmp, pname
        else:
            raw_price, used_price_name = tmp, pname

raw_sell, raw_price, used_sell_name, used_price_name = _maybe_swap_roles(raw_sell, raw_price, used_sell_name, used_price_name)

# Verarbeitung
if (raw_sell is not None) and (raw_price is not None):
    try:
        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)

        # Zeitraumfilter mit Woche-Snapping (Mo–So)
        filtered_sell_df = sell_df
        if {"StartDatum","EndDatum"}.issubset(sell_df.columns) and not sell_df["StartDatum"].isna().all():
            st.subheader("Periode wählen")
            min_date = sell_df["StartDatum"].min().date()
            max_date = (sell_df["EndDatum"].dropna().max() if "EndDatum" in sell_df else sell_df["StartDatum"].max()).date()

            if "date_range" not in st.session_state:
                st.session_state["date_range"] = (min_date, max_date)

            col_range, col_btn = st.columns([3,1])
            with col_range:
                date_value = st.date_input(
                    "Zeitraum (DD.MM.YYYY) – Auswahl wird automatisch auf volle Kalenderwochen erweitert",
                    value=st.session_state["date_range"],
                    min_value=min_date,
                    max_value=max_date,
                    format="DD.MM.YYYY",
                )
            with col_btn:
                st.write(""); st.write("")
                if st.button("Gesamten Zeitraum"):
                    st.session_state["date_range"] = (min_date, max_date)
                    st.rerun()

            if isinstance(date_value, tuple):
                start_date, end_date = date_value
            else:
                start_date = end_date = date_value

            start_snapped = start_date - timedelta(days=start_date.weekday())        # Montag
            end_snapped   = end_date + timedelta(days=(6 - end_date.weekday()))      # Sonntag
            st.session_state["date_range"] = (start_snapped, end_snapped)

            if (start_snapped != start_date) or (end_snapped != end_date):
                st.caption(f"📅 Auswahl auf ganze Wochen erweitert: {start_snapped.strftime('%d.%m.%Y')} – {end_snapped.strftime('%d.%m.%Y')}")

            sdt = sell_df["StartDatum"].dt.date
            edt = (sell_df["EndDatum"].fillna(sell_df["StartDatum"])).dt.date
            mask = ~((edt < start_snapped) | (sdt > end_snapped))
            filtered_sell_df = sell_df.loc[mask].copy()

            for col in ["Einkaufsmenge","Verkaufsmenge"]:
                if col in filtered_sell_df:
                    filtered_sell_df[col] = (
                        pd.to_numeric(filtered_sell_df[col], errors="coerce").fillna(0).clip(0, MAX_QTY)
                    )

        st.caption(f"🔎 Verwendete Dateien: {used_sell_name or '—'} / {used_price_name or '—'}")

        with st.spinner("🔗 Matche & berechne Werte…"):
            try:
                with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
                    detail, totals, ts_source = enrich_and_merge(filtered_sell_df, price_df, latest_stock_baseline_df=sell_df)
            except FloatingPointError:
                st.warning("Zahlüberlauf – Eingaben werden geclippt und erneut berechnet.")
                for col in ["Einkaufsmenge","Verkaufsmenge"]:
                    if col in filtered_sell_df:
                        filtered_sell_df[col] = pd.to_numeric(filtered_sell_df[col], errors="coerce").fillna(0).clip(0, MAX_QTY)
                for col in ["Einkaufspreis","Verkaufspreis","Lagermenge"]:
                    if col in price_df.columns:
                        lim = MAX_PRICE if "preis" in col.lower() else MAX_QTY
                        price_df[col] = pd.to_numeric(price_df[col], errors="coerce").fillna(0).clip(0, lim)
                detail, totals, ts_source = enrich_and_merge(filtered_sell_df, price_df, latest_stock_baseline_df=sell_df)

        # ------- Chart -------
        st.markdown("### 📈 Verkaufsverlauf nach Kategorie (Woche)")
        if not ts_source.empty:
            ts = ts_source.dropna(subset=["StartDatum"]).copy()
            ts["Periode"]   = ts["StartDatum"].dt.to_period("W").dt.start_time
            ts["Kategorie"] = ts["Kategorie"].astype("string")
            all_cats = sorted(ts["Kategorie"].unique())
            sel_cats = st.multiselect("Kategorien filtern", options=all_cats, default=all_cats)
            if sel_cats:
                ts = ts[ts["Kategorie"].isin(sel_cats)]
            ts_agg = (ts.groupby(["Kategorie","Periode"], as_index=False)["Verkaufswert (CHF)"]
                        .sum()
                        .rename(columns={"Verkaufswert (CHF)":"Wert (CHF)"}))
            ts_agg["Periode"]    = pd.to_datetime(ts_agg["Periode"])
            ts_agg["Kategorie"]  = ts_agg["Kategorie"].astype(str)
            ts_agg["Wert (CHF)"] = pd.to_numeric(ts_agg["Wert (CHF)"], errors="coerce").fillna(0.0).astype(float)

            hover_cat = alt.selection_single(fields=["Kategorie"], on="mouseover", nearest=True, empty="none")
            hover_pt  = alt.selection_single(fields=["Periode","Kategorie"], on="mouseover", nearest=True, empty="none")

            base = alt.Chart(ts_agg)
            lines = (
                base.mark_line(point=alt.OverlayMarkDef(size=30), interpolate="linear")
                    .encode(
                        x=alt.X("Periode:T", title="Woche"),
                        y=alt.Y("Wert (CHF):Q", title="Verkaufswert (CHF) pro Woche", stack=None),
                        color=alt.Color("Kategorie:N", title="Kategorie"),
                        opacity=alt.condition(hover_cat, alt.value(1.0), alt.value(0.25)),
                        strokeWidth=alt.condition(hover_cat, alt.value(3), alt.value(1.5)),
                        tooltip=[
                            alt.Tooltip("Periode:T", title="Woche"),
                            alt.Tooltip("Kategorie:N", title="Kategorie"),
                            alt.Tooltip("Wert (CHF):Q", title="Verkaufswert (CHF)", format=",.0f"),
                        ],
                    )
                    .add_selection(hover_cat)
            )
            points = (
                base.mark_point(size=70, opacity=0)
                    .encode(x="Periode:T", y="Wert (CHF):Q", color="Kategorie:N")
                    .add_selection(hover_pt)
            )
            popup = (
                base.transform_filter(hover_pt)
                    .mark_text(align='left', dx=6, dy=-8, fontSize=12, fontWeight='bold')
                    .encode(x="Periode:T", y="Wert (CHF):Q", text="Kategorie:N", color="Kategorie:N")
            )
            end_labels = (
                base.transform_window(
                        row_number='row_number()',
                        sort=[alt.SortField(field='Periode', order='descending')],
                        groupby=['Kategorie']
                    )
                    .transform_filter(alt.datum.row_number == 0)
                    .mark_text(align='left', dx=6, dy=-6, fontSize=11)
                    .encode(x='Periode:T', y='Wert (CHF):Q', text='Kategorie:N', color='Kategorie:N',
                            opacity=alt.condition(hover_cat, alt.value(1.0), alt.value(0.6)))
            )
            st.altair_chart((lines + points + popup + end_labels).properties(height=400), use_container_width=True)
        else:
            st.info("Für den Verlauf werden gültige Startdaten und nicht-leere Kategorien benötigt.")
    
        # ------- Tabellen & Downloads -------
        detail_renamed = pd.DataFrame()
        totals_renamed = pd.DataFrame()

        show_detail = st.checkbox("Detailtabelle anzeigen", value=False)
        if show_detail:
            st.subheader("Detailtabelle")
            detail_renamed = detail.rename(columns={
                "Einkaufswert":"Einkaufswert (CHF)",
                "Verkaufswert":"Verkaufswert (CHF)",
                "Lagerwert":"Lagerwert (CHF)"
            })
            detail_display = append_total_row_for_display(detail_renamed)
            _, d_styler = style_numeric(detail_display)
            st.dataframe(d_styler, use_container_width=True)

        st.subheader("Summen pro Artikel")
        totals_renamed = totals.rename(columns={
            "Einkaufswert":"Einkaufswert (CHF)",
            "Verkaufswert":"Verkaufswert (CHF)",
            "Lagerwert":"Lagerwert (CHF)"
        })
        totals_display = append_total_row_for_display(totals_renamed)
        _, t_styler = style_numeric(totals_display)
        st.dataframe(t_styler, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Detail (CSV)",
                data=(detail_renamed if show_detail else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
                file_name="detail.csv", mime="text/csv", disabled=not show_detail
            )
        with dl2:
            st.download_button(
                "⬇️ Summen (CSV)",
                data=totals_renamed.to_csv(index=False).encode("utf-8"),
                file_name="summen.csv", mime="text/csv"
            )

    except KeyError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
else:
    st.info("Bitte beide Dateien hochladen or in den Ordner `data/` legen. "
            "Es werden zuerst `sellout.xlsx`/`preisliste.xlsx` geladen, sonst Auto-Erkennung.")
