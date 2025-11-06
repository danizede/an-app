import os
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from datetime import datetime
from collections.abc import Mapping
import warnings

# =====================
# Globale Settings
# =====================
warnings.filterwarnings("ignore", message="overflow encountered in multiply")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered")
np.seterr(all="ignore")

st.set_page_config(page_title="Analyse", layout="wide")
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

THOUSANDS_SEP = "'"
NUM_COLS_DEFAULT = [
    "Einkaufsmenge", "Einkaufswert (CHF)",
    "Verkaufsmenge", "Verkaufswert (CHF)",
    "Lagermenge",   "Lagerwert (CHF)"
]
MAX_QTY, MAX_PRICE = 1_000_000, 1_000_000

# =====================
# Auth (Passcode)
# =====================
def _to_plain_mapping(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        try:
            return dict(obj)
        except Exception:
            pass
        try:
            return {k: obj[k] for k in obj.keys()}
        except Exception:
            return {}
    return {}

def _auth_cfg() -> dict:
    try:
        raw = st.secrets.get("auth", {})
    except Exception:
        raw = {}
    return _to_plain_mapping(raw)

def auth_enabled() -> bool:
    return bool(_auth_cfg().get("require_login", True))

def _get_passcode() -> str | None:
    # query param
    try:
        qp = st.query_params
        if "code" in qp and str(qp["code"]).strip():
            return str(qp["code"]).strip()
    except Exception:
        pass
    # secrets.auth.*
    auth = _auth_cfg()
    aliases = ("code", "password", "passcode", "pw", "passwort", "secret")
    for k in aliases:
        v = auth.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    # root secrets
    try:
        root = _to_plain_mapping(st.secrets)
        for k in aliases:
            v = root.get(k)
            if isinstance(v, (str, int)) and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    # env
    for k in ("AUTH_CODE", "AUTH_PASSWORD", "AUTH_PASSCODE", "STREAMLIT_AUTH_CODE"):
        v = os.environ.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return None

def _login_view():
    st.title("🔐 Zugang")
    with st.form("login-passcode", clear_on_submit=False):
        code = st.text_input("Code / Passwort", type="password")
        ok = st.form_submit_button("Anmelden")

    expected = _get_passcode()
    if expected is None:
        st.error("Kein Passcode gefunden. Bitte in st.secrets['auth']['code'] oder Env hinterlegen.")
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

def ensure_auth():
    if not auth_enabled():
        return True
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

# Gate
if not ensure_auth():
    st.stop()
logout_button()

# =====================
# Anzeige-Helfer
# =====================
def _fmt_thousands(x, sep=THOUSANDS_SEP):
    if pd.isna(x): return ""
    try:
        return f"{int(round(float(x))):,}".replace(",", sep)
    except Exception:
        return str(x)

def style_numeric(df: pd.DataFrame, num_cols=NUM_COLS_DEFAULT, sep=THOUSANDS_SEP):
    out = df.copy()
    present = [c for c in num_cols if c in out.columns]
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    fmt = {c: (lambda v, s=sep: _fmt_thousands(v, s)) for c in present}
    return out, out.style.format(fmt)

def append_total_row_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    num_targets = [
        "Einkaufsmenge","Einkaufswert (CHF)",
        "Verkaufsmenge","Verkaufswert (CHF)",
        "Lagermenge","Lagerwert (CHF)"
    ]
    num_cols = [c for c in num_targets if c in cols]
    label_col = next(
        (c for c in ["Bezeichnung_anzeige","Bezeichnung","ArtikelNr","Kategorie"] if c in cols),
        cols[0]
    )
    total_row = {c: "" for c in cols}
    total_row[label_col] = "Gesamt"
    for c in num_cols:
        total_row[c] = pd.to_numeric(df[c], errors="coerce").sum()
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# =====================
# Parsing Helpers
# =====================
def read_excel_flat(upload) -> pd.DataFrame:
    raw = pd.read_excel(upload, header=None, dtype=object)
    if raw.empty:
        return pd.DataFrame()
    header_idx = int(raw.notna().mean(axis=1).idxmax())
    headers = raw.iloc[header_idx].fillna("").astype(str).tolist()
    headers = [re.sub(r"\s+", " ", h).strip() for h in headers]
    n = raw.shape[1]
    if len(headers) < n:
        headers += [f"col_{i}" for i in range(len(headers), n)]
    else:
        headers = headers[:n]
    df = raw.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = headers
    # doppelte Spalten entschärfen
    seen = {}
    newcols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            newcols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            newcols.append(c)
    df.columns = newcols
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .map(lambda c: unicodedata.normalize("NFKC", str(c)))
        .map(lambda c: re.sub(r"\s+", " ", c).strip())
    )
    return df

def normalize_key(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True) -> str | None:
    cols = list(df.columns)
    # first: literal match
    for cand in candidates:
        if cand in cols:
            return cand

    # robust match
    def _norm(s: str) -> str:
        tmp = re.sub(r"\s+", "", str(s))
        tmp = tmp.translate(str.maketrans({
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2015": "-",
        }))
        tmp = re.sub(r"[\-*/.]+", "", tmp)
        return tmp.lower()

    canon = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in canon:
            return canon[key]

    if required:
        raise KeyError(
            f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
            f"Verfügbare Spalten: {cols}"
        )
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i", "u", "f"):
        return s
    def _clean(x):
        if pd.isna(x): return np.nan
        x = str(x).strip().replace("’", "").replace("'", "").replace(" ", "").replace(",", ".")
        if x.count(".") > 1:
            parts = x.split(".")
            x = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return float(x)
        except Exception:
            return np.nan
    return s.map(_clean)

def parse_date_series_us(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    nums = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(nums, origin="1899-12-30", unit="d", errors="coerce")
    return dt1.combine_first(dt2)

def sanitize_numbers(qty: pd.Series, price: pd.Series) -> tuple[pd.Series, pd.Series]:
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

# =====================
# Farben / Familie
# =====================
_COLOR_MAP = {
    "weiss":"Weiss","weiß":"Weiss","white":"White","offwhite":"Off-White",
    "cream":"Cream","ivory":"Ivory",
    "schwarz":"Schwarz","black":"Black","grau":"Grau","gray":"Grau",
    "anthrazit":"Anthrazit","charcoal":"Anthrazit","graphite":"Graphit","silver":"Silber",
    "blau":"Blau","blue":"Blau","navy":"Dunkelblau","light blue":"Hellblau",
    "dark blue":"Dunkelblau","sky blue":"Hellblau",
    "rot":"Rot","red":"Rot","bordeaux":"Bordeaux","burgundy":"Bordeaux",
    "pink":"Pink","magenta":"Magenta",
    "lila":"Lila","violett":"Violett","purple":"Violett","fuchsia":"Fuchsia",
    "grün":"Grün","gruen":"Grün","green":"Grün","mint":"Mint",
    "türkis":"Türkis","tuerkis":"Türkis","turquoise":"Türkis",
    "petrol":"Petrol","olive":"Olivgrün","gelb":"Gelb","yellow":"Gelb",
    "orange":"Orange","braun":"Braun","brown":"Braun","beige":"Beige","sand":"Sand",
    "gold":"Gold","rose gold":"Roségold","rosegold":"Roségold",
    "kupfer":"Kupfer","copper":"Kupfer","bronze":"Bronze",
    "transparent":"Transparent","clear":"Transparent"
}
_COLOR_WORDS = set(_COLOR_MAP.keys()) | set(map(str.lower, _COLOR_MAP.values()))
_STOP_TOKENS = {"eu","ch","us","uk","mobile","little","bundle","set","kit"}

def _looks_like_not_a_color(token: str) -> bool:
    t = (token or "").strip().lower()
    return (
        (not t) or
        (t in {"eu","ch","us","uk"}) or
        any(x in t for x in ["ml","db","m²","m2"]) or
        bool(re.search(r"\d", t))
    )

def _strip_parens_units(name: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", str(name))
    s = re.sub(r"\b\d+([.,]\d+)?\s*(ml|db|m²|m2|l/24h)\b", " ", s, flags=re.I)
    return s

def make_family_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = _strip_parens_units(name.lower())
    s = re.sub(r"\b[o0]-\d+\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split()
            if t and (t not in _STOP_TOKENS) and (t not in _COLOR_WORDS)]
    return "".join(toks[:2]) if toks else ""

def extract_color_from_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # 1) Farbangabe am Ende nach "-" oder "/"
    m = re.search(r"(?:-|/)\s*([A-Za-z äöüÄÖÜß]+)\s*$", name.strip())
    if m:
        cand = m.group(1).strip().lower()
        if not _looks_like_not_a_color(cand):
            return _COLOR_MAP.get(cand, cand.title())
    # 2) Farbe irgendwo im Namen
    for w in sorted(_COLOR_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", name, flags=re.I):
            if not _looks_like_not_a_color(w):
                return _COLOR_MAP.get(w, w.title())
    return ""

# =====================
# Parsing Preislisten
# =====================
PRICE_COL_CANDIDATES = ["Preis","VK","Netto","NETTO","Einkaufspreis",
                        "Verkaufspreis","NETTO NETTO","NETTO NE"]
BUY_PRICE_CANDIDATES  = ["Einkaufspreis","Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis","VK","Preis","NETTO","NETTO NE"]

ARTNR_CANDIDATES = [
    "Artikelnummer","Artikelnr","ArtikelNr","Artikel-Nr.","Hersteller-Nr.",
    "Produkt ID","ProdNr","ArtNr","ArtikelNr.","Artikel","Artikelnumm",
    "Herstellerartikelnummer","Herstellerartikel","Hersteller Artikelnummer","Herstellernr",
    "Produktnr","Produktnummer","erstelle","Hersteller Nr.","Hersteller Nr",
    "HerstellerNr.","Herstellernr.","Hersteller-Nr","HerstellerNr"
]
EAN_CANDIDATES  = ["EAN","GTIN","BarCode","Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung","Produktname","Name","Titel","Artikelname"]
CAT_CANDIDATES  = ["Kategorie","Warengruppe","Zusatz","Category","KategorieName"]
STOCK_CANDIDATES= ["Bestand","Verfügbar","verfügbar","Verfuegbar",
                   "Lagerbestand","Lagermenge","Available"]
COLOR_CANDIDATES= ["Farbe","Color","Colour","Variante","Variant",
                   "Farbvariante","Farbname"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES,  "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")

    col_cat = find_column(df, CAT_CANDIDATES, "Kategorie", required=False)
    if not col_cat:
        try:
            col_cat = df.columns[6]
        except Exception:
            col_cat = None

    col_stock = find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_any   = None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str)
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Familie"]         = out["Bezeichnung"].map(make_family_key)

    if col_cat:
        out["Kategorie"] = (
            df[col_cat].astype(str)
            .replace({"nan":"","NaN":"","None":""})
            .str.strip()
        )
    else:
        out["Kategorie"] = ""

    # Farbe
    cand_color_col = find_column(df, COLOR_CANDIDATES, "Farbe/Variante", required=False)
    if cand_color_col:
        out["Farbe"] = df[cand_color_col].astype(str).map(
            lambda v: _COLOR_MAP.get(str(v).lower(), str(v))
        )
    else:
        out["Farbe"] = out["Bezeichnung"].map(extract_color_from_name)
    out["Farbe"] = out["Farbe"].fillna("").astype(str)

    # Lagermenge aus PL (optional)
    out["Lagermenge"] = (
        parse_number_series(df[col_stock]).fillna(0).astype("Int64")
        if col_stock else pd.Series([0]*len(out), dtype="Int64")
    )

    # Preise
    if col_buy:
        out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell:
        out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        anyp = parse_number_series(df[col_any])
        out["Einkaufspreis"] = anyp
        out["Verkaufspreis"] = anyp
    if "Einkaufspreis" not in out:
        out["Einkaufspreis"] = out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out:
        out["Verkaufspreis"] = out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))

    # Dedupe: prefer Zeilen mit Kategorie, dann Preis, dann Farbe
    out = out.assign(
        _cat=out["Kategorie"].astype(bool).astype(int),
        _price=out["Verkaufspreis"].notna().astype(int),
        _color=out["Farbe"].astype(bool).astype(int),
    )
    out = out.sort_values(
        ["ArtikelNr_key","_cat","_price","_color"],
        ascending=[True, False, False, False]
    )
    out = out.drop_duplicates(
        subset=["ArtikelNr_key"], keep="first"
    ).drop(columns=["_cat","_price","_color"])

    return out

# =====================
# Parsing Sellout
# =====================
NAME_CANDIDATES_SO   = ["Bezeichnung","Name","Artikelname","Bezeichnung_Sales","Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty","Verkauf","Verkaufte Menge","Menge verkauft",
                        "Absatz","Stück","Menge","Verkaufsmenge"]
BUY_QTY_CANDIDATES   = ["Einkauf","Einkaufsmenge","Menge Einkauf"]
DATE_START_CANDS     = ["Start","Startdatum","Start Date","Anfangs datum",
                        "Anfangsdatum","Von","Period Start"]
DATE_END_CANDS       = ["Ende","Enddatum","End Date","Bis","Period End"]
STOCK_SO_CANDIDATES  = ["Lagermenge","Lagerbestand","Bestand","Verfügbar",
                        "verfügbar","Verfuegbar","Available"]

ART_EXACT_EQUIV  = {"e008":"e009","j031":"j030","m057":"m051","s054":"s054"}
ART_PREFIX_EQUIV = {"o061":"o061","o013":"o013"}

def _apply_hints_to_row(name_raw: str) -> dict:
    s = (name_raw or "").lower()
    h = {"hint_family":"","hint_color":"","hint_art_exact":"","hint_art_prefix":""}
    for fam in ["finn mobile","charly little","duftöl","duftoel","duft oil"]:
        if fam in s:
            h["hint_family"] = "finn" if fam=="finn mobile" else ("charly" if "charly" in fam else "duftol")
    for fam in ["finn","theo","robert","peter","julia","albert","roger",
                "mia","simon","otto","oskar","tim","charly","oliver"]:
        if fam in s:
            h["hint_family"] = h["hint_family"] or fam
    if "tim" in s and "schwarz" in s: h["hint_color"]="weiss"
    if "mia" in s and "gold" in s:    h["hint_color"]="schwarz"
    if "oskar" in s and "little" in s: h["hint_art_prefix"]="o061"
    if "simon" in s: h["hint_art_exact"]="s054"
    if "otto"  in s: h["hint_art_prefix"]="o013"
    if "eva" in s and "e-008" in s: h["hint_art_exact"]="e008"
    if "julia" in s and "j-031" in s: h["hint_art_exact"]="j031"
    if "mia" in s and "m-057" in s: h["hint_art_exact"]="m057"
    return h

def _fallback_col_by_index(df: pd.DataFrame, idx0: int) -> str | None:
    try:
        return df.columns[idx0]
    except Exception:
        return None

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES,   "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES,     "EAN/GTIN",      required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung",   required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)

    col_stock_so = find_column(df, STOCK_SO_CANDIDATES, "Lagermenge (Sell-out)", required=False)
    if not col_stock_so and df.shape[1] >= 7:
        col_stock_so = _fallback_col_by_index(df, 6)

    col_start = find_column(df, DATE_START_CANDS, "Startdatum (Spalte I)", required=False)
    col_end   = find_column(df, DATE_END_CANDS,   "Enddatum (Spalte J)",   required=False)
    if not col_start and df.shape[1] >= 9:
        col_start = _fallback_col_by_index(df, 8)
    if not col_end and df.shape[1] >= 10:
        col_end   = _fallback_col_by_index(df, 9)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","", str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Familie"]         = out["Bezeichnung"].map(make_family_key)

    hints = out["Bezeichnung"].map(_apply_hints_to_row)
    out["Hint_Family"]   = hints.map(lambda h: h["hint_family"])
    out["Hint_Color"]    = hints.map(lambda h: h["hint_color"])
    out["Hint_ArtExact"] = hints.map(lambda h: h["hint_art_exact"])
    out["Hint_ArtPref"]  = hints.map(lambda h: h["hint_art_prefix"])

    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"] = (
        parse_number_series(df[col_buy]).fillna(0).astype("Int64")
        if col_buy else pd.Series([0]*len(df), dtype="Int64")
    )

    if col_stock_so:
        out["SellLagermenge"] = pd.to_numeric(df[col_stock_so], errors="coerce")

    if col_start:
        out["StartDatum"] = parse_date_series_us(df[col_start])
    if col_end:
        out["EndDatum"] = parse_date_series_us(df[col_end])

    if "StartDatum" in out and "EndDatum" in out:
        out.loc[out["EndDatum"].isna(), "EndDatum"] = out.loc[out["EndDatum"].isna(), "StartDatum"]

    # Schlüssel für Dedupe (wenn gleiche Woche nochmal hochgeladen wird)
    out["PeriodKey"] = (
        out["ArtikelNr_key"].astype(str).fillna("") + "|" +
        out.get("StartDatum", pd.NaT).astype(str) + "|" +
        out.get("EndDatum",   pd.NaT).astype(str)
    )
    out["UploadTimestampUTC"] = datetime.utcnow().isoformat()

    return out

# =====================
# Matching-Funktionen
# =====================
def _tokenize_name_for_match(name: str) -> set[str]:
    if not isinstance(name, str):
        name = str(name or "")
    cleaned = re.sub(r"\([^)]*\)", " ", name.lower())
    cleaned = re.sub(r"\b\d+([.,]\d+)?\s*(ml|db|m²|m2|l/24h)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    toks = [
        t for t in cleaned.split()
        if t and t not in _STOP_TOKENS and t not in _COLOR_WORDS
    ]
    return set(toks)

def _match_best_row_for_single_item(row_sell: pd.Series, price_df: pd.DataFrame) -> pd.Series | None:
    art_key   = str(row_sell.get("ArtikelNr_key","") or "").strip().lower()
    ean_key   = str(row_sell.get("EAN_key","") or "").strip()
    fam_sell  = str(row_sell.get("Familie","") or "").strip().lower()
    hint_fam  = str(row_sell.get("Hint_Family","") or "").strip().lower()

    sell_color_hint = (
        str(row_sell.get("Hint_Color","") or "").strip()
        or str(row_sell.get("Farbe","") or "").strip()
    )
    sell_color_hint_low = sell_color_hint.lower()

    def pf_by(keys: dict) -> pd.DataFrame:
        df = price_df
        for k,v in keys.items():
            if v is None:
                continue
            if k == "Farbe" and v.strip():
                df = df.loc[df["Farbe"].str.strip().str.lower() == v.strip().lower()]
            else:
                df = df.loc[df[k].astype(str).str.strip().str.lower() == v.strip().lower()]
        return df

    # 1: ArtikelNr_key + Farbe
    if art_key and sell_color_hint_low:
        hit = pf_by({"ArtikelNr_key": art_key, "Farbe": sell_color_hint_low})
        if len(hit) >= 1:
            return hit.iloc[0]
    # 2: ArtikelNr_key
    if art_key:
        hit = pf_by({"ArtikelNr_key": art_key})
        if len(hit) >= 1:
            if len(hit) > 1 and sell_color_hint_low:
                sub = hit.loc[hit["Farbe"].str.strip().str.lower() == sell_color_hint_low]
                if not sub.empty:
                    return sub.iloc[0]
            return hit.iloc[0]
    # 3: EAN + Farbe
    if ean_key and sell_color_hint_low:
        hit = price_df.loc[
            (price_df["EAN_key"].str.strip() == ean_key) &
            (price_df["Farbe"].str.strip().str.lower() == sell_color_hint_low)
        ]
        if len(hit) >= 1:
            return hit.iloc[0]
    # 4: EAN
    if ean_key:
        hit = price_df.loc[price_df["EAN_key"].str.strip() == ean_key]
        if len(hit) >= 1:
            if len(hit) > 1 and sell_color_hint_low:
                sub = hit.loc[hit["Farbe"].str.strip().str.lower() == sell_color_hint_low]
                if not sub.empty:
                    return sub.iloc[0]
            return hit.iloc[0]
    # 5: Familie (+evtl. Farbe)
    fam_candidate = (hint_fam or fam_sell).strip().lower()
    if fam_candidate:
        fam_hit = price_df.loc[price_df["Familie"].str.strip().str.lower() == fam_candidate]
        if not fam_hit.empty:
            if sell_color_hint_low:
                sub = fam_hit.loc[fam_hit["Farbe"].str.strip().str.lower() == sell_color_hint_low]
                if not sub.empty:
                    fam_hit = sub
            return fam_hit.iloc[0]
    # 6: Fuzzy Name
    sell_tokens = _tokenize_name_for_match(str(row_sell.get("Bezeichnung","")))
    if sell_tokens:
        best_idx, best_score = None, 0.0
        for idx_price, row_price in price_df.iterrows():
            price_tokens = _tokenize_name_for_match(str(row_price.get("Bezeichnung","")))
            if not price_tokens:
                continue
            inter = len(sell_tokens & price_tokens)
            union = len(sell_tokens | price_tokens)
            score = inter / union if union else 0.0
            if sell_color_hint_low and str(row_price.get("Farbe","")).strip():
                if str(row_price["Farbe"]).strip().lower() == sell_color_hint_low:
                    score += 0.15
            if score > best_score:
                best_score, best_idx = score, idx_price
        if best_idx is not None and best_score >= 0.5:
            return price_df.loc[best_idx]
    return None

def match_sell_to_price(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    sell_df = sell_df.copy()
    for col in ["Einkaufspreis","Verkaufspreis","Kategorie","Farbe","ArtikelNr","ArtikelNr_key","Familie","Lagermenge"]:
        if col not in sell_df.columns:
            sell_df[col] = np.nan

    match_flags, cat_list, col_list, art_list, fam_list, ekpreis_list, vkpreis_list, stock_list = ([] for _ in range(8))

    for _, row in sell_df.iterrows():
        hit = _match_best_row_for_single_item(row, price_df)
        if hit is None:
            match_flags.append(True);  cat_list.append(""); col_list.append("")
            art_list.append(row.get("ArtikelNr", "")); fam_list.append(row.get("Familie", ""))
            ekpreis_list.append(np.nan); vkpreis_list.append(np.nan); stock_list.append(np.nan)
        else:
            match_flags.append(False)
            cat_list.append(hit.get("Kategorie","")); col_list.append(hit.get("Farbe",""))
            art_list.append(hit.get("ArtikelNr",""));  fam_list.append(hit.get("Familie",""))
            ekpreis_list.append(hit.get("Einkaufspreis", np.nan)); vkpreis_list.append(hit.get("Verkaufspreis", np.nan))
            stock_list.append(hit.get("Lagermenge", np.nan))

    sell_df["MatchFehler"]    = match_flags
    sell_df["Kategorie"]      = cat_list
    sell_df["Farbe"]          = col_list
    sell_df["ArtikelNr"]      = art_list
    sell_df["Familie"]        = fam_list
    sell_df["Einkaufspreis"]  = ekpreis_list
    sell_df["Verkaufspreis"]  = vkpreis_list
    sell_df["Lagermenge_PL"]  = stock_list
    return sell_df

# =====================
# Merge + Werte
# =====================
def enrich_and_merge(filtered_sell_df: pd.DataFrame,
                     price_df: pd.DataFrame,
                     latest_stock_baseline_df: pd.DataFrame | None = None):

    if (filtered_sell_df is None or price_df is None or
        filtered_sell_df.empty or price_df.empty):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sell_for_stock = latest_stock_baseline_df if latest_stock_baseline_df is not None else filtered_sell_df

    merged = filtered_sell_df.copy()
    stock_merged = sell_for_stock.copy()

    # Fallback aus Preisliste pro ArtikelNr_key
    price_keyed = price_df.drop_duplicates("ArtikelNr_key").set_index("ArtikelNr_key")

    def enrich_missing_from_price(df2: pd.DataFrame):
        for idx, row in df2.iterrows():
            key = str(row.get("ArtikelNr_key","")).strip().lower()
            if not key or key not in price_keyed.index:
                continue
            plrow = price_keyed.loc[key]
            for tgt in ["Einkaufspreis","Verkaufspreis","Kategorie","Farbe","Familie","ArtikelNr"]:
                if tgt in df2.columns and (pd.isna(df2.at[idx, tgt]) or str(df2.at[idx, tgt]).strip() == ""):
                    if tgt in plrow and not pd.isna(plrow[tgt]):
                        df2.at[idx, tgt] = plrow[tgt]
        return df2

    merged = enrich_missing_from_price(merged)
    stock_merged = enrich_missing_from_price(stock_merged)

    # Strings säubern
    for df2 in (merged, stock_merged):
        df2["Kategorie"] = (
            df2.get("Kategorie","").fillna("").astype(str).replace({"nan":"","NaN":"","None":""}).str.strip()
        )
        df2["Bezeichnung"] = df2.get("Bezeichnung"," ").fillna("")
        df2["Farbe"]       = df2.get("Farbe"," ").fillna("")

    # Anzeige-Name (Farbe bei Duplikaten anhängen)
    merged["Bezeichnung_anzeige"] = merged["Bezeichnung"]
    def _invalid_color(token: str) -> bool:
        t = (token or "").strip().lower()
        return (not t) or (t in {"eu","ch","us","uk"}) or any(x in t for x in ["ml","db","m²","m2"]) or bool(re.search(r"\d", t))
    dup = merged.duplicated(subset=["Bezeichnung"], keep=False)
    valid_color = merged["Farbe"].astype(str).str.strip().map(lambda t: (t != "") and (not _invalid_color(t)))
    merged.loc[dup & valid_color, "Bezeichnung_anzeige"] = (
        merged.loc[dup & valid_color, "Bezeichnung"] + " – " + merged.loc[dup & valid_color, "Farbe"].astype(str).str.strip()
    )

    # Werte berechnen
    q_buy,  p_buy  = sanitize_numbers(merged.get("Einkaufsmenge", 0), merged.get("Einkaufspreis", 0))
    q_sell, p_sell = sanitize_numbers(merged.get("Verkaufsmenge", 0), merged.get("Verkaufspreis", 0))
    merged["Einkaufswert"] = safe_mul(q_buy.fillna(0.0),  p_buy.fillna(0.0))
    merged["Verkaufswert"] = safe_mul(q_sell.fillna(0.0), p_sell.fillna(0.0))

    # === Lagerstand-Logik ===
    def _mk_grpkey(df2):
        a = df2.get("ArtikelNr_key","").astype(str).fillna("")
        e = df2.get("EAN_key","").astype(str).fillna("")
        use_art = a.str.len() > 0
        return np.where(use_art, "A:" + a, "E:" + e)

    def _row_date(df2):
        if ("EndDatum" in df2.columns) and ("StartDatum" in df2.columns):
            d = df2["EndDatum"].fillna(df2["StartDatum"])
        elif "StartDatum" in df2.columns:
            d = df2["StartDatum"]
        elif "EndDatum" in df2.columns:
            d = df2["EndDatum"]
        else:
            d = pd.to_datetime(pd.NaT)
        return pd.to_datetime(d, errors="coerce")

    merged["_rowdate"]       = _row_date(merged)
    stock_merged["_rowdate"] = _row_date(stock_merged)
    merged["_grpkey"]        = _mk_grpkey(merged)
    stock_merged["_grpkey"]  = _mk_grpkey(stock_merged)

    # Für die Lagerberechnung: SellLagermenge numerisch; fehlende Spalte initialisieren mit NaN
    stock_merged["SellLagermenge"] = pd.to_numeric(stock_merged.get("SellLagermenge", np.nan), errors="coerce")

    # Zeitraumfenster bestimmen (für die Lagerlogik)
    period_min = pd.to_datetime(merged["_rowdate"]).min()
    period_max = pd.to_datetime(merged["_rowdate"]).max()

    # Nur Zeilen mit gültiger SellLagermenge berücksichtigen
    sv = stock_merged.loc[stock_merged["SellLagermenge"].notna()].copy()

    if not sv.empty:
        # Nach Datum filtern, wenn Datumsspalte vorhanden und gültig
        if "_rowdate" in sv.columns:
            if pd.notna(period_min) and pd.notna(period_max):
                sv = sv.loc[
                    (sv["_rowdate"] >= period_min) &
                    (sv["_rowdate"] <= period_max)
                ]
            elif pd.notna(period_max):
                sv = sv.loc[sv["_rowdate"] <= period_max]

        # Falls Filter zu keinem Ergebnis führt, die jüngsten Zeilen nehmen
        if sv.empty and "_rowdate" in stock_merged.columns and pd.notna(period_max):
            sv = stock_merged.loc[
                stock_merged["SellLagermenge"].notna() &
                (stock_merged["_rowdate"] <= period_max)
            ].copy()
        # Wenn immer noch leer, dann einfach alle Zeilen mit SellLagermenge
        if sv.empty:
            sv = stock_merged.loc[stock_merged["SellLagermenge"].notna()].copy()

        sv = sv.sort_values(["_grpkey","_rowdate"], ascending=[True, True])
        last_rows = sv.groupby("_grpkey", as_index=False).tail(1)
        latest_qty_map = dict(zip(last_rows["_grpkey"], last_rows["SellLagermenge"]))
    else:
        latest_qty_map = {}

    # Fallback Preis pro ArtikelNr_key
    price_map = (
        pd.to_numeric(
            price_df.drop_duplicates("ArtikelNr_key").set_index("ArtikelNr_key")["Verkaufspreis"],
            errors="coerce"
        ).fillna(0.0).clip(lower=0, upper=MAX_PRICE).to_dict()
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

    # === Ausgabe-Tabellen ===
    detail = merged[[
        "ArtikelNr","Bezeichnung_anzeige","Kategorie",
        "Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert"
    ]].copy()
    detail["Lagermenge"] = merged["Lagermenge_latest"]
    detail["Lagerwert"]  = merged["Lagerwert_latest"]

    totals = (
        detail.groupby(["ArtikelNr","Bezeichnung_anzeige","Kategorie"], dropna=False, as_index=False)
              .agg({"Einkaufsmenge":"sum","Einkaufswert":"sum","Verkaufsmenge":"sum","Verkaufswert":"sum",
                    "Lagermenge":"max","Lagerwert":"max"})
    )

    ts_source = pd.DataFrame()
    if "StartDatum" in merged.columns:
        ts_source = merged[["StartDatum","Kategorie","Verkaufswert"]].copy()
        ts_source["Kategorie"] = ts_source["Kategorie"].fillna("").astype(str).str.strip()
        ts_source.rename(columns={"Verkaufswert":"Verkaufswert (CHF)"}, inplace=True)

    merged.drop(columns=["_rowdate","_grpkey"], errors="ignore", inplace=True)
    return detail, totals, ts_source

# =====================
# Persistenz-Logik
# =====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SELLOUT_PATH   = DATA_DIR / "master_sellout.parquet"
MASTER_PRICELIST_PATH = DATA_DIR / "master_preisliste.parquet"

def load_master_sellout() -> pd.DataFrame:
    if MASTER_SELLOUT_PATH.exists():
        return pd.read_parquet(MASTER_SELLOUT_PATH)
    return pd.DataFrame()

def save_master_sellout(df: pd.DataFrame):
    df.to_parquet(MASTER_SELLOUT_PATH, index=False)

def load_master_pricelist() -> pd.DataFrame:
    if MASTER_PRICELIST_PATH.exists():
        return pd.read_parquet(MASTER_PRICELIST_PATH)
    return pd.DataFrame()

def save_master_pricelist(df: pd.DataFrame):
    df.to_parquet(MASTER_PRICELIST_PATH, index=False)

def merge_into_master_sellout(master_df: pd.DataFrame,
                              new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Historie anhängen.
    Wenn derselbe Artikel (ArtikelNr_key) für denselben Zeitraum (PeriodKey)
    erneut hochgeladen wird, gewinnt der neuere UploadTimestampUTC.
    """
    if master_df.empty:
        return new_df.copy()

    comb = pd.concat([master_df, new_df], ignore_index=True)

    comb["__dedupe_key"] = np.where(
        comb["PeriodKey"].astype(str).str.strip().ne(""),
        comb["PeriodKey"].astype(str),
        comb["ArtikelNr_key"].astype(str)
    )

    comb["__ts_rank"] = comb.groupby("__dedupe_key")["UploadTimestampUTC"] \
                            .transform(lambda s: (s == s.max()))

    comb = comb.loc[comb["__ts_rank"]].copy()

    comb.drop(columns=["__dedupe_key","__ts_rank"], inplace=True, errors="ignore")
    comb.reset_index(drop=True, inplace=True)
    return comb

# =====================
# UI
# =====================
st.title("Analyse")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="sell")
with c2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="price")

# 1. Master laden
master_sell_df  = load_master_sellout()
master_price_df = load_master_pricelist()

new_sell_df = None
new_price_df = None

# 2. Neue Uploads einlesen und in Master persistieren
if sell_file is not None:
    uploaded_sell_raw = read_excel_flat(sell_file)
    parsed_new_sell   = prepare_sell_df(uploaded_sell_raw)
    master_sell_df    = merge_into_master_sellout(master_sell_df, parsed_new_sell)
    save_master_sellout(master_sell_df)
    new_sell_df = parsed_new_sell

if price_file is not None:
    uploaded_price_raw = read_excel_flat(price_file)
    parsed_new_price   = prepare_price_df(uploaded_price_raw)
    master_price_df    = parsed_new_price.copy()  # neueste PL ersetzt
    save_master_pricelist(master_price_df)
    new_price_df = parsed_new_price

# 3. Hinweis-Box Datenquelle
with st.expander("ℹ️ Daten-Quelle"):
    if new_sell_df is not None:
        st.write(f"✅ Neuer Sell-out von Upload '{sell_file.name}' wurde übernommen und im Master gespeichert.")
    else:
        if master_sell_df.empty:
            st.write("❌ Kein Sell-out vorhanden (Master leer).")
        else:
            st.write("📂 Verwende gespeicherten Master-Sell-out (kumuliert aus bisherigen Uploads).")

    if new_price_df is not None:
        st.write(f"✅ Neue Preisliste '{price_file.name}' wurde übernommen und als aktuelle Preisliste gespeichert.")
    else:
        if master_price_df.empty:
            st.write("❌ Keine Preisliste vorhanden (Master leer).")
        else:
            st.write("📂 Verwende gespeicherte Preisliste aus dem Master.")

# 4. Wenn noch nichts da ist -> abbrechen
if master_sell_df.empty or master_price_df.empty:
    st.info("Bitte mindestens einen Sell-out und eine Preisliste hochladen, damit die Analyse berechnet werden kann.")
    if master_sell_df.empty or master_price_df.empty:
        st.stop()

# 5. Zeitraumfilter nur zur Anzeige
sell_df_all = master_sell_df.copy()

if {"StartDatum","EndDatum"}.issubset(sell_df_all.columns) and not sell_df_all["StartDatum"].isna().all():
    st.subheader("Periode wählen")

    min_date = sell_df_all["StartDatum"].min().date()
    max_date = (
        sell_df_all["EndDatum"].dropna().max()
        if "EndDatum" in sell_df_all
        else sell_df_all["StartDatum"].max()
    ).date()

    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (min_date, max_date)

    col_range, col_btn = st.columns([3,1])
    with col_range:
        date_value = st.date_input(
            "Zeitraum (DD.MM.YYYY)",
            value=st.session_state["date_range"],
            min_value=min_date,
            max_value=max_date,
            format="DD.MM.YYYY",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Gesamten Zeitraum"):
            st.session_state["date_range"] = (min_date, max_date)
            st.rerun()

    if isinstance(date_value, tuple):
        start_date, end_date = date_value
    else:
        start_date = end_date = date_value

    st.session_state["date_range"] = (start_date, end_date)

    mask = ~(
        (sell_df_all["EndDatum"].dt.date < start_date) |
        (sell_df_all["StartDatum"].dt.date > end_date)
    )
    filtered_sell_df = sell_df_all.loc[mask].copy()
else:
    filtered_sell_df = sell_df_all.copy()

# 6. Neues Matching, dann Berechnung/Werte
with st.spinner("🔗 Matche & berechne Werte…"):
    matched_filtered = match_sell_to_price(filtered_sell_df, master_price_df)
    matched_all      = match_sell_to_price(sell_df_all,      master_price_df)

    detail, totals, ts_source = enrich_and_merge(
        matched_filtered,
        master_price_df,
        latest_stock_baseline_df=matched_all
    )

# 7. Match-Qualität anzeigen
unmatched = matched_all.loc[matched_all["MatchFehler"] == True].copy()
if not unmatched.empty:
    st.error(
        f"Achtung: {len(unmatched)} Position(en) aus dem Sell-out konnten nicht in der Preisliste gefunden werden. "
        "Bitte Preisliste ergänzen (Kategorie/Preis/Farbe)."
    )
    st.dataframe(
        unmatched[[
            "ArtikelNr","Bezeichnung","Farbe","Verkaufsmenge","Einkaufsmenge",
            "StartDatum","EndDatum","Hint_Family","Hint_Color","EAN","EAN_key"
        ]],
        use_container_width=True
    )
else:
    st.success("Alle Artikel aus dem Sell-out sind in der Preisliste gefunden worden und kategorisiert.")

# 8. Chart
st.markdown("### 📈 Verkaufsverlauf nach Kategorie (Woche)")
if not ts_source.empty:
    ts = ts_source.dropna(subset=["StartDatum"]).copy()
    ts["Periode"]   = ts["StartDatum"].dt.to_period("W").dt.start_time
    ts["Kategorie"] = ts["Kategorie"].astype("string")

    all_cats = sorted(ts["Kategorie"].unique())
    sel_cats = st.multiselect(
        "Kategorien filtern",
        options=all_cats,
        default=all_cats
    )
    if sel_cats:
        ts = ts[ts["Kategorie"].isin(sel_cats)]

    ts_agg = (
        ts.groupby(["Kategorie","Periode"], as_index=False)["Verkaufswert (CHF)"]
          .sum()
          .rename(columns={"Verkaufswert (CHF)":"Wert (CHF)"})
    )
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
        base.transform_window(row_number='row_number()',
                              sort=[alt.SortField(field='Periode', order='descending')],
                              groupby=['Kategorie'])
            .transform_filter(alt.datum.row_number == 0)
            .mark_text(align='left', dx=6, dy=-6, fontSize=11)
            .encode(x='Periode:T', y='Wert (CHF):Q', text='Kategorie:N',
                    color='Kategorie:N', opacity=alt.condition(hover_cat, alt.value(1.0), alt.value(0.6)))
    )
    chart = (lines + points + popup + end_labels).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Für den Verlauf werden gültige Startdaten benötigt.")

# 9. Tabellen
show_detail = st.checkbox("Detailtabelle anzeigen", value=False)
if show_detail:
    st.subheader("Detailtabelle")
    detail_renamed = detail.rename(columns={
        "Einkaufswert": "Einkaufswert (CHF)",
        "Verkaufswert": "Verkaufswert (CHF)",
        "Lagerwert":    "Lagerwert (CHF)"
    })
    detail_display = append_total_row_for_display(detail_renamed)
    d_rounded, d_styler = style_numeric(detail_display)
    st.dataframe(d_styler, use_container_width=True)

st.subheader("Summen pro Artikel")
totals_renamed = totals.rename(columns={
    "Einkaufswert": "Einkaufswert (CHF)",
    "Verkaufswert": "Verkaufswert (CHF)",
    "Lagerwert":    "Lagerwert (CHF)"
})
totals_display = append_total_row_for_display(totals_renamed)
t_rounded, t_styler = style_numeric(totals_display)
st.dataframe(t_styler, use_container_width=True)

# 10. Downloads
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ Detail (CSV)",
        data=(detail_renamed if show_detail else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        file_name="detail.csv",
        mime="text/csv",
        disabled=not show_detail
    )
with dl2:
    st.download_button(
        "⬇️ Summen (CSV)",
        data=totals_renamed.to_csv(index=False).encode("utf-8"),
        file_name="summen.csv",
        mime="text/csv"
    )
