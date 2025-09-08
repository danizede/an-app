# app.py — Galaxus Sellout Analyse
# - Robustes Matching (ArtNr → EAN → Name → Familie → Hints → Fuzzy)
# - EU-Datumsfilter, Detailtabelle optional
# - Summen pro Artikel (Lagerwert = letzter verfügbarer Stand je Artikel, NICHT aufsummiert)
# - Interaktives Linienchart (Woche) mit Hover-Highlight & Pop-up-Label
# - Auto-Load aus /data (sellout.xlsx, preisliste.xlsx) + optionales Persistieren von Uploads
# - Sichere Multiplikation (safe_mul) + np.seterr(all="ignore")
# - Kategorie primär aus Spalte G; leere/NaN-Kategorien bleiben leer und erscheinen nicht im Chart
# - Summenzeile Σ Gesamt (nur Anzeige) – Währungsangaben in Spaltennamen (CHF)

import os
import io
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ---------- Global ----------
np.seterr(all='ignore')
st.set_page_config(page_title="Galaxus Sellout Analyse", layout="wide")

# Altair große Datensätze erlauben
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

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
    """Σ-Gesamt ans Ende anhängen (nur UI-Anzeige)."""
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
    if len(headers) < n:
        headers += [f"col_{i}" for i in range(len(headers), n)]
    else:
        headers = headers[:n]
    df = raw.iloc[header_idx+1:].reset_index(drop=True)
    df.columns = headers
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
    df.columns = (df.columns
                  .map(lambda c: unicodedata.normalize("NFKC", str(c)))
                  .map(lambda c: re.sub(r"\s+"," ", c).strip()))
    return df

def normalize_key(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    return re.sub(r"[^a-z0-9]+","", s)

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True) -> str|None:
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
def sanitize_numbers(qty: pd.Series, price: pd.Series) -> tuple[pd.Series,pd.Series]:
    q = pd.to_numeric(qty, errors="coerce").astype("float64").clip(lower=0, upper=MAX_QTY)
    p = pd.to_numeric(price, errors="coerce").astype("float64").clip(lower=0, upper=MAX_PRICE)
    return q, p

def safe_mul(a: pd.Series, b: pd.Series, max_a=MAX_QTY, max_b=MAX_PRICE) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce").astype("float64")
    b = pd.to_numeric(b, errors="coerce").astype("float64")
    a = np.clip(a, 0, max_a)
    b = np.clip(b, 0, max_b)
    with np.errstate(all='ignore'):
        out = np.multiply(a, b, dtype="float64")
    out = np.where(np.isfinite(out), out, 0.0).astype("float64")
    return pd.Series(out, index=a.index)

# =========================
# Farben & Familie
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
_STOP_TOKENS = {"eu","ch","us","uk","mobile","little","bundle","set","kit"}

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
    if not m: m = re.search(r"[-–—]\s*([A-Za-zäöüÄÖÜß]+)$", name.strip())
    if not m: m = re.search(r"/\s*([A-Za-zäöüÄÖÜß]+)$", name.strip())
    if m:
        cand = m.group(1).strip().lower()
        if not _looks_like_not_a_color(cand):
            return _COLOR_MAP.get(cand, cand.title())
    for w in sorted(_COLOR_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", name, flags=re.I):
            if not _looks_like_not_a_color(w):
                return _COLOR_MAP.get(w, w.title())
    return ""

# =========================
# Parsing – Preislisten
# =========================
PRICE_COL_CANDIDATES = ["Preis","VK","Netto","NETTO","Einkaufspreis","Verkaufspreis","NETTO NETTO","Einkauf"]
BUY_PRICE_CANDIDATES  = ["Einkaufspreis","Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis","VK","Preis"]

ARTNR_CANDIDATES = ["Artikelnummer","Artikelnr","ArtikelNr","Artikel-Nr.","Hersteller-Nr.","Produkt ID","ProdNr","ArtNr","ArtikelNr.","Artikel"]
EAN_CANDIDATES  = ["EAN","GTIN","BarCode","Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung","Produktname","Name","Titel","Artikelname"]
CAT_CANDIDATES  = ["Kategorie","Warengruppe","Zusatz"]
STOCK_CANDIDATES= ["Bestand","Verfügbar","verfügbar","Verfuegbar","Lagerbestand","Lagermenge","Available"]
COLOR_CANDIDATES= ["Farbe","Color","Colour","Variante","Variant","Farbvariante","Farbname"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES,  "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")

    # Kategorie primär aus Spalte G (Index 6)
    col_cat = None
    try:
        maybe_g = df.columns[6]
        if maybe_g in df.columns:
            col_cat = maybe_g
    except Exception:
        pass
    if not col_cat:
        col_cat = find_column(df, CAT_CANDIDATES, "Kategorie", required=False)

    col_stock = find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_color = find_column(df, COLOR_CANDIDATES, "Farbe/Variante", required=False)
    col_any=None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str)
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","",str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Familie"]         = out["Bezeichnung"].map(make_family_key)

    # Kategorie bereinigen (leer statt NaN/None)
    if col_cat:
        out["Kategorie"] = (
            df[col_cat].astype(str)
            .replace({"nan":"", "NaN":"", "None":""})
            .str.strip()
        )
    else:
        out["Kategorie"] = ""

    # Farbe
    if col_color:
        out["Farbe"] = df[col_color].astype(str).map(lambda v: _COLOR_MAP.get(str(v).lower(), str(v)))
    else:
        out["Farbe"] = out["Bezeichnung"].map(extract_color_from_name)
    out["Farbe"] = out["Farbe"].fillna("").astype(str)

    out["Lagermenge"] = parse_number_series(df[col_stock]).fillna(0).astype("Int64") if col_stock else pd.Series([0]*len(out), dtype="Int64")
    if col_buy:  out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell: out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        anyp = parse_number_series(df[col_any]); out["Einkaufspreis"]=anyp; out["Verkaufspreis"]=anyp
    if "Einkaufspreis" not in out: out["Einkaufspreis"]=out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out: out["Verkaufspreis"]=out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))

    # Dedupliziere nach ArtikelNr_key (bevorzuge mit Preis)
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

# Einfache Äquivalenzen / Regeln (Beispiele)
ART_EXACT_EQUIV  = {"e008":"e009","j031":"j030","m057":"m051","s054":"s054"}
ART_PREFIX_EQUIV = {"o061":"o061","o013":"o013"}

def _apply_hints_to_row(name_raw: str) -> dict:
    s = (name_raw or "").lower()
    h = {"hint_family":"","hint_color":"","hint_art_exact":"","hint_art_prefix":""}
    for fam in ["finn mobile","charly little","duftöl","duftoel","duft oil"]:
        if fam in s: h["hint_family"] = "finn" if fam=="finn mobile" else ("charly" if "charly" in fam else "duftol")
    for fam in ["finn","theo","robert","peter","julia","albert","roger","mia","simon","otto","oskar","tim","charly"]:
        if fam in s: h["hint_family"] = h["hint_family"] or fam
    if "tim" in s and "schwarz" in s: h["hint_color"]="weiss"
    if "mia" in s and "gold" in s:    h["hint_color"]="schwarz"
    if "oskar" in s and "little" in s: h["hint_art_prefix"]="o061"
    if "simon" in s: h["hint_art_exact"]="s054"
    if "otto"  in s: h["hint_art_prefix"]="o013"
    if "eva" in s and "e-008" in s: h["hint_art_exact"]="e008"
    if "julia" in s and "j-031" in s: h["hint_art_exact"]="j031"
    if "mia" in s and "m-057" in s: h["hint_art_exact"]="m057"
    return h

def _fallback_col_by_index(df: pd.DataFrame, idx0: int) -> str|None:
    try: return df.columns[idx0]
    except: return None

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES,   "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES,     "EAN/GTIN",      required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung",   required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)

    # Lager aus Sell-out (G = Index 6) – Header- oder Positions-Fallback
    col_stock_so = find_column(df, STOCK_SO_CANDIDATES, "Lagermenge (Sell-out)", required=False)
    if not col_stock_so and df.shape[1] >= 7:
        col_stock_so = _fallback_col_by_index(df, 6)  # Spalte G

    # Datumsfelder (I/J)
    col_start = find_column(df, DATE_START_CANDS, "Startdatum (Spalte I)", required=False)
    col_end   = find_column(df, DATE_END_CANDS,   "Enddatum (Spalte J)",   required=False)
    if not col_start and df.shape[1]>=9:  col_start=_fallback_col_by_index(df,8)   # Spalte I
    if not col_end   and df.shape[1]>=10: col_end  =_fallback_col_by_index(df,9)   # Spalte J

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","",str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Familie"]         = out["Bezeichnung"].map(make_family_key)

    hints = out["Bezeichnung"].map(_apply_hints_to_row)
    out["Hint_Family"]   = hints.map(lambda h: h["hint_family"])
    out["Hint_Color"]    = hints.map(lambda h: h["hint_color"])
    out["Hint_ArtExact"] = hints.map(lambda h: h["hint_art_exact"])
    out["Hint_ArtPref"]  = hints.map(lambda h: h["hint_art_prefix"])

    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"] = parse_number_series(df[col_buy]).fillna(0).astype("Int64") if col_buy else pd.Series([0]*len(df), dtype="Int64")

    # Lagermenge aus Sell-out
    if col_stock_so:
        # 0 ist ein gültiger gemeldeter Bestand
        out["SellLagermenge"] = pd.to_numeric(df[col_stock_so], errors="coerce")

    if col_start: out["StartDatum"] = parse_date_series_us(df[col_start])
    if col_end:   out["EndDatum"]   = parse_date_series_us(df[col_end])
    if "StartDatum" in out and "EndDatum" in out:
        out.loc[out["EndDatum"].isna(),"EndDatum"] = out.loc[out["EndDatum"].isna(),"StartDatum"]
    return out

# =========================
# Matching-Backstops
# =========================
def _assign_from_price_row(merged: pd.DataFrame, i, row: pd.Series):
    for col in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]:
        merged.at[i, col] = row.get(col, merged.at[i, col])

def _token_set(s: str) -> set:
    s = _strip_parens_units(s.lower())
    s = re.sub(r"[^a-z0-9]+"," ", s)
    toks = [t for t in s.split() if t and (t not in _STOP_TOKENS) and (t not in _COLOR_WORDS)]
    return set(toks)

def _best_fuzzy_in_candidates(name: str, cand_series: pd.Series) -> int|None:
    base = _token_set(name)
    if not len(base): return None
    best_idx, best_score = None, 0.0
    for idx, val in cand_series.items():
        cand = _token_set(str(val))
        if not cand: continue
        inter = len(base & cand); union = len(base | cand)
        score = inter/union if union else 0.0
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx if best_score >= 0.5 else None

def _family_match(row: pd.Series, price_df: pd.DataFrame, prefer_color: str|None):
    fam = row.get("Hint_Family") or row.get("Familie") or ""
    fam = fam.strip()
    if not fam: return None
    grp = price_df.loc[price_df["Familie"]==fam]
    if grp.empty: grp = price_df.loc[price_df["Familie"].str.contains(re.escape(fam), na=False)]
    if grp.empty: return None
    if prefer_color:
        g2 = grp.loc[grp["Farbe"].str.lower()==prefer_color.lower()]
        if not g2.empty: grp = g2
    return grp.iloc[0]

def _apply_equivalences(hint_art_exact: str, hint_art_pref: str) -> str|None:
    if hint_art_exact:
        return ART_EXACT_EQUIV.get(hint_art_exact.lower(), hint_art_exact.lower())
    if hint_art_pref:
        p = hint_art_pref.lower()
        return ART_PREFIX_EQUIV.get(p, p)
    return None

def _final_backstops(merged: pd.DataFrame, price_df: pd.DataFrame):
    need = merged["Verkaufspreis"].isna()
    if not need.any(): return
    for i in merged.index[need]:
        art_key = _apply_equivalences(str(merged.at[i,"Hint_ArtExact"] or ""), str(merged.at[i,"Hint_ArtPref"] or ""))
        if art_key:
            hit = price_df.loc[price_df["ArtikelNr_key"].str.startswith(art_key, na=False)]
            if not hit.empty:
                _assign_from_price_row(merged,i, hit.iloc[0]); continue
        pref_color = str(merged.at[i,"Hint_Color"] or "")
        hit = _family_match(merged.loc[i], price_df, pref_color if pref_color else None)
        if hit is not None:
            _assign_from_price_row(merged,i, hit); continue
        idx = _best_fuzzy_in_candidates(str(merged.at[i,"Bezeichnung"]), price_df["Bezeichnung"])
        if idx is not None:
            _assign_from_price_row(merged,i, price_df.loc[idx]); continue

# =========================
# Merge & Werte (+ Quelle für Chart)
# =========================
@st.cache_data(show_spinner=False)
def enrich_and_merge(filtered_sell_df: pd.DataFrame, price_df: pd.DataFrame, latest_stock_baseline_df: pd.DataFrame|None=None):
    if filtered_sell_df is None or price_df is None or filtered_sell_df.empty or price_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sell_for_stock = latest_stock_baseline_df if latest_stock_baseline_df is not None else filtered_sell_df

    # Merge für Umsatz
    merged = filtered_sell_df.merge(price_df, on=["ArtikelNr_key"], how="left", suffixes=("", "_pl"))

    # Merge für Lagerstand (UNGEFILTERTE Basis)
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

    # Fallback-Matches (nur auf Umsatz-Merge)
    need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if need.any():
        tmp = merged.loc[need, ["EAN_key"]].merge(
            price_df[["EAN_key","Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]],
            on="EAN_key", how="left"
        )
        idx = merged.index[need]; tmp.index = idx
        for c in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]:
            merged.loc[idx,c] = merged.loc[idx,c].fillna(tmp[c])
    need = merged["Verkaufspreis"].isna()
    if need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        for i,k in zip(merged.index[need], merged.loc[need,"Bezeichnung_key"]):
            if k in name_map.index: _assign_from_price_row(merged,i, name_map.loc[k])
    need = merged["Verkaufspreis"].isna()
    if need.any():
        fam_map = price_df.drop_duplicates("Familie").set_index("Familie")
        for i,f in zip(merged.index[need], merged.loc[need,"Familie"]):
            if f and f in fam_map.index: _assign_from_price_row(merged,i, fam_map.loc[f])
    _final_backstops(merged, price_df)

    # Strings säubern
    for df in (merged, stock_merged):
        df["Kategorie"] = (
            df["Kategorie"].fillna("")
              .astype(str)
              .replace({"nan":"", "NaN":"", "None":""})
              .str.strip()
        )
        df["Bezeichnung"] = df["Bezeichnung"].fillna("")
        df["Farbe"]       = df.get("Farbe","").fillna("")

    # Anzeige-Bezeichnung (bei Duplikaten Farbe anhängen)
    merged["Bezeichnung_anzeige"] = merged["Bezeichnung"]
    def _looks_like_not_a_color(token: str) -> bool:
        t=(token or "").strip().lower()
        return (not t) or (t in {"eu","ch","us","uk"}) or any(x in t for x in ["ml","db","m²","m2"]) or bool(re.search(r"\d",t))
    dup = merged.duplicated(subset=["Bezeichnung"], keep=False)
    valid_color = merged["Farbe"].astype(str).str.strip().map(lambda t: (t!="") and (not _looks_like_not_a_color(t)))
    merged.loc[dup & valid_color, "Bezeichnung_anzeige"] = merged.loc[dup & valid_color,"Bezeichnung"] + " – " + merged.loc[dup & valid_color,"Farbe"].astype(str).str.strip()

    # Umsatz-Werte
    q_buy,p_buy   = sanitize_numbers(merged["Einkaufsmenge"], merged["Einkaufspreis"])
    q_sell,p_sell = sanitize_numbers(merged["Verkaufsmenge"], merged["Verkaufspreis"])
    merged["Einkaufswert"] = safe_mul(q_buy.fillna(0.0),  p_buy.fillna(0.0))
    merged["Verkaufswert"] = safe_mul(q_sell.fillna(0.0), p_sell.fillna(0.0))

    # Aktuellster Lagerstand aus Sell-out
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

    # Tabellen
    detail = merged[["ArtikelNr","Bezeichnung_anzeige","Kategorie",
                     "Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert"]].copy()
    detail["Lagermenge"] = merged["Lagermenge_latest"]
    detail["Lagerwert"]  = merged["Lagerwert_latest"]

    totals = (detail.groupby(["ArtikelNr","Bezeichnung_anzeige","Kategorie"], dropna=False, as_index=False)
                  .agg({
                      "Einkaufsmenge":"sum",
                      "Einkaufswert":"sum",
                      "Verkaufsmenge":"sum",
                      "Verkaufswert":"sum",
                      "Lagermenge":"max",
                      "Lagerwert":"max"
                  }))

    ts_source = pd.DataFrame()
    if "StartDatum" in merged.columns:
        ts_source = merged[["StartDatum","Kategorie","Verkaufswert"]].copy()
        # Leere Kategorien nicht anzeigen
        ts_source["Kategorie"] = ts_source["Kategorie"].fillna("").astype(str).str.strip()
        ts_source = ts_source[ts_source["Kategorie"] != ""]
        ts_source.rename(columns={"Verkaufswert":"Verkaufswert (CHF)"}, inplace=True)

    merged.drop(columns=["_rowdate","_grpkey"], errors="ignore", inplace=True)
    return detail, totals, ts_source

# =========================
# Datenquellen / Persistieren
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" if (BASE_DIR.name == "galaxus_tool_source") else BASE_DIR / "data"
DEFAULT_SELL_PATH  = DATA_DIR / "sellout.xlsx"
DEFAULT_PRICE_PATH = DATA_DIR / "preisliste.xlsx"
DATA_DIR.mkdir(exist_ok=True, parents=True)

def _persist_upload(uploaded_file, target_path: Path):
    if uploaded_file is None: return
    try:
        content = uploaded_file.getvalue()
    except Exception:
        content = uploaded_file.read()
    with open(target_path, "wb") as f:
        f.write(content)

# =========================
# UI
# =========================
st.title("Galaxus Sellout Analyse")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="sell")
with c2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="price")

# Auto-Load + Fallback
raw_sell = None
raw_price = None
if sell_file is not None:
    raw_sell = read_excel_flat(sell_file)
    st.session_state["sell_last"]  = {"name": sell_file.name}
    _persist_upload(sell_file, DEFAULT_SELL_PATH)
elif DEFAULT_SELL_PATH.exists():
    with open(DEFAULT_SELL_PATH, "rb") as f:
        raw_sell = read_excel_flat(io.BytesIO(f.read()))
    st.session_state["sell_last"]  = {"name": DEFAULT_SELL_PATH.name}

if price_file is not None:
    raw_price = read_excel_flat(price_file)
    st.session_state["price_last"] = {"name": price_file.name}
    _persist_upload(price_file, DEFAULT_PRICE_PATH)
elif DEFAULT_PRICE_PATH.exists():
    with open(DEFAULT_PRICE_PATH, "rb") as f:
        raw_price = read_excel_flat(io.BytesIO(f.read()))
    st.session_state["price_last"] = {"name": DEFAULT_PRICE_PATH.name}

# Verarbeitung
if (raw_sell is not None) and (raw_price is not None):
    try:
        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)

        # Zeitraumfilter
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
                    "Zeitraum (DD.MM.YYYY)",
                    value=st.session_state["date_range"],
                    min_value=min_date,
                    max_value=max_date,
                    format="DD.MM.YYYY",
                )
            with col_btn:
                st.write(""); st.write("")
                if st.button("Gesamten Zeitraum"):
                    st.session_state["date_range"] = (min_date, max_date)
                    st.experimental_rerun()

            if isinstance(date_value, tuple):
                start_date, end_date = date_value
            else:
                start_date = end_date = date_value

            st.session_state["date_range"] = (start_date, end_date)

            mask = ~((sell_df["EndDatum"].dt.date < start_date) |
                     (sell_df["StartDatum"].dt.date > end_date))
            filtered_sell_df = sell_df.loc[mask].copy()

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals, ts_source = enrich_and_merge(filtered_sell_df, price_df, latest_stock_baseline_df=sell_df)

        used_sell = st.session_state.get("sell_last", {}).get("name", "sellout.xlsx")
        used_price = st.session_state.get("price_last", {}).get("name", "preisliste.xlsx")
        st.info(f"Verwendete Dateien: {used_sell} / {used_price}")

        # ------- Chart -------
        st.markdown("### 📈 Verkaufsverlauf nach Kategorie (Woche)")
        if not ts_source.empty:
            ts = ts_source.dropna(subset=["StartDatum"]).copy()
            ts["Periode"] = ts["StartDatum"].dt.to_period("W").dt.start_time
            ts["Kategorie"] = ts["Kategorie"].astype("string")

            # Filter-Auswahl
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
            chart = (lines + points + popup + end_labels).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Für den Verlauf werden gültige Startdaten und nicht-leere Kategorien benötigt.")

        # ------- Tabellen -------
        show_detail = st.checkbox("Detailtabelle anzeigen", value=False)
        if show_detail:
            st.subheader("Detailtabelle")
            detail_renamed = detail.rename(columns={
                "Einkaufswert":"Einkaufswert (CHF)",
                "Verkaufswert":"Verkaufswert (CHF)",
                "Lagerwert":"Lagerwert (CHF)"
            })
            detail_display = append_total_row_for_display(detail_renamed)
            d_rounded, d_styler = style_numeric(detail_display)
            st.dataframe(d_styler, use_container_width=True)

        st.subheader("Summen pro Artikel")
        totals_renamed = totals.rename(columns={
            "Einkaufswert":"Einkaufswert (CHF)",
            "Verkaufswert":"Verkaufswert (CHF)",
            "Lagerwert":"Lagerwert (CHF)"
        })
        totals_display = append_total_row_for_display(totals_renamed)
        t_rounded, t_styler = style_numeric(totals_display)
        st.dataframe(t_styler, use_container_width=True)

        # ------- Downloads (ohne Σ-Gesamtzeile) -------
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
    st.info("Bitte beide Dateien hochladen oder in den Ordner `data/` legen (Sell-out = `sellout.xlsx`, Preisliste = `preisliste.xlsx`).")
