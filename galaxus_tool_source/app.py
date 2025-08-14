# app.py — Galaxus Sellout Analyse
# Robustes Matching (ArtNr → EAN → Name → Familie → Hints → Fuzzy),
# EU‑Datumsfilter, Detailtabelle optional, Summen pro Artikel,
# EIN Linienchart (eine Linie je Kategorie) – Verkaufswert-Verlauf (Monat),
# Overflow‑Fix.

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Galaxus Sellout Analyse", layout="wide")

# Altair: große Datensätze zulassen (optional robust)
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

# =========================
# Anzeige-Helfer
# =========================
THOUSANDS_SEP = "'"
NUM_COLS_DEFAULT = [
    "Einkaufsmenge","Einkaufswert",
    "Verkaufsmenge","Verkaufswert",
    "Lagermenge","Lagerwert"
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

# =========================
# Robust: Excel einlesen (tolerant ggü. Kopfzeilen)
# =========================
def read_excel_flat(upload) -> pd.DataFrame:
    raw = pd.read_excel(upload, header=None, dtype=object)
    if raw.empty: return pd.DataFrame()
    header_idx = int(raw.notna().mean(axis=1).idxmax())
    headers = raw.iloc[header_idx].fillna("").astype(str).tolist()
    headers = [re.sub(r"\s+"," ",h).strip() for h in headers]
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
            seen[c]+=1; newcols.append(f"{c}.{seen[c]}")
        else:
            seen[c]=0; newcols.append(c)
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
        x=str(x).strip().replace("’","").replace("'","").replace(" ","").replace(",",".")
        if x.count(".")>1:
            parts=x.split("."); x="".join(parts[:-1])+"."+parts[-1]
        try: return float(x)
        except Exception: return np.nan
    return s.map(_clean)

def parse_date_series_us(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64): return s
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    nums = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(nums, origin="1899-12-30", unit="d", errors="coerce")
    return dt1.combine_first(dt2)

# Obergrenzen (Overflow-Fix)
MAX_QTY, MAX_PRICE = 1_000_000, 1_000_000
def sanitize_numbers(qty: pd.Series, price: pd.Series) -> tuple[pd.Series,pd.Series]:
    q = pd.to_numeric(qty, errors="coerce").astype("float64").clip(lower=0, upper=MAX_QTY)
    p = pd.to_numeric(price, errors="coerce").astype("float64").clip(lower=0, upper=MAX_PRICE)
    return q, p

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
    t=(token or "").strip().lower()
    return (not t) or (t in {"eu","ch","us","uk"}) or any(x in t for x in ["ml","db","m²","m2"]) or bool(re.search(r"\d",t))

def _strip_parens_units(name: str) -> str:
    s = re.sub(r"\([^)]*\)"," ", name)
    s = re.sub(r"\b\d+([.,]\d+)?\s*(ml|db|m²|m2)\b"," ", s, flags=re.I)
    return s

def make_family_key(name: str) -> str:
    if not isinstance(name,str): return ""
    s = _strip_parens_units(name.lower())
    s = re.sub(r"\b[o0]-\d+\b"," ", s)   # Codes wie O-061 etc. ausblenden
    s = re.sub(r"[^a-z0-9]+"," ", s)
    toks = [t for t in s.split() if t and (t not in _STOP_TOKENS) and (t not in _COLOR_WORDS)]
    return "".join(toks[:2]) if toks else ""

def extract_color_from_name(name: str) -> str:
    if not isinstance(name,str): return ""
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
STOCK_CANDIDATES= ["Bestand","Verfügbar","Lagerbestand"]
COLOR_CANDIDATES= ["Farbe","Color","Colour","Variante","Variant","Farbvariante","Farbname"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES,  "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat   = find_column(df, CAT_CANDIDATES,  "Kategorie", required=False)
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
    out["Kategorie"]       = df[col_cat].astype(str) if col_cat else ""

    # Farbe falls vorhanden, sonst aus Name extrahieren
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

# Einfache Äquivalenzen / Regeln
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
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame):
    merged = sell_df.merge(price_df, on=["ArtikelNr_key"], how="left", suffixes=("", "_pl"))

    # Fallback per EAN
    need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if need.any():
        tmp = merged.loc[need, ["EAN_key"]].merge(
            price_df[["EAN_key","Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]],
            on="EAN_key", how="left"
        )
        idx = merged.index[need]; tmp.index = idx
        for c in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Familie","Farbe","Kategorie","ArtikelNr","ArtikelNr_key"]:
            merged.loc[idx,c] = merged.loc[idx,c].fillna(tmp[c])

    # Fallback per Bezeichnung_key
    need = merged["Verkaufspreis"].isna()
    if need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        for i,k in zip(merged.index[need], merged.loc[need,"Bezeichnung_key"]):
            if k in name_map.index: _assign_from_price_row(merged,i, name_map.loc[k])

    # Fallback per Familie
    need = merged["Verkaufspreis"].isna()
    if need.any():
        fam_map = price_df.drop_duplicates("Familie").set_index("Familie")
        for i,f in zip(merged.index[need], merged.loc[need,"Familie"]):
            if f and f in fam_map.index: _assign_from_price_row(merged,i, fam_map.loc[f])

    # Backstops (Äquivalenzen, Familie, Fuzzy)
    _final_backstops(merged, price_df)

    # Strings & Anzeige
    merged["Kategorie"]   = merged["Kategorie"].fillna("")
    merged["Bezeichnung"] = merged["Bezeichnung"].fillna("")
    merged["Farbe"]       = merged.get("Farbe","").fillna("")
    merged["Bezeichnung_anzeige"] = merged["Bezeichnung"]

    dup = merged.duplicated(subset=["Bezeichnung"], keep=False)
    valid_color = merged["Farbe"].astype(str).str.strip().map(lambda t: (t!="") and (not _looks_like_not_a_color(t)))
    merged.loc[dup & valid_color, "Bezeichnung_anzeige"] = merged.loc[dup & valid_color,"Bezeichnung"] + " – " + merged.loc[dup & valid_color,"Farbe"].astype(str).str.strip()

    # Werte berechnen (Overflow-sicher)
    q_buy,p_buy   = sanitize_numbers(merged["Einkaufsmenge"], merged["Einkaufspreis"])
    q_sell,p_sell = sanitize_numbers(merged["Verkaufsmenge"], merged["Verkaufspreis"])
    q_stock,_     = sanitize_numbers(merged["Lagermenge"],  merged["Verkaufspreis"])
    q_buy=q_buy.fillna(0.0); p_buy=p_buy.fillna(0.0)
    q_sell=q_sell.fillna(0.0); p_sell=p_sell.fillna(0.0)
    q_stock=q_stock.fillna(0.0)
    with np.errstate(over='ignore', invalid='ignore'):
        merged["Einkaufswert"] = (q_buy*p_buy).astype("float64")
        merged["Verkaufswert"] = (q_sell*p_sell).astype("float64")
        merged["Lagerwert"]    = (q_stock*p_sell).astype("float64")

    # Tabellen
    display_cols = [c for c in ["ArtikelNr","Bezeichnung_anzeige","Kategorie","Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert","Lagermenge","Lagerwert"] if c in merged.columns]
    detail = merged[display_cols].copy()
    totals = (detail.groupby(["ArtikelNr","Bezeichnung_anzeige","Kategorie"], dropna=False, as_index=False)
                   .agg({"Einkaufsmenge":"sum","Einkaufswert":"sum","Verkaufsmenge":"sum","Verkaufswert":"sum","Lagermenge":"sum","Lagerwert":"sum"}))

    # Zeitquelle für das Linien-Diagramm
    ts_source = pd.DataFrame()
    if "StartDatum" in merged.columns:
        ts_source = merged[["StartDatum","Kategorie","Verkaufswert"]].copy()
        ts_source["Kategorie"] = ts_source["Kategorie"].replace({"": "— ohne Kategorie —"})
    return detail, totals, ts_source

# =========================
# UI
# =========================
st.title("📊 Galaxus Sellout Analyse")
st.caption("Summenansicht, robustes Matching (ArtNr → EAN → Name → Familie → Hints → Fuzzy), EU‑Datumsfilter. Detailtabelle optional. Linien‑Überblick pro Kategorie.")

c1,c2 = st.columns(2)
with c1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="sell")
    if "sell_last" in st.session_state and st.session_state["sell_last"]:
        st.text(f"Letzter Sell-out: {st.session_state['sell_last']['name']}")
with c2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="price")
    if "price_last" in st.session_state and st.session_state["price_last"]:
        st.text(f"Letzte Preisliste: {st.session_state['price_last']['name']}")

if sell_file and price_file:
    try:
        st.session_state["sell_last"]  = {"name": sell_file.name}
        st.session_state["price_last"] = {"name": price_file.name}
        raw_sell  = read_excel_flat(sell_file)
        raw_price = read_excel_flat(price_file)

        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)

        # ========= Zeitraumfilter mit Button „Gesamten Zeitraum“ =========
        filtered_sell_df = sell_df
        if {"StartDatum","EndDatum"}.issubset(sell_df.columns) and not sell_df["StartDatum"].isna().all():
            st.subheader("Periode wählen")
            min_date = sell_df["StartDatum"].min().date()
            max_date = (sell_df["EndDatum"].dropna().max()
                        if "EndDatum" in sell_df else sell_df["StartDatum"].max()).date()

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
        # =================================================================

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals, ts_source = enrich_and_merge(filtered_sell_df, price_df)

        # =============== EIN LINIEN-CHART (eine Linie je Kategorie) ===============
        st.markdown("### 📈 Verkaufsverlauf nach Kategorie")
        if not ts_source.empty:
            ts = ts_source.dropna(subset=["StartDatum"]).copy()
            ts["Periode"] = ts["StartDatum"].dt.to_period("M").dt.start_time  # für Woche: .dt.to_period("W")
            ts = (ts.groupby(["Kategorie","Periode"], as_index=False)["Verkaufswert"]
                    .sum()
                    .rename(columns={"Verkaufswert":"Wert"}))
            ts["Periode"]   = pd.to_datetime(ts["Periode"])
            ts["Kategorie"] = ts["Kategorie"].astype(str)
            ts["Wert"]      = pd.to_numeric(ts["Wert"], errors="coerce").fillna(0.0).astype(float)

            chart = (
                alt.Chart(ts)
                  .mark_line(point=True)
                  .encode(
                      x=alt.X(field="Periode", type="temporal", title="Periode (Monat)"),
                      y=alt.Y(field="Wert", type="quantitative", title="Verkaufswert"),
                      color=alt.Color(field="Kategorie", type="nominal", title="Kategorie"),
                      tooltip=[
                          alt.Tooltip(field="Periode", type="temporal", title="Periode"),
                          alt.Tooltip(field="Kategorie", type="nominal", title="Kategorie"),
                          alt.Tooltip(field="Wert", type="quantitative", title="Verkaufswert", format=",.0f"),
                      ],
                  )
                  .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Für den Verlauf werden gültige Startdaten benötigt.")

        # =============== Tabellen ===============
        show_detail = st.checkbox("Detailtabelle anzeigen", value=False)
        if show_detail:
            st.subheader("Detailtabelle")
            d_rounded, d_styler = style_numeric(detail)
            st.dataframe(d_styler, use_container_width=True)

        st.subheader("Summen pro Artikel (Varianten: Farbe bei Dubletten)")
        t_rounded, t_styler = style_numeric(totals)
        st.dataframe(t_styler, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Detail (CSV)",
                data=(detail if show_detail else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
                file_name="detail.csv", mime="text/csv", disabled=not show_detail
            )
        with dl2:
            st.download_button(
                "⬇️ Summen (CSV)",
                data=t_rounded.to_csv(index=False).encode("utf-8"),
                file_name="summen.csv", mime="text/csv"
            )

    except KeyError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
else:
    st.info("Bitte beide Dateien hochladen (Sell-out & Preisliste).")
