# app.py — Galaxus Sell‑out Aggregator
# Robustes Matching: ArtNr → EAN → Bezeichnung → Kurzname → Hints (deine Regeln)
# EU‑Datumsfilter (US->EU), Detailtabelle optional, Varianten: Farbe nur bei Dubletten
# Overflow-sicher (float64 + Sanity-Clean), Header-/Scope-Fixes

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Galaxus Sell‑out Aggregator", layout="wide")

# =========================
# Anzeige-Helfer
# =========================
THOUSANDS_SEP = "'"
NUM_COLS_DEFAULT = [
    "Einkaufsmenge", "Einkaufswert",
    "Verkaufsmenge", "Verkaufswert",
    "Lagermenge",   "Lagerwert",
]

def _fmt_thousands(x, sep=THOUSANDS_SEP):
    if pd.isna(x):
        return ""
    try:
        return f"{int(round(float(x))):,}".replace(",", sep)
    except Exception:
        return str(x)

def style_numeric(df: pd.DataFrame, num_cols=NUM_COLS_DEFAULT, sep=THOUSANDS_SEP):
    out = df.copy()
    # FIX: kein Scope-Fehler; keine freie Variable in Comprehensions
    for c in [col for col in num_cols if col in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    def _fmt_func(v, s=sep):
        return _fmt_thousands(v, s)
    fmt = {col: _fmt_func for col in num_cols if col in out.columns}
    styler = out.style.format(fmt)
    return out, styler

# =========================
# Robust: Excel einlesen (verhindert "Length mismatch")
# =========================
def read_excel_flat(upload) -> pd.DataFrame:
    raw = pd.read_excel(upload, header=None, dtype=object)
    if raw.empty:
        return pd.DataFrame()
    non_empty_ratio = raw.notna().mean(axis=1)
    header_idx = int(non_empty_ratio.idxmax())

    headers = raw.iloc[header_idx].fillna("").astype(str).tolist()
    headers = [re.sub(r"\s+", " ", h).strip() for h in headers]
    n = raw.shape[1]
    if len(headers) < n:
        headers += [f"col_{i}" for i in range(len(headers), n)]
    else:
        headers = headers[:n]

    df = raw.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = headers

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

# =========================
# Utilities
# =========================
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
    for cand in candidates:
        if cand in cols:
            return cand
    canon = {re.sub(r"[\s\-_/\.]+", "", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-_/\.]+", "", cand).lower()
        if key in canon:
            return canon[key]
    if required:
        raise KeyError(f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}")
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i", "u", "f"):
        return s
    def _clean(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        x = x.replace("’", "").replace("'", "").replace(" ", "")
        x = x.replace(",", ".")
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

# Sanity-Clean vor Multiplikation
MAX_QTY   = 1_000_000
MAX_PRICE = 1_000_000
def sanitize_numbers(qty: pd.Series, price: pd.Series) -> tuple[pd.Series, pd.Series]:
    q = pd.to_numeric(qty, errors="coerce").astype("float64")
    p = pd.to_numeric(price, errors="coerce").astype("float64")
    q = q.where((q >= 0) & (q <= MAX_QTY))
    p = p.where((p >= 0) & (p <= MAX_PRICE))
    return q, p

# =========================
# Farben (Erkennung + Normierung)
# =========================
_COLOR_MAP = {
    "weiss":"Weiss","weiß":"Weiss","white":"White","offwhite":"Off-White","cream":"Cream","ivory":"Ivory",
    "schwarz":"Schwarz","black":"Black","graphite":"Graphit","gray":"Grau","grau":"Grau",
    "anthrazit":"Anthrazit","charcoal":"Anthrazit","silver":"Silber",
    "blau":"Blau","blue":"Blau","navy":"Dunkelblau","light blue":"Hellblau","dark blue":"Dunkelblau","sky blue":"Hellblau",
    "rot":"Rot","red":"Rot","bordeaux":"Bordeaux","burgundy":"Bordeaux","pink":"Pink","magenta":"Magenta",
    "lila":"Lila","violett":"Violett","purple":"Violett","fuchsia":"Fuchsia",
    "grün":"Grün","gruen":"Grün","green":"Grün","mint":"Mint","türkis":"Türkis","tuerkis":"Türkis","turquoise":"Türkis",
    "petrol":"Petrol","olive":"Olivgrün","gelb":"Gelb","yellow":"Gelb","orange":"Orange",
    "braun":"Braun","brown":"Braun","beige":"Beige","sand":"Sand",
    "gold":"Gold","rose gold":"Roségold","rosegold":"Roségold","kupfer":"Kupfer","copper":"Kupfer","bronze":"Bronze",
    "transparent":"Transparent","clear":"Transparent",
}
_COLOR_KEYS_SORTED = sorted(_COLOR_MAP.keys(), key=len, reverse=True)
_COLOR_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, _COLOR_KEYS_SORTED)) + r")\b", re.IGNORECASE)
_COLOR_REGEXES = [r"\(([^)]+)\)$", r"[-–—]\s*([A-Za-zäöüÄÖÜß]+)$", r"/\s*([A-Za-zäöüÄÖÜß]+)$"]

def _canon_color_from_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    m = _COLOR_PATTERN.search(text.lower())
    return _COLOR_MAP.get(m.group(1).lower(), "") if m else ""

def _looks_like_not_a_color(token: str) -> bool:
    t = (token or "").strip().lower()
    return (not t) or (t in {"eu","ch","us","uk"}) or any(x in t for x in ["ml","db","m²","m2"]) or bool(re.search(r"\d", t))

def extract_color_from_name(name: str) -> str:
    if not isinstance(name, str): return ""
    for rgx in _COLOR_REGEXES:
        m = re.search(rgx, name.strip())
        if m:
            cand = m.group(1).strip()
            if not _looks_like_not_a_color(cand):
                canon = _canon_color_from_text(cand)
                return canon or cand
    canon = _canon_color_from_text(name)
    return canon if canon and not _looks_like_not_a_color(canon) else ""

def guess_color_from_row(row: pd.Series, all_columns: list[str]) -> str:
    for col in all_columns:
        if re.search(r"(farb|color|colour|varian)", col, re.IGNORECASE):
            canon = _canon_color_from_text(str(row.get(col, "")))
            if canon and not _looks_like_not_a_color(canon):
                return canon
    for col in all_columns:
        if col.lower() in {"ean","gtin","barcode","artikelnummer","artikelnr","artnr","produkt id"}:
            continue
        val = str(row.get(col, "") or "").strip()
        if not val or re.fullmatch(r"[0-9 .,'’-]+", val):
            continue
        c = extract_color_from_name(val) or _canon_color_from_text(val)
        if c and not _looks_like_not_a_color(c):
            return c
    return ""

# Kurz-Keys (1–2 Hauptwörter)
_STOP_TOKENS = {"eu","ch","us","uk","mobile","little","bundle"}
def _make_short_key(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\b[o0]-\d+\b", " ", s)
    s = re.sub(r"\b\d+([.,]\d+)?\s*(ml|db|m²|m2)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t and t not in _STOP_TOKENS]
    return "".join(toks[:2]) if toks else ""

# =========================
# Parsing – Preislisten
# =========================
PRICE_COL_CANDIDATES = ["Preis","VK","Netto","NETTO","Einkaufspreis","Verkaufspreis","NETTO NETTO","Einkauf"]
BUY_PRICE_CANDIDATES  = ["Einkaufspreis","Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis","VK","Preis"]

ARTNR_CANDIDATES = [
    "Artikelnummer","Artikelnr","ArtikelNr","Artikel-Nr.",
    "Hersteller-Nr.","Produkt ID","ProdNr","ArtNr","ArtikelNr.","Artikel"
]
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
    col_any = None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str)
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Kurz_key"]        = out["Bezeichnung"].map(_make_short_key)
    out["Kategorie"]       = df[col_cat].astype(str) if col_cat else ""

    if col_color:
        out["Farbe"] = df[col_color].astype(str).map(lambda v: _canon_color_from_text(str(v)) or str(v))
    else:
        out["Farbe"] = out["Bezeichnung"].map(extract_color_from_name)
        if out["Farbe"].isna().any() or (out["Farbe"].astype(str).str.strip() == "").any():
            all_cols = list(df.columns)
            out["Farbe"] = out.apply(
                lambda r: r["Farbe"] if str(r.get("Farbe","")).strip()
                else guess_color_from_row(df.loc[r.name], all_cols),
                axis=1
            )
    out["Farbe"] = out["Farbe"].fillna("").astype(str)

    out["Lagermenge"] = (parse_number_series(df[col_stock]).fillna(0).astype("Int64")
                         if col_stock else pd.Series([0]*len(df), dtype="Int64"))
    if col_buy:
        out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell:
        out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        price_any = parse_number_series(df[col_any])
        out["Einkaufspreis"] = price_any
        out["Verkaufspreis"] = price_any
    if "Einkaufspreis" not in out:
        out["Einkaufspreis"] = out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out:
        out["Verkaufspreis"] = out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))
    return out

# =========================
# Parsing – Sell‑out (+ Hints)
# =========================
NAME_CANDIDATES_SO   = ["Bezeichnung","Name","Artikelname","Bezeichnung_Sales","Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty","Verkauf","Verkaufte Menge","Menge verkauft","Absatz","Stück","Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf","Einkaufsmenge","Menge Einkauf"]
DATE_START_CANDS     = ["Start","Startdatum","Start Date","Anfangs datum","Anfangsdatum","Von","Period Start"]
DATE_END_CANDS       = ["Ende","Enddatum","End Date","Bis","Period End"]

def _fallback_col_by_index(df: pd.DataFrame, idx0: int) -> str | None:
    try:
        return df.columns[idx0]
    except Exception:
        return None

def _apply_hints_to_row(name_raw: str) -> dict:
    """
    Deine Regeln, um Varianten zuverlässig auf einen Preislisteintrag abzubilden.
    Rückgabe-Kandidaten:
      - hint_artnr_exact: exakte ArtikelNr (z.B. 's054')
      - hint_artnr_prefix: ArtikelNr-Präfix (z.B. 'o013' → 'O-013*')
      - hint_kurz: Familienname/Kurzname für Gruppensuche
      - hint_color: gewünschte Farbe (bevorzugte Variante)
    """
    s = (name_raw or "").lower()
    h = {"hint_kurz":"", "hint_color":"", "hint_artnr_exact":"", "hint_artnr_prefix":""}

    # Sonderfälle/Familien laut Vorgabe:
    # Tim schwarz = Tim weiss Preis
    if "tim" in s and "schwarz" in s:
        h["hint_kurz"]  = "tim"
        h["hint_color"] = "weiss"

    # Finn, Finn mobile, Theo, Robert, Peter, Julia, Albert – per Namen matchen
    for fam in ["finn", "theo", "robert", "peter", "julia", "albert"]:
        if fam in s:
            h["hint_kurz"] = fam
    if "finn mobile" in s:
        h["hint_kurz"] = "finn"

    # Roger/Peter/Robert: auf Roger (schwarz) normalisieren
    if any(x in s for x in ["roger", "peter", "robert"]):
        h["hint_kurz"]  = "roger"
        h["hint_color"] = h["hint_color"] or "schwarz"

    # Oskar little → Preis von O-061
    if "oskar" in s and "little" in s:
        h["hint_artnr_prefix"] = "o061"

    # Mia Gold → gleicher Preis wie Schwarz
    if "mia" in s and "gold" in s:
        h["hint_kurz"]  = "mia"
        h["hint_color"] = "schwarz"

    # Simon → S-054, Otto → O‑013
    if "simon" in s: h["hint_artnr_exact"] = "s054"
    if "otto"  in s: h["hint_artnr_prefix"] = "o013"

    # Charly little zusammenführen (selber Artikel)
    if "charly" in s:
        h["hint_kurz"] = "charly"

    # Duftöle: alle gleicher Netto-Netto (familienweit)
    if ("duftöl" in s) or ("duftoel" in s) or ("duft oil" in s):
        h["hint_kurz"] = "duftol"  # vereinheitlichter Kurzkey

    return h

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES,  "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES,    "EAN/GTIN",      required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO,"Bezeichnung",   required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES,"Verkaufsmenge",required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,  "Einkaufsmenge",required=False)

    col_start = find_column(df, DATE_START_CANDS, "Startdatum (Spalte I)", required=False)
    col_end   = find_column(df, DATE_END_CANDS,   "Enddatum (Spalte J)",   required=False)
    if not col_start and df.shape[1] >= 9:  col_start = _fallback_col_by_index(df, 8)
    if not col_end   and df.shape[1] >= 10: col_end   = _fallback_col_by_index(df, 9)

    out = pd.DataFrame()
    out["ArtikelNr"]     = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)
    out["EAN"]           = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]       = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"]   = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"]=out["Bezeichnung"].map(normalize_key)
    out["Kurz_key"]      = out["Bezeichnung"].map(_make_short_key)

    # Hints je Zeile
    hints = out["Bezeichnung"].map(_apply_hints_to_row)
    out["Hint_Kurz"]      = hints.map(lambda h: h["hint_kurz"])
    out["Hint_Color"]     = hints.map(lambda h: h["hint_color"])
    out["Hint_ArtNr"]     = hints.map(lambda h: h["hint_artnr_exact"])
    out["Hint_ArtNr_Pre"] = hints.map(lambda h: h["hint_artnr_prefix"])

    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"] = (parse_number_series(df[col_buy]).fillna(0).astype("Int64")
                            if col_buy else pd.Series([0]*len(df), dtype="Int64"))

    if col_start: out["StartDatum"] = parse_date_series_us(df[col_start])
    if col_end:   out["EndDatum"]   = parse_date_series_us(df[col_end])
    if "StartDatum" in out and "EndDatum" in out:
        mask = out["EndDatum"].isna()
        out.loc[mask, "EndDatum"] = out.loc[mask, "StartDatum"]
    return out

# =========================
# Merge & Berechnung
# =========================
def _assign_from_price_row(merged: pd.DataFrame, i, row: pd.Series):
    for col in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Kurz_key","Farbe","Kategorie","ArtikelNr"]:
        merged.at[i, col] = row.get(col, merged.at[i, col])

def _match_with_hints(merged: pd.DataFrame, price_df: pd.DataFrame):
    need = merged["Verkaufspreis"].isna()
    if not need.any(): return

    idxs = merged.index[need]
    price_by_kurz = price_df.groupby("Kurz_key")

    for i in idxs:
        hint_artnr = str(merged.at[i,"Hint_ArtNr"] or "")
        hint_pref  = str(merged.at[i,"Hint_ArtNr_Pre"] or "")
        hint_kurz  = (merged.at[i,"Hint_Kurz"] or "").strip()
        hint_color = (merged.at[i,"Hint_Color"] or "").strip()

        # 1) Exakte ArtNr
        if hint_artnr:
            ak = normalize_key(hint_artnr)
            hit = price_df.loc[price_df["ArtikelNr_key"] == ak]
            if not hit.empty:
                _assign_from_price_row(merged, i, hit.iloc[0])
                continue

        # 2) Präfix ArtNr
        if hint_pref:
            pref = hint_pref.lower()
            hit = price_df.loc[price_df["ArtikelNr_key"].str.startswith(pref, na=False)]
            if not hit.empty:
                if hint_color:
                    hitc = hit.loc[hit["Farbe"].str.lower() == hint_color.lower()]
                    if not hitc.empty:
                        hit = hitc
                _assign_from_price_row(merged, i, hit.iloc[0])
                continue

        # 3) Kurzname (Familie)
        if hint_kurz:
            grp = price_by_kurz.get_group(hint_kurz) if hint_kurz in price_by_kurz.groups else pd.DataFrame()
            if not grp.empty:
                # Duftöle: alle gleicher Netto – nimm erste Zeile
                if hint_kurz in {"duftol"}:
                    _assign_from_price_row(merged, i, grp.iloc[0])
                    continue
                # Sonst Farbe bevorzugen
                if hint_color:
                    g2 = grp.loc[grp["Farbe"].str.lower() == hint_color.lower()]
                    if not g2.empty:
                        grp = g2
                _assign_from_price_row(merged, i, grp.iloc[0])
                continue

@st.cache_data(show_spinner=False)
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = sell_df.merge(price_df, on=["ArtikelNr_key"], how="left", suffixes=("", "_pl"))

    # 1) EAN-Fallback
    need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if need.any():
        tmp = merged.loc[need, ["EAN_key"]].merge(
            price_df[["EAN_key","Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Kurz_key","Farbe","Kategorie","ArtikelNr"]],
            on="EAN_key", how="left"
        )
        idx = merged.index[need]; tmp.index = idx
        for c in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Kurz_key","Farbe","Kategorie","ArtikelNr"]:
            merged.loc[idx, c] = merged.loc[idx, c].fillna(tmp[c])

    # 2) Name-Fallback
    need = merged["Verkaufspreis"].isna()
    if need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        for i, k in zip(merged.index[need], merged.loc[need, "Bezeichnung_key"]):
            if k in name_map.index:
                _assign_from_price_row(merged, i, name_map.loc[k])

    # 3) Kurzname-Fallback
    need = merged["Verkaufspreis"].isna()
    if need.any():
        short_map = price_df.drop_duplicates("Kurz_key").set_index("Kurz_key")
        for i, k in zip(merged.index[need], merged.loc[need, "Kurz_key"]):
            if k and k in short_map.index:
                _assign_from_price_row(merged, i, short_map.loc[k])

    # 4) Deine Hints
    _match_with_hints(merged, price_df)

    # Strings + Anzeige
    merged["Kategorie"]   = merged["Kategorie"].fillna("")
    merged["Bezeichnung"] = merged["Bezeichnung"].fillna("")
    merged["Farbe"]       = merged.get("Farbe", "").fillna("")
    merged["Bezeichnung_anzeige"] = merged["Bezeichnung"]

    # Farbe nur bei Dubletten anhängen
    dup = merged.duplicated(subset=["Bezeichnung"], keep=False)
    valid_color = merged["Farbe"].astype(str).str.strip().map(lambda t: (t != "") and (not _looks_like_not_a_color(t)))
    merged.loc[dup & valid_color, "Bezeichnung_anzeige"] = (
        merged.loc[dup & valid_color, "Bezeichnung"] + " – " + merged.loc[dup & valid_color, "Farbe"].astype(str).str.strip()
    )

    # sichere Berechnung
    qty_buy,  pr_buy  = sanitize_numbers(merged["Einkaufsmenge"], merged["Einkaufspreis"])
    qty_sell, pr_sell = sanitize_numbers(merged["Verkaufsmenge"], merged["Verkaufspreis"])
    qty_stock, _      = sanitize_numbers(merged["Lagermenge"], merged["Verkaufspreis"])
    qty_buy = qty_buy.fillna(0.0);  pr_buy  = pr_buy.fillna(0.0)
    qty_sell= qty_sell.fillna(0.0); pr_sell = pr_sell.fillna(0.0)
    qty_stock=qty_stock.fillna(0.0)

    with np.errstate(over='ignore', invalid='ignore'):
        merged["Einkaufswert"] = (qty_buy   * pr_buy).astype("float64")
        merged["Verkaufswert"] = (qty_sell  * pr_sell).astype("float64")
        merged["Lagerwert"]    = (qty_stock * pr_sell).astype("float64")

    display_cols = [
        "ArtikelNr", "Bezeichnung_anzeige", "Kategorie",
        "Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert","Lagermenge","Lagerwert"
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    detail = merged[display_cols].copy()

    totals = (
        detail.groupby(["ArtikelNr","Bezeichnung_anzeige","Kategorie"], dropna=False, as_index=False)
              .agg({
                    "Einkaufsmenge":"sum",
                    "Einkaufswert":"sum",
                    "Verkaufsmenge":"sum",
                    "Verkaufswert":"sum",
                    "Lagermenge":"sum",
                    "Lagerwert":"sum"
              })
    )
    return detail, totals

# =========================
# UI
# =========================
st.title("📦 Galaxus Sell‑out Aggregator")
st.caption("Summenansicht, robustes Matching (ArtNr → EAN → Name → Kurzname → Hints), EU‑Datumsfilter. Detailtabelle optional.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag and drop file here", type=["xlsx"], key="sell")
    if "sell_last" in st.session_state and st.session_state["sell_last"]:
        st.text(f"Letzter Sell-out: {st.session_state['sell_last']['name']}")
with col2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag and drop file here", type=["xlsx"], key="price")
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

        # Zeitraumfilter (Start/Ende aus I/J; EU-Formatanzeige)
        filtered_sell_df = sell_df
        if {"StartDatum", "EndDatum"}.issubset(sell_df.columns) and not sell_df["StartDatum"].isna().all():
            min_date = sell_df["StartDatum"].min().date()
            max_date = (sell_df["EndDatum"].dropna().max() if "EndDatum" in sell_df else sell_df["StartDatum"].max()).date()

            st.subheader("Periode wählen")
            date_value = st.date_input("Zeitraum (DD.MM.YYYY)", value=(min_date, max_date),
                                       min_value=min_date, max_value=max_date)
            if isinstance(date_value, tuple):
                start_date, end_date = date_value
            else:
                start_date = end_date = date_value

            overlaps = ~(
                (sell_df["EndDatum"].dt.date < start_date) |
                (sell_df["StartDatum"].dt.date > end_date)
            )
            filtered_sell_df = sell_df.loc[overlaps].copy()

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals = enrich_and_merge(filtered_sell_df, price_df)

        # Detailtabelle optional
        show_detail = st.checkbox("Detailtabelle anzeigen", value=False)
        if show_detail:
            st.subheader("Detailtabelle")
            d_rounded, d_styler = style_numeric(detail)
            st.dataframe(d_styler, use_container_width=True)

        st.subheader("Summen pro Artikel (Varianten: Farbe bei Dubletten)")
        t_rounded, t_styler = style_numeric(totals)
        st.dataframe(t_styler, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Detail (CSV)",
                data=(detail if show_detail else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
                file_name="detail.csv",
                mime="text/csv",
                disabled=not show_detail
            )
        with c2:
            st.download_button(
                "⬇️ Summen (CSV)",
                data=t_rounded.to_csv(index=False).encode("utf-8"),
                file_name="summen.csv",
                mime="text/csv",
            )

    except KeyError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
else:
    st.info("Bitte beide Dateien hochladen (Sell-out & Preisliste).")
