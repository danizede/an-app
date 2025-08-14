# app.py — Galaxus Sell‑out Aggregator
# - Robustes Excel-Header-Handling (kein Length-mismatch)
# - US→EU Datum (Start=Spalte I, Ende=Spalte J), Zeitraumfilter (Überschneidung)
# - Zahlformat (0 Nachkommastellen, ’ als Tausender)
# - Detailtabelle standardmäßig ausgeblendet
# - Varianten: Farbe NUR bei Dubletten an den Namen anhängen (kein EU/Art.-Nr.-Fallback)
# - Bessere Farberkennung: explizite Spalten, Bezeichnung-Heuristik + Spaltenscan mit Synonym-Mapping
# - Overflow-Fix: Berechnungen in float64

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Galaxus Sell‑out Aggregator", layout="wide")

# =========================
# Anzeige-Helfer (Runden + Tausender)
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
    for c in (col for col in num_cols if col in out.columns):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    fmt = {c: (lambda v, s=sep: _fmt_thousands(v, s)) for c in num_cols if c in out.columns}
    styler = out.style.format(fmt)
    return out, styler

# =========================
# Robust: Excel einlesen & Spalten fixieren
# =========================
def read_excel_flat(upload) -> pd.DataFrame:
    """Robustes Einlesen – verhindert 'Length mismatch' auch bei mehrzeiligen Headern."""
    raw = pd.read_excel(upload, header=None, dtype=object)
    if raw.empty:
        return pd.DataFrame()
    non_empty_ratio = raw.notna().mean(axis=1)
    header_idx = int(non_empty_ratio.idxmax())

    header_vals = raw.iloc[header_idx].fillna("").astype(str).tolist()
    header_vals = [re.sub(r"\s+", " ", h).strip() for h in header_vals]
    n_cols = raw.shape[1]
    if len(header_vals) < n_cols:
        header_vals += [f"col_{i}" for i in range(len(header_vals), n_cols)]
    elif len(header_vals) > n_cols:
        header_vals = header_vals[:n_cols]

    df = raw.iloc[header_idx+1:].reset_index(drop=True)
    df.columns = header_vals

    # Duplikate eindeutig machen
    seen, newcols = {}, []
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
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

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
        raise KeyError(
            f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}"
        )
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i", "u", "f"):
        return s
    def _clean(x):
        if pd.isna(x): return np.nan
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
    """US-Format (MM/DD/YYYY) sicher parsen + Excel-Seriennummern."""
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    nums = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(nums, origin="1899-12-30", unit="d", errors="coerce")
    return dt1.combine_first(dt2)

# ===== Farb-Logik (Heuristik + Vokabular) =====
_COLOR_REGEXES = [
    r"\(([^)]+)\)$",                # ... (Weiss)
    r"[-–—]\s*([A-Za-zäöüÄÖÜß]+)$", # ... – White
    r"/\s*([A-Za-zäöüÄÖÜß]+)$",     # ... / Black
]
def _looks_like_not_a_color(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t: return True
    if t in {"eu","ch","us","uk"}: return True
    if any(x in t for x in ["ml","db","m²","m2"]): return True
    if re.search(r"\d", t): return True
    return False

# Synonyme & Mapping auf Kanon
_COLOR_MAP = {
    # weiss / weiß
    "weiss":"Weiss","weiß":"Weiss","white":"White","offwhite":"Off-White","cream":"Cream","ivory":"Ivory",
    # schwarz
    "schwarz":"Schwarz","black":"Black","jet black":"Black",
    # grau
    "grau":"Grau","gray":"Grau","anthrazit":"Anthrazit","charcoal":"Anthrazit","graphite":"Graphit","silver":"Silber",
    # blau
    "blau":"Blau","blue":"Blau","navy":"Dunkelblau","sky blue":"Hellblau","light blue":"Hellblau","dark blue":"Dunkelblau",
    # rot / rosa / violett
    "rot":"Rot","red":"Rot","bordeaux":"Bordeaux","burgundy":"Bordeaux","pink":"Pink","magenta":"Magenta",
    "lila":"Lila","violett":"Violett","purple":"Violett","fuchsia":"Fuchsia",
    # grün
    "grün":"Grün","gruen":"Grün","green":"Grün","mint":"Mint","türkis":"Türkis","tuerkis":"Türkis","turquoise":"Türkis",
    "petrol":"Petrol","olive":"Olivgrün",
    # gelb / orange / braun / beige
    "gelb":"Gelb","yellow":"Gelb","orange":"Orange","braun":"Braun","brown":"Braun","beige":"Beige","sand":"Sand",
    # metallic / speziell
    "gold":"Gold","rose gold":"Roségold","rosegold":"Roségold","kupfer":"Kupfer","copper":"Kupfer","bronze":"Bronze",
    "transparent":"Transparent","clear":"Transparent"
}

# für Regex eine Liste der Keys, längere zuerst (damit 'rose gold' vor 'gold' matched)
_COLOR_KEYS_SORTED = sorted(_COLOR_MAP.keys(), key=len, reverse=True)
_COLOR_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, _COLOR_KEYS_SORTED)) + r")\b", re.IGNORECASE)

def _canon_color_from_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    m = _COLOR_PATTERN.search(text.lower())
    if m:
        key = m.group(1).lower()
        return _COLOR_MAP.get(key, key.title())
    return ""

def extract_color_from_name(name: str) -> str:
    """Heuristik: Farbe am Ende der Bezeichnung (in Klammern, nach Strich, nach Slash)."""
    if not isinstance(name, str): return ""
    for rgx in _COLOR_REGEXES:
        m = re.search(rgx, name.strip())
        if m:
            cand = m.group(1).strip()
            if not _looks_like_not_a_color(cand):
                # auch hier Synonyme normalisieren
                canon = _canon_color_from_text(cand)
                return canon or cand
    # falls nicht am Ende, versuche irgendwo im Text via Vokabular
    canon = _canon_color_from_text(name)
    if canon and not _looks_like_not_a_color(canon):
        return canon
    return ""

def guess_color_from_row(row: pd.Series, all_columns: list[str]) -> str:
    """Durchsucht mehrere Textspalten (Bezeichnung, Kategorie, Zusatz, etc.) nach Farbworten."""
    # Bevorzugt dedizierte Farbfelder
    for col in all_columns:
        if re.search(r"(farb|color|colour|varian)", col, re.IGNORECASE):
            val = row.get(col, "")
            canon = _canon_color_from_text(str(val))
            if canon and not _looks_like_not_a_color(canon):
                return canon
    # Allgemeiner Scan über typische Textspalten
    candidates = []
    for col in all_columns:
        if col.lower() in {"ean","gtin","barcode","artikelnummer","artikelnr","artnr","produkt id"}:
            continue
        val = str(row.get(col, "") or "").strip()
        if not val:
            continue
        # nur in Textspalten suchen (keine reinen Zahlen-/Mengenfelder)
        if re.fullmatch(r"[0-9 .,'’-]+", val):
            continue
        c = extract_color_from_name(val) or _canon_color_from_text(val)
        if c and not _looks_like_not_a_color(c):
            candidates.append(c)
    # wähle die erste gefundene
    return candidates[0] if candidates else ""

# =========================
# Parsing – Preislisten
# =========================
PRICE_COL_CANDIDATES = ["Preis", "VK", "Netto", "NETTO", "Einkaufspreis", "Verkaufspreis", "NETTO NETTO", "Einkauf"]
BUY_PRICE_CANDIDATES  = ["Einkaufspreis", "Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis", "VK", "Preis"]

ARTNR_CANDIDATES = [
    "Artikelnummer", "Artikelnr", "ArtikelNr", "Artikel-Nr.",
    "Hersteller-Nr.", "Produkt ID", "ProdNr", "ArtNr", "ArtikelNr.", "Artikel"
]
EAN_CANDIDATES  = ["EAN", "GTIN", "BarCode", "Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung", "Produktname", "Name", "Titel", "Artikelname"]
CAT_CANDIDATES  = ["Kategorie", "Warengruppe", "Zusatz"]
STOCK_CANDIDATES= ["Bestand", "Verfügbar", "Lagerbestand"]
COLOR_CANDIDATES= ["Farbe", "Color", "Colour", "Variante", "Variant", "Farbvariante", "Farbname"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES,   "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat   = find_column(df, CAT_CANDIDATES,   "Kategorie", required=False)
    col_stock = find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_color = find_column(df, COLOR_CANDIDATES, "Farbe/Variante", required=False)
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
    out["Kategorie"]       = df[col_cat].astype(str) if col_cat else ""

    # --- Farbe bestimmen ---
    if col_color:
        out["Farbe"] = df[col_color].astype(str).map(lambda x: _canon_color_from_text(str(x)) or str(x))
    else:
        # erst Bezeichnung-Heuristik …
        out["Farbe"] = out["Bezeichnung"].map(extract_color_from_name)
        # … dann Spaltenscan, falls noch leer
        if out["Farbe"].isna().any() or (out["Farbe"].astype(str).str.strip() == "").any():
            # zeilenweiser Scan
            all_cols = list(df.columns)
            out["Farbe"] = out.apply(
                lambda r: r["Farbe"] if str(r.get("Farbe","")).strip() else guess_color_from_row(df.loc[r.name], all_cols),
                axis=1
            )
    out["Farbe"] = out["Farbe"].fillna("").astype(str)

    # Lager & Preise
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
# Parsing – Sell‑out‑Report (Spalte I/J für Start/Ende, US-Format)
# =========================
NAME_CANDIDATES_SO   = ["Bezeichnung", "Name", "Artikelname", "Bezeichnung_Sales", "Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty", "Verkauf", "Verkaufte Menge", "Menge verkauft", "Absatz", "Stück", "Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf", "Einkaufsmenge", "Menge Einkauf"]
DATE_START_CANDS     = ["Start", "Startdatum", "Start Date", "Anfangs datum", "Anfangsdatum", "Von", "Period Start"]
DATE_END_CANDS       = ["Ende", "Enddatum", "End Date", "Bis", "Period End"]

def _fallback_col_by_index(df: pd.DataFrame, index_zero_based: int) -> str | None:
    try:
        return df.columns[index_zero_based]
    except Exception:
        return None

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)

    col_art   = find_column(df, ARTNR_CANDIDATES,  "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES,    "EAN/GTIN",       required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO,"Bezeichnung",    required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)

    col_start = find_column(df, DATE_START_CANDS, "Startdatum (Spalte I)", required=False)
    col_end   = find_column(df, DATE_END_CANDS,   "Enddatum (Spalte J)",   required=False)
    if not col_start and df.shape[1] >= 9:  col_start = _fallback_col_by_index(df, 8)   # I
    if not col_end   and df.shape[1] >= 10: col_end   = _fallback_col_by_index(df, 9)   # J

    out = pd.DataFrame()
    out["ArtikelNr"] = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)
    out["EAN"] = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"] = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)

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
def _f64(s: pd.Series) -> pd.Series:
    """Sicher zu float64 konvertieren (gegen Integer-Overflow)."""
    return pd.to_numeric(s, errors="coerce").astype("float64")

@st.cache_data(show_spinner=False)
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = sell_df.merge(
        price_df,
        on=["ArtikelNr_key"],
        how="left",
        suffixes=("", "_pl")
    )

    # Fallback via EAN
    mask_need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if mask_need.any():
        tmp = merged.loc[mask_need, ["EAN_key"]].merge(
            price_df[["EAN_key", "Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Farbe", "Kategorie", "ArtikelNr"]],
            on="EAN_key", how="left"
        )
        idx = merged.index[mask_need]
        tmp.index = idx
        for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Farbe", "Kategorie", "ArtikelNr"]:
            merged.loc[idx, col] = merged.loc[idx, col].fillna(tmp[col])

    # Fallback via normalisierte Bezeichnung
    mask_need = merged["Verkaufspreis"].isna()
    if mask_need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        idx = merged.index[mask_need]
        keys = merged.loc[idx, "Bezeichnung_key"]
        for i, k in zip(idx, keys):
            if k in name_map.index:
                row = name_map.loc[k]
                for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Farbe", "Kategorie", "ArtikelNr"]:
                    if pd.isna(merged.at[i, col]) or col in ("ArtikelNr", "Bezeichnung", "Kategorie", "Farbe"):
                        merged.at[i, col] = row.get(col, merged.at[i, col])

    # numerisch säubern
    merged["Kategorie"]   = merged["Kategorie"].fillna("")
    merged["Bezeichnung"] = merged["Bezeichnung"].fillna("")
    merged["Farbe"] = merged.get("Farbe", "").fillna("")

    # Sichtbarer Name: standardmäßig Bezeichnung
    merged["Bezeichnung_anzeige"] = merged["Bezeichnung"]

    # Nur bei Dubletten Farbe anhängen (wenn valide Farbe vorhanden)
    dup = merged.duplicated(subset=["Bezeichnung"], keep=False)
    valid_color = merged["Farbe"].astype(str).str.strip().map(lambda t: (t != "") and (not _looks_like_not_a_color(t)))
    merged.loc[dup & valid_color, "Bezeichnung_anzeige"] = (
        merged.loc[dup & valid_color, "Bezeichnung"] + " – " + merged.loc[dup & valid_color, "Farbe"].astype(str).str.strip()
    )

    # Overflow-sicher: alles in float64 berechnen
    qty_buy   = _f64(merged["Einkaufsmenge"]).fillna(0.0)
    qty_sell  = _f64(merged["Verkaufsmenge"]).fillna(0.0)
    qty_stock = _f64(merged["Lagermenge"]).fillna(0.0)
    pr_buy    = _f64(merged["Einkaufspreis"]).fillna(0.0)
    pr_sell   = _f64(merged["Verkaufspreis"]).fillna(0.0)
    pr_buy  = pr_buy.mask(pr_buy.abs()  > 1e6, np.nan).fillna(0.0)
    pr_sell = pr_sell.mask(pr_sell.abs() > 1e6, np.nan).fillna(0.0)

    merged["Einkaufswert"] = qty_buy   * pr_buy
    merged["Verkaufswert"] = qty_sell  * pr_sell
    merged["Lagerwert"]    = qty_stock * pr_sell

    display_cols = [
        "ArtikelNr", "Bezeichnung_anzeige", "Kategorie",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge", "Lagerwert"
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    detail = merged[display_cols].copy()

    totals = (
        detail.groupby(["ArtikelNr", "Bezeichnung_anzeige", "Kategorie"], dropna=False, as_index=False)
              .agg({
                    "Einkaufsmenge": "sum",
                    "Einkaufswert": "sum",
                    "Verkaufsmenge": "sum",
                    "Verkaufswert": "sum",
                    "Lagermenge": "sum",
                    "Lagerwert": "sum"
              })
    )
    return detail, totals

# =========================
# UI
# =========================
st.title("📦 Galaxus Sell‑out Aggregator")
st.caption("Summenansicht, robustes Matching (ArtNr → EAN → 1./2. Wort), EU‑Datumsfilter. Detailtabelle optional.")

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

        # Zeitraumfilter (Start/Ende aus I/J)
        filtered_sell_df = sell_df
        if {"StartDatum", "EndDatum"}.issubset(sell_df.columns) and not sell_df["StartDatum"].isna().all():
            min_date = sell_df["StartDatum"].min().date()
            max_date = (sell_df["EndDatum"].dropna().max() if "EndDatum" in sell_df else sell_df["StartDatum"].max()).date()

            st.subheader("Periode wählen")
            date_value = st.date_input("Zeitraum (DD.MM.YYYY)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_value, tuple):
                start_date, end_date = date_value
            else:
                start_date = end_date = date_value

            mask = ~(
                (sell_df["EndDatum"].dt.date < start_date) |
                (sell_df["StartDatum"].dt.date > end_date)
            )
            filtered_sell_df = sell_df.loc[mask].copy()

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals = enrich_and_merge(filtered_sell_df, price_df)

        # Detailtabelle standardmäßig ausblenden
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
