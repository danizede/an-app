# app.py — Galaxus Sell‑out Aggregator (Summenansicht, robustes Matching)
# Matching-Reihenfolge: ArtikelNr -> EAN -> erste 2 Wörter (ohne Klammern)
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Galaxus Sell‑out Aggregator", layout="wide")

# ========= Anzeige-Helfer =========
THOUSANDS_SEP = "'"
NUM_COLS_DEFAULT = ["Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert","Lagermenge","Lagerwert"]

def _fmt_thousands(x, sep=THOUSANDS_SEP):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}".replace(",", sep)
    except Exception: return str(x)

def style_numeric(df: pd.DataFrame, num_cols=NUM_COLS_DEFAULT, sep=THOUSANDS_SEP):
    out = df.copy()
    for c in (col for col in num_cols if col in out.columns):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    fmt = {c: (lambda v, s=sep: _fmt_thousands(v, s)) for c in num_cols if c in out.columns}
    return out, out.style.format(fmt)

# ========= Utilities =========
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
        .map(lambda c: unicodedata.normalize("NFKC", str(c)))
        .map(lambda c: re.sub(r"\s+", " ", c).strip()))
    return df

def normalize_key(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+","", s)
    return s

def strip_parens(txt: str) -> str:
    return re.sub(r"\([^)]*\)", "", txt or "")

def first2_words_key(name: str) -> str:
    """Erste zwei Wörter ohne Klammern, normalisiert zu a-z0-9."""
    base = strip_parens(str(name))
    tokens = re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKD", base))
    first2 = "".join(tokens[:2]).lower()
    return first2

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True) -> str|None:
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols: return cand
    canon = {re.sub(r"[\s\-_/]+","", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-_/]+","", cand).lower()
        if key in canon: return canon[key]
    if required:
        raise KeyError(f'Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}')
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i","u","f"): return s
    def _clean(x):
        if pd.isna(x): return np.nan
        x = str(x).strip().replace("’","").replace("'","").replace(" ","").replace(",",".")
        if x.count(".")>1:
            parts = x.split("."); x = "".join(parts[:-1]) + "." + parts[-1]
        try: return float(x)
        except Exception: return np.nan
    return s.map(_clean)

def parse_date_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64): return s
    return pd.to_datetime(s, errors="coerce")

# ========= Parsing – Preislisten =========
PRICE_COL_CANDIDATES = ["Preis","VK","Netto","NETTO","Einkaufspreis","Verkaufspreis","NETTO NETTO","Einkauf"]
BUY_PRICE_CANDIDATES = ["Einkaufspreis","Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis","VK","Preis"]

ARTNR_CANDIDATES = ["Artikelnummer","Artikelnr","ArtikelNr","Artikel-Nr.","Hersteller-Nr.","Produkt ID","ProdNr","ArtNr","ArtikelNr.","Artikel"]
EAN_CANDIDATES = ["EAN","GTIN","BarCode","Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung","Produktname","Name","Titel","Artikelname"]
CAT_CANDIDATES = ["Kategorie","Warengruppe","Zusatz"]
STOCK_CANDIDATES = ["Bestand","Verfügbar","Lagerbestand"]
COLOR_CANDIDATES = ["Farbe","Color","Colour","Farben","Farbvariante","Variante","Colorway","Farbton","Farbe / Color","Zusatz"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES, "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat   = find_column(df, CAT_CANDIDATES, "Kategorie", required=False)
    col_stock = find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_color = find_column(df, COLOR_CANDIDATES, "Farbe", required=False)
    col_any   = None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]        = df[col_art].astype(str)
    out["ArtikelNr_key"]    = out["ArtikelNr"].map(normalize_key)
    out["EAN"]              = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]          = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","", str(x)))
    out["Bezeichnung"]      = df[col_name].astype(str)
    out["Bezeichnung_key"]  = out["Bezeichnung"].map(normalize_key)
    out["First2_key"]       = out["Bezeichnung"].map(first2_words_key)
    out["Kategorie"]        = (df[col_cat].astype(str) if col_cat else "").replace(["nan","None"], "").fillna("")
    out["Farbe"]            = (df[col_color].astype(str) if col_color else "").replace(["nan","None"], "").fillna("")

    if col_stock:
        out["Lagermenge"] = parse_number_series(df[col_stock]).fillna(0).astype("Int64")
    else:
        out["Lagermenge"] = pd.Series([0]*len(out), dtype="Int64")

    if col_buy:  out["Einkaufspreis"]  = parse_number_series(df[col_buy])
    if col_sell: out["Verkaufspreis"]  = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        price_any = parse_number_series(df[col_any])
        out["Einkaufspreis"] = price_any
        out["Verkaufspreis"] = price_any
    if "Einkaufspreis" not in out:   out["Einkaufspreis"]  = out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out:   out["Verkaufspreis"]  = out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))

    def build_display(row):
        base = row["Bezeichnung"].strip()
        color = str(row["Farbe"]).strip()
        return f"{base} ({color})" if color else base
    out["DisplayName"] = out.apply(build_display, axis=1)
    return out

# ========= Parsing – Sell-out =========
NAME_CANDIDATES_SO = ["Bezeichnung","Name","Artikelname","Bezeichnung_Sales","Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty","Verkauf","Verkaufte Menge","Menge verkauft","Absatz","Stück","Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf","Einkaufsmenge","Menge Einkauf"]

DATE_CANDIDATES   = ["Datum","Date","Verkaufsdatum","Bestelldatum","Belegdatum"]
START_DATE_CANDS  = ["Von","Start","Startdatum","Period Start","Zeitraum Start","Beginn"]
END_DATE_CANDS    = ["Bis","Ende","Enddatum","Period End","Zeitraum Ende","Schluss"]
KW_CANDIDATES     = ["KW","Kalenderwoche","Week"]
YEAR_CANDIDATES   = ["Jahr","Year"]

def _first_existing(df, cands):
    for c in cands:
        col = find_column(df,[c],c,required=False)
        if col: return col
    return None

def _find_any_date_column(df):
    for col in df.columns:
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().sum() > 0:
            return col
    return None

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES, "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung", required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)

    # Datum/Periode erkennen
    col_date  = find_column(df, DATE_CANDIDATES, "Datum", required=False)
    col_start = _first_existing(df, START_DATE_CANDS)
    col_kw    = _first_existing(df, KW_CANDIDATES)
    col_year  = _first_existing(df, YEAR_CANDIDATES)

    out = pd.DataFrame()
    out["ArtikelNr"]        = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"]    = out["ArtikelNr"].map(normalize_key)
    out["EAN"]              = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]          = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","", str(x)))
    out["Bezeichnung"]      = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"]  = out["Bezeichnung"].map(normalize_key)
    out["First2_key"]       = out["Bezeichnung"].map(first2_words_key)
    out["Verkaufsmenge"]    = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"]    = parse_number_series(df[col_buy]).fillna(0).astype("Int64") if col_buy else pd.Series([0]*len(out), dtype="Int64")

    if col_date:
        out["Datum"] = parse_date_series(df[col_date])
    elif col_start:
        out["Datum"] = parse_date_series(df[col_start])
    elif col_kw and col_year:
        kw = pd.to_numeric(df[col_kw], errors="coerce").astype("Int64")
        yr = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
        def kw_to_date(k,y):
            if pd.isna(k) or pd.isna(y): return pd.NaT
            return pd.to_datetime(f"{int(y)}-W{int(k):02d}-1", format="%G-W%V-%u", errors="coerce")
        out["Datum"] = [kw_to_date(k,y) for k,y in zip(kw,yr)]
    else:
        col_any = _find_any_date_column(df)
        if col_any: out["Datum"] = parse_date_series(df[col_any])

    return out

# ========= Merge & Berechnung =========
@st.cache_data(show_spinner=False)
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame):
    merged = sell_df.merge(price_df, on=["ArtikelNr_key"], how="left", suffixes=("","_pl"))

    def _fallback_join(key):
        mask = merged["Verkaufspreis"].isna() & merged[key].astype(bool)
        if not mask.any(): return
        cols = ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Kategorie","ArtikelNr","Farbe","DisplayName"]
        tmp = merged.loc[mask, [key]].merge(price_df[[key]+cols], on=key, how="left")
        idx = merged.index[mask]; tmp.index = idx
        for c in cols:
            merged.loc[idx, c] = merged.loc[idx, c].fillna(tmp[c])

    # Fallback 1: EAN
    _fallback_join("EAN_key")
    # Fallback 2: erste zwei Wörter aus Name (ohne Klammern)
    _fallback_join("First2_key")
    # Fallback 3: vollständiger, normalisierter Name (letzte Rettung)
    if merged["Verkaufspreis"].isna().any():
        key = "Bezeichnung_key"
        mask = merged["Verkaufspreis"].isna() & merged[key].astype(bool)
        if mask.any():
            name_map = price_df.drop_duplicates(key).set_index(key)
            for i, k in zip(merged.index[mask], merged.loc[mask, key]):
                if k in name_map.index:
                    row = name_map.loc[k]
                    for c in ["Einkaufspreis","Verkaufspreis","Lagermenge","Bezeichnung","Kategorie","ArtikelNr","Farbe","DisplayName"]:
                        if pd.isna(merged.at[i, c]) or c in ("ArtikelNr","Bezeichnung","Kategorie","Farbe","DisplayName"):
                            merged.at[i, c] = row.get(c, merged.at[i, c])

    # Strings säubern – Kategorie nie 'nan'
    merged["Kategorie"]  = merged.get("Kategorie","").astype(str).replace(["nan","None"],"").fillna("")
    merged["Bezeichnung"]= merged.get("Bezeichnung","").astype(str).replace(["nan","None"],"").fillna("")
    merged["Farbe"]      = merged.get("Farbe","").astype(str).replace(["nan","None"],"").fillna("")
    # DisplayName sicherstellen
    def _disp(row):
        base = str(row.get("Bezeichnung","")).strip()
        col  = str(row.get("Farbe","")).strip()
        return f"{base} ({col})" if base and col else base or ""
    merged["DisplayName"] = merged.get("DisplayName","").astype(str).replace(["nan","None"],"").fillna("")
    merged.loc[merged["DisplayName"]=="","DisplayName"] = merged[merged["DisplayName"]==""].apply(_disp, axis=1)

    # Werte berechnen
    for pcol in ["Einkaufspreis","Verkaufspreis"]:
        merged[pcol] = pd.to_numeric(merged[pcol], errors="coerce")
    merged["Einkaufswert"] = (merged["Einkaufsmenge"].astype("Int64").fillna(0) * merged["Einkaufspreis"]).astype(float)
    merged["Verkaufswert"] = (merged["Verkaufsmenge"].astype("Int64").fillna(0) * merged["Verkaufspreis"]).astype(float)
    merged["Lagerwert"]    = (merged["Lagermenge"].astype("Int64").fillna(0)    * merged["Verkaufspreis"]).astype(float)

    display_cols = ["ArtikelNr","DisplayName","Kategorie","Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert","Lagermenge","Lagerwert"]
    detail = merged[[c for c in display_cols if c in merged.columns]].copy()

    totals = (detail.groupby(["ArtikelNr","DisplayName","Kategorie"], dropna=False, as_index=False)
                    .agg({"Einkaufsmenge":"sum","Einkaufswert":"sum","Verkaufsmenge":"sum","Verkaufswert":"sum","Lagermenge":"sum","Lagerwert":"sum"}))

    # Liste der nicht gematchten Positionen (falls vorhanden)
    unmatched = merged[merged["Verkaufspreis"].isna()][["ArtikelNr","EAN","Bezeichnung","First2_key","Verkaufsmenge","Einkaufsmenge"]].copy() if "Verkaufspreis" in merged else pd.DataFrame()
    return detail, totals, unmatched

# ========= UI =========
st.title("📦 Galaxus Sell‑out Aggregator")
st.caption("Summenansicht mit robustem Matching (ArtNr → EAN → erste 2 Wörter).")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Datei wählen", type=["xlsx"], key="sell")
with c2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Datei wählen", type=["xlsx"], key="price")

if sell_file and price_file:
    try:
        raw_sell  = pd.read_excel(sell_file)
        raw_price = pd.read_excel(price_file)

        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)

        # Perioden-Filter (KW/Jahr) mit „Gesamten Zeitraum“
        filtered_sell_df = sell_df
        period_enabled = False
        if "Datum" in sell_df.columns and not sell_df["Datum"].isna().all():
            iso = sell_df["Datum"].dt.isocalendar()
            sell_df["KW"] = iso["week"]; sell_df["KW_Year"] = iso["year"]
            sell_df["Period"] = sell_df["KW"].astype(str)+"/"+sell_df["KW_Year"].astype(str)
            periods = sorted(sell_df["Period"].dropna().unique().tolist())
            if periods:
                period_enabled = True
                st.subheader("Periode wählen")
                key_period = "period_select"
                sel = st.selectbox("Kalenderwoche (KW/Jahr) oder 'Alle'", options=["Alle"]+periods, index=0, key=key_period)
                def set_all(): st.session_state[key_period] = "Alle"
                st.button("Gesamten Zeitraum", on_click=set_all)
                sel = st.session_state.get(key_period, sel)
                if sel != "Alle":
                    filtered_sell_df = sell_df[sell_df["Period"]==sel].copy()
        if not period_enabled:
            st.info("Hinweis: Keine Datum-/KW-Infos erkannt – Filter ausgeblendet.")

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals, unmatched = enrich_and_merge(filtered_sell_df, price_df)

        # NUR Summen anzeigen
        st.subheader("Summen pro Artikel")
        t_rounded, t_styler = style_numeric(totals)
        st.dataframe(t_styler, use_container_width=True)
        st.download_button("⬇️ Summen (CSV)", data=t_rounded.to_csv(index=False).encode("utf-8"), file_name="summen.csv", mime="text/csv")

        # Hinweis + optionaler Download für nicht gematchte Positionen
        if unmatched is not None and len(unmatched)>0:
            st.warning(f"{len(unmatched)} Position(en) ohne Preis-Match. (ArtNr/EAN/erste 2 Wörter prüfen)")
            st.download_button("⬇️ Ungematchte (CSV)", data=unmatched.to_csv(index=False).encode("utf-8"), file_name="ungematcht.csv", mime="text/csv")

    except KeyError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
else:
    st.info("Bitte beide Dateien hochladen (Sell-out & Preisliste).")
