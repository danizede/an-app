# app.py
# Galaxus Sell‑out Aggregator – Summenansicht mit KW-Dropdown, Ein-Klick-"Gesamter Zeitraum",
# robustem Farb-/Kategorie-Handling und geprüfter Berechnung.

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
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True) -> str | None:
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    canon = {re.sub(r"[\s\-_/]+", "", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-_/]+", "", cand).lower()
        if key in canon:
            return canon[key]
    if required:
        raise KeyError(
            f'Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}'
        )
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

def parse_date_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

# =========================
# Parsing – Preislisten
# =========================
PRICE_COL_CANDIDATES = [
    "Preis", "VK", "Netto", "NETTO", "Einkaufspreis", "Verkaufspreis", "NETTO NETTO", "Einkauf"
]
BUY_PRICE_CANDIDATES = ["Einkaufspreis", "Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis", "VK", "Preis"]

ARTNR_CANDIDATES = [
    "Artikelnummer", "Artikelnr", "ArtikelNr", "Artikel-Nr.",
    "Hersteller-Nr.", "Produkt ID", "ProdNr", "ArtNr", "ArtikelNr.", "Artikel"
]
EAN_CANDIDATES = ["EAN", "GTIN", "BarCode", "Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung", "Produktname", "Name", "Titel", "Artikelname"]
CAT_CANDIDATES = ["Kategorie", "Warengruppe", "Zusatz"]
STOCK_CANDIDATES = ["Bestand", "Verfügbar", "Lagerbestand"]

# → erweitert, damit Farbe wirklich gefunden wird
COLOR_CANDIDATES = [
    "Farbe", "Color", "Colour", "Farben", "Farbvariante",
    "Variante", "Colorway", "Farbton", "Farbe / Color", "Zusatz"
]

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
    out["ArtikelNr"]     = df[col_art].astype(str)
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)
    out["EAN"]     = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"]      = df[col_name].astype(str)
    out["Bezeichnung_key"]  = out["Bezeichnung"].map(normalize_key)
    out["Kategorie"] = (df[col_cat].astype(str) if col_cat else "").replace(["nan","None"], "").fillna("")

    if col_color:
        out["Farbe"] = df[col_color].astype(str).replace(["nan","None"], "").fillna("")
    else:
        out["Farbe"] = ""

    if col_stock:
        out["Lagermenge"] = parse_number_series(df[col_stock]).fillna(0).astype("Int64")
    else:
        out["Lagermenge"] = pd.Series([0]*len(out), dtype="Int64")

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

    def build_display(row):
        base = row["Bezeichnung"].strip()
        color = str(row["Farbe"]).strip()
        return f"{base} ({color})" if color else base
    out["DisplayName"] = out.apply(build_display, axis=1)

    return out

# =========================
# Parsing – Sell-out-Report
# =========================
NAME_CANDIDATES_SO = ["Bezeichnung", "Name", "Artikelname", "Bezeichnung_Sales", "Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty", "Verkauf", "Verkaufte Menge", "Menge verkauft", "Absatz", "Stück", "Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf", "Einkaufsmenge", "Menge Einkauf"]

DATE_CANDIDATES   = ["Datum", "Date", "Verkaufsdatum", "Bestelldatum", "Belegdatum"]
START_DATE_CANDS  = ["Von", "Start", "Startdatum", "Period Start", "Zeitraum Start", "Beginn"]
END_DATE_CANDS    = ["Bis", "Ende", "Enddatum", "Period End", "Zeitraum Ende", "Schluss"]
KW_CANDIDATES     = ["KW", "Kalenderwoche", "Week"]
YEAR_CANDIDATES   = ["Jahr", "Year"]

def _first_existing(df, candidates):
    for c in candidates:
        col = find_column(df, [c], c, required=False)
        if col:
            return col
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

    col_date  = find_column(df, DATE_CANDIDATES, "Datum", required=False)
    col_start = _first_existing(df, START_DATE_CANDS)
    col_end   = _first_existing(df, END_DATE_CANDS)
    col_kw    = _first_existing(df, KW_CANDIDATES)
    col_year  = _first_existing(df, YEAR_CANDIDATES)

    out = pd.DataFrame()
    out["ArtikelNr"] = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)
    out["EAN"] = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"] = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"] = parse_number_series(df[col_buy]).fillna(0).astype("Int64") if col_buy else pd.Series([0]*len(out), dtype="Int64")

    if col_date:
        out["Datum"] = parse_date_series(df[col_date])
    elif col_start:
        out["Datum"] = parse_date_series(df[col_start])
    elif col_kw and col_year:
        kw = pd.to_numeric(df[col_kw], errors="coerce").astype("Int64")
        yr = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
        def kw_to_date(k, y):
            if pd.isna(k) or pd.isna(y):
                return pd.NaT
            return pd.to_datetime(f"{int(y)}-W{int(k):02d}-1", format="%G-W%V-%u", errors="coerce")
        out["Datum"] = [kw_to_date(k, y) for k, y in zip(kw, yr)]
    else:
        col_any = _find_any_date_column(df)
        if col_any:
            out["Datum"] = parse_date_series(df[col_any])

    return out

# =========================
# Merge & Berechnung
# =========================
@st.cache_data(show_spinner=False)
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = sell_df.merge(
        price_df,
        on=["ArtikelNr_key"],
        how="left",
        suffixes=("", "_pl"),
    )

    # Fallback via EAN
    mask_need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if mask_need.any():
        tmp = merged.loc[mask_need, ["EAN_key"]].merge(
            price_df[[
                "EAN_key", "Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung",
                "Kategorie", "ArtikelNr", "Farbe", "DisplayName"
            ]],
            on="EAN_key", how="left"
        )
        idx = merged.index[mask_need]
        tmp.index = idx
        for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr", "Farbe", "DisplayName"]:
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
                for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr", "Farbe", "DisplayName"]:
                    if pd.isna(merged.at[i, col]) or col in ("ArtikelNr", "Bezeichnung", "Kategorie", "Farbe", "DisplayName"):
                        merged.at[i, col] = row.get(col, merged.at[i, col])

    # Strings säubern – Kategorie nie 'nan'
    merged["Kategorie"]  = merged["Kategorie"].astype(str).replace(["nan","None"], "").fillna("")
    merged["Bezeichnung"] = merged["Bezeichnung"].astype(str).replace(["nan","None"], "").fillna("")
    merged["Farbe"] = merged.get("Farbe", "").astype(str).replace(["nan","None"], "").fillna("")
    # DisplayName final sicherstellen (auch wenn Merge/Mapping das Feld leeren sollte)
    def build_display_row(row):
        base = str(row.get("Bezeichnung", "")).strip()
        col  = str(row.get("Farbe", "")).strip()
        return f"{base} ({col})" if base and col else base or ""
    merged["DisplayName"] = merged.get("DisplayName", "")
    merged["DisplayName"] = merged["DisplayName"].astype(str).replace(["nan","None"], "").fillna("")
    missing_disp = merged["DisplayName"] == ""
    if missing_disp.any():
        merged.loc[missing_disp, "DisplayName"] = merged.loc[missing_disp].apply(build_display_row, axis=1)

    # Werte berechnen
    for pcol in ["Einkaufspreis", "Verkaufspreis"]:
        merged[pcol] = pd.to_numeric(merged[pcol], errors="coerce")
    merged["Einkaufswert"] = (merged["Einkaufsmenge"].astype("Int64").fillna(0) * merged["Einkaufspreis"].fillna(0)).astype(float)
    merged["Verkaufswert"] = (merged["Verkaufsmenge"].astype("Int64").fillna(0) * merged["Verkaufspreis"].fillna(0)).astype(float)
    merged["Lagerwert"]    = (merged["Lagermenge"].astype("Int64").fillna(0)    * merged["Verkaufspreis"].fillna(0)).astype(float)

    display_cols = [
        "ArtikelNr", "DisplayName", "Kategorie",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge", "Lagerwert",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    detail = merged[display_cols].copy()

    totals = (
        detail.groupby(["ArtikelNr", "DisplayName", "Kategorie"], dropna=False, as_index=False)
              .agg({
                    "Einkaufsmenge": "sum",
                    "Einkaufswert": "sum",
                    "Verkaufsmenge": "sum",
                    "Verkaufswert": "sum",
                    "Lagermenge": "sum",
                    "Lagerwert": "sum",
              })
    )
    return detail, totals

# =========================
# UI
# =========================
st.title("📦 Galaxus Sell‑out Aggregator")
st.caption("Summenansicht mit Periodenfilter (KW/Jahr). Zahlen sind gerundet und mit Tausendertrennzeichen formatiert.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="sell")
    if "sell_last" in st.session_state and st.session_state["sell_last"]:
        st.text(st.session_state["sell_last"]["name"])
with col2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="price")
    if "price_last" in st.session_state and st.session_state["price_last"]:
        st.text(st.session_state["price_last"]["name"])

if sell_file and price_file:
    try:
        st.session_state["sell_last"]  = {"name": sell_file.name}
        st.session_state["price_last"] = {"name": price_file.name}

        raw_sell  = pd.read_excel(sell_file)
        raw_price = pd.read_excel(price_file)

        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)

        # Periodenfilter (KW/Jahr) + Ein-Klick-Button "Gesamter Zeitraum"
        filtered_sell_df = sell_df
        period_enabled = False
        if "Datum" in sell_df.columns and not sell_df["Datum"].isna().all():
            iso = sell_df["Datum"].dt.isocalendar()
            sell_df["KW"] = iso["week"]
            sell_df["KW_Year"] = iso["year"]
            sell_df["Period"] = sell_df["KW"].astype(str) + "/" + sell_df["KW_Year"].astype(str)
            periods = sorted(sell_df["Period"].dropna().unique().tolist())
            if periods:
                period_enabled = True
                st.subheader("Periode wählen")
                key_period = "period_select"
                selected_period = st.selectbox(
                    "Kalenderwoche (KW/Jahr) oder 'Alle'",
                    options=["Alle"] + periods,
                    index=0,
                    key=key_period,
                )
                # Ein Klick => Gesamter Zeitraum
                def _set_all():
                    st.session_state[key_period] = "Alle"
                st.button("Gesamten Zeitraum", on_click=_set_all)

                sel = st.session_state.get(key_period, selected_period)
                if sel != "Alle":
                    filtered_sell_df = sell_df[sell_df["Period"] == sel].copy()
        if not period_enabled:
            st.info("Hinweis: Keine Datum-/KW-Information erkannt – Filter ausgeblendet. "
                    "Akzeptierte Spalten: "
                    f"{DATE_CANDIDATES + START_DATE_CANDS + END_DATE_CANDS + KW_CANDIDATES + YEAR_CANDIDATES}")

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals = enrich_and_merge(filtered_sell_df, price_df)

        st.subheader("Summen pro Artikel")
        t_rounded, t_styler = style_numeric(totals)
        st.dataframe(t_styler, use_container_width=True)

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
