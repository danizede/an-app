# app.py
# Galaxus Sell-out Aggregator – robust & stabil
# - flexible Spaltenerkennung (Preislisten & Sell-out-Report)
# - Schweizer Zahlenformat (Apostroph-Tausender, Komma-Dezimal) wird korrekt geparst
# - Merge-Fallbacks: Artikelnummer -> EAN -> normalisierter Name
# - Einkaufs-/Verkaufs-/Lager-Mengen & -Werte werden getrennt berechnet
# - Anzeige: rundet auf ganze Zahlen + Tausendertrennzeichen

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")

# =========================
# Anzeige-Helfer (Runden + Tausender)
# =========================
THOUSANDS_SEP = "'"  # Schweizer Format
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
    for c in (col for col in num_cols if c in out.columns):
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
    # exakte Treffer
    for cand in candidates:
        if cand in cols:
            return cand
    # case-insensitive + Leerzeichen/Bindestrich tolerant
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

    # zuerst Apostroph als Tausender raus, dann Komma -> Punkt
    def _clean(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        # gängige Tausendertrenner entfernen
        x = x.replace("’", "").replace("'", "").replace(" ", "")
        # Dezimal-Komma -> Punkt
        x = x.replace(",", ".")
        # vereinzelte Punkte als Tausender rausholen, wenn mehr als ein Punkt
        if x.count(".") > 1:
            # alles bis auf die letzte Punktinstanz entfernen
            parts = x.split(".")
            x = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return float(x)
        except Exception:
            return np.nan

    return s.map(_clean)

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

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)

    col_art  = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean  = find_column(df, EAN_CANDIDATES, "EAN/GTIN", required=False)
    col_name = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat  = find_column(df, CAT_CANDIDATES, "Kategorie", required=False)
    col_stock= find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)

    # Preise
    col_buy  = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_any  = None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]   = df[col_art].astype(str)
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)

    out["EAN"] = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))

    out["Bezeichnung"] = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)

    out["Kategorie"] = df[col_cat].astype(str) if col_cat else ""

    if col_stock:
        out["Lagermenge"] = parse_number_series(df[col_stock]).fillna(0).astype("Int64")
    else:
        out["Lagermenge"] = pd.Series([0]*len(out), dtype="Int64")

    # Preise: Einkauf + Verkauf wenn vorhanden, sonst eine Preis-Spalte verwenden
    if col_buy:
        out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell:
        out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        price_any = parse_number_series(df[col_any])
        out["Einkaufspreis"] = price_any
        out["Verkaufspreis"] = price_any

    # Defaults falls eine Seite fehlt
    if "Einkaufspreis" not in out:
        out["Einkaufspreis"] = out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out:
        out["Verkaufspreis"] = out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))

    return out

# =========================
# Parsing – Sell-out-Report
# =========================
NAME_CANDIDATES_SO = ["Bezeichnung", "Name", "Artikelname", "Bezeichnung_Sales", "Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty", "Verkauf", "Verkaufte Menge", "Menge verkauft", "Absatz", "Stück", "Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf", "Einkaufsmenge", "Menge Einkauf"]

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)

    col_art  = find_column(df, ARTNR_CANDIDATES, "Artikelnummer", required=False)
    col_ean  = find_column(df, EAN_CANDIDATES, "EAN/GTIN", required=False)
    col_name = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung", required=False)
    col_sales= find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy  = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)

    out = pd.DataFrame()
    out["ArtikelNr"] = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)

    out["EAN"] = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))

    out["Bezeichnung"] = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)

    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    if col_buy:
        out["Einkaufsmenge"] = parse_number_series(df[col_buy]).fillna(0).astype("Int64")
    else:
        out["Einkaufsmenge"] = pd.Series([0]*len(out), dtype="Int64")

    return out

# =========================
# Merge & Berechnung
# =========================
@st.cache_data(show_spinner=False)
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gibt (detail_tabelle, totals) zurück."""
    # 1) exakter Merge auf ArtikelNr
    merged = sell_df.merge(
        price_df,
        on=["ArtikelNr_key"],
        how="left",
        suffixes=("", "_pl")
    )

    # 2) fehlende Preise/Infos via EAN nachziehen
    mask_need = merged["Verkaufspreis"].isna() & (sell_df["EAN_key"].astype(bool))
    if mask_need.any():
        tmp = sell_df.loc[mask_need, ["EAN_key"]].merge(
            price_df[["EAN_key", "Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr"]],
            on="EAN_key", how="left"
        )
        idx = merged.index[mask_need]
        for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr"]:
            merged.loc[idx, col] = merged.loc[idx, col].fillna(tmp[col].values)

    # 3) letzter Fallback: normalisierte Bezeichnung
    mask_need = merged["Verkaufspreis"].isna()
    if mask_need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        idx = merged.index[mask_need]
        keys = merged.loc[idx, "Bezeichnung_key"]
        for i, k in zip(idx, keys):
            if k in name_map.index:
                row = name_map.loc[k]
                for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr"]:
                    if pd.isna(merged.at[i, col]) or col in ("ArtikelNr", "Bezeichnung", "Kategorie"):
                        merged.at[i, col] = row.get(col, merged.at[i, col])

    # Sicherstellen: Preise numerisch
    for pcol in ["Einkaufspreis", "Verkaufspreis"]:
        merged[pcol] = pd.to_numeric(merged[pcol], errors="coerce")

    # Restliche Defaults
    merged["Kategorie"] = merged["Kategorie"].fillna("")
    merged["Bezeichnung"] = merged["Bezeichnung"].replace("", sell_df["Bezeichnung"]).fillna("")

    # 4) Werte berechnen (mit vorhandenen Spalten)
    merged["Einkaufswert"] = (merged["Einkaufsmenge"].astype("Int64").fillna(0) * merged["Einkaufspreis"].fillna(0)).astype(float)
    merged["Verkaufswert"] = (merged["Verkaufsmenge"].astype("Int64").fillna(0) * merged["Verkaufspreis"].fillna(0)).astype(float)
    merged["Lagerwert"]    = (merged["Lagermenge"].astype("Int64").fillna(0)    * merged["Verkaufspreis"].fillna(0)).astype(float)

    # Anzeige- und Exporttabelle (geordnet)
    display_cols = [
        "ArtikelNr", "Bezeichnung", "Kategorie",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge", "Lagerwert"
    ]
    # evtl. nicht existierende Spalten filtern
    display_cols = [c for c in display_cols if c in merged.columns]
    detail = merged[display_cols].copy()

    # Totals (über Artikel zusammengefasst) – falls gewünscht pro Artikel
    totals = (
        detail.groupby(["ArtikelNr", "Bezeichnung", "Kategorie"], dropna=False, as_index=False)
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
st.title("📦 Galaxus Sell-out Aggregator")
st.caption("Lädt Preislisten & Sell-out-Daten, matcht sie robust und berechnet Einkaufs-/Verkaufs-/Lagerwerte. Anzeige ist gerundet mit Tausendertrennzeichen.")

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

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals = enrich_and_merge(sell_df, price_df)

        # Anzeige – gerundet + Tausendertrennzeichen
        st.subheader("Detailtabelle")
        d_rounded, d_styler = style_numeric(detail)
        st.dataframe(d_styler, use_container_width=True)

        st.subheader("Summen pro Artikel")
        t_rounded, t_styler = style_numeric(totals)
        st.dataframe(t_styler, use_container_width=True)

        # Optional: Downloads (Export bleibt numerisch, aber gerundet auf ganze Werte)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Detail (CSV)",
                data=d_rounded.to_csv(index=False).encode("utf-8"),
                file_name="detail.csv",
                mime="text/csv",
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
