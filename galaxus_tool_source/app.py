# ────────────────────────────────────────────────────────────────────────────────
# Galaxus Sell-out Aggregator – Streamlit App
# Passwortschutz: Nofava22caro!
# Matching nach Artikelnr. → Bezeichnung (PL Spalte C), Zusatz (PL Spalte D), Preis (PL Spalte F)
# Berechnet Einkaufs-, Verkaufs- und Lagerwerte (CHF)
# ────────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd

# ───────────── PASSWORTSCHUTZ ──────────────────────────────────────────────────
PW = "Nofava22caro!"
pw = st.text_input("🔐 Passwort eingeben", type="password")
if pw != PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

# ───────────── HEADER ───────────────────────────────────────────────────────────
st.title("📦 Galaxus Sell-out Aggregator")

# ───────────── DATEI-UPLOAD ─────────────────────────────────────────────────────
sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
price_file = st.file_uploader("Preisliste (.xlsx)",   type="xlsx", key="price")

if not sell_file or not price_file:
    st.info("Bitte Sell-out-Report und Preisliste hochladen, um die Auswertung zu starten.")
    st.stop()

# ───────────── DATEIEN EINLESEN ─────────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Dateien laden …")
def load_xlsx(bin_io):
    return pd.read_excel(bin_io, engine="openpyxl")

sell_df  = load_xlsx(sell_file)
price_df = load_xlsx(price_file)

# ───────────── SPALTENFINDER ────────────────────────────────────────────────────
def find_col(df, candidates, label):
    """Gibt den ersten Treffer aus candidates zurück oder wirft KeyError."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte «{label}» fehlt – gesucht: {candidates}")

# Kandidaten-Listen für PL-Spalten
ALIAS_NR    = ["Artikelnummer", "Artikelnr", "Artikelnr.", "Artikelnr"]   # Spalte A
ALIAS_EAN   = ["GTIN", "EAN", "ean"]                                      # Spalte B
ALIAS_BEZ   = ["Bezeichnung", "Bezeichnung"]                             # Spalte C
ALIAS_ZUSATZ= ["Zusatz", "Kategorie", "Warengruppe"]                     # Spalte D
ALIAS_PREIS = ["NETTO NETTO", "Preis", "VK", "Verkaufspreis"]             # Spalte F

# tatsächliche PL-Spaltennamen
p_nr     = find_col(price_df, ALIAS_NR,    "Artikelnr")
p_ean    = find_col(price_df, ALIAS_EAN,   "GTIN")
p_bez    = find_col(price_df, ALIAS_BEZ,   "Bezeichnung")
p_zusatz = find_col(price_df, ALIAS_ZUSATZ,"Zusatz")
p_pr     = find_col(price_df, ALIAS_PREIS, "Preis")

# Vereinheitlichte Preis-Tabelle
price = (
    price_df
    .rename(columns={
        p_nr:    "Artikelnr",
        p_ean:   "EAN",
        p_bez:   "Bezeichnung",
        p_zusatz:"Zusatz",
        p_pr:    "Preis"
    })
    [["Artikelnr","EAN","Bezeichnung","Zusatz","Preis"]]
)

# ───────────── ENRICHMENT ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Match über Hersteller-Nr. → Artikelnr
    merged = sell.merge(
        price[["Artikelnr","Bezeichnung","Zusatz","Preis"]],
        left_on="Hersteller-Nr.",
        right_on="Artikelnr",
        how="left"
    )

    # 2) Fallback über EAN/GTIN
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        tmp = (
            merged[mask]
            .merge(
                price[["EAN","Bezeichnung","Zusatz","Preis"]],
                on="EAN", how="left"
            )
        )
        # Werte in merged zurückschreiben
        merged.loc[mask, ["Bezeichnung","Zusatz","Preis"]] = tmp[["Bezeichnung","Zusatz","Preis"]].values

    # (optional) 3) Fuzzy-Matching via erste zwei Wörter… etc.

    return merged

# ───────────── AGGREGATION ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="📊 Aggregation …")
def aggregate(df: pd.DataFrame):
    tbl = (
        df
        .groupby(
            ["Hersteller-Nr.","Bezeichnung","EAN","Zusatz"],
            dropna=False,
            as_index=False
        )
        .agg(
            Einkauf        =("Einkaufsmenge", "sum"),
            Einkaufswert   =("Einkaufswert",  "sum"),
            Verkauf        =("Verkaufsmenge", "sum"),
            Verkaufswert   =("Verkaufswert",  "sum"),
            Verfügbar      =("Lagermenge",    "sum"),
            Lagerwert      =("Lagerwert",     "sum"),
        )
    )
    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# ───────────── HAUPT-WORKFLOW ──────────────────────────────────────────────────
enriched = enrich(sell_df, price)
agg_tbl, tot = aggregate(enriched)

# Metriken oben
c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)",   f"{tot['VK']:,.0f}".replace(",","."))
c2.metric("Einkaufswert (CHF)",   f"{tot['EK']:,.0f}".replace(",","."))
c3.metric("Lagerwert (CHF)",      f"{tot['LG']:,.0f}".replace(",","."))

# Detail-Tabelle
st.dataframe(agg_tbl)
