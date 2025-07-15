# galaxus_tool_source/app.py

import streamlit as st
import pandas as pd
from typing import List

# ---------- Hilfsfunktionen -----------------------------------------

def find_column(df: pd.DataFrame, candidates: List[str], purpose: str) -> str:
    """
    Sucht in df.columns einen Namen aus 'candidates'.
    Falls keiner passt, wirft es einen KeyError mit klarer Fehlermeldung.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte für {purpose} fehlt – gesucht unter {candidates}")

@st.cache_data(show_spinner="📥 Lade Excel-Dateien…")
def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    # Pandas kann Byte-Objekte direkt verarbeiten
    return pd.read_excel(uploaded_file)

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Spaltennamen in beiden Tabellen erkennen
    col_nr    = find_column(price,    ["Artikelnr","Hersteller-Nr.","Artikelnummer"], "Artikelnr")
    col_ean   = find_column(price,    ["EAN","GTIN"],                     "EAN / GTIN")
    col_bez   = find_column(price,    ["Bezeichnung","Bez","Bezeichnung"],"Bezeichnung")
    col_zus   = find_column(price,    ["Kategorie","Zusatz","Warengruppe"],"Kategorie/Zusatz")
    col_pr    = find_column(price,    ["Preis","VK","Verkaufspreis","NETTO NETTO"], "Preis")
    col_best  = find_column(sell,     ["Bestand","Lagerbestand"],         "Bestand")
    col_sell  = find_column(sell,     ["Verkauf","Sales"],                "Verkaufsmenge")
    col_eink  = find_column(sell,     ["Einkauf","E","Einkaufsmenge"],     "Einkaufsmenge")

    # 2) Erste Anreicherung über Hersteller-Nr. / Artikelnr
    merged = sell.merge(
        price[[col_nr, col_bez, col_zus, col_pr]],
        left_on=col_nr, right_on=col_nr, how="left"
    )

    # 3) Wenn noch kein Preis, über EAN / GTIN mergen
    mask = merged[col_pr].isna() & merged[col_ean].notna()
    if mask.any():
        df2 = (
            merged[mask]
            .merge(
                price.drop_duplicates(col_ean)[[col_ean, col_bez, col_zus, col_pr]],
                left_on=col_ean, right_on=col_ean, how="left"
            )
        )
        # für jede Spalte einzeln zuweisen, damit die Indizes passen
        for col in [col_bez, col_zus, col_pr]:
            merged.loc[mask, col] = df2[col].values

    # 4) (Optional) Fuzzy-Matching auf ersten zwei Wörter o.ä. …
    #    ließe sich hier ergänzen, wenn noch Lücken bleiben

    # 5) Spalten für die Auswertung umbenennen / vereinheitlichen
    merged = merged.rename(columns={
        col_bez: "Bezeichnung",
        col_zus: "Kategorie",
        col_pr:   "Preis",
        col_nr:   "Artikelnr",
        col_ean:  "EAN",
        col_best: "Bestand",
        col_sell: "Verkauf",
        col_eink: "Einkauf"
    })

    return merged

@st.cache_data(show_spinner="📊 Aggregiere Daten…")
def compute_agg(df: pd.DataFrame):
    # Lagerwert = Bestand * Preis
    df["Lagerwert"] = df["Bestand"] * df["Preis"]
    # Verkaufswert = Verkauf * Preis, Einkaufswert = Einkauf * Preis
    df["Verkaufswert"] = df["Verkauf"] * df["Preis"]
    df["Einkaufswert"] = df["Einkauf"] * df["Preis"]

    # Gruppieren nach Artikelnummer / Bezeichnung / Kategorie
    agg = (
        df
        .groupby(["Artikelnr", "Bezeichnung", "Kategorie"], as_index=False)
        .agg(
            {
                "Einkauf":       "sum",
                "Verkauf":       "sum",
                "Bestand":       "sum",
                "Einkaufswert":  "sum",
                "Verkaufswert":  "sum",
                "Lagerwert":     "sum",
            }
        )
    )

    totals = {
        "EK": agg["Einkaufswert"].sum(),
        "VK": agg["Verkaufswert"].sum(),
        "LG": agg["Lagerwert"].sum(),
    }
    return agg, totals

# ---------- UI --------------------------------------------------------
st.set_page_config(layout="wide", page_title="Galaxus Aggregator")

st.title("📦 Galaxus Sell-out Aggregator")

sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)",     type="xlsx")

if not sell_file or not price_file:
    st.info("Bitte beide Dateien hochladen, um die Auswertung zu starten.")
    st.stop()

# Excel laden
sell_df  = load_xlsx(sell_file)
price_df = load_xlsx(price_file)

# Anreichern und Aggregation
enriched, totals = compute_agg(enrich(sell_df, price_df))

# Metrics oben
c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
c3.metric("Lagerwert (CHF)",    f"{totals['LG']:,.0f}")

# Tabelle anzeigen
st.dataframe(enriched, use_container_width=True)
