# app.py
import streamlit as st
import pandas as pd
import numpy as np
from thefuzz import process

# -----------------------------------
# Hilfsfunktion: flexible Spaltensuche
# -----------------------------------
def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """
    Sucht in df.columns nach der ersten Übereinstimmung
    aus candidates. Wenn keine gefunden wird, wird ein KeyError geworfen.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte für {label} fehlt – gesucht unter {candidates}")

@st.cache_data
def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    """Liest ein Excel-File (BytesIO) in einen DataFrame."""
    return pd.read_excel(uploaded_file)

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- 1) Spalten in der Preisliste finden -----------------------------
    p_nr  = find_column(price_df, ["Artikelnummer","Artikelnr","Hersteller-Nr."], "Artikelnr")
    p_ean = find_column(price_df, ["EAN","GTIN","ean"],               "EAN")
    p_bez = find_column(price_df, ["Bezeichnung","Bez"],              "Bezeichnung")
    p_cat = find_column(price_df, ["Zusatz","Warengruppe","Kategorie"],"Kategorie")
    p_pr  = find_column(price_df, ["Preis","VK","Verkaufspreis"],      "Preis")

    # --- 2) Spalten im Sell-out-Report finden ---------------------------
    s_nr    = find_column(sell_df, ["Hersteller-Nr.","Artikelnr","Artikelnummer"], "Hersteller-Nr.")
    s_ean   = find_column(sell_df, ["EAN","GTIN","ean"],                         "EAN")
    s_bez   = find_column(sell_df, ["Bezeichnung","Bez"],                        "Bezeichnung")
    s_best  = find_column(sell_df, ["Verfügbar","Bestand","Lagerbestand"],       "Bestand")
    s_sell  = find_column(sell_df, ["Verkauf","Sales","Verkaufsmenge"],          "Verkauf")
    s_eink  = find_column(sell_df, ["Einkauf","E","Einkaufsmenge"],              "Einkauf")

    # --- 3) Preisliste und Sell-out umbenennen --------------------------
    price = price_df.rename(columns={
        p_nr:   "Artikelnr",
        p_ean:  "EAN",
        p_bez:  "Bezeichnung",
        p_cat:  "Kategorie",
        p_pr:   "Preis"
    })

    sell = sell_df.rename(columns={
        s_nr:   "Hersteller-Nr.",
        s_ean:  "EAN",
        s_bez:  "Bezeichnung",
        s_best: "Verfügbar",
        s_sell: "Verkauf",
        s_eink: "Einkauf"
    })

    # --- 4) 1st pass: Merge auf Hersteller-Nr. -------------------------
    merged = pd.merge(
        sell,
        price[["Artikelnr", "Bezeichnung", "Kategorie", "Preis"]],
        left_on="Hersteller-Nr.",
        right_on="Artikelnr",
        how="left",
        suffixes=("","_p1")
    )

    # --- 5) 2nd pass: Merge per EAN, wo Preis noch fehlt ---------------
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        df2 = pd.merge(
            merged[mask],
            price.drop_duplicates("EAN")[["EAN","Bezeichnung","Kategorie","Preis"]],
            on="EAN",
            how="left",
            suffixes=("","_p2")
        )
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask, col] = df2[f"{col}_p2"].values

    # --- 6) 3rd pass: Fuzzy-Match auf die ersten 2 Wörter der Bezeichnung
    # Build lookup table
    price["__key"] = (
        price["Bezeichnung"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )
    merged["__key"] = (
        merged["Bezeichnung"]
        .str.split()
        .str[:2]
        .str.join(" ")
        .str.lower()
    )

    mask2 = merged["Preis"].isna() & merged["__key"].notna()
    if mask2.any():
        choices = price.set_index("__key")[["Bezeichnung","Kategorie","Preis"]]
        def lookup(k):
            match, score = process.extractOne(k, choices.index)
            return choices.loc[match]
        df3 = merged.loc[mask2, "__key"].apply(lookup).apply(pd.Series)
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask2, col] = df3[col].values

    # Aufräumen
    merged = merged.drop(columns="__key")
    return merged.fillna({"Preis": 0})

@st.cache_data(show_spinner="🔢 Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    # Werte berechnen
    df["Lagerwert"]      = df["Verfügbar"] * df["Preis"]
    df["Einkaufswert"]   = df["Einkauf"]   * df["Preis"]
    df["Verkaufswert"]   = df["Verkauf"]   * df["Preis"]

    # Gruppieren / Summieren
    tbl = (
        df.groupby(
            ["Hersteller-Nr.","Bezeichnung","Kategorie","Preis"],
            dropna=False,
        )
        .agg(
            Einkaufsmenge  = ("Einkauf",     "sum"),
            Einkaufswert   = ("Einkaufswert","sum"),
            Verkaufsmenge  = ("Verkauf",     "sum"),
            Verkaufswert   = ("Verkaufswert","sum"),
            Verfügbar      = ("Verfügbar",   "sum"),
            Lagerwert      = ("Lagerwert",   "sum"),
        )
        .reset_index()
    )

    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum()
    }
    return tbl, totals

# ----------------- UI -----------------
def main():
    st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")

    st.title("📦 Galaxus Sell-out Aggregator")

    sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
    price_file = st.file_uploader("Preisliste (.xlsx)" , type="xlsx", key="price")

    if sell_file and price_file:
        sell_df  = load_xlsx(sell_file)
        price_df = load_xlsx(price_file)

        with st.spinner("🔧 Daten anreichern…"):
            enriched = enrich(sell_df, price_df)

        with st.spinner("🔢 Aggregieren…"):
            agg_tbl, totals = compute_agg(enriched)

        # Kennzahlen
        c1, c2, c3 = st.columns(3)
        c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}".replace(",", " "))
        c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}".replace(",", " "))
        c3.metric("Lagerwert (CHF)",   f"{totals['LG']:,.0f}".replace(",", " "))

        st.markdown("---")
        st.dataframe(
            agg_tbl,
            use_container_width=True,
            column_config={
                "Einkaufsmenge":  st.column_config.NumberColumn("Einkaufsmenge"),
                "Verkaufsmenge":  st.column_config.NumberColumn("Verkaufsmenge"),
                "Verfügbar":      st.column_config.NumberColumn("Verfügbar"),
                "Einkaufswert":   st.column_config.NumberColumn("Einkaufswert", format="€0.00"),
                "Verkaufswert":   st.column_config.NumberColumn("Verkaufswert", format="€0.00"),
                "Lagerwert":      st.column_config.NumberColumn("Lagerwert", format="€0.00"),
            }
        )
    else:
        st.info("Bitte sowohl Sell-out-Report als auch Preisliste hochladen, um zu starten.")

if __name__ == "__main__":
    main()
