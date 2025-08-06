import streamlit as st
import pandas as pd

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df.columns einen der Kandidaten und gibt ihn zurück.
    Falls keiner passt, wird ein KeyError mit allen verfügbaren Spalten geworfen.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(ttl=3600)
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten in sell-out-Report finden
    sell_nr_col     = find_column(sell_df,     ["Hersteller-Nr.", "Produkt ID", "EAN"],      "Artikelnr im Sell-Out-Report")
    purchase_col    = find_column(sell_df,     ["Einkauf"],                                    "Einkaufsmenge")
    sold_col        = find_column(sell_df,     ["Verkauf"],                                    "Verkaufsmenge")
    name_col        = find_column(sell_df,     ["Produktname", "Bezeichnung"],                 "Bezeichnung")
    
    # 2) Spalten in Preisliste finden
    price_nr_col    = find_column(price_df,    ["Artikelnummer", "Hersteller-Nr.", "GTIN", "EAN"], "Artikelnr in Preisliste")
    stock_col       = find_column(price_df,    ["Bestand", "Verfügbar"],                       "Lagermenge")
    price_col       = find_column(price_df,    ["NETTO NETTO", "Netto", "Preis"],              "Preis in PL")
    category_col    = find_column(price_df,    ["Zusatz", "Kategorie", "Warengruppe"],         "Kategorie")
    
    # 3) Umbenennen für einheitliches Merge-Feld
    sell = sell_df.rename(columns={
        sell_nr_col:  "ArtNr",
        purchase_col: "Einkaufsmenge",
        sold_col:     "Verkaufsmenge",
        name_col:     "Bezeichnung"
    })
    price = price_df.rename(columns={
        price_nr_col:   "ArtNr",
        stock_col:      "Lagermenge",
        price_col:      "Preis in PL",
        category_col:   "Kategorie"
    })
    
    # 4) Merge auf ArtNr
    merged = pd.merge(
        sell[["ArtNr", "Bezeichnung", "Einkaufsmenge", "Verkaufsmenge"]],
        price[["ArtNr", "Kategorie", "Lagermenge", "Preis in PL"]],
        on="ArtNr",
        how="left"
    )
    
    # 5) Berechnungen
    merged["Einkaufswert"]  = merged["Einkaufsmenge"]  * merged["Preis in PL"]
    merged["Verkaufswert"]  = merged["Verkaufsmenge"]  * merged["Preis in PL"]
    merged["Lagerwert"]     = merged["Lagermenge"]     * merged["Preis in PL"]
    
    # 6) Nur die gewünschten Spalten in dieser Reihenfolge
    return merged[
        [
            "ArtNr", "Bezeichnung", "Kategorie",
            "Einkaufsmenge", "Einkaufswert",
            "Verkaufsmenge", "Verkaufswert",
            "Lagermenge", "Lagerwert"
        ]
    ]

@st.cache_data(ttl=3600)
def compute_agg(enriched: pd.DataFrame) -> pd.DataFrame:
    return (
        enriched
        .groupby(["ArtNr", "Bezeichnung", "Kategorie"], as_index=False)
        .agg({
            "Einkaufsmenge": "sum",
            "Einkaufswert":  "sum",
            "Verkaufsmenge": "sum",
            "Verkaufswert":  "sum",
            "Lagermenge":    "sum",
            "Lagerwert":     "sum"
        })
    )

def main():
    st.set_page_config(layout="wide")
    st.title("📦 Galaxus Sell-out Aggregator")
    
    col1, col2 = st.columns(2)
    with col1:
        sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx")
    
    if sell_file and price_file:
        try:
            sell_df  = pd.read_excel(sell_file)
            price_df = pd.read_excel(price_file)
            
            enriched = enrich(sell_df, price_df)
            agg      = compute_agg(enriched)
            
            st.subheader("Aggregierte Kennzahlen pro Artikel")
            st.dataframe(agg, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fehler während Berechnung:\n{e}")
    else:
        st.info("Bitte beide Dateien hochladen, um die Auswertung zu starten.")

if __name__ == "__main__":
    main()
