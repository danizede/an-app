import streamlit as st
import pandas as pd

# Hilfsfunktion: findet eine Spalte anhand von Kandidaten (case-insensitive, substring-match)
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    cols = list(df.columns)
    for col in cols:
        low = col.lower()
        for cand in candidates:
            c = cand.lower()
            if c == low or c in low or low in c:
                return col
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- Sell-Out-Report Spalten ---
    col_buy_qty = find_column(sell_df, ["Einkauf"],       "Einkaufsmenge im Sell-Out-Report")
    col_sell_qty= find_column(sell_df, ["Verkauf"],       "Verkaufsmenge im Sell-Out-Report")
    col_stock   = find_column(sell_df, ["Verfügbar","Bestand"], "Lagermenge im Sell-Out-Report")
    col_nr_s    = find_column(sell_df, ["Hersteller-Nr.","Produkt ID","Artikelnummer"], "Artikelnummer im Sell-Out-Report")
    col_ean_s   = find_column(sell_df, ["EAN"],           "EAN im Sell-Out-Report")
    col_name_s  = find_column(sell_df, ["Produktname","Bezeichnung"], "Bezeichnung im Sell-Out-Report")

    sell = sell_df.rename(columns={
        col_buy_qty : "Einkaufsmenge",
        col_sell_qty: "Verkaufsmenge",
        col_stock   : "Lagermenge",
        col_nr_s    : "ArtNr",
        col_ean_s   : "GTIN",
        col_name_s  : "Bezeichnung"
    })[["ArtNr","GTIN","Bezeichnung","Einkaufsmenge","Verkaufsmenge","Lagermenge"]]

    # --- Preisliste Spalten ---
    col_nr_p    = find_column(price_df, ["Artikelnummer","Hersteller-Nr."], "Artikelnummer in Preisliste")
    col_ean_p   = find_column(price_df, ["EAN"],           "EAN in Preisliste")
    col_name_p  = find_column(price_df, ["Bezeichnung","Produktname"], "Bezeichnung in Preisliste")
    col_cat     = find_column(price_df, ["Zusatz","Warengruppe","Kategorie"], "Kategorie in Preisliste")
    col_price   = find_column(price_df, ["Netto","Einkauf","Preis","VK","Verkaufspreis"], "Preis in Preisliste")

    price = price_df.rename(columns={
        col_nr_p  : "ArtNr",
        col_ean_p : "GTIN",
        col_name_p: "Bezeichnung_PL",
        col_cat   : "Kategorie",
        col_price : "Preis"
    })[["ArtNr","GTIN","Bezeichnung_PL","Kategorie","Preis"]]

    # Zusammenführen via ArtNr + EAN (falls beides vorhanden)
    merged = pd.merge(
        sell,
        price,
        on=["ArtNr","GTIN"],
        how="left",
        validate="m:1"  # viele Sell-Out-Zeilen auf genau einen Preis
    )
    return merged

@st.cache_data(show_spinner="📊 Auswertung …")
def compute(enriched: pd.DataFrame) -> pd.DataFrame:
    df = enriched.copy()
    # Fehlende Preise auffüllen mit 0, um Rechenfehler zu vermeiden
    df["Preis"] = df["Preis"].fillna(0.0)

    # Wertspalten berechnen
    df["Einkaufswert"]  = df["Einkaufsmenge"]  * df["Preis"]
    df["Verkaufswert"]  = df["Verkaufsmenge"]  * df["Preis"]
    df["Lagerwert"]     = df["Lagermenge"]     * df["Preis"]

    # Endergebnis
    return df[[
        "ArtNr","Bezeichnung","Kategorie",
        "Einkaufsmenge","Einkaufswert",
        "Verkaufsmenge","Verkaufswert",
        "Lagermenge","Lagerwert"
    ]]

def main():
    st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
    st.title("📦 Galaxus Sell-out Aggregator")

    # Uploads
    sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte zuerst beide Dateien hochladen: Sell-out-Report und Preisliste.")
        st.stop()

    # Daten laden
    sell_df  = pd.read_excel(sell_file)
    price_df = pd.read_excel(price_file)

    # Matching & Anreicherung
    with st.spinner("🔗 Matching & Anreicherung …"):
        enriched = enrich(sell_df, price_df)

    # Berechnungen
    with st.spinner("📊 Auswertung …"):
        result = compute(enriched)

    # Anzeige
    st.dataframe(result, use_container_width=True)
    # Optional: als Excel zum Download anbieten
    st.download_button(
        "📥 Als Excel herunterladen",
        data=result.to_excel(index=False).encode("utf-8"),
        file_name="aggregat.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
