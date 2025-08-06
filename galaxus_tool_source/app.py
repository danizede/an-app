import streamlit as st
import pandas as pd

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

def clean_numeric(series: pd.Series) -> pd.Series:
    """Strings bereinigen und in float umwandeln."""
    return (
        series
        .astype(str)
        .str.replace("'", "", regex=False)   # Apostroph-Tausender
        .str.replace(" ", "", regex=False)   # Leerzeichen
        .str.replace(",", ".", regex=False)  # Dezimalkomma → Punkt
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

@st.cache_data(ttl=3600)
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten finden
    sn = find_column(sell_df,  ["Hersteller-Nr.", "Produkt ID", "EAN"], "ArtNr im Sell-Out")
    pe = find_column(sell_df,  ["Einkauf"],                        "Einkaufsmenge")
    ps = find_column(sell_df,  ["Verkauf"],                        "Verkaufsmenge")
    nb = find_column(sell_df,  ["Produktname", "Bezeichnung"],      "Bezeichnung")
    
    pn = find_column(price_df, ["Artikelnummer", "Hersteller-Nr.", "GTIN", "EAN"], "ArtNr in Preisliste")
    pl = find_column(price_df, ["NETTO NETTO", "Netto", "Preis"],    "Preis in PL")
    stc= find_column(price_df, ["Bestand", "Verfügbar"],            "Lagermenge")
    cat= find_column(price_df, ["Zusatz", "Kategorie", "Warengruppe"],"Kategorie")
    
    # 2) Umbenennen auf Standard
    sell = sell_df.rename(columns={
        sn:  "ArtNr",
        pe:  "Einkaufsmenge",
        ps:  "Verkaufsmenge",
        nb:  "Bezeichnung"
    })
    price = price_df.rename(columns={
        pn:  "ArtNr",
        pl:  "Preis in PL",
        stc: "Lagermenge",
        cat: "Kategorie"
    })
    
    # 3) Bereinigen & Typkonvertierung
    sell["Einkaufsmenge"]  = clean_numeric(sell["Einkaufsmenge"])
    sell["Verkaufsmenge"]  = clean_numeric(sell["Verkaufsmenge"])
    price["Lagermenge"]    = clean_numeric(price["Lagermenge"])
    price["Preis in PL"]   = clean_numeric(price["Preis in PL"])
    
    # 4) Zusammenführen
    merged = pd.merge(
        sell[["ArtNr", "Bezeichnung", "Einkaufsmenge", "Verkaufsmenge"]],
        price[["ArtNr", "Kategorie", "Lagermenge", "Preis in PL"]],
        on="ArtNr",
        how="left",
        validate="m:1"
    )
    
    # 5) Werte berechnen
    merged["Einkaufswert"] = merged["Einkaufsmenge"] * merged["Preis in PL"]
    merged["Verkaufswert"] = merged["Verkaufsmenge"] * merged["Preis in PL"]
    merged["Lagerwert"]    = merged["Lagermenge"]    * merged["Preis in PL"]
    
    return merged[[
        "ArtNr", "Bezeichnung", "Kategorie",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge", "Lagerwert"
    ]]

@st.cache_data(ttl=3600)
def compute_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .groupby(["ArtNr", "Bezeichnung", "Kategorie"], as_index=False)
        .agg({
            "Einkaufsmenge": "sum",
            "Einkaufswert":  "sum",
            "Verkaufsmenge": "sum",
            "Verkaufswert":  "sum",
            "Lagermenge":    "sum",
            "Lagerwert":     "sum",
        })
    )

def main():
    st.set_page_config(layout="wide")
    st.title("📦 Galaxus Sell-out Aggregator")
    
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    with c2:
        f2 = st.file_uploader("Preisliste (.xlsx)",    type="xlsx")
    
    if f1 and f2:
        try:
            sell_df  = pd.read_excel(f1)
            price_df = pd.read_excel(f2)
            
            enriched = enrich(sell_df, price_df)
            agg      = compute_agg(enriched)
            
            st.subheader("Aggregierte Kennzahlen pro Artikel")
            st.dataframe(agg, use_container_width=True)
        except Exception as e:
            st.error(f"Fehler während Berechnung: {e}")
    else:
        st.info("Bitte beide Dateien hochladen, um zu starten.")

if __name__ == "__main__":
    main()
