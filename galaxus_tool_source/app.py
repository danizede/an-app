import streamlit as st
import pandas as pd
from thefuzz import process

st.set_page_config(layout="wide")

@st.cache_data
def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    purpose: str
) -> str:
    """
    Sucht in df.columns nach einer der Kandidaten.
    Gibt den ersten passenden Spaltennamen zurück oder feuert KeyError.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- 1) Spalten in Sell-out-Report finden -----------------------
    s_nr    = find_column(sell_df, ["ArtikelNr","Hersteller-Nr.","ArtNr"], "Artikelnr")
    s_ean   = find_column(sell_df, ["EAN","GTIN"], "EAN")
    s_name  = find_column(sell_df, ["Bezeichnung","Bez","Name"], "Bezeichnung")
    s_best  = find_column(sell_df, ["Bestand","Lagerbestand"], "Lagerbestand")
    s_sell  = find_column(sell_df, ["Verkauf","Sell-Out von","Sell-Out bis"], "Verkauf")

    # --- 2) Spalten in Preisliste finden ----------------------------
    p_nr    = find_column(price_df, ["Artikelnummer","Hersteller-Nr.","ArtNr"], "Artikelnr")
    p_ean   = find_column(price_df, ["EAN","GTIN"], "EAN")
    p_name  = find_column(price_df, ["Produktname","Bezeichnung","Bez"], "Bezeichnung")
    p_cat   = find_column(price_df, ["Zusatz","Kategorie","Warengruppe"], "Kategorie")
    p_price = find_column(price_df, ["Preis","NETTO","VK"], "Preis")

    # --- 3) Spalten umbenennen & Merge 1: Artikelnr -----------------
    sell  = sell_df.rename(columns={
        s_nr:"Artikelnr", s_ean:"EAN", s_name:"Bezeichnung",
        s_best:"Lagerbestand", s_sell:"Verkauf"
    })
    price = price_df.rename(columns={
        p_nr:"Artikelnr", p_ean:"EAN",
        p_name:"Bezeichnung", p_cat:"Kategorie", p_price:"Preis"
    })[
        ["Artikelnr","EAN","Bezeichnung","Kategorie","Preis"]
    ]
    merged = sell.merge(price, on="Artikelnr", how="left")

    # --- 4) Fallback-Match 1: EAN -----------------------------------
    mask1 = merged["Preis"].isna() & merged["EAN"].notna()
    if mask1.any():
        fb = (
            merged[mask1]
            .merge(
                price.drop_duplicates("EAN"),
                left_on="EAN", right_on="EAN",
                how="left"
            )
        )
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask1, col] = fb[col].values

    # --- 5) Fallback-Match 2: Fuzzy über 1.2 Worte der Bezeichnung ---
    # Tokens lower-case
    price["tkn"]  = price["Bezeichnung"].str.lower().str.split().str[:2].str.join(" ")
    merged["tkn"] = merged["Bezeichnung"].str.lower().str.split().str[:2].str.join(" ")
    mask2 = merged["Preis"].isna() & merged["tkn"].notna()
    if mask2.any():
        # für jede fehlende Zeile einen fuzzy-Score holen
        for idx in merged[mask2].index:
            t = merged.at[idx, "tkn"]
            match, score = process.extractOne(t, price["tkn"], score_cutoff=80) or (None,0)
            if match:
                row = price[price["tkn"] == match].iloc[0]
                merged.at[idx, ["Bezeichnung","Kategorie","Preis"]] = (
                    row[["Bezeichnung","Kategorie","Preis"]].values
                )
    merged.drop(columns="tkn", inplace=True)

    # --- 6) Mengen & Werte berechnen --------------------------------
    # Verkaufs- und Lagermengen aus den Original-Spalten
    merged["Verkaufsmenge"] = merged["Verkauf"]
    merged["Lagermenge"]    = merged["Lagerbestand"]
    # Einkaufsmenge = alles, was reingekommen ist
    merged["Einkaufsmenge"] = merged["Verkaufsmenge"] + merged["Lagermenge"]

    # Werte = Menge * Preis
    merged["Verkaufswert"]  = merged["Verkaufsmenge"] * merged["Preis"]
    merged["Lagerwert"]     = merged["Lagermenge"]    * merged["Preis"]
    merged["Einkaufswert"]  = merged["Einkaufsmenge"] * merged["Preis"]

    return merged

@st.cache_data
def compute_agg(df: pd.DataFrame):
    tbl = (
        df
        .groupby(["Artikelnr","Bezeichnung","Kategorie"], as_index=False)
        .agg(
            Einkaufsmenge=("Einkaufsmenge","sum"),
            Einkaufswert  =("Einkaufswert","sum"),
            Verkaufsmenge =("Verkaufsmenge","sum"),
            Verkaufswert  =("Verkaufswert","sum"),
            Lagermenge    =("Lagermenge","sum"),
            Lagerwert     =("Lagerwert","sum"),
        )
    )
    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    price_file = st.file_uploader("Preisliste (.xlsx)",   type="xlsx")

    if sell_file and price_file:
        sell_df  = load_xlsx(sell_file)
        price_df = load_xlsx(price_file)

        enriched = enrich(sell_df, price_df)
        tbl, tot = compute_agg(enriched)

        # KPI-Metriken
        c1,c2,c3 = st.columns(3)
        c1.metric("Einkaufswert",  f"CHF {tot['EK']:,.0f}")
        c2.metric("Verkaufswert",  f"CHF {tot['VK']:,.0f}")
        c3.metric("Lagerwert",     f"CHF {tot['LG']:,.0f}")

        # Ausgabetabelle_FULLWIDTH
        st.dataframe(tbl, use_container_width=True)

    else:
        st.info("Bitte sowohl Sell-out-Report als auch Preisliste hochladen.")

if __name__ == "__main__":
    main()
