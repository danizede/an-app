import streamlit as st
import pandas as pd
from thefuzz import process

st.set_page_config(layout="wide")

@st.cache_data
def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Versucht nacheinander jeden Namen in `candidates`, ob er in df.columns ist.
    Gibt den Spaltennamen zurück oder wirft KeyError mit allen verfügbaren Spalten.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(show_spinner="🔗 Matching & Enrichment …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # ——— 1) Sell-out-Report Spalten ermitteln ————————————————
    s_nr    = find_column(sell_df, ["Artikelnummer","Artikelnr","Hersteller-Nr.","Produkt ID"], "Artikelnr")
    s_ean   = find_column(sell_df, ["EAN","GTIN"], "EAN")
    s_name  = find_column(sell_df, ["Bezeichnung","Bez","Name","Produktname"], "Bezeichnung")
    s_best  = find_column(sell_df, ["Verfügbar","Bestand","Lagerbestand"], "Bestand")
    s_sell  = find_column(sell_df, ["Verkauf","Sell-Out von","Sell-Out bis"], "Verkauf")

    # ——— 2) Preisliste Spalten ermitteln ————————————————
    p_nr    = find_column(price_df, ["Artikelnummer","Artikelnr","Hersteller-Nr.","Produkt ID"], "Artikelnr")
    p_ean   = find_column(price_df, ["EAN","GTIN"], "EAN")
    p_name  = find_column(price_df, ["Bezeichnung","Bez","Name","Produktname"], "Bezeichnung")
    p_cat   = find_column(price_df, ["Kategorie","Zusatz","Warengruppe"], "Kategorie")
    p_price = find_column(price_df, ["Preis","NETTO","VK"], "Preis")

    # ——— 3) RENAME für Standard-Namen ————————————————
    sell = sell_df.rename(columns={
        s_nr:"Artikelnr",
        s_ean:"EAN",
        s_name:"Bezeichnung",
        s_best:"Lagerbestand",
        s_sell:"Verkauf"
    })
    price = price_df.rename(columns={
        p_nr:"Artikelnr",
        p_ean:"EAN",
        p_name:"Bezeichnung",
        p_cat:"Kategorie",
        p_price:"Preis"
    })[["Artikelnr","EAN","Bezeichnung","Kategorie","Preis"]]

    # ——— 4) Merge 1: über Art.Nr. ——————————————————————
    merged = sell.merge(price, on="Artikelnr", how="left")

    # ——— 5) Fallback über EAN ——————————————————————
    mask1 = merged["Preis"].isna() & merged["EAN"].notna()
    if mask1.any():
        fb = (
            merged[mask1]
            .merge(price.drop_duplicates("EAN"), on="EAN", how="left")
        )
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask1, col] = fb[col].values

    # ——— 6) Fallback fuzzy über erste 2 Wörter ——————————
    price["tkn"]   = price["Bezeichnung"].str.lower().str.split().str[:2].str.join(" ")
    merged["tkn"]  = merged["Bezeichnung"].str.lower().str.split().str[:2].str.join(" ")
    mask2 = merged["Preis"].isna() & merged["tkn"].notna()
    if mask2.any():
        for idx in merged[mask2].index:
            tok = merged.at[idx,"tkn"]
            match = process.extractOne(tok, price["tkn"], score_cutoff=80)
            if match:
                best, score = match
                row = price[price["tkn"]==best].iloc[0]
                merged.at[idx, ["Bezeichnung","Kategorie","Preis"]] = (
                    row[["Bezeichnung","Kategorie","Preis"]].values
                )
    merged.drop(columns="tkn", inplace=True)

    # ——— 7) Mengen & Werte ——————————————————————————
    merged["Verkaufsmenge"] = merged["Verkauf"].fillna(0)
    merged["Lagermenge"]    = merged["Lagerbestand"].fillna(0)
    merged["Einkaufsmenge"] = merged["Verkaufsmenge"] + merged["Lagermenge"]

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
            Einkaufsmenge = ("Einkaufsmenge","sum"),
            Einkaufswert   = ("Einkaufswert","sum"),
            Verkaufsmenge  = ("Verkaufsmenge","sum"),
            Verkaufswert   = ("Verkaufswert","sum"),
            Lagermenge     = ("Lagermenge","sum"),
            Lagerwert      = ("Lagerwert","sum"),
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
    price_file = st.file_uploader("Preisliste (.xlsx)",    type="xlsx")

    if not (sell_file and price_file):
        st.info("Bitte beides hochladen: Sell-out-Report & Preisliste.")
        return

    sell_df  = load_xlsx(sell_file)
    price_df = load_xlsx(price_file)

    enriched = enrich(sell_df, price_df)
    tbl, tot = compute_agg(enriched)

    c1,c2,c3 = st.columns(3)
    c1.metric("Einkaufswert",  f"CHF {tot['EK']:,.0f}")
    c2.metric("Verkaufswert",  f"CHF {tot['VK']:,.0f}")
    c3.metric("Lagerwert",     f"CHF {tot['LG']:,.0f}")

    st.dataframe(tbl, use_container_width=True)

if __name__=="__main__":
    main()
