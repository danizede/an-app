import streamlit as st
import pandas as pd

st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")

# ------------------ Helferfunktionen ------------------

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df.columns nacheinander nach den Kandidaten und
    liefert den ersten gefundenen Spaltennamen zurück.
    """
    for c in candidates:
        if c in df.columns:
            return c
    st.error(f"❌ Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
             f"Verfügbare Spalten: {list(df.columns)}")
    st.stop()

@st.cache_data(show_spinner="🔗 Dateien einlesen …")
def load_data(sell_file: bytes, price_file: bytes):
    sell_df = pd.read_excel(sell_file, engine="openpyxl")
    price_df = pd.read_excel(price_file, engine="openpyxl")
    return sell_df, price_df

@st.cache_data(show_spinner="🔗 Daten matchen & anreichern …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten ermitteln
    s_nr    = find_column(sell_df,  ["Hersteller-Nr.","Produkt ID","ArtNr"],       "Hersteller-Nr.")
    s_ean   = find_column(sell_df,  ["EAN"],                                        "EAN")
    s_eink  = find_column(sell_df,  ["Einkauf","Einkaufsmenge"],                    "Einkauf")
    s_sell  = find_column(sell_df,  ["Verkauf","Sell-Out"],                         "Verkauf")
    s_best  = find_column(sell_df,  ["Verfügbar","Lagerbestand","Bestand"],         "Bestand")
    #
    p_nr    = find_column(price_df, ["Artikelnummer","Hersteller-Nr.","Produkt ID"],"Artikelnummer")
    p_ean   = find_column(price_df, ["GTIN","EAN"],                                 "EAN")
    p_name  = find_column(price_df, ["Bezeichnung","Produktname","Name"],            "Bezeichnung")
    p_add   = find_column(price_df, ["Zusatz","Warengruppe"],                       "Zusatz")
    p_best  = find_column(price_df, ["Bestand","Verfügbar"],                        "Lagerbestand")
    p_price = find_column(price_df, ["NETTO NETTO","NETTO","Einkauf"],              "Preis")

    # 2) Merge auf Hersteller-Nr.
    merged = sell_df.merge(
        price_df[[p_nr,p_ean,p_name,p_add,p_price]],
        left_on=s_nr, right_on=p_nr, how="left", suffixes=("","_pl")
    )

    # 3) Fallback Merge auf EAN, wo noch kein Preis gefunden
    mask = merged[p_price].isna() & merged[s_ean].notna()
    if mask.any():
        fallback = merged.loc[mask, :].merge(
            price_df[[p_ean,p_name,p_add,p_price]],
            left_on=s_ean, right_on=p_ean, how="left"
        )
        for col in [p_name, p_add, p_price]:
            merged.loc[mask, col] = fallback[col].values

    return merged, (s_eink, s_sell, s_best, p_price, p_name, p_add)

@st.cache_data(show_spinner="🔢 Aggregation berechnen …")
def compute_agg(merged: pd.DataFrame, cols):
    s_eink, s_sell, s_best, p_price, p_name, p_add = cols

    # 4) Wert-Spalten berechnen
    merged["Einkaufswert"] = merged[s_eink] * merged[p_price]
    merged["Verkaufswert"] = merged[s_sell] * merged[p_price]
    merged["Lagerwert"]    = merged[s_best] * merged[p_price]

    # 5) Gruppieren
    tbl = (
        merged
        .groupby([p_name, p_add], dropna=False)
        .agg(
            Einkaufsmenge  = (s_eink,  "sum"),
            Einkaufswert   = ("Einkaufswert", "sum"),
            Verkaufsmenge  = (s_sell,  "sum"),
            Verkaufswert   = ("Verkaufswert", "sum"),
            Lagermenge     = (s_best,  "sum"),
            Lagerwert      = ("Lagerwert", "sum"),
        )
        .reset_index()
    )

    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# ------------------ UI & Main ------------------

def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    st.markdown("**1. Sell-out-Report (.xlsx)**")
    sell_file  = st.file_uploader("Drag and drop hier den Sell-out-Report", type="xlsx", key="sell")
    st.markdown("**2. Preisliste (.xlsx)**")
    price_file = st.file_uploader("Drag and drop hier die Preisliste",       type="xlsx", key="price")

    if not sell_file or not price_file:
        st.info("📌 Bitte beide Dateien hochladen, um fortzufahren.")
        return

    sell_df, price_df = load_data(sell_file, price_file)

    merged, cols = enrich(sell_df, price_df)
    tbl, totals = compute_agg(merged, cols)

    # Kennzahlen oben
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
    c2.metric("💰 Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
    c3.metric("💰 Lagerwert (CHF)",    f"{totals['LG']:,.0f}")

    # Tabelle breit ausgeben
    st.write("### Detailauswertung nach Kategorie")
    st.dataframe(tbl, use_container_width=True)

if __name__ == "__main__":
    main()
