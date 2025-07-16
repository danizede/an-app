import streamlit as st
import pandas as pd
import difflib

# 1) Fuzzy-Matching-Helfer
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    cols = list(df.columns)
    low2orig = {c.lower(): c for c in cols}
    # a) exakte (case-insensitive)
    for cand in candidates:
        key = cand.lower()
        if key in low2orig:
            return low2orig[key]
    # b) fuzzy
    for cand in candidates:
        m = difflib.get_close_matches(cand, cols, n=1, cutoff=0.6)
        if m:
            return m[0]
    # c) nichts gefunden
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )

# 2) Excel einlesen
def load_xlsx(uploader) -> pd.DataFrame:
    return pd.read_excel(uploader, engine="openpyxl")

# 3) Daten anreichern
@st.cache_data(show_spinner="🔗 Matching & Enrichment …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- Sell-Out-Report: nur Verkaufsmenge + Schlüsselspalten ---
    c_s_nr   = find_column(sell_df, ["Artikelnummer","Hersteller-Nr.","ArtNr"], "Artikel-Nr.")
    c_s_ean  = find_column(sell_df, ["EAN","GTIN"],                      "EAN")
    c_s_bez  = find_column(sell_df, ["Produktname","Bezeichnung","Bez"],  "Bezeichnung")
    c_s_sell = find_column(sell_df, ["Verkauf","Sell-Out von","Sold"],    "Verkaufs-Menge")

    sell = sell_df.rename(columns={
        c_s_nr:   "HerstellerNr",
        c_s_ean:  "EAN",
        c_s_bez:  "Bezeichnung",
        c_s_sell: "Verkauf"
    })[["HerstellerNr","EAN","Bezeichnung","Verkauf"]]

    # --- Preisliste: Einkaufspreis, Verkaufspreis, Bestand, Kategorie ---
    c_p_nr    = find_column(price_df, ["Artikelnummer","Hersteller-Nr.","ArtNr"], "Artikel-Nr.")
    c_p_ean   = find_column(price_df, ["EAN","GTIN"],                           "EAN")
    c_p_bez   = find_column(price_df, ["Produktname","Bezeichnung","Bez"],       "Bezeichnung")
    c_p_cat   = find_column(price_df, ["Zusatz","Kategorie","Warengruppe"],      "Kategorie")
    c_p_stock = find_column(price_df, ["Bestand","Verfügbar"],                  "Lagerbestand")
    c_p_price = find_column(price_df, ["Preis","VK","Netto"],                   "Verkaufspreis")
    c_p_cost  = find_column(price_df, ["Einkauf","EK","Cost"],                  "Einkaufspreis")

    price = price_df.rename(columns={
        c_p_nr:    "HerstellerNr",
        c_p_ean:   "EAN",
        c_p_bez:   "Bezeichnung",
        c_p_cat:   "Kategorie",
        c_p_stock: "Bestand",
        c_p_price: "Preis",
        c_p_cost:  "Einkauf"
    })[["HerstellerNr","EAN","Bezeichnung","Kategorie","Bestand","Preis","Einkauf"]]

    # --- Zusammenführen auf EAN ---
    merged = pd.merge(
        sell,
        price,
        on="EAN",
        how="left",
        suffixes=("","_price")
    )

    # falls Kategorie fehlt, aus Preis-Bezeichnung nachziehen
    merged["Kategorie"] = merged["Kategorie"].fillna(merged["Bezeichnung_price"])

    # --- Werte berechnen ---
    merged["Verkaufswert"]  = merged["Verkauf"] * merged["Preis"]
    merged["Einkaufswert"]  = merged["Verkauf"] * merged["Einkauf"]
    merged["Lagerwert"]     = merged["Bestand"] * merged["Preis"]

    return merged

# 4) Aggregation und Totals
@st.cache_data(show_spinner="⚖️ Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,float]]:
    agg = (
        df
        .groupby(
            ["HerstellerNr","Bezeichnung","EAN","Kategorie"],
            as_index=False
        )
        .agg(
            Verkauf      = ("Verkauf",      "sum"),
            Verkaufswert = ("Verkaufswert", "sum"),
            Einkaufswert = ("Einkaufswert", "sum"),
            Lagerwert    = ("Lagerwert",    "first")
        )
    )
    totals = {
        "VK": agg["Verkaufswert"].sum(),
        "EK": agg["Einkaufswert"].sum(),
        "LG": agg["Lagerwert"].sum(),
    }
    return agg, totals

# 5) Streamlit-UI
def main():
    st.title("📦 Galaxus Sell-out Aggregator")
    st.subheader("1) Sell-Out-Report hochladen")
    sell_file  = st.file_uploader("Drag & drop hier den Sell-Out-Report", type="xlsx", key="sell")
    st.subheader("2) Preisliste hochladen")
    price_file = st.file_uploader("Drag & drop hier die Preisliste",   type="xlsx", key="price")

    if sell_file and price_file:
        with st.spinner("⏳ Lese Dateien …"):
            sell_df  = load_xlsx(sell_file)
            price_df = load_xlsx(price_file)
        with st.spinner("🔗 Anreichern …"):
            enriched = enrich(sell_df, price_df)
        with st.spinner("⚖️ Aggregieren …"):
            table, totals = compute_agg(enriched)

        # Top-Metriken
        c1,c2,c3 = st.columns(3)
        c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
        c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
        c3.metric("Lagerwert (CHF)",    f"{totals['LG']:,.0f}")

        st.dataframe(table, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
