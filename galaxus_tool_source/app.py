import streamlit as st
import pandas as pd

st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")

# -------------------------------------------------------------------
# Hilfsfunktion zum flexiblen Finden von Spalten anhand mehrerer Kandidaten
# -------------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Versucht, in df.columns eine Spalte zu finden, 
    deren Name in der Liste candidates steht.
    purpose dient nur für aussagekräftige Fehlermeldungen.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte für {purpose} fehlt – gesucht unter {candidates}")

# -------------------------------------------------------------------
# Enrichment: Artikel-Nr. und EAN matchen, fehlende Bezeichnung/Kategorie/Preis füllen
# -------------------------------------------------------------------
@st.cache_data(show_spinner="🔗 Daten anreichern …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # Spaltennamen in der Preisliste ermitteln
    p_nr  = find_column(price,    ["Artikelnr","Hersteller-Nr."],    "Artikelnr")
    p_ean = find_column(price,    ["GTIN","EAN"],                    "EAN")
    p_bez = find_column(price,    ["Bezeichnung","Bez"],            "Bezeichnung")
    p_cat = find_column(price,    ["Kategorie","Zusatz"],           "Kategorie")
    p_pr  = find_column(price,    ["Preis","VK","Verkaufspreis"],   "Preis")

    # Normierte Spaltennamen für den Merge
    price = price.rename(columns={
        p_nr:  "Artikelnr",
        p_ean: "GTIN",
        p_bez: "Bezeichnung",
        p_cat: "Kategorie",
        p_pr:  "Preis"
    })

    # 1) Matching über Hersteller-Nr. / Artikelnr
    s_nr = find_column(sell, ["Artikelnr","Hersteller-Nr."], "Artikelnr")
    merged = sell.merge(
        price[["Artikelnr","Bezeichnung","Kategorie","Preis"]],
        left_on = s_nr,
        right_on = "Artikelnr",
        how = "left"
    )

    # 2) Matching über GTIN / EAN, wenn Preis noch fehlt
    s_ean = find_column(merged, ["GTIN","EAN"], "EAN")
    mask = merged["Preis"].isna() & merged[s_ean].notna()
    if mask.any():
        df2 = (
            merged[mask]
            .merge(
                price.drop_duplicates("GTIN"),
                left_on  = s_ean,
                right_on = "GTIN",
                how      = "left"
            )
            [["Bezeichnung","Kategorie","Preis"]]
        )
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask, col] = df2[col].values

    # 3) (Optional) Fuzzy-Match etc. – hier weglassen oder nachrüsten

    return merged

# -------------------------------------------------------------------
# Aggregation: Summieren von Einkauf, Verkauf, Bestand und Wertegrössen
# -------------------------------------------------------------------
@st.cache_data(show_spinner="📊 Aggregieren …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,float]]:
    # Spaltennamen im Sell-out-Report
    col_nr   = find_column(df, ["Artikelnr","Hersteller-Nr."], "Artikelnr")
    col_bez  = "Bezeichnung"
    col_cat  = "Kategorie"
    col_best = find_column(df, ["Bestand","Lagerbestand","Verfügbar"], "Bestand")
    col_sell = find_column(df, ["Verkauf","Verkaufte Menge","VerkaufteStück"], "Verkauf")
    col_buy  = find_column(df, ["Einkauf","Bestellmenge"], "Einkauf")

    # Wertspalten berechnen
    df["Lagerwert"]      = df[col_best] * df["Preis"]
    df["Verkaufswert"]   = df[col_sell] * df["Preis"]
    df["Einkaufswert"]   = df[col_buy]  * df["Preis"]

    # Gruppieren auf Artikel-Ebene
    tbl = (
        df
        .groupby([col_nr, col_bez, col_cat], dropna=False)
        .agg({
            col_buy:       "sum",
            col_sell:      "sum",
            col_best:      "sum",
            "Einkaufswert":   "sum",
            "Verkaufswert":    "sum",
            "Lagerwert":       "sum",
        })
        .reset_index()
    )

    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum()
    }

    return tbl, totals

# -------------------------------------------------------------------
# UI: Dateien hochladen und Ergebnis anzeigen
# -------------------------------------------------------------------
st.title("📦 Galaxus Sell-out Aggregator")

sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx")

if sell_file and price_file:
    sell_df  = pd.read_excel(sell_file)
    price_df = pd.read_excel(price_file)

    enriched, = (enrich(sell_df, price_df),)  # entpacken
    details, totals = compute_agg(enriched)

    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
    c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
    c3.metric("Lagerwert (CHF)",    f"{totals['LG']:,.0f}")

    st.dataframe(details, use_container_width=True)
else:
    st.info("Bitte Sell-out-Report und Preisliste hochladen, um die Auswertung zu starten.")
