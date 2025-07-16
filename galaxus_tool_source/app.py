import streamlit as st
import pandas as pd
import difflib

# Hilfsfunktion: robustes Finden einer Spalte anhand mehrerer Kandidaten-Namen
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    cols = list(df.columns)
    # 1) exakte, fallunabhängige Übereinstimmung
    low2orig = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low2orig:
            return low2orig[cand.lower()]
    # 2) Fuzzy-Matching über difflib
    for cand in candidates:
        matches = difflib.get_close_matches(
            cand, cols, n=1, cutoff=0.6
        )
        if matches:
            return matches[0]
    # 3) nichts gefunden -> Fehlermeldung
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )

# Excel-Datei einlesen (Streamlit-Binärobjekt -> DataFrame)
def load_xlsx(uploaded) -> pd.DataFrame:
    return pd.read_excel(uploaded, engine="openpyxl")

# Daten anreichern: Spalten vereinheitlichen, joinen, neue Spalten berechnen
@st.cache_data(show_spinner="🔗 Matching & Enrichment …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- Spalten in Sell-Out-Report finden ---
    col_s_nr   = find_column(sell_df, ["Hersteller-Nr.", "Artikelnummer", "ArtNr"],  "Hersteller-Nr.")
    col_s_ean  = find_column(sell_df, ["EAN", "GTIN"],                          "EAN")
    col_s_bez  = find_column(sell_df, ["Produktname", "Bezeichnung", "Bez"],     "Bezeichnung")
    col_s_sell = find_column(sell_df, ["Verkauf", "Sell-Out von", "Sold"],       "Verkaufs-Menge")
    col_s_buy  = find_column(sell_df, ["Einkauf", "EK", "Cost"],                 "Einkaufspreis")

    # --- Spalten in Preisliste finden ---
    col_p_nr    = find_column(price_df, ["Artikelnummer", "Hersteller-Nr.", "ArtNr"], "Artikel-Nr.")
    col_p_ean   = find_column(price_df, ["EAN", "GTIN"],                          "EAN")
    col_p_bez   = find_column(price_df, ["Bezeichnung", "Produktname", "Name"],   "Bezeichnung")
    col_p_cat   = find_column(price_df, ["Zusatz", "Kategorie", "Warengruppe"],    "Kategorie")
    col_p_stock = find_column(price_df, ["Bestand", "Verfügbar"],                 "Lagerbestand")
    col_p_price = find_column(price_df, ["NETTO NETTO", "Preis", "VK"],           "Netto-Verkaufspreis")

    # --- Standardisierte Spaltennamen ---
    sell = sell_df.rename(columns={
        col_s_nr:   "HerstellerNr",
        col_s_ean:  "EAN",
        col_s_bez:  "Bezeichnung",
        col_s_sell: "Verkauf",
        col_s_buy:  "Einkauf"
    })[["HerstellerNr", "EAN", "Bezeichnung", "Verkauf", "Einkauf"]]

    price = price_df.rename(columns={
        col_p_nr:    "HerstellerNr",
        col_p_ean:   "EAN",
        col_p_bez:   "Bezeichnung",
        col_p_cat:   "Kategorie",
        col_p_stock: "Bestand",
        col_p_price: "Preis"
    })[["HerstellerNr", "EAN", "Bezeichnung", "Kategorie", "Bestand", "Preis"]]

    # --- Joinen auf EAN (falls EAN fehlt, könnte man alternativ auf HerstellerNr. joinen) ---
    merged = pd.merge(
        sell,
        price,
        on="EAN",
        how="left",
        suffixes=("", "_price")
    )

    # Fehlende Kategorien aus Preis-Liste übernehmen
    merged["Kategorie"] = merged["Kategorie"].fillna(merged["Bezeichnung_price"])

    # --- Berechnete Spalten ---
    merged["Verkaufswert"] = merged["Verkauf"] * merged["Preis"]
    merged["Einkaufswert"] = merged["Verkauf"] * merged["Einkauf"]
    merged["Lagerwert"]    = merged["Bestand"] * merged["Preis"]

    return merged

# Aggregationstabelle + Gesamtsummen
@st.cache_data(show_spinner="⚖️ Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,float]]:
    # Gruppieren nach Artikelidentität
    agg = (
        df.groupby(
            ["HerstellerNr", "Bezeichnung", "EAN", "Kategorie"],
            as_index=False,
        )
        .agg(
            Verkauf      = ("Verkauf",      "sum"),
            Verkaufswert = ("Verkaufswert", "sum"),
            Einkaufswert = ("Einkaufswert", "sum"),
            Lagerwert    = ("Lagerwert",    "first"),  # Bestand*Preis ist pro Artikel konstant
        )
    )

    totals = {
        "VK": agg["Verkaufswert"].sum(),
        "EK": agg["Einkaufswert"].sum(),
        "LG": agg["Lagerwert"].sum(),
    }
    return agg, totals

# ---- Streamlit UI ----
def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    st.subheader("Sell-out-Report (.xlsx)")
    sell_file  = st.file_uploader("📂 Datei hier ablegen", type="xlsx", key="sell")

    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("📂 Datei hier ablegen", type="xlsx", key="price")

    if sell_file and price_file:
        with st.spinner("🔍 Lese Dateien …"):
            sell_df  = load_xlsx(sell_file)
            price_df = load_xlsx(price_file)

        with st.spinner("🔗 Daten anreichern …"):
            enriched = enrich(sell_df, price_df)

        with st.spinner("⚖️ Werte aggregieren …"):
            table, totals = compute_agg(enriched)

        # Metriken oben
        c1, c2, c3 = st.columns(3)
        c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
        c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
        c3.metric("Lagerwert (CHF)",    f"{totals['LG']:,.0f}")

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()
