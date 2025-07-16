import streamlit as st
import pandas as pd

# -----------------------------------------------------------
# Hilfsfunktion, um eine Spalte aus mehreren Kandidaten zu finden
# Bei Fehlen zeigt sie eine verständliche Fehlermeldung und bricht ab.
# -----------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    st.error(
        f"❌ Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )
    st.stop()

# -----------------------------------------------------------
# XLSX‐Loader mit Caching
# -----------------------------------------------------------
@st.cache_data
def load_xlsx(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)

# -----------------------------------------------------------
# Enrichment: Zusammenführen von Sell‐Out und Preisliste
# -----------------------------------------------------------
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # 1) Spaltendetektion im Sell‐Out‐Report
    col_nr   = find_column(sell_df, ["Artikelnummer", "ArtNr", "Hersteller-Nr."], "ArtNr")
    col_ean  = find_column(sell_df, ["GTIN", "EAN"],                          "EAN")
    col_bez  = find_column(sell_df, ["Bezeichnung", "Bez", "Name"],            "Bezeichnung")
    col_best = find_column(sell_df, ["Bestand", "Lagerbestand"],               "Bestand")
    col_sell = find_column(sell_df, ["Verkauf", "Sell-Out", "Stück"],          "Verkauf")

    # 2) Spaltendetektion in der Preisliste
    p_nr     = find_column(price_df, ["Artikelnummer", "ArtNr", "Hersteller-Nr."], "ArtNr")
    p_ean    = find_column(price_df, ["GTIN", "EAN"],                             "EAN")
    p_bez    = find_column(price_df, ["Bezeichnung", "Bez", "Name"],               "Bezeichnung")
    p_cat    = find_column(price_df, ["Zusatz", "Kategorie", "Warengruppe"],       "Kategorie")
    p_price  = find_column(price_df, ["Preis", "Verkaufspreis", "NETTO NETTO", "VK"], "Preis")

    # 3) Einheitliche Spaltennamen
    sell = sell_df.rename(columns={
        col_nr:   "ArtNr",
        col_ean:  "EAN",
        col_bez:  "Bezeichnung",
        col_best: "Bestand",
        col_sell: "Verkauf",
    })
    price = price_df.rename(columns={
        p_nr:    "ArtNr",
        p_ean:   "EAN",
        p_bez:   "Bezeichnung",
        p_cat:   "Kategorie",
        p_price: "Preis",
    })

    # 4) 1. Merge‐Versuch: nach ArtNr
    merged = sell.merge(
        price[["ArtNr", "Bezeichnung", "Kategorie", "Preis"]],
        on="ArtNr", how="left"
    )

    # 5) 2. Merge‐Versuch: nach EAN, falls Preis noch fehlt
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        df2 = merged.loc[mask, ["EAN"]].merge(
            price[["EAN", "Bezeichnung", "Kategorie", "Preis"]],
            on="EAN", how="left"
        )
        # direkt Werte übertragen
        for col in ["Bezeichnung", "Kategorie", "Preis"]:
            merged.loc[mask, col] = df2[col].values

    # 6) 3. Fuzzy‐Fallback: nach den ersten zwei Wörtern in der Bezeichnung
    def first_two_words(s: str) -> str:
        parts = str(s).split()
        return " ".join(parts[:2]) if len(parts) >= 2 else s

    merged["tkn"] = merged["Bezeichnung"].apply(first_two_words)
    price["tkn"] = price["Bezeichnung"].apply(first_two_words)
    mask2 = merged["Preis"].isna() & merged["tkn"].notna()
    if mask2.any():
        df3 = merged.loc[mask2, ["tkn"]].merge(
            price[["tkn", "Preis"]].drop_duplicates("tkn"),
            on="tkn", how="left"
        )
        merged.loc[mask2, "Preis"] = df3["Preis"].values

    # 7) Rechnungen
    merged["Lagerwert"]    = merged["Bestand"] * merged["Preis"]
    merged["Verkaufswert"] = merged["Verkauf"] * merged["Preis"]

    return merged

# -----------------------------------------------------------
# Aggregation und Kennzahlen
# -----------------------------------------------------------
@st.cache_data(show_spinner="📊 Aggregation …")
def compute_agg(df: pd.DataFrame):
    # Stelle sicher, dass alle Felder existieren
    for col in ["Lagerwert", "Verkaufswert"]:
        if col not in df.columns:
            df[col] = 0.0

    # Gruppieren auf Artikel‐Ebene (hier nach Artikelnummer + Zusatz, falls vorhanden)
    group_keys = ["ArtNr"]
    if "Kategorie" in df.columns:
        group_keys.append("Kategorie")
    if "Bezeichnung" in df.columns:
        group_keys.append("Bezeichnung")

    tbl = (
        df
        .groupby(group_keys, dropna=False)
        .agg(
            Verkaufswert=("Verkaufswert", "sum"),
            Lagerwert=("Lagerwert", "sum"),
        )
        .reset_index()
    )

    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# -----------------------------------------------------------
# Hauptteil der App
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Galaxus Sell‐out Aggregator", layout="wide")
    st.title("📦 Galaxus Sell‐out Aggregator")

    st.markdown("**Sell-out-Report (.xlsx)**")
    sell_file  = st.file_uploader("Drag and drop hier den Sell-out-Report", type="xlsx")
    st.markdown("**Preisliste (.xlsx)**")
    price_file = st.file_uploader("Drag and drop hier die Preisliste", type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte lade zuerst beide Dateien hoch, um fortzufahren.")
        st.stop()

    # Daten einlesen
    sell_df  = load_xlsx(sell_file)
    price_df = load_xlsx(price_file)

    # Anreicherung + Aggregation
    enriched, totals = compute_agg(enrich(sell_df, price_df))

    # Kennzahlen‐Metriken
    c1, c2 = st.columns(2)
    c1.metric("Verkaufswert (CHF)",     f"{totals['VK']:,.0f}".replace(",", "'"))
    c2.metric("Lagerwert (CHF)",        f"{totals['LG']:,.0f}".replace(",", "'"))

    # Detail‐Tabelle
    st.markdown("---")
    st.dataframe(enriched, use_container_width=True)

if __name__ == "__main__":
    main()
