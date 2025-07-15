import streamlit as st
import pandas as pd

# ----------------------------------------
# Hilfsfunktion: Spalte suchen
# ----------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df.columns nacheinander alle Einträge aus candidates
    und liefert den ersten Treffer zurück.
    Wenn nichts gefunden wird, wirft es einen KeyError mit allen Kandidaten.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

# ----------------------------------------
# Daten einlesen
# ----------------------------------------
@st.cache_data(show_spinner="📥 Daten laden …")
def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)

# ----------------------------------------
# Matching & Anreicherung
# ----------------------------------------
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten in der Preisliste finden
    col_nr    = find_column(price, ["Artikelnr", "Artikelnummer", "Hersteller-Nr.", "ArtNr"],     "Artikelnr")
    col_ean   = find_column(price, ["EAN", "GTIN"],                                              "EAN/GTIN")
    col_bez   = find_column(price, ["Bezeichnung", "Bez", "Name"],                                "Bezeichnung")
    col_cat   = find_column(price, ["Kategorie", "Zusatz", "Warengruppe"],                        "Kategorie")
    col_price = find_column(price, ["Preis", "EK", "Netto Netto", "NETTO NETTO"],                 "Preis")

    # Aufräumen & umbenennen
    price_sel = (
        price
        .loc[:, [col_nr, col_ean, col_bez, col_cat, col_price]]
        .rename(columns={
            col_nr:    "Artikelnr",
            col_ean:   "EAN",
            col_bez:   "Bezeichnung",
            col_cat:   "Kategorie",
            col_price: "Preis",
        })
    )

    # --- 1) Matching über Hersteller-Nr. ---
    merged = sell.merge(
        price_sel,
        left_on="Hersteller-Nr.", right_on="Artikelnr",
        how="left",
        suffixes=("", "_price")
    )

    # --- 2) Falls noch kein Preis, Matching über GTIN/EAN ---
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        df2 = (
            merged.loc[mask, ["EAN"]]
            .merge(
                price_sel.drop_duplicates("EAN"),
                on="EAN",
                how="left"
            )[
                ["Bezeichnung", "Kategorie", "Preis"]
            ]
        )
        # Spaltenweise ersetzen
        for col in ["Bezeichnung", "Kategorie", "Preis"]:
            merged.loc[mask, col] = df2[col].values

    # Hier könnten noch weitere Fuzzy-Matches folgen...

    return merged

# ----------------------------------------
# Aggregation
# ----------------------------------------
@st.cache_data(show_spinner="🔢 Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    tbl = (
        df
        .groupby(["Artikelnr", "Bezeichnung", "Kategorie"], as_index=False)
        .agg(
            Verkaufswert=("Verkaufswert", "sum"),
            Einkaufswert=("Einkaufswert", "sum"),
            Lagerwert=("Lagerwert", "sum"),
        )
    )
    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# ----------------------------------------
# App-UI
# ----------------------------------------
def main():
    st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
    st.title("📦 Galaxus Sell-out Aggregator")

    sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    price_file = st.file_uploader("Preisliste (.xlsx)",     type="xlsx")

    if sell_file and price_file:
        sell_df  = load_xlsx(sell_file)
        price_df = load_xlsx(price_file)

        # Anreichern & Aggrieren
        enriched, totals = compute_agg(enrich(sell_df, price_df))

        # Kennzahlen
        c1, c2, c3 = st.columns(3)
        c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
        c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
        c3.metric("Lagerwert    (CHF)", f"{totals['LG']:,.0f}")

        # Tabelle anzeigen
        st.dataframe(enriched, use_container_width=True)

    else:
        st.info("Bitte beide Dateien hochladen, damit der Sell-out-Report angereichert und aggregiert werden kann.")

if __name__ == "__main__":
    main()
