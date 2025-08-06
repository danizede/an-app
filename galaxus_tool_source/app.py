import pandas as pd
import streamlit as st

# Kandidaten-Listen für Spaltennamen
ALIAS_NR    = ["Artikelnr", "Hersteller-Nr.", "ArtNr"]
ALIAS_EAN   = ["EAN", "GTIN"]
ALIAS_NAME  = ["Bezeichnung", "Bez", "Name", "Produktname"]
ALIAS_CAT   = ["Warengruppe", "Kategorie", "Zusatz"]
ALIAS_PREIS = ["Preis", "VK", "NETTO"]

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """Sucht in df.columns nach einer der Kandidaten und gibt den gefundenen Spaltennamen zurück."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # 1) Leitspalten festlegen
    s_nr    = find_column(sell_df,  ALIAS_NR,    "Hersteller-Nr.")
    s_ean   = find_column(sell_df,  ALIAS_EAN,   "EAN")
    s_name  = find_column(sell_df,  ALIAS_NAME,  "Bezeichnung")

    p_nr    = find_column(price_df, ALIAS_NR,    "Hersteller-Nr.")
    p_ean   = find_column(price_df, ALIAS_EAN,   "EAN")
    p_name  = find_column(price_df, ALIAS_NAME,  "Bezeichnung")
    p_cat   = find_column(price_df, ALIAS_CAT,   "Kategorie")
    p_price = find_column(price_df, ALIAS_PREIS, "Preis")

    # 2) Erstes Join: über Hersteller-Nr.
    merged = sell_df.merge(
        price_df[[p_nr, p_name, p_cat, p_price]],
        left_on  = s_nr,
        right_on = p_nr,
        how       = "left",
        suffixes  = ("", "_pr1")
    )

    # 3) Zweites Join für fehlende Preise: über EAN
    mask = merged[p_price].isna() & merged[s_ean].notna()
    if mask.any():
        fallback = (
            price_df[[p_ean, p_name, p_cat, p_price]]
            .drop_duplicates(p_ean)
        )
        merged.loc[mask, [p_name, p_cat, p_price]] = (
            merged[mask]
            .merge(fallback, left_on=s_ean, right_on=p_ean, how="left")
            [[p_name, p_cat, p_price]]
            .values
        )

    # 4) (Optional) Fuzzy-Matching oder zusätzliche Backups könnten hier folgen

    # 5) Standard-Spaltennamen setzen
    merged = merged.rename(columns={
        s_nr:      "Artikelnr",
        s_ean:     "EAN",
        s_name:    "Bezeichnung",
        p_cat:     "Kategorie",
        p_price:   "Preis",
    })

    return merged

@st.cache_data(show_spinner="🔢 Aggregiere …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Werte-Spalten neu berechnen
    df = df.assign(
        Einkaufswert = df["Einkauf"]  * df["Preis"],
        Verkaufswert  = df["Verkauf"] * df["Preis"],
        Lagerwert     = df["Verfügbar"] * df["Preis"],
    )
    # Gruppierung & Aggregation
    tbl = (
        df
        .groupby(["Artikelnr", "Bezeichnung", "Kategorie"], as_index=False)
        .agg(
            Einkaufsmenge = ("Einkauf",   "sum"),
            Einkaufswert  = ("Einkaufswert", "sum"),
            Verkaufsmenge = ("Verkauf",   "sum"),
            Verkaufswert  = ("Verkaufswert",  "sum"),
            Lagermenge    = ("Verfügbar", "sum"),
            Lagerwert     = ("Lagerwert",    "sum"),
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
    price_file = st.file_uploader("Preisliste (.xlsx)",  type="xlsx")

    if sell_file and price_file:
        sell_df  = pd.read_excel(sell_file)
        price_df = pd.read_excel(price_file)

        enriched = enrich(sell_df, price_df)
        tbl, totals = compute_agg(enriched)

        st.dataframe(tbl, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Einkaufswert (CHF)", f"{totals['EK']:,.2f}")
        c2.metric("Verkaufswert (CHF)", f"{totals['VK']:,.2f}")
        c3.metric("Lagerwert    (CHF)", f"{totals['LG']:,.2f}")

if __name__ == "__main__":
    main()
