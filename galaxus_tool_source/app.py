import pandas as pd
import streamlit as st

# Kandidaten-Listen für Spaltennamen
ALIAS_NR    = ["Artikelnr", "Hersteller-Nr.", "ArtNr", "Artikelnummer"]
ALIAS_EAN   = ["EAN", "GTIN"]
ALIAS_NAME  = ["Bezeichnung", "Bez", "Name", "Produktname"]
ALIAS_CAT   = ["Warengruppe", "Kategorie", "Zusatz"]
ALIAS_PREIS = ["Preis", "VK", "NETTO", "Einkauf", "Verkauf"]

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht zuerst nach exaktem Match, 
    wenn nichts gefunden wird: case-insensitive substring match.
    """
    # 1) Exakte Übereinstimmung
    for c in candidates:
        if c in df.columns:
            return c

    # 2) Substring-Match in beliebiger Spalte (case-ins.)
    lowered_cols = [col.lower() for col in df.columns]
    for c in candidates:
        c_low = c.lower()
        for idx, col_low in enumerate(lowered_cols):
            if c_low in col_low:
                return df.columns[idx]

    # 3) Fehlermeldung
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # Sell-Out-Spalten finden
    s_nr  = find_column(sell_df, ALIAS_NR,  "Artikelnr im Sell-Out")
    s_ean = find_column(sell_df, ALIAS_EAN, "EAN im Sell-Out")
    s_qty = find_column(sell_df, ["Verkauf", "Sell-Out von", "Sell-Out bis"], "Verkaufsmenge")

    # PL-Spalten finden
    p_nr    = find_column(price_df, ALIAS_NR,     "Artikelnr in der PL")
    p_ean   = find_column(price_df, ALIAS_EAN,    "EAN in der PL")
    p_name  = find_column(price_df, ALIAS_NAME,   "Bezeichnung in der PL")
    p_cat   = find_column(price_df, ALIAS_CAT,    "Kategorie in der PL")
    p_price = find_column(price_df, ALIAS_PREIS,  "Preis in der PL")
    p_stock = find_column(price_df, ["Verfügbar","Bestand"], "Lagerbestand in der PL")

    # 1) Erstes Join via Hersteller-Nr.
    merged = sell_df.merge(
        price_df[[p_nr, p_name, p_cat, p_price, p_stock]],
        left_on  = s_nr,
        right_on = p_nr,
        how      = "left",
        suffixes = ("", "_pl")
    )

    # 2) Fallback Join via EAN, falls Preis/Bestand noch fehlt
    mask_missing = merged[p_price].isna() & merged[s_ean].notna()
    if mask_missing.any():
        fallback = price_df[[p_ean, p_name, p_cat, p_price, p_stock]].drop_duplicates(p_ean)
        merged.loc[mask_missing, [p_name, p_cat, p_price, p_stock]] = (
            merged[mask_missing]
            .merge(fallback, left_on=s_ean, right_on=p_ean, how="left")
            [[p_name, p_cat, p_price, p_stock]]
            .values
        )

    # 3) Einheitliche Spaltennamen vergeben
    merged = merged.rename(columns={
        s_nr:    "Artikelnr",
        s_ean:   "EAN",
        s_qty:   "Verkauf",
        p_name:  "Bezeichnung",
        p_cat:   "Kategorie",
        p_price: "Preis",
        p_stock: "Verfügbar",
    })

    return merged

@st.cache_data(show_spinner="🔢 Aggregieren …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Sicherstellen, dass unsere Kennzahlen numerisch sind
    for col in ["Verkauf", "Verfügbar", "Preis"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Neue Kennzahlen berechnen
    df = df.assign(
        Einkaufsmenge = df["Verkauf"],               
        Einkaufswert  = df["Verkauf"] * df["Preis"], 
        Verkaufsmenge = df["Verkauf"],
        Verkaufswert  = df["Verkauf"] * df["Preis"],
        Lagermenge    = df["Verfügbar"],
        Lagerwert     = df["Verfügbar"] * df["Preis"],
    )

    # Gruppieren & Summen bilden
    tbl = (
        df
        .groupby(["Artikelnr", "Bezeichnung", "Kategorie"], as_index=False)
        .agg(
            Einkaufsmenge=("Einkaufsmenge", "sum"),
            Einkaufswert =("Einkaufswert",  "sum"),
            Verkaufsmenge=("Verkaufsmenge", "sum"),
            Verkaufswert =("Verkaufswert",  "sum"),
            Lagermenge   =("Lagermenge",    "sum"),
            Lagerwert    =("Lagerwert",     "sum"),
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

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen.")
        return

    sell_df  = pd.read_excel(sell_file)
    price_df = pd.read_excel(price_file)

    # Matching
    try:
        enriched = enrich(sell_df, price_df)
    except KeyError as e:
        st.error(f"🔍 Matching-Fehler:\n{e}")
        return

    # Aggregation
    try:
        tbl, totals = compute_agg(enriched)
    except Exception as e:
        st.error(f"📊 Aggregations-Fehler:\n{e}")
        return

    # Ausgabe
    st.dataframe(tbl, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Einkaufswert (CHF)", f"{totals['EK']:,.2f}")
    c2.metric("Verkaufswert (CHF)", f"{totals['VK']:,.2f}")
    c3.metric("Lagerwert    (CHF)", f"{totals['LG']:,.2f}")

if __name__ == "__main__":
    main()
