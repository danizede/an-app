import re
import pandas as pd
import streamlit as st

# Hilfsfunktion zur Normalisierung (Klein, keine Sonderzeichen)
def _normalize(s: str) -> str:
    return re.sub(r"\W+", "", s).lower()

# Findet die richtige Spalte im DataFrame anhand mehrerer Kandidaten
def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    cols = list(df.columns)
    norm_cols = {col: _normalize(col) for col in cols}

    # 1) exakte Übereinstimmung (case-sensitive)
    for cand in candidates:
        if cand in cols:
            return cand

    # 2) exakte Übereinstimmung (case-insensitive)
    for cand in candidates:
        for col in cols:
            if col.lower() == cand.lower():
                return col

    # 3) normalized substring matching
    for cand in candidates:
        cand_norm = _normalize(cand)
        for col, col_norm in norm_cols.items():
            if cand_norm in col_norm or col_norm in cand_norm:
                return col

    # Wenn nichts passt, Fehler mit Info über alle verfügbaren Spalten
    raise KeyError(
        f"Spalte für «{label}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )

@st.cache_data(show_spinner="🔗 Daten anreichern …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- 1) Spalten in sell_df finden ---
    s_nr   = find_column(sell_df, ["Hersteller-Nr.","ArtikelNr","ArtNr","Artikelnummer"], "Artikelnr")
    s_ean  = find_column(sell_df, ["EAN","GTIN"], "EAN")
    s_name = find_column(sell_df, ["Bezeichnung","Bez","Produktname","Name"], "Bezeichnung")
    s_cat  = find_column(sell_df, ["Kategorie","Warengruppe","Zusatz"], "Kategorie")
    s_buy  = find_column(sell_df, ["Einkauf","Einkaufsmenge","Purchase"], "Einkaufsmenge")
    s_sell = find_column(sell_df, ["Verkauf","Sell-Out von","Sell-Out","Verkaufsmenge"], "Verkaufsmenge")
    s_avail= find_column(sell_df, ["Verfügbar","Bestand","Lagerbestand"], "Lagermenge")

    # --- 2) Spalten in price_df finden ---
    p_nr   = find_column(price_df, ["Hersteller-Nr.","ArtikelNr","ArtNr","Artikelnummer"], "Artikelnr")
    p_ean  = find_column(price_df, ["EAN","GTIN"], "EAN")
    p_name = find_column(price_df, ["Bezeichnung","Bez","Produktname","Name"], "Bezeichnung")
    p_cat  = find_column(price_df, ["Kategorie","Warengruppe","Zusatz"], "Kategorie")
    p_price= find_column(price_df, ["Preis","NETTO","VK","Sell-Out bis"], "Preis")

    # 3) Mergen über Hersteller-Nr.
    merged = sell_df.merge(
        price_df[[p_nr, p_name, p_cat, p_price]],
        left_on=s_nr, right_on=p_nr, how="left", suffixes=("", "_p")
    )

    # 4) Fallback via EAN/GTIN, falls Preis noch fehlt
    mask = merged[p_price].isna() & merged[s_ean].notna()
    if mask.any():
        fallback = (
            price_df.drop_duplicates(subset=p_ean)
                    [[p_ean, p_name, p_cat, p_price]]
        )
        merged.loc[mask, [p_name, p_cat, p_price]] = (
            merged[mask]
            .merge(fallback, left_on=s_ean, right_on=p_ean, how="left")
            [[p_name, p_cat, p_price]]
            .values
        )

    # 5) Spalten umbenennen auf unsere Standardnamen
    return (
        merged
        .rename(columns={
            s_nr:       "Artikelnr",
            s_name:     "Bezeichnung",
            s_cat:      "Kategorie",
            s_buy:      "Einkaufsmenge",
            s_sell:     "Verkaufsmenge",
            s_avail:    "Lagermenge",
            p_price:    "Preis",
            s_ean:      "EAN"
        })
        .loc[:, [
            "Artikelnr", "EAN", "Bezeichnung", "Kategorie",
            "Einkaufsmenge", "Verkaufsmenge", "Lagermenge",
            "Preis"
        ]]
    )

@st.cache_data(show_spinner="📊 Aggregieren …")
def compute_agg(df: pd.DataFrame):
    # Werte berechnen
    df["Einkaufswert"] = df["Einkaufsmenge"] * df["Preis"]
    df["Verkaufswert"] = df["Verkaufsmenge"]  * df["Preis"]
    df["Lagerwert"]    = df["Lagermenge"]     * df["Preis"]

    # Gruppieren und Gesamttotals
    tbl = df.groupby(
        ["Artikelnr","Bezeichnung","Kategorie"], as_index=False
    ).agg(
        Einkaufsmenge = ("Einkaufsmenge", "sum"),
        Einkaufswert  = ("Einkaufswert",  "sum"),
        Verkaufsmenge = ("Verkaufsmenge", "sum"),
        Verkaufswert  = ("Verkaufswert",  "sum"),
        Lagermenge    = ("Lagermenge",    "sum"),
        Lagerwert     = ("Lagerwert",     "sum"),
    )

    tots = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, tots

def main():
    st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
    st.title("📦 Galaxus Sell-out Aggregator")

    col1, col2 = st.columns(2)

    with col1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen, um fortzufahren.")
        st.stop()

    # Excel-Dateien einlesen
    sell_df  = pd.read_excel(sell_file,  engine="openpyxl")
    price_df = pd.read_excel(price_file, engine="openpyxl")

    # Anreichern + Aggregieren
    enriched = enrich(sell_df, price_df)
    agg_tbl, totals = compute_agg(enriched)

    # Metriken
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Einkaufswert (CHF)", f"{totals['EK']:, .0f}".replace(",", "'"))
    c2.metric("Verkaufswert (CHF)", f"{totals['VK']:, .0f}".replace(",", "'"))
    c3.metric("Lagerwert (CHF)",    f"{totals['LG']:, .0f}".replace(",", "'"))

    # Tabelle breit anzeigen
    st.dataframe(agg_tbl, use_container_width=True)

if __name__ == "__main__":
    main()
