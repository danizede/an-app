import streamlit as st
import pandas as pd

# ---------- Hilfsfunktionen -----------------------------------------

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df.columns nach einer der Kandidaten (case-insensitive).
    Gibt den erste Treffer zurück oder wirft KeyError mit Klartext.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(show_spinner="📥 Dateien laden …")
def load_df(uploader) -> pd.DataFrame:
    """Lädt eine Excel-Datei ins DataFrame."""
    return pd.read_excel(uploader, engine="openpyxl")

@st.cache_data(show_spinner="🔗 Matching & Enrichment …", max_entries=5)
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Matcht Sell-Out-Report und Preisliste, rechnet Mengen in Werte um."""
    # 1) Spalten in Sell-Out finden
    c_artnr   = find_column(sell_df,  ["Artikelnr","Artikelnummer","Hersteller-Nr.","Produkt ID"], "Artikelnr")
    c_ean     = find_column(sell_df,  ["EAN","GTIN"],                                  "EAN")
    c_sold    = find_column(sell_df,  ["Verkauf","Sell-Out von","Sell-Out bis","Verkaufsmenge"], "Verkaufsmenge")
    c_stock   = find_column(sell_df,  ["Verfügbar","Bestand","Lagermenge"],             "Lagermenge")
    c_buy     = find_column(sell_df,  ["Einkauf","EK","Cost"],                          "Einkaufsmenge")

    # 2) Spalten in Preisliste finden
    p_artnr   = find_column(price_df, ["Artikelnr","Artikelnummer","Hersteller-Nr."], "Artikelnr")
    p_ean     = find_column(price_df, ["EAN","GTIN"],                                "EAN")
    p_name    = find_column(price_df, ["Bezeichnung","Produktname","Name"],           "Bezeichnung")
    p_cat     = find_column(price_df, ["Zusatz","Warengruppe","Kategorie"],           "Kategorie")
    p_price   = find_column(price_df, ["Preis","VK","NETTO"],                         "Preis")

    # 3) Preisliste umbenennen auf eindeutige Spalten
    prices = price_df.rename(columns={
        p_artnr: "Artikelnr",
        p_ean:   "EAN",
        p_name:  "Bezeichnung",
        p_cat:   "Kategorie",
        p_price: "Preis"
    })[["Artikelnr","EAN","Bezeichnung","Kategorie","Preis"]]

    # 4) Sell-Out-Daten umbenennen
    sell = sell_df.rename(columns={
        c_artnr: "Artikelnr",
        c_ean:   "EAN",
        c_sold:  "Verkaufsmenge",
        c_stock: "Lagermenge",
        c_buy:   "Einkaufsmenge"
    })[["Artikelnr","EAN","Verkaufsmenge","Lagermenge","Einkaufsmenge"]]

    # 5) Merge über Artikelnr
    merged = sell.merge(prices, on="Artikelnr", how="left")

    # 6) Fehlende Preise via EAN nachtragen
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        tmp = merged[mask].merge(
            prices[["EAN","Bezeichnung","Kategorie","Preis"]].drop_duplicates("EAN"),
            on="EAN", how="left"
        )
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask, col] = tmp[col].values

    # 7) Fehlende Preise = 0
    merged["Preis"] = merged["Preis"].fillna(0)

    # 8) Werte berechnen
    merged["Einkaufswert"]  = merged["Einkaufsmenge"]  * merged["Preis"]
    merged["Verkaufswert"]  = merged["Verkaufsmenge"]  * merged["Preis"]
    merged["Lagerwert"]     = merged["Lagermenge"]     * merged["Preis"]

    return merged

@st.cache_data(show_spinner="📊 Aggregation …", max_entries=5)
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregiert nach Artikel und summiert alle Mengen & Werte."""
    tbl = df.groupby(["Artikelnr","Bezeichnung","Kategorie"], dropna=False).agg(
        Einkaufsmenge   = ("Einkaufsmenge",  "sum"),
        Einkaufswert    = ("Einkaufswert",   "sum"),
        Verkaufsmenge   = ("Verkaufsmenge",  "sum"),
        Verkaufswert    = ("Verkaufswert",   "sum"),
        Lagermenge      = ("Lagermenge",     "sum"),
        Lagerwert       = ("Lagerwert",      "sum"),
    ).reset_index()
    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# ---------------- Hauptprogramm ---------------------------------------

def main():
    st.set_page_config(page_title="Galaxus Sell-out", layout="wide")
    st.title("📦 Galaxus Sell-out Aggregator")

    col1, col2 = st.columns(2)
    with col1:
        sell_file  = st.file_uploader("Sell-Out-Report (.xlsx)", type="xlsx")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen, um zu starten.")
        return

    sell_df  = load_df(sell_file)
    price_df = load_df(price_file)

    enriched = enrich(sell_df, price_df)

    # —— Diagnostik: Artikel mit Menge aber Preis=0 ——
    no_price = enriched.query("Preis == 0 and (Einkaufsmenge>0 or Verkaufsmenge>0 or Lagermenge>0)")
    if not no_price.empty:
        st.error(f"{len(no_price)} Zeilen haben Menge > 0, aber Preis=0 → kein Wert.")
        st.dataframe(
            no_price[["Artikelnr","EAN","Einkaufsmenge","Verkaufsmenge","Lagermenge"]],
            use_container_width=True
        )
        st.stop()

    # —— Aggregation & Anzeige ——
    agg_tbl, totals = compute_agg(enriched)

    # Kennzahlen in einer Zeile
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("EK-Menge",   f"{agg_tbl['Einkaufsmenge'].sum():,.0f}")
    k2.metric("EK-Wert",    f"{totals['EK']:,.0f}")
    k3.metric("VK-Menge",   f"{agg_tbl['Verkaufsmenge'].sum():,.0f}")
    k4.metric("VK-Wert",    f"{totals['VK']:,.0f}")
    k5.metric("LG-Menge",   f"{agg_tbl['Lagermenge'].sum():,.0f}")
    k6.metric("LG-Wert",    f"{totals['LG']:,.0f}")

    st.dataframe(agg_tbl, use_container_width=True)

if __name__ == "__main__":
    main()
