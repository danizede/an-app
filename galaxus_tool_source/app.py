import streamlit as st
import pandas as pd

# ---------------- Spaltenerkennung ----------------
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Durchsuche df.columns nach einer der candidates.
    Liefert den ersten Treffer, sonst KeyError mit Hinweis.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {list(df.columns)}")

# ---------------- Anreicherung ----------------
@st.cache_data(show_spinner="🔗 Matching & Enrichment …", max_entries=5)
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # Sell-Out-Report Spalten
    c_id        = find_column(sell,  ["Hersteller-Nr.","Produkt ID"],           "Artikelnr")
    c_ean       = find_column(sell,  ["EAN"],                                     "EAN")
    c_name_s    = find_column(sell,  ["Produktname","Name"],                      "Bezeichnung (Sell-Out)")
    c_buy_qty   = find_column(sell,  ["Einkauf"],                                 "Einkauf")
    c_avail_qty = find_column(sell,  ["Verfügbar","Bestand"],                     "Verfügbar")
    c_sold_qty  = find_column(sell,  ["Verkauf","Sell-Out"],                      "Verkauf")

    # Preisliste Spalten
    p_id        = find_column(price, ["Artikelnummer","Hersteller-Nr."],          "Artikelnr")
    p_ean       = find_column(price, ["GTIN","EAN"],                              "EAN")
    p_name_p    = find_column(price, ["Bezeichnung","Produktname"],               "Bezeichnung")
    p_cat       = find_column(price, ["Zusatz","Warengruppe"],                    "Kategorie")
    p_price     = find_column(price, ["NETTO NETTO","VK","Preis","Einkauf"],      "Preis")

    # 1) Merge über Hersteller-Nr. / Artikelnummer
    merged = sell.merge(
        price[[p_id, p_name_p, p_cat, p_price]],
        left_on  = c_id,
        right_on = p_id,
        how      = "left",
    ).rename(columns={p_name_p: "Bezeichnung", p_cat: "Kategorie", p_price: "Preis"})

    # 2) Fehlende Preise via EAN füllen
    mask = merged["Preis"].isna() & merged[c_ean].notna()
    if mask.any():
        fill = (
            merged[mask]
            .merge(
                price.drop_duplicates(p_ean)[[p_ean, p_name_p, p_cat, p_price]],
                left_on  = c_ean,
                right_on = p_ean,
                how      = "left",
            )
            .rename(columns={p_name_p: "Bezeichnung", p_cat: "Kategorie", p_price: "Preis"})
        )
        for col in ["Bezeichnung", "Kategorie", "Preis"]:
            merged.loc[mask, col] = fill[col].values

    # 3) Werte berechnen
    merged["Einkaufswert"]  = merged[c_buy_qty].fillna(0) * merged["Preis"].fillna(0)
    merged["Verkaufswert"]  = merged[c_sold_qty].fillna(0) * merged["Preis"].fillna(0)
    merged["Lagerwert"]     = merged[c_avail_qty].fillna(0) * merged["Preis"].fillna(0)

    # Spalten sauber umbenennen
    merged = merged.rename(columns={
        c_id:        "Artikelnr",
        c_ean:       "EAN",
        c_name_s:    "Produktname",
        c_buy_qty:   "Einkauf",
        c_avail_qty: "Verfügbar",
        c_sold_qty:  "Verkauf"
    })

    return merged

# ---------------- Aggregation ----------------
@st.cache_data(show_spinner="📊 Aggregating …", max_entries=5)
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Gruppieren nach Artikel, Bezeichnung, Kategorie
    agg = df.groupby(["Artikelnr", "Produktname", "Kategorie"], dropna=False).agg(
        Einkauf      = ("Einkauf",      "sum"),
        Verfügbar    = ("Verfügbar",    "sum"),
        Verkauf      = ("Verkauf",      "sum"),
        Einkaufswert = ("Einkaufswert", "sum"),
        Verkaufswert = ("Verkaufswert", "sum"),
        Lagerwert    = ("Lagerwert",    "sum")
    ).reset_index()

    totals = {
        "EK": agg["Einkaufswert"].sum(),
        "VK": agg["Verkaufswert"].sum(),
        "LG": agg["Lagerwert"].sum()
    }
    return agg, totals

# ---------------- Hauptprogramm ----------------
def main():
    st.set_page_config("Galaxus Sell-out Aggregator", "📦")
    st.title("📦 Galaxus Sell-out Aggregator")

    col1, col2 = st.columns(2)
    with col1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx", key="price")

    if not sell_file or not price_file:
        st.info("Bitte beideseite eine Datei hochladen, um die Auswertung zu starten.")
        st.stop()

    # Daten laden
    sell_df  = pd.read_excel(sell_file,  engine="openpyxl")
    price_df = pd.read_excel(price_file, engine="openpyxl")

    # Enrich & Aggregate
    try:
        enriched, totals = compute_agg(enrich(sell_df, price_df))
    except KeyError as e:
        st.error(e)
        st.stop()

    # Dashboard
    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
    c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
    c3.metric("Lagerwert (CHF)",   f"{totals['LG']:,.0f}")

    st.dataframe(
        enriched.style.format({
            "Einkaufswert": "€{:,.2f}",
            "Verkaufswert": "€{:,.2f}",
            "Lagerwert":    "€{:,.2f}"
        }),
        use_container_width=True
    )

if __name__ == "__main__":
    main()
