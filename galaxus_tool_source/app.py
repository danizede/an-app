import streamlit as st
import pandas as pd

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """Ermittelt aus df.columns den ersten Namen aus candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {list(df.columns)}"
    )

@st.cache_data(show_spinner="🔗 Matching & Enrichment …", max_entries=5)
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten erkennen
    c_id        = find_column(sell,  ["Hersteller-Nr.","Produkt ID"],    "Artikelnr")
    c_ean       = find_column(sell,  ["EAN"],                            "EAN")
    c_name_s    = find_column(sell,  ["Bezeichnung","Produktname","Name"], "DescrSell")
    c_buy_qty   = find_column(sell,  ["Einkauf"],                        "Einkauf")
    c_avail_qty = find_column(sell,  ["Verfügbar","Bestand"],            "Verfügbar")
    c_sold_qty  = find_column(sell,  ["Verkauf","Sell-Out"],             "Verkauf")

    p_id        = find_column(price, ["Artikelnummer","Hersteller-Nr."],  "Artikelnr")
    p_ean       = find_column(price, ["GTIN","EAN"],                     "EAN")
    p_name_p    = find_column(price, ["Bezeichnung","Produktname"],      "Bezeichnung")
    p_cat       = find_column(price, ["Zusatz","Warengruppe"],           "Kategorie")
    p_price     = find_column(price, ["NETTO NETTO","VK","Preis","Einkauf"], "Preis")

    # 2) Preistabelle umbenennen auf eindeutige temporäre Namen
    price2 = price.rename(columns={
        p_id:       "Artikelnr_PL",
        p_ean:      "EAN_PL",
        p_name_p:   "Bezeichnung_PL",
        p_cat:      "Kategorie_PL",
        p_price:    "Preis_PL"
    })

    # 3) Erstes Merge über Artikelnr
    merged = sell.merge(
        price2[["Artikelnr_PL","Bezeichnung_PL","Kategorie_PL","Preis_PL"]],
        left_on  = c_id,
        right_on = "Artikelnr_PL",
        how      = "left",
    )

    # 4) Fehlende Preise über EAN nachschlagen
    mask = merged["Preis_PL"].isna() & merged[c_ean].notna()
    if mask.any():
        ean_ref = price2[["EAN_PL","Bezeichnung_PL","Kategorie_PL","Preis_PL"]].drop_duplicates("EAN_PL")
        tmp = (
            merged.loc[mask]
                  .merge(ean_ref,
                         left_on  = c_ean,
                         right_on = "EAN_PL",
                         how      = "left",
                         suffixes=("","_EAN"))
        )
        # Update nur dort, wo wir was gefunden haben
        for col in ["Bezeichnung_PL","Kategorie_PL","Preis_PL"]:
            merged.loc[mask, col] = tmp[col].values

    # 5) Finales Zielschema zusammenbauen
    merged["Bezeichnung"]   = merged["Bezeichnung_PL"].fillna(merged[c_name_s])
    merged["Kategorie"]     = merged["Kategorie_PL"].fillna("—")
    merged["Preis"]         = merged["Preis_PL"].fillna(0)

    # 6) Werte berechnen
    merged["Einkaufswert"]  = merged[c_buy_qty].fillna(0) * merged["Preis"]
    merged["Verkaufswert"]  = merged[c_sold_qty].fillna(0) * merged["Preis"]
    merged["Lagerwert"]     = merged[c_avail_qty].fillna(0) * merged["Preis"]

    # 7) Rename der Sell-Spalten auf Eure Standardnamen
    merged = merged.rename(columns={
        c_id:        "Artikelnr",
        c_ean:       "EAN",
        c_name_s:    "Orig_Bezeichnung",
        c_buy_qty:   "Einkauf",
        c_avail_qty: "Verfügbar",
        c_sold_qty:  "Verkauf"
    })

    # 8) Nur die finalen Spalten zurückgeben
    return merged[[
        "Artikelnr","EAN","Bezeichnung","Kategorie",
        "Einkauf","Verfügbar","Verkauf",
        "Einkaufswert","Verkaufswert","Lagerwert"
    ]]

@st.cache_data(show_spinner="📊 Aggregating …", max_entries=5)
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tbl = df.groupby(
        ["Artikelnr","Bezeichnung","Kategorie"], dropna=False
    ).agg(
        Einkauf      = ("Einkauf",      "sum"),
        Verfügbar    = ("Verfügbar",    "sum"),
        Verkauf      = ("Verkauf",      "sum"),
        Einkaufswert = ("Einkaufswert", "sum"),
        Verkaufswert = ("Verkaufswert", "sum"),
        Lagerwert    = ("Lagerwert",    "sum"),
    ).reset_index()
    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum()
    }
    return tbl, totals

def main():
    st.set_page_config("Galaxus Sell-out Aggregator", "📦")
    st.title("📦 Galaxus Sell-out Aggregator")

    c1,c2 = st.columns(2)
    with c1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
    with c2:
        price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx", key="price")

    if not sell_file or not price_file:
        st.info("Bitte beideseitig die Dateien hochladen, um fortzufahren.")
        st.stop()

    sell_df  = pd.read_excel(sell_file,  engine="openpyxl")
    price_df = pd.read_excel(price_file, engine="openpyxl")

    try:
        enriched, totals = compute_agg(enrich(sell_df, price_df))
    except KeyError as e:
        st.error(e)
        st.stop()

    # KPI‐Metriken
    m1,m2,m3 = st.columns(3)
    m1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
    m2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
    m3.metric("Lagerwert   (CHF)", f"{totals['LG']:,.0f}")

    st.dataframe(enriched, use_container_width=True)

if __name__=="__main__":
    main()
