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
    # --- 1) Spalten erkennen aus beiden Dateien ---
    c_id        = find_column(sell,  ["Hersteller-Nr.","Produkt ID"],    "Artikelnr")
    c_ean       = find_column(sell,  ["EAN"],                            "EAN")
    c_name_s    = find_column(sell,  ["Bezeichnung","Produktname","Name"], "DescrSell")
    c_buy_qty   = find_column(sell,  ["Einkauf"],                        "Einkaufsmenge")
    c_avail_qty = find_column(sell,  ["Verfügbar","Bestand"],            "Lagermenge")
    c_sold_qty  = find_column(sell,  ["Verkauf","Sell-Out"],             "Verkaufsmenge")

    p_id        = find_column(price, ["Artikelnummer","Hersteller-Nr."],  "Artikelnr")
    p_ean       = find_column(price, ["GTIN","EAN"],                     "EAN")
    p_name_p    = find_column(price, ["Bezeichnung","Produktname"],      "Bezeichnung")
    p_cat       = find_column(price, ["Zusatz","Warengruppe"],           "Kategorie")
    p_price     = find_column(price, ["NETTO NETTO","VK","Preis","Einkauf"], "Preis")

    # --- 2) Preisliste umbenennen auf eindeutige temporäre Namen ---
    price2 = price.rename(columns={
        p_id:       "Artikelnr_PL",
        p_ean:      "EAN_PL",
        p_name_p:   "Bezeichnung_PL",
        p_cat:      "Kategorie_PL",
        p_price:    "Preis_PL"
    })

    # --- 3) Merge über Artikelnr ---
    merged = sell.merge(
        price2[["Artikelnr_PL","Bezeichnung_PL","Kategorie_PL","Preis_PL"]],
        left_on  = c_id,
        right_on = "Artikelnr_PL",
        how      = "left",
    )

    # --- 4) Fehlende Preise via EAN nachschlagen ---
    mask = merged["Preis_PL"].isna() & merged[c_ean].notna()
    if mask.any():
        ean_ref = price2[["EAN_PL","Bezeichnung_PL","Kategorie_PL","Preis_PL"]].drop_duplicates("EAN_PL")
        tmp = (
            merged.loc[mask]
                  .merge(ean_ref,
                         left_on  = c_ean,
                         right_on = "EAN_PL",
                         how      = "left",
                         suffixes=("", "_EAN"))
        )
        for col in ["Bezeichnung_PL","Kategorie_PL","Preis_PL"]:
            merged.loc[mask, col] = tmp[col].values

    # --- 5) Finale Spalten bauen ---
    merged["Bezeichnung"]   = merged["Bezeichnung_PL"].fillna(merged[c_name_s])
    merged["Kategorie"]     = merged["Kategorie_PL"].fillna("—")
    merged["Preis"]         = merged["Preis_PL"].fillna(0)

    # --- 6) Werte berechnen ---
    merged["Einkaufswert"]  = merged[c_buy_qty].fillna(0) * merged["Preis"]
    merged["Verkaufswert"]  = merged[c_sold_qty].fillna(0) * merged["Preis"]
    merged["Lagerwert"]     = merged[c_avail_qty].fillna(0) * merged["Preis"]

    # --- 7) Spalten umbenennen auf Standardnamen ---
    merged = merged.rename(columns={
        c_id:        "Artikelnr",
        c_ean:       "EAN",
        c_name_s:    "Orig_Bezeichnung",
        c_buy_qty:   "Einkaufsmenge",
        c_avail_qty: "Lagermenge",
        c_sold_qty:  "Verkaufsmenge"
    })

    # --- 8) Nur finale Spalten zurückgeben ---
    return merged[[
        "Artikelnr","EAN","Bezeichnung","Kategorie",
        "Einkaufsmenge","Einkaufswert",
        "Verkaufsmenge","Verkaufswert",
        "Lagermenge","Lagerwert"
    ]]

@st.cache_data(show_spinner="📊 Aggregating …", max_entries=5)
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tbl = df.groupby(
        ["Artikelnr","Bezeichnung","Kategorie"], dropna=False
    ).agg(
        Einkaufsmenge   = ("Einkaufsmenge", "sum"),
        Einkaufswert    = ("Einkaufswert",  "sum"),
        Verkaufsmenge   = ("Verkaufsmenge", "sum"),
        Verkaufswert    = ("Verkaufswert",  "sum"),
        Lagermenge      = ("Lagermenge",    "sum"),
        Lagerwert       = ("Lagerwert",     "sum"),
    ).reset_index()
    totals = {
        "EKmenge":  tbl["Einkaufsmenge"].sum(),
        "EKwert":   tbl["Einkaufswert"].sum(),
        "VKmenge":  tbl["Verkaufsmenge"].sum(),
        "VKwert":   tbl["Verkaufswert"].sum(),
        "LGmenge":  tbl["Lagermenge"].sum(),
        "LGwert":   tbl["Lagerwert"].sum(),
    }
    return tbl, totals

def main():
    st.set_page_config(
        page_title="Galaxus Sell-out Aggregator",
        page_icon="📦",
        layout="wide"               # <<< hier: wide layout
    )
    st.title("📦 Galaxus Sell-out Aggregator")

    c1,c2 = st.columns(2)
    with c1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
    with c2:
        price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx", key="price")

    if not sell_file or not price_file:
        st.info("Bitte beidseitig die Dateien hochladen, um fortzufahren.")
        st.stop()

    sell_df  = pd.read_excel(sell_file,  engine="openpyxl")
    price_df = pd.read_excel(price_file, engine="openpyxl")

    try:
        enriched = enrich(sell_df, price_df)
        # Warnung: Menge vorhanden aber Preis=0?
        missing = enriched.query("Einkaufsmenge>0 and Einkaufswert==0")
        if not missing.empty:
            st.warning(
                f"{len(missing)} Zeilen haben eine Einkaufsmenge > 0, "
                "aber Einkaufswert==0 (Preis ungleich gefunden)."
            )
        # Aggregation
        agg_tbl, totals = compute_agg(enriched)
    except KeyError as e:
        st.error(e)
        st.stop()

    # KPI‐Metriken nebeneinander
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("EK-Menge", f"{totals['EKmenge']:,.0f}")
    k2.metric("EK-Wert (CHF)", f"{totals['EKwert']:,.0f}")
    k3.metric("VK-Menge", f"{totals['VKmenge']:,.0f}")
    k4.metric("VK-Wert (CHF)", f"{totals['VKwert']:,.0f}")
    k5.metric("LG-Menge", f"{totals['LGmenge']:,.0f}")
    k6.metric("LG-Wert (CHF)", f"{totals['LGwert']:,.0f}")

    # DataFrame breit anzeigen
    st.dataframe(
        agg_tbl,
        use_container_width=True
    )

if __name__ == "__main__":
    main()
