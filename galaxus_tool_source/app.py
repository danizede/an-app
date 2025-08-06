import streamlit as st
import pandas as pd
import numpy as np
from difflib import get_close_matches

# --- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Galaxus Sell-out Aggregator",
    layout="wide",
)

# --- Hilfsfunktionen -------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """
    Sucht in df.columns nach einer Spalte, die case-insensitive mit einem Eintrag aus candidates
    übereinstimmt. Gibt den exakten Spaltennamen zurück oder wirft einen KeyError mit Info.
    """
    cols = df.columns.tolist()
    for cand in candidates:
        for col in cols:
            if col.strip().lower() == cand.strip().lower():
                return col
    raise KeyError(
        f"Spalte für «{label}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )

@st.cache_data(show_spinner="🔗 Daten anreichern …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # Kopien anlegen und Spalten trimmen
    sell = sell_df.copy()
    price = price_df.copy()
    sell.columns = sell.columns.str.strip()
    price.columns = price.columns.str.strip()

    # Spalten finden
    # Sell-Out Report
    c_nr_s    = find_column(sell,  ["Hersteller-Nr.","Artikelnr","Artikelnummer"], "Artikelnr")
    c_ean_s   = find_column(sell,  ["EAN","GTIN"],                      "EAN")
    c_qty_s   = find_column(sell,  ["Verkauf","Sell-Out bis","Menge"],  "Verkaufsmenge")
    c_stock_s = find_column(sell,  ["Bestand","Verfügbar","Lagerbestand"], "Lagerbestand")

    # Preisliste
    c_nr_p    = find_column(price, ["Artikelnr","Artikelnummer","Hersteller-Nr."], "Artikelnr")
    c_ean_p   = find_column(price, ["EAN","GTIN"],                        "EAN")
    c_name_p  = find_column(price, ["Bezeichnung","Bez","Name","Produktname"], "Bezeichnung")
    c_cat_p   = find_column(price, ["Zusatz","Warengruppe","Kategorie"],   "Kategorie")
    c_price_p = find_column(price, ["Preis","NETTO","VK"],                "Preis")

    # Spalten umbenennen auf Standard
    sell = sell.rename(columns={
        c_nr_s:    "Artikelnr",
        c_ean_s:   "EAN",
        c_qty_s:   "Verkauf",
        c_stock_s: "Lagerbestand",
    })
    price = price.rename(columns={
        c_nr_p:    "Artikelnr",
        c_ean_p:   "EAN",
        c_name_p:  "Bezeichnung",
        c_cat_p:   "Kategorie",
        c_price_p: "Preis",
    })

    # 1) Merge auf Artikel-Nr.
    merged = sell.merge(
        price[["Artikelnr","Bezeichnung","Kategorie","Preis"]],
        on="Artikelnr",
        how="left",
    )

    # 2) Fallback via EAN/GTIN
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        price_ean = price.drop_duplicates("EAN")
        tmp = (
            merged.loc[mask, ["EAN"]]
            .merge(price_ean[["EAN","Bezeichnung","Kategorie","Preis"]],
                   on="EAN", how="left")
        )
        merged.loc[mask, ["Bezeichnung","Kategorie","Preis"]] = tmp[["Bezeichnung","Kategorie","Preis"]].values

    # 3) Fuzzy-Match über ersten 2 Wörter der Bezeichnung
    mask = merged["Preis"].isna()
    if mask.any():
        # Token generieren
        price["token"] = price["Bezeichnung"].str.lower().str.split().str[:2].str.join(" ")
        for idx in merged[mask].index:
            tok = str(merged.at[idx, "Bezeichnung"]).lower().split()[:2]
            tok = " ".join(tok)
            choices = price["token"].dropna().unique().tolist()
            best = get_close_matches(tok, choices, n=1, cutoff=0.8)
            if best:
                row = price[price["token"] == best[0]].iloc[0]
                merged.at[idx, "Bezeichnung"] = row["Bezeichnung"]
                merged.at[idx, "Kategorie"]  = row["Kategorie"]
                merged.at[idx, "Preis"]      = row["Preis"]

    # Berechne Werte
    merged["Einkaufsmenge"] = merged["Verkauf"]         # gleich Verkauf
    merged["Einkaufswert"]  = merged["Verkauf"] * merged["Preis"]
    merged["Verkaufsmenge"] = merged["Verkauf"]
    merged["Verkaufswert"]  = merged["Verkauf"] * merged["Preis"]
    merged["Lagermenge"]    = merged["Lagerbestand"]
    merged["Lagerwert"]     = merged["Lagerbestand"] * merged["Preis"]

    return merged

@st.cache_data(show_spinner="🔢 Aggregieren …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    grp = (
        df
        .groupby(["Artikelnr","Bezeichnung","Kategorie"], as_index=False)
        .agg(
            Einkaufsmenge  = ("Einkaufsmenge","sum"),
            Einkaufswert   = ("Einkaufswert","sum"),
            Verkaufsmenge  = ("Verkaufsmenge","sum"),
            Verkaufswert   = ("Verkaufswert","sum"),
            Lagermenge     = ("Lagermenge","sum"),
            Lagerwert      = ("Lagerwert","sum"),
        )
    )
    totals = {
        "Einkaufswert": grp["Einkaufswert"].sum(),
        "Verkaufswert": grp["Verkaufswert"].sum(),
        "Lagerwert":    grp["Lagerwert"].sum(),
    }
    return grp, totals

# --- Haupt-Workflow --------------------------------------------------------

def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    c1, c2 = st.columns(2)
    with c1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", key="sell",  type="xlsx")
    with c2:
        price_file = st.file_uploader("Preisliste (.xlsx)",    key="price", type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen, um fortzufahren.")
        return

    try:
        sell_df  = pd.read_excel(sell_file)
        price_df = pd.read_excel(price_file)
        merged   = enrich(sell_df, price_df)
        table, totals = compute_agg(merged)
    except KeyError as e:
        st.error(e)
        return

    # Kennzahlen
    k1, k2, k3 = st.columns(3)
    k1.metric("Einkaufswert (CHF)", f"{totals['Einkaufswert']:,.2f}")
    k2.metric("Verkaufswert (CHF)", f"{totals['Verkaufswert']:,.2f}")
    k3.metric("Lagerwert (CHF)",    f"{totals['Lagerwert']:,.2f}")

    # Tabelle full-width anzeigen
    st.dataframe(table, use_container_width=True)

if __name__ == "__main__":
    main()
