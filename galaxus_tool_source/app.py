import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# Helferfunktion: flexible Spaltenfindung per Substring, case-insensitive
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    cols = df.columns.tolist()
    # erst exakte Übereinstimmung
    for cand in candidates:
        if cand in cols:
            return cand
    # dann substring match
    for cand in candidates:
        for col in cols:
            if cand.lower() in col.lower():
                return col
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )

@st.cache_data
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # ---- Spalten in price_df finden ----
    # Artikelnummer in PL
    nr_p = find_column(price_df, ["Artikelnummer", "Hersteller-Nr.", "ArtNr"], "Artikelnr in PL")
    # GTIN/Barcode in PL
    ean_p = find_column(price_df, ["GTIN", "EAN"], "EAN in PL")
    # Bezeichnung/Kurzbezeichnung in PL
    name_p = find_column(price_df, ["Bezeichnung", "Produktname"], "Bezeichnung in PL")
    # Kategorie in PL (heißt hier 'Zusatz')
    cat_p  = find_column(price_df, ["Zusatz", "Warengruppe", "Kategorie"], "Kategorie in PL")
    # Netto-Einstandspreis in PL
    price_p = find_column(price_df,
                          ["NETTO NETTO", "Einkaufspreis", "Einkauf", "Netto", "VK", "Preis"],
                          "Preis in PL")

    # Einheitlich umbenennen
    price_df = price_df.rename(columns={
        nr_p:    "Artikelnr",
        ean_p:   "EAN_PL",
        name_p:  "Bezeichnung",
        cat_p:   "Kategorie",
        price_p: "Einstandspreis",
        # Bestand schon ‚Bestand‘ oder ‚Verfügbar‘?
    })
    # Bestandsspalte umbenennen, falls nötig
    if "Bestand" not in price_df.columns:
        bcol = find_column(price_df, ["Bestand", "Verfügbar"], "Bestand in PL")
        price_df = price_df.rename(columns={bcol: "Bestand"})
    
    # ---- Spalten in sell_df finden ----
    nr_s   = find_column(sell_df, ["Hersteller-Nr.", "Produkt ID", "Artikelnr", "Artikelnummer"], "Artikelnr im Sell-Out-Report")
    ean_s  = find_column(sell_df, ["EAN", "GTIN"], "EAN im Sell-Out-Report")
    name_s = find_column(sell_df, ["Produktname", "Bezeichnung", "Name"], "Bezeichnung im Sell-Out-Report")
    sold_s = find_column(sell_df, ["Verkauf", "Sell-Out"], "Verkaufsmenge im Sell-Out-Report")

    # Umbenennen
    sell_df = sell_df.rename(columns={
        nr_s:   "Artikelnr",
        ean_s:  "EAN_Sales",
        name_s: "Bezeichnung_Sales",
        sold_s: "Verkauf"
    })

    # Den Name aus PL benutzen, falls Sell-Out-Bezeichnung fehlt
    sell_df["Bezeichnung"] = sell_df["Bezeichnung_Sales"].fillna("")
    
    # ---- Merge 1: per Artikelnr ----
    merged = sell_df.merge(
        price_df,
        how="left",
        on="Artikelnr",
        suffixes=("_sales", "_pl"),
        validate="many_to_one"
    )

    # ---- Fallback per EAN, falls per Nr nichts gefunden ----
    mask = merged["Einstandspreis"].isna() & merged["EAN_Sales"].notna()
    if mask.any():
        fallback = price_df.set_index("EAN_PL")
        # Achtung: EAN_PL kann numeric sein, casten auf str
        merged.loc[mask, "Einstandspreis"] = (
            merged.loc[mask, "EAN_Sales"]
            .astype(str)
            .map(fallback["Einstandspreis"])
        )
        merged.loc[mask, "Bestand"] = (
            merged.loc[mask, "EAN_Sales"]
            .astype(str)
            .map(fallback["Bestand"])
        )
        merged.loc[mask, "Bezeichnung"] = merged.loc[mask, "Bezeichnung"].mask(
            merged.loc[mask, "Bezeichnung"] == "",
            merged.loc[mask, "EAN_Sales"].astype(str).map(fallback["Bezeichnung"])
        )
        merged.loc[mask, "Kategorie"] = (
            merged.loc[mask, "EAN_Sales"]
            .astype(str)
            .map(fallback["Kategorie"])
        )

    # ---- Nun fertige Tabelle ----
    return merged[[
        "Artikelnr",
        "Bezeichnung",
        "Kategorie",
        "Verkauf",
        "Einstandspreis",
        "Bestand"
    ]].copy()


def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    col1, col2 = st.columns(2)
    with col1:
        sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type=["xlsx"], key="sell")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)", type=["xlsx"], key="price")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen, um fortzufahren.")
        return

    # DataFrames einlesen
    sell_df  = pd.read_excel(sell_file)
    price_df = pd.read_excel(price_file)

    # Merge / Enrichment
    enriched = enrich(sell_df, price_df)

    # Gruppieren
    grp = enriched.groupby(
        ["Artikelnr", "Bezeichnung", "Kategorie"],
        dropna=False,
        as_index=False
    ).agg(
        Verkaufsmenge = ("Verkauf", "sum"),
        Lagermenge    = ("Bestand", "first")
    )

    # Werte berechnen
    grp["Einkaufsmenge"] = grp["Verkaufsmenge"]
    grp["Verkaufswert"]  = grp["Verkaufsmenge"] * enriched["Einstandspreis"].iloc[0]
    grp["Einkaufswert"]  = grp["Verkaufsmenge"] * enriched["Einstandspreis"].iloc[0]
    grp["Lagerwert"]     = grp["Lagermenge"]    * enriched["Einstandspreis"].iloc[0]

    # Spalten in gewünschter Reihenfolge
    out = grp[[
        "Artikelnr", "Bezeichnung", "Kategorie",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge",   "Lagerwert"
    ]]

    # Breite Tabelle
    st.dataframe(out, use_container_width=True)


if __name__ == "__main__":
    main()
