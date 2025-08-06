import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")


def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df erst exakten Spaltennamen, dann case-insensitive Substring.
    Gibt den gefundenen Namen zurück oder wirft einen KeyError mit klarer Fehlermeldung.
    """
    cols = df.columns.tolist()
    # Exakte Übereinstimmung
    for c in candidates:
        if c in cols:
            return c
    # Substring-Match
    for c in candidates:
        for col in cols:
            if c.lower() in col.lower():
                return col
    raise KeyError(
        f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
        f"Verfügbare Spalten: {cols}"
    )


@st.cache_data
def enrich_and_merge(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # ---------- Spalten in der Preisliste finden & umbenennen ----------
    price_cols = {
        find_column(price, ["Artikelnummer", "Artikelnr", "Hersteller-Nr."], "Artikelnr in PL"): "Artikelnr",
        find_column(price, ["GTIN", "EAN"], "EAN in PL"):                 "EAN",
        find_column(price, ["Bezeichnung", "Produktname"], "Bezeichnung in PL"): "Bezeichnung",
        find_column(price, ["Zusatz", "Kategorie", "Warengruppe"], "Kategorie in PL"): "Kategorie",
        find_column(price, ["NETTO NETTO", "Einkaufspreis", "Einkauf", "Netto"], "Einstandspreis in PL"): "Einstandspreis",
        find_column(price, ["Bestand", "Verfügbar"], "Bestand in PL"):   "Bestand"
    }
    price = price.rename(columns=price_cols)[
        ["Artikelnr", "EAN", "Bezeichnung", "Kategorie", "Einstandspreis", "Bestand"]
    ]

    # ---------- Spalten im Sell-Out-Report finden & umbenennen ----------
    sell_cols = {
        find_column(sell, ["Artikelnr", "Artikelnummer", "Hersteller-Nr.", "Produkt ID"], "Artikelnr im Sales"): "Artikelnr",
        find_column(sell, ["GTIN", "EAN"], "EAN im Sales"):              "EAN",
        find_column(sell, ["Bezeichnung", "Produktname", "Name"], "Bezeichnung im Sales"): "Bezeichnung_Sales",
        find_column(sell, ["Verkauf", "Sell-Out", "Absatz"], "Verkaufsmenge im Sales"): "SalesQty"
    }
    sell = sell.rename(columns=sell_cols)[
        ["Artikelnr", "EAN", "Bezeichnung_Sales", "SalesQty"]
    ]

    # Wenn Bezeichnung im Sales fehlt, aus PL ziehen
    # (Dazu brauchen wir Merge per Artikelnr & EAN, daher später)

    # ---------- 1. Merge per Artikelnummer ----------
    m1 = pd.merge(
        sell,
        price,
        on="Artikelnr",
        how="left",
        suffixes=("_sales", "_pl"),
        validate="many_to_one",
        indicator=True
    )

    # ---------- 2. Fallback-Merge per EAN für alle, die in m1 kein price gefunden haben ----------
    no_price = m1["_merge"] == "left_only"
    if no_price.any():
        # Select nur die Rows ohne match
        fallback = pd.merge(
            m1.loc[no_price, ["Artikelnr", "EAN", "Bezeichnung_Sales", "SalesQty"]],
            price,
            left_on="EAN",
            right_on="EAN",
            how="left",
            suffixes=("", "_pl2"),
            validate="many_to_one"
        )
        # Bezeichnung aus Sales falls vorhanden, sonst PL
        fallback["Bezeichnung"] = fallback["Bezeichnung_Sales"].fillna(fallback["Bezeichnung_pl2"])
        # Bestand, Einstandspreis, Kategorie aus PL übernehmen
        for col in ["Bestand", "Einstandspreis", "Kategorie"]:
            fallback[col] = fallback[col].combine_first(fallback[f"{col}_pl2"])
        # Drop unnötige
        fallback = fallback[ m1.columns.drop("_merge") ]  # selbe Struktur wie m1 ohne _merge

        # Nun alte rows in m1 überschreiben
        m1 = pd.concat([
            m1.loc[~no_price, m1.columns != "_merge"],
            fallback
        ], ignore_index=True)

    else:
        m1 = m1.drop(columns="_merge")

    # ---------- Finale Spalten aufräumen ----------
    # Wenn in m1 weder Bezeichnung_Sales noch PL-Bezeichnung existieren, bleibt NaN
    m1["Bezeichnung"] = m1["Bezeichnung_Sales"].fillna(m1["Bezeichnung"])

    return m1[
        ["Artikelnr", "EAN", "Bezeichnung", "Kategorie", "SalesQty", "Einstandspreis", "Bestand"]
    ]


def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    col1, col2 = st.columns(2)
    with col1:
        sell_file  = st.file_uploader("Sell-Out-Report (.xlsx)", type="xlsx", key="u1")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)",       type="xlsx", key="u2")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen.")
        return

    sell_df  = pd.read_excel(sell_file)
    price_df = pd.read_excel(price_file)

    # Merge & Enrichment
    merged = enrich_and_merge(sell_df, price_df)

    # ---------- Aggregation ----------
    agg = merged.groupby(
        ["Artikelnr", "Bezeichnung", "Kategorie"],
        as_index=False
    ).agg(
        Verkaufsmenge = ("SalesQty", "sum"),
        # Lagermenge nur einmal pro Artikel: nehmen wir das Maximum des (verfügbaren) Bestands
        Lagermenge    = ("Bestand",    "max"),
        # Einstandspreis sollte pro Artikel gleich sein: nehmen wir das erste
        Einstandspreis = ("Einstandspreis", "first")
    )

    # ---------- Werte berechnen ----------
    agg["Einkaufsmenge"] = agg["Verkaufsmenge"]
    agg["Verkaufswert"]  = agg["Verkaufsmenge"] * agg["Einstandspreis"]
    agg["Einkaufswert"]  = agg["Einkaufsmenge"] * agg["Einstandspreis"]
    agg["Lagerwert"]     = agg["Lagermenge"]    * agg["Einstandspreis"]

    # ---------- Ergebnis-Tabelle in exakter Spaltenreihenfolge ----------
    result = agg[
        [
            "Artikelnr", "Bezeichnung", "Kategorie",
            "Einkaufsmenge", "Einkaufswert",
            "Verkaufsmenge", "Verkaufswert",
            "Lagermenge",   "Lagerwert"
        ]
    ]

    st.dataframe(result, use_container_width=True)


if __name__ == "__main__":
    main()
