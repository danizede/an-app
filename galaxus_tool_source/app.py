import streamlit as st
import pandas as pd

# ---------------------------------------------
# 1) Hilfsfunktionen
# ---------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """Suche in df.columns nach dem ersten Treffer aus candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte für {purpose} fehlt – gesucht unter {candidates}")

def clean_numeric(series: pd.Series) -> pd.Series:
    """Entfernt Tausender-Trennzeichen und wandelt in float um."""
    s = series.astype(str)
    # Apostrophen entfernen, Komma->Punkt, alles außer Ziffern/.- weglassen
    s = (
        s
        .str.replace("'", "")
        .str.replace(",", ".")
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# ---------------------------------------------
# 2) UI: Upload
# ---------------------------------------------
st.set_page_config(page_title="Galaxus Sell-out", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

sell_up  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_up = st.file_uploader("Preisliste      (.xlsx)", type="xlsx")
if not sell_up or not price_up:
    st.info("Bitte beide Dateien hochladen.")
    st.stop()

# ---------------------------------------------
# 3) Excel laden (gecached)
# ---------------------------------------------
@st.cache_data(show_spinner="📥 Excel laden …")
def load_xlsx(u) -> pd.DataFrame:
    return pd.read_excel(u, engine="openpyxl")

sell = load_xlsx(sell_up)
price= load_xlsx(price_up)

# ---------------------------------------------
# 4) Spalten standardisieren
# ---------------------------------------------
# Sell-out-Report
col_nr     = find_column(sell, ["Hersteller-Nr.","Artikelnr","ArtNr"],  "Hersteller-Nr.")
col_ean    = find_column(sell, ["EAN","GTIN"],                    "EAN")
col_eink   = find_column(sell, ["Einkauf","Menge Einkauf"],       "Einkauf")
col_verk   = find_column(sell, ["Verkauf","Menge Verkauf"],       "Verkauf")
col_best   = find_column(sell, ["Bestand","Lagerbestand"],        "Bestand")

sell = sell.rename(columns={
    col_nr:   "Hersteller-Nr.",
    col_ean:  "EAN",
    col_eink: "Einkauf",
    col_verk: "Verkauf",
    col_best: "Bestand",
})

# Preisliste
col_pnr    = find_column(price, ["Artikelnummer","Artikelnr","ArtNr"], "Artikelnr")
col_pgtin  = find_column(price, ["GTIN","EAN"],                      "GTIN")
col_pbez   = find_column(price, ["Bezeichnung","Name"],             "Bezeichnung")
col_pkat   = find_column(price, ["Zusatz","Kategorie","Warengruppe"],"Kategorie")
col_pr     = find_column(price, ["NETTO NETTO","Preis","VK","Verkaufspreis"], "Preis")

price = price.rename(columns={
    col_pnr:   "Artikelnr",
    col_pgtin: "GTIN",
    col_pbez:  "Bezeichnung",
    col_pkat:  "Kategorie",
    col_pr:    "Preis",
})

# ---------------------------------------------
# 5) Numeric cleaning
# ---------------------------------------------
sell["Einkauf"]  = clean_numeric(sell["Einkauf"])
sell["Verkauf"]  = clean_numeric(sell["Verkauf"])
sell["Bestand"]  = clean_numeric(sell["Bestand"])
price["Preis"]   = clean_numeric(price["Preis"])

# ---------------------------------------------
# 6) Matching & Anreicherung (gecached)
# ---------------------------------------------
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # 6.1 Merge nach Hersteller-Nr.
    m1 = pd.merge(
        sell_df,
        price_df[["Artikelnr","Bezeichnung","Kategorie","Preis"]],
        left_on="Hersteller-Nr.",
        right_on="Artikelnr",
        how="left",
        suffixes=("","_p1")
    )

    # 6.2 Für fehlende Preise: GTIN-Merge
    need_gtin = m1["Preis"].isna() & m1["EAN"].notna()
    if need_gtin.any():
        m2 = pd.merge(
            m1[need_gtin],
            price_df[["GTIN","Bezeichnung","Kategorie","Preis"]],
            left_on="EAN",
            right_on="GTIN",
            how="left",
            suffixes=("","_p2")
        )
        # Coalesce: nimm erst _p1, fällt das aus, nimm _p2
        for col in ["Bezeichnung","Kategorie","Preis"]:
            m1.loc[need_gtin, col] = (
                m2[f"{col}_p1"].fillna(m2[f"{col}_p2"])
            ).values

    return m1

enriched = enrich(sell, price)

# ---------------------------------------------
# 7) Werte berechnen
# ---------------------------------------------
enriched["Einkaufswert"] = enriched["Einkauf"] * enriched["Preis"]
enriched["Verkaufswert"] = enriched["Verkauf"] * enriched["Preis"]
enriched["Lagerwert"]    = enriched["Bestand"] * enriched["Preis"]

# ---------------------------------------------
# 8) Aggregation & Summen (gecached)
# ---------------------------------------------
@st.cache_data(show_spinner="📊 Berechne Kennzahlen …")
def aggregate(df: pd.DataFrame):
    tbl = (
        df
        .groupby(
            ["Hersteller-Nr.","Bezeichnung","Kategorie"],
            dropna=False,
            as_index=False
        )
        .agg(
            Einkaufsmenge   = ("Einkauf",      "sum"),
            Verkaufsmenge   = ("Verkauf",      "sum"),
            Bestand         = ("Bestand",      "sum"),
            Einkaufswert    = ("Einkaufswert", "sum"),
            Verkaufswert    = ("Verkaufswert", "sum"),
            Lagerwert       = ("Lagerwert",    "sum"),
        )
    )
    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

agg_tbl, totals = aggregate(enriched)

# ---------------------------------------------
# 9) Ausgabe
# ---------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
c3.metric("Lagerwert (CHF)",     f"{totals['LG']:,.0f}")

st.markdown("### Detailauswertung pro Artikel")
st.dataframe(agg_tbl, use_container_width=True)
