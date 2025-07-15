# ──────────────────────────────────────────────────────────────────────────────
# Galaxus Sell-out Aggregator – Streamlit App
#  • Matching nach Artikelnummer → Bezeichnung aus PL
#  • Berechnet Verkaufs-, Einkaufs- und Lagerwerte (CHF)
#  • Robust gegen unterschiedliche Spaltennamen
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
from thefuzz import process

# ------------------ Helfer‐Funktionen ------------------------------------------

def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    """Lädt eine .xlsx-Datei als DataFrame."""
    return pd.read_excel(uploaded_file)

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df.columns nach einer der Kandidaten-Lables und gibt
    den gefundenen tatsächlichen Spaltennamen zurück.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte für {purpose} fehlt – gesucht unter {candidates}")

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """
    Anreichern des Sell-out-Reports mit PL-Informationen:
     1) Merge über Hersteller-Nr.
     2) Merge über EAN/GTIN
     3) Optional: Fuzzy-Match auf Bezeichnung (erste zwei Wörter)
    """
    # 1) Spaltennamen finden
    col_nr    = find_column(price, ["Artikelnr", "Hersteller-Nr.", "ArtNr"], "Artikelnr")
    col_gtin  = find_column(price, ["GTIN", "EAN"],                   "EAN/GTIN")
    col_bez   = find_column(price, ["Bezeichnung", "Name"],           "Bezeichnung")
    col_cat   = find_column(price, ["Zusatz", "Kategorie"],           "Kategorie")
    col_pr    = find_column(price, ["Preis","NETTO NETTO","VK"],      "Preis")

    # für Sell-out:
    s_nr      = find_column(sell,  ["Hersteller-Nr.","Artikelnr"],   "Artikelnr")
    s_gtin    = find_column(sell,  ["EAN","GTIN"],                   "EAN/GTIN")
    s_qty     = find_column(sell,  ["Verkauf","Menge","Stück"],      "Verkauf")
    s_stock   = find_column(sell,  ["Bestand","Lagerbestand"],       "Bestand")

    prices = price[[col_nr, col_gtin, col_bez, col_cat, col_pr]].rename(columns={
        col_nr:   "Artikelnr",
        col_gtin: "EAN",
        col_bez:  "Bez",
        col_cat:  "Kategorie",
        col_pr:   "Preis"
    })

    # 2) Merge über Hersteller-Nr.
    merged = sell.merge(
        prices,
        left_on = s_nr,
        right_on= "Artikelnr",
        how     = "left"
    )

    # 3) Merge über EAN/GTIN, wenn Preis noch fehlt
    mask = merged["Preis"].isna() & merged[s_gtin].notna()
    if mask.any():
        df2 = merged[mask].merge(
            prices.drop_duplicates("EAN"),
            left_on = s_gtin, right_on = "EAN", how = "left"
        )[['Bez','Kategorie','Preis']]
        for col in ["Bez","Kategorie","Preis"]:
            merged.loc[mask, col] = df2[col].values

    # 4) Fuzzy-Fallback: wenn immer noch kein Preis, dann match über Bezeichnung
    mask2 = merged["Preis"].isna() & merged["Bez"].notna()
    if mask2.any():
        # Erst zwei Wörter extrahieren
        def first_two(w): return " ".join(w.split()[:2])
        choices = prices["Bez"].unique().tolist()
        tmp = merged.loc[mask2, :].copy()
        tok = tmp[s_nr].astype(str).apply(first_two)  # nutze Hersteller-Nr. als Proxy
        mapped = tok.apply(lambda t: process.extractOne(t, choices, score_cutoff=70) or ("",0))
        merged.loc[mask2, "Bez"] = mapped.apply(lambda x: x[0])
        # nun Kategorie + Preis aus prices ziehen
        merged = merged.merge(
            prices[['Bez','Kategorie','Preis']].drop_duplicates('Bez'),
            on='Bez', how='left', suffixes=('','_fuzzy')
        )
        # Falls original noch nan, fülle von fuzzy:
        for col in ['Kategorie','Preis']:
            merged[col] = merged[col].fillna(merged[f"{col}_fuzzy"])
            merged.drop(columns=[f"{col}_fuzzy"], inplace=True)

    return merged

@st.cache_data(show_spinner="⚙️ Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aggregiert nach Artikel (Hersteller-Nr., Bezeichnung, Preis):
    • Summe Verkauf, Bestand
    • Berechnet Verkaufswert, Einkaufswert, Lagerwert
    • Liefert Detail-Tabelle + Totals
    """
    # Key-Spalte bauen
    df['_key'] = df.apply(
        lambda row: (row['Hersteller-Nr.'], row['Bez'], row['Preis']),
        axis=1
    )
    grp = df.groupby('_key').agg({
        'Verkauf':    'sum',
        'Bestand':    'sum',
    }).reset_index()

    # aus _key zurücksplitten
    grp[['Hersteller-Nr.','Bez','Preis']] = pd.DataFrame(grp['_key'].tolist(), index=grp.index)
    grp.drop(columns=['_key'], inplace=True)

    # Werte berechnen
    grp['Verkaufswert']   = grp['Verkauf'] * grp['Preis']
    grp['Einkaufswert']   = grp['Bestand'] * grp['Preis'] * 0.83  # z.B. EK = 83% vom VK
    grp['Lagerwert']      = grp['Bestand'] * grp['Preis']

    totals = {
        'VK': grp['Verkaufswert'].sum(),
        'EK': grp['Einkaufswert'].sum(),
        'LG': grp['Lagerwert'].sum()
    }
    return grp, totals

# ------------------ Streamlit-UI ------------------------------------------------

st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")

st.title("📦 Galaxus Sell-out Aggregator")

# Dateiupload
sell_file  = st.file_uploader("Sell-out-Report (.xlsx)",  type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

if not sell_file or not price_file:
    st.info("Bitte Sell-out-Report und Preisliste hochladen, um zu starten.")
    st.stop()

# Laden & Anreichern
sell_df  = load_xlsx(sell_file)
price_df = load_xlsx(price_file)

enriched, totals = compute_agg(enrich(sell_df, price_df))

# Kennzahlen
c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}".replace(",", "'"))
c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}".replace(",", "'"))
c3.metric("Lagerwert    (CHF)", f"{totals['LG']:,.0f}".replace(",", "'"))

# Detail-Tabelle
st.dataframe(
    enriched[['Hersteller-Nr.','Bez','Preis','Verkauf','Verkaufswert','Bestand','Lagerwert']],
    use_container_width=True
)
