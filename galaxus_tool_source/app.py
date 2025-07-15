# galaxus_tool_source/app.py

import streamlit as st
import pandas as pd

# ----------------- Passwortschutz -----------------
CORRECT_PW = "Nofava22caro!"

if "auth" not in st.session_state:
    st.session_state.auth = False

pw = st.text_input("🔐 Passwort eingeben", type="password")
if not st.session_state.auth:
    if pw == CORRECT_PW:
        st.session_state.auth = True
        st.experimental_rerun()
    else:
        st.warning("Bitte gültiges Passwort eingeben.")
        st.stop()

# --------- Hilfsfunktionen ---------
@st.cache_data(show_spinner="📥 Lade Excel-Datei …")
def load_xlsx(bin_data: bytes) -> pd.DataFrame:
    """Lädt eine Excel-Datei via openpyxl."""
    return pd.read_excel(bin_data, engine="openpyxl")

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich_with_prices(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Verknüpft Sell-Out und Preisliste über 'Hersteller-Nr.' ➔ 'Artikelnr'
    und holt Bezeichnung, Zusatz (PL-Spalte früher 'Kategorie') und Preis."""
    merged = sell_df.merge(
        price_df[["Artikelnr", "Bezeichnung", "Zusatz", "Preis"]],
        left_on="Hersteller-Nr.",
        right_on="Artikelnr",
        how="left"
    )
    return merged

@st.cache_data(show_spinner="🔢 Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregiert nach Artikel und summiert Mengen & Werte,
    gibt das DataFrame plus eine Totals-Dict zurück."""
    tbl = df.groupby(
        ["Hersteller-Nr.", "Bezeichnung", "Zusatz"], 
        as_index=False
    ).agg(
        Einkaufsmenge = ("Einkauf", "sum"),
        Einkaufswert  = ("Einkaufswert", "sum"),
        Verkaufsmenge = ("Verkauf", "sum"),
        Verkaufswert  = ("Verkaufswert", "sum"),
        Lagermenge    = ("Verfügbar", "sum"),
        Lagerwert     = ("Lagerwert", "sum"),
    )

    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }

    return tbl, totals

# ----------------- Haupt-UI -----------------
st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx")

if sell_file and price_file:
    sell_df  = load_xlsx(sell_file)
    price_df = load_xlsx(price_file)

    enriched = enrich_with_prices(sell_df, price_df)
    data, totals = compute_agg(enriched)

    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)",     f"{totals['VK']:,.0f}")
    c2.metric("Einkaufswert (CHF)",      f"{totals['EK']:,.0f}")
    c3.metric("Lagerwert (CHF)",         f"{totals['LG']:,.0f}")

    st.dataframe(data, use_container_width=True)

else:
    st.info("Bitte Sell-out-Report und Preisliste hochladen, um die Auswertung zu starten.")
