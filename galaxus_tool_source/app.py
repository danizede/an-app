# galāxus_tool_source/app.py

import streamlit as st
import pandas as pd

# ────── 1) Passwortschutz ──────
PW = "Nofava22caro!"

if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    pw = st.text_input("🔐 Passwort eingeben", type="password")
    if pw != PW:
        st.warning("Bitte gültiges Passwort eingeben.")
        st.stop()
    st.session_state.authed = True

# ────── 2) Excel-Loader ──────
@st.cache_data(show_spinner="📥 Dateien laden …")
def load_xlsx(uploaded_file: bytes) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, engine="openpyxl")

# ────── 3) Matching & Anreicherung ──────
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # Merge Sell-Out ↔ Preisliste
    merged = sell.merge(
        price[["Artikelnr", "Bezeichnung", "Zusatz", "Preis"]],
        left_on="Hersteller-Nr.",
        right_on="Artikelnr",
        how="left"
    )
    return merged

# ────── 4) Aggregation ──────
@st.cache_data(show_spinner="🔢 Aggregation …")
def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tbl = (
        df
        .groupby(
            ["Hersteller-Nr.", "Bezeichnung", "Zusatz"],
            as_index=False
        )
        .agg(
            Einkaufsmenge = ("Einkauf", "sum"),
            Einkaufswert  = ("Einkaufswert", "sum"),
            Verkaufsmenge = ("Verkauf", "sum"),
            Verkaufswert  = ("Verkaufswert", "sum"),
            Lagermenge    = ("Verfügbar", "sum"),
            Lagerwert     = ("Lagerwert", "sum"),
        )
    )
    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# ────── 5) UI ──────
st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

sell_upl  = st.file_uploader("Sell-out Report (.xlsx)", type="xlsx")
price_upl = st.file_uploader("Preisliste (.xlsx)",    type="xlsx")

if sell_upl and price_upl:
    sell_df  = load_xlsx(sell_upl)
    price_df = load_xlsx(price_upl)

    enriched = enrich(sell_df, price_df)
    data, tot = aggregate(enriched)

    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)", f"{tot['VK']:,.0f}")
    c2.metric("Einkaufswert (CHF)",  f"{tot['EK']:,.0f}")
    c3.metric("Lagerwert (CHF)",     f"{tot['LG']:,.0f}")

    st.dataframe(data, use_container_width=True)
else:
    st.info("Bitte Sell-out-Report und Preisliste hochladen, um die Auswertung zu starten.")
