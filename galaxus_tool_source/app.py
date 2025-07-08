# --------------------------------------------------------------------
#  📦  Galaxus Sell-out Aggregator  –  Streamlit App
# --------------------------------------------------------------------
#  * Passwortschutz (Nofava22caro!)
#  * Matching-Reihenfolge:  Artikel-Nr. ➜ EAN ➜ erste 2 Wörter
#  * Liefert Bez., Kategorie, Preis, Mengen- & Wert-Spalten
# --------------------------------------------------------------------
#  Benötigte Pakete (requirements.txt):
#     streamlit>=1.35
#     pandas>=2.2
#     openpyxl
#     et-xmlfile
# --------------------------------------------------------------------
import streamlit as st
import pandas as pd
from io import BytesIO

# ---------- PASSWORTSCHUTZ ----------------------------------------------------
CORRECT_PW = "Nofava22caro!"

st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    pw = st.text_input("🔐 Passwort eingeben", type="password")
    if pw == CORRECT_PW:
        st.session_state.auth = True
        st.rerun()

    st.stop()                     # blockt alles, bis PW korrekt eingegeben

# ---------- Hilfsfunktionen ---------------------------------------------------
def first_two_words(text: str) -> str:
    if pd.isna(text):
        return ""
    return " ".join(str(text).split()[:2]).lower()

@st.cache_data(show_spinner=False)
def load_excel(uploaded_file: BytesIO) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)

def enrich_with_prices(sell: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    # -------- 1) Spalten harmonisieren ---------------------------------------
    prices = prices.rename(
        columns={
            "Artikelnummer": "ArtNr",
            "Bezeichnung": "Bez",
            "NETTO NETTO": "Preis",
            "GTIN": "EAN",
        }
    )
    if "Kategorie" not in prices.columns:
        prices["Kategorie"] = ""

    sell = sell.copy()

    for col in ["Bez", "Kategorie", "Preis"]:
        if col not in sell.columns:
            sell[col] = pd.NA

    # -------- 2) 3-stufiges Matching -----------------------------------------
    # 2.1 Artikel-Nr.
    sell = sell.merge(
        prices[["ArtNr", "Bez", "Kategorie", "Preis"]],
        left_on="Hersteller-Nr.",
        right_on="ArtNr",
        how="left",
        suffixes=("", "_pl1"),
    )
    mask = sell["Preis"].isna() & sell["Preis_pl1"].notna()
    sell.loc[mask, ["Bez", "Kategorie", "Preis"]] = sell.loc[
        mask, ["Bez_pl1", "Kategorie_pl1", "Preis_pl1"]
    ].values
    sell.drop(columns=[c for c in sell.columns if c.endswith("_pl1")], inplace=True)

    # 2.2 EAN/GTIN
    sell = sell.merge(
        prices[["EAN", "Bez", "Kategorie", "Preis"]],
        on="EAN",
        how="left",
        suffixes=("", "_pl2"),
    )
    mask = sell["Preis"].isna() & sell["Preis_pl2"].notna()
    sell.loc[mask, ["Bez", "Kategorie", "Preis"]] = sell.loc[
        mask, ["Bez_pl2", "Kategorie_pl2", "Preis_pl2"]
    ].values
    sell.drop(columns=[c for c in sell.columns if c.endswith("_pl2")], inplace=True)

    # 2.3 erste 2 Wörter (Fuzzy)
    sell["match_key"] = sell["Produktname"].apply(first_two_words)
    prices["match_key"] = prices["Bez"].apply(first_two_words)
    sell = sell.merge(
        prices[["match_key", "Bez", "Kategorie", "Preis"]],
        on="match_key",
        how="left",
        suffixes=("", "_pl3"),
    )
    mask = sell["Preis"].isna() & sell["Preis_pl3"].notna()
    sell.loc[mask, ["Bez", "Kategorie", "Preis"]] = sell.loc[
        mask, ["Bez_pl3", "Kategorie_pl3", "Preis_pl3"]
    ].values
    sell.drop(columns=[c for c in sell.columns if c.endswith("_pl3")] + ["match_key"], inplace=True)

    return sell

def compute_aggregation(df: pd.DataFrame):
    df["Einkaufswert"] = df["Einkauf"] * df["Preis"]
    df["Verkaufswert"] = df["Verkauf"] * df["Preis"]
    df["Lagerwert"]    = df["Verfügbar"] * df["Preis"]

    totals = {
        "Verkaufswert": df["Verkaufswert"].sum(),
        "Einkaufswert": df["Einkaufswert"].sum(),
        "Lagerwert":    df["Lagerwert"].sum(),
    }

    cols_out = [
        "Hersteller-Nr.",
        "Bez",
        "EAN",
        "Einkauf",
        "Einkaufswert",
        "Verkauf",
        "Verkaufswert",
        "Verfügbar",
        "Lagerwert",
    ]
    return df[cols_out], totals

# ---------- Upload-Widgets (Dateien bleiben in Session) -----------------------
sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
price_file = st.file_uploader("Preisliste (.xlsx)",    type="xlsx", key="price")

if sell_file and price_file:
    # Dateien in der Session hinterlegen / aktualisieren
    if (
        "sell_df"   not in st.session_state
        or st.session_state.get("sell_name") != sell_file.name
    ):
        st.session_state.sell_df   = load_excel(sell_file)
        st.session_state.sell_name = sell_file.name

    if (
        "price_df"  not in st.session_state
        or st.session_state.get("price_name") != price_file.name
    ):
        st.session_state.price_df   = load_excel(price_file)
        st.session_state.price_name = price_file.name

    sell_df  = st.session_state.sell_df
    price_df = st.session_state.price_df

    with st.spinner("🔄 Daten werden angereichert …"):
        enriched = enrich_with_prices(sell_df, price_df)
        table, totals = compute_aggregation(enriched)

    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)",  f"{totals['Verkaufswert']:,.0f}")
    c2.metric("Einkaufswert (CHF)", f"{totals['Einkaufswert']:,.0f}")
    c3.metric("Lagerwert (CHF)",    f"{totals['Lagerwert']:,.0f}")

    st.dataframe(table, use_container_width=True)
else:
    st.info("📥 Bitte **Sell-out-Report** und **Preisliste** hochladen, "
            "um die Auswertung zu starten.")
