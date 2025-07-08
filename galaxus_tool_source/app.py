# -----------------------------------------------------------
#  Galaxus Sell-out Aggregator  ·  Streamlit App
#  Passwort:  Nofava22caro!   (Zeile 22 anpassen)
#  Matching-Reihenfolge:
#      1. Artikelnummer   2. EAN/GTIN   3. erste 2 Wörter
#  Ergebnis-Spalten:
#      Artikel-Nr · Bez · Kategorie · EAN ·
#      Einkaufsmenge/Wert · Verkaufsmenge/Wert · Lagermenge/Wert
# -----------------------------------------------------------
# Benötigte Pakete → requirements.txt
#   streamlit>=1.35
#   pandas>=2.2
#   openpyxl
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import re
from pathlib import Path

# ---------- Passwortschutz ----------
CORRECT_PW = {"Nofava22caro!"}

pw = st.text_input("🔒 Passwort eingeben", type="password")
if pw not in CORRECT_PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

# ---------- Hilfsfunktionen ----------

def first_two_words(txt: str) -> str:
    if pd.isna(txt):
        return ""
    tokens = re.findall(r"[A-Za-z0-9äöüÄÖÜß]+", str(txt).lower())
    return " ".join(tokens[:2])

@st.cache_data(show_spinner=False)
def enrich_with_prices(sell: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.rename(
        columns={
            "Artikelnummer": "ArtNr",
            "GTIN": "EAN",
            "Bezeichnung": "Bez",
            "Zusatz": "Kategorie",
            "NETTO NETTO": "Preis",
        }
    )[["ArtNr", "EAN", "Bez", "Kategorie", "Preis"]]

    sell = sell.rename(
        columns={
            "Hersteller-Nr.": "ArtNr",
            "EAN": "EAN",
            "Produktname": "Prod",
        }
    )

    # --- 1) Match Artikelnummer ---------------------------------
    merged = sell.merge(
        prices,
        on="ArtNr",
        how="left",
        suffixes=("", "_pl"),
    )

    # --- 2) Match EAN für noch fehlende --------------------------
    mask_missing = merged["Preis"].isna() & merged["EAN"].notna()
    if mask_missing.any():
        df_ean = (
            merged.loc[mask_missing, ["EAN"]]
            .merge(prices, on="EAN", how="left")
            [["Bez", "Kategorie", "Preis"]]
        )
        merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = df_ean.values

    # --- 3) Fuzzy: erste 2 Wörter --------------------------------
    prices["tkn"] = prices["Bez"].apply(first_two_words)
    merged["tkn"] = merged["Prod"].apply(first_two_words)

    price_by_tkn = prices.drop_duplicates("tkn").set_index("tkn")[["Bez", "Kategorie", "Preis"]]

    mask_missing = merged["Preis"].isna() & merged["tkn"].notna()
    if mask_missing.any():
        df_fuzzy = merged.loc[mask_missing, "tkn"].map(price_by_tkn).apply(pd.Series)
        merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = df_fuzzy.values

    # Fallback: Bezeichnung = Original-Produktname
    merged["Bez"] = merged["Bez"].fillna(merged["Prod"])
    return merged.drop(columns="tkn")

def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Summen
    sum_tbl = (
        df.groupby("ArtNr", as_index=False)
        .agg(
            Einkaufsmenge=("Einkauf", "sum"),
            Verkaufsmenge=("Verkauf", "sum"),
            Lagermenge=("Verfügbar", "last"),
            Bez=("Bez", "last"),
            Kategorie=("Kategorie", "last"),
            EAN=("EAN", "last"),
            Preis=("Preis", "last"),
        )
    )
    sum_tbl["Einkaufswert"] = sum_tbl["Einkaufsmenge"] * sum_tbl["Preis"]
    sum_tbl["Verkaufswert"] = sum_tbl["Verkaufsmenge"] * sum_tbl["Preis"]
    sum_tbl["Lagerwert"] = sum_tbl["Lagermenge"] * sum_tbl["Preis"]

    totals = {
        "Verkaufswert": sum_tbl["Verkaufswert"].sum(),
        "Einkaufswert": sum_tbl["Einkaufswert"].sum(),
        "Lagerwert": sum_tbl["Lagerwert"].sum(),
    }
    return sum_tbl, totals

# ---------- UI ----------

st.title("📦 Galaxus Sell-out Aggregator")

sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx", key="sell")
price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx", key="price")

# Session-State: zuletzt hochgeladene behalten
if "sell_df" not in st.session_state:
    st.session_state.sell_df = None
if "price_df" not in st.session_state:
    st.session_state.price_df = None

if sell_file:
    st.session_state.sell_df = pd.read_excel(sell_file)
if price_file:
    st.session_state.price_df = pd.read_excel(price_file)

if st.session_state.sell_df is None or st.session_state.price_df is None:
    st.info("Bitte Sell-out-Report **und** Preisliste hochladen, um die Auswertung zu starten.")
    st.stop()

sell_df = st.session_state.sell_df.copy()
price_df = st.session_state.price_df.copy()

st.success("Dateien geladen ✔")

# ---------------- Auswertung ----------------

enriched = enrich_with_prices(sell_df, price_df)
agg_tbl, totals = compute_agg(enriched)

c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{totals['Verkaufswert']:,.0f}")
c2.metric("Einkaufswert (CHF)", f"{totals['Einkaufswert']:,.0f}")
c3.metric("Lagerwert (CHF)", f"{totals['Lagerwert']:,.0f}")

st.markdown("### Detail")
st.dataframe(
    agg_tbl[
        [
            "ArtNr",
            "Bez",
            "Kategorie",
            "EAN",
            "Einkaufsmenge",
            "Einkaufswert",
            "Verkaufsmenge",
            "Verkaufswert",
            "Lagermenge",
            "Lagerwert",
            "Preis",
        ]
    ],
    hide_index=True,
)
