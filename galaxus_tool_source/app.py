# ──────────────────────────────────────────────────────────────
#  Galaxus Sell-out Aggregator  ·  Streamlit-App
#  · Passwort (Nofava22caro!)
#  · Matching:
#      1. Artikelnummer
#      2. EAN / GTIN
#      3. Fuzzy – erste 2 Wörter der Bezeichnung
#  · Berechnet EK-, VK- und Lagerwerte (CHF)
# ──────────────────────────────────────────────────────────────
#  requirements.txt:
#     streamlit>=1.35
#     pandas>=2.2
#     openpyxl>=3.1
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd

# ------------------ Passwortschutz ---------------------------
CORRECT_PW = {"Nofava22caro!"}

pw = st.text_input("🔑 Passwort eingeben", type="password")
if pw not in CORRECT_PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

# ------------------ Hilfsfunktionen --------------------------
def first_two_words(txt: str) -> str:
    """erste 2 Wörter einer Bezeichnung als fuzzy-Key"""
    parts = str(txt).lower().split()
    return " ".join(parts[:2])

@st.cache_data(show_spinner="🚚 Dateien einlesen …")
def load_xlsx(file) -> pd.DataFrame:
    return pd.read_excel(file)

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich_with_prices(sell: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    sell = sell.copy()
    prices = prices.copy()

    # ------------ 1) Hersteller-Nr. ---------------------------
    merged = sell.merge(
        prices[["Artikelnr", "Bez", "Kategorie", "Preis"]],
        left_on="Hersteller-Nr.",
        right_on="Artikelnr",
        how="left",
        suffixes=("", "_p")
    )

    # ------------ 2) EAN / GTIN ------------------------------
    mask_missing = merged["Preis"].isna() & merged["EAN"].notna()
    if mask_missing.any():
        df_ean = (
            merged[mask_missing]
            .merge(
                prices[["GTIN", "Bez", "Kategorie", "Preis"]],
                on="GTIN",
                how="left",
            )[["Bez", "Kategorie", "Preis"]]
        )
        merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = df_ean.values

    # ------------ 3) Fuzzy (erste 2 Wörter) ------------------
    prices["tkn"] = prices["Bez"].apply(first_two_words)
    merged["tkn"] = merged["Bez"].apply(first_two_words)

    mask_missing = merged["Preis"].isna()
    if mask_missing.any():
        df_fuzzy = (
            merged[mask_missing]
            .merge(
                prices[["tkn", "Bez", "Kategorie", "Preis"]],
                on="tkn",
                how="left",
            )[["Bez", "Kategorie", "Preis"]]
        )
        merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = df_fuzzy.values

    # ------------ Werte berechnen ---------------------------
    merged["Einkaufswert"] = merged["Einkauf"] * merged["Preis"]
    merged["Verkaufswert"] = merged["Verkauf"] * merged["Preis"]
    merged["Lagerwert"] = merged["Verfügbar"] * merged["Preis"]

    return merged.drop(columns=["Artikelnr", "tkn"])

# -------- neue Fassung: KEINE Aggregation der Zeilen --------
@st.cache_data
def compute_agg(df: pd.DataFrame):
    """liefert Detail-Tabelle & KPI-Summen (keine Gruppierung)."""
    details = df.copy().sort_values("Verkaufswert", ascending=False)

    totals = {
        "vk": details["Verkaufswert"].sum(),
        "ek": details["Einkaufswert"].sum(),
        "stock": details["Lagerwert"].sum(),
    }
    return details, totals

# ------------------ Streamlit UI -----------------------------
st.title("📦 Galaxus Sell-out Aggregator")

sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx")

if not sell_file or not price_file:
    st.info("Bitte Sell-out-Report **und** Preisliste hochladen, um die Auswertung zu starten.")
    st.stop()

# --------- Daten einlesen, anreichern, KPIs ------------------
sell_df = load_xlsx(sell_file)
price_df = load_xlsx(price_file)

details, totals = compute_agg(enrich_with_prices(sell_df, price_df))

c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{totals['vk']:,.0f}")
c2.metric("Einkaufswert (CHF)", f"{totals['ek']:,.0f}")
c3.metric("Lagerwert (CHF)", f"{totals['stock']:,.0f}")

st.dataframe(
    details,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Bez": st.column_config.Column(width="medium"),
    },
)
