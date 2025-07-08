# ──────────────────────────────────────────────────────────────
#  Galaxus Sell-out Aggregator  ·  Streamlit App
#  – Passwortschutz (Nofava22caro!)
#  – Matching ① Artikelnr ② EAN/GTIN ③ erste 2 Wörter
#  – Berechnet Einkaufs-, Verkaufs-, Lagerwerte (CHF)
# ──────────────────────────────────────────────────────────────
#  requirements.txt:
#     streamlit>=1.35
#     pandas>=2.2
#     openpyxl>=3.1
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from io import BytesIO

# ---------- PASSWORTSCHUTZ -----------------------------------
CORRECT_PW = {"Nofava22caro!"}

pw = st.text_input("🔒 Passwort eingeben", type="password")
if pw not in CORRECT_PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

# ---------- HEADLINE -----------------------------------------
st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

# ---------- Datei-Uploads (mit Session-State zum Merken) ------
ss = st.session_state
sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx",
                             key="sell",
                             help="Report von Galaxus hochladen")
price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx",
                              key="price",
                              help="Aktuelle PL hochladen")

# Zwischenspeichern → bei Neu-Laden wieder vorbefüllt
if sell_file:  ss["last_sell"]  = sell_file.getvalue()
if price_file: ss["last_price"] = price_file.getvalue()

def read_xlsx(binary: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(binary))

# ---------- Hilfsfunktionen ----------------------------------
ALIAS_NR   = ["Artikelnr", "Artikelnummer", "Hersteller-Nr.", "Hersteller Nr"]
ALIAS_EAN  = ["EAN", "GTIN", "EAN Code", "EAN_Code"]
ALIAS_BEZ  = ["Bez", "Bezeichnung", "Produktname"]
ALIAS_CAT  = ["Kategorie", "Warengruppe"]
ALIAS_PREIS= ["Preis", "NETTO NETTO", "Netto", "VK"]

def _col(df, cand, lbl):
    for c in cand:
        if c in df.columns: return c
    raise KeyError(f"Spalte für {lbl} fehlt – gesucht: {cand}")

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich_with_prices(sell_bin: bytes, price_bin: bytes):
    sell   = read_xlsx(sell_bin)
    prices = read_xlsx(price_bin)

    s_nr  = _col(sell  , ALIAS_NR  , "Hersteller-Nr.")
    s_ean = _col(sell  , ALIAS_EAN , "EAN / GTIN")
    p_nr  = _col(prices, ALIAS_NR  , "Hersteller-Nr.")
    p_ean = _col(prices, ALIAS_EAN , "EAN / GTIN")
    p_bez = _col(prices, ALIAS_BEZ , "Bezeichnung")
    p_cat = _col(prices, ALIAS_CAT , "Kategorie")
    p_pr  = _col(prices, ALIAS_PREIS, "Preis")

    prices = prices.rename(columns={
        p_nr:"nr", p_ean:"ean", p_bez:"Bez",
        p_cat:"Kategorie", p_pr:"Preis"
    })[["nr","ean","Bez","Kategorie","Preis"]]

    # ① Artikelnr-Match
    merged = (sell
              .merge(prices.drop_duplicates("nr"),
                     left_on=s_nr, right_on="nr", how="left",
                     suffixes=("", "_p")))

    # ② EAN / GTIN-Match
    m2 = merged["Preis"].isna() & merged[s_ean].notna()
    if m2.any():
        df2 = (merged[m2]
               .merge(prices.drop_duplicates("ean"),
                      left_on=s_ean, right_on="ean", how="left")
               [["Bez","Kategorie","Preis"]])
        merged.loc[m2, ["Bez","Kategorie","Preis"]] = df2.values

    # ③ Fuzzy: erste zwei Wörter
    def tkn(txt): return " ".join(str(txt).lower().split()[:2])
    prices["tkn"] = prices["Bez"].apply(tkn)
    merged["tkn"] = merged["Bez"].apply(tkn)
    m3 = merged["Preis"].isna()
    if m3.any():
        df3 = (merged[m3]
               .merge(prices.drop_duplicates("tkn")[["tkn","Bez","Kategorie","Preis"]],
                      on="tkn", how="left")[["Bez","Kategorie","Preis"]])
        merged.loc[m3, ["Bez","Kategorie","Preis"]] = df3.values

    # Werte in CHF
    merged["Einkaufswert"] = merged["Einkauf"]   * merged["Preis"]
    merged["Verkaufswert"] = merged["Verkauf"]   * merged["Preis"]
    merged["Lagerwert"]    = merged["Verfügbar"] * merged["Preis"]

    return merged.drop(columns=["nr","ean","tkn"])

@st.cache_data(show_spinner="📊 Aggregation …")
def compute_agg(df: pd.DataFrame):
    agg = (df.groupby(["Hersteller-Nr.", "Bez"], as_index=False)
             .agg(Einkauf=("Einkauf","sum"),
                  Verkaufsmenge=("Verkauf","sum"),
                  Verfügbar=("Verfügbar","sum"),
                  Einkaufswert=("Einkaufswert","sum"),
                  Verkaufswert=("Verkaufswert","sum"),
                  Lagerwert=("Lagerwert","sum")))
    totals = {
        "VK":    df["Verkaufswert"].sum(),
        "EK":    df["Einkaufswert"].sum(),
        "Lager": df["Lagerwert"].sum()
    }
    return agg, totals

# ---------- Haupt-Workflow -----------------------------------
if "last_sell" in ss and "last_price" in ss:
    sell_df = ss["last_sell"]; price_df = ss["last_price"]
    details, tot = compute_agg(enrich_with_prices(sell_df, price_df))

    c1,c2,c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)", f"{tot['VK']:,.0f}".replace(","," "))
    c2.metric("Einkaufswert (CHF)", f"{tot['EK']:,.0f}".replace(","," "))
    c3.metric("Lagerwert (CHF)",    f"{tot['Lager']:,.0f}".replace(","," "))

    st.dataframe(details, use_container_width=True)
else:
    st.info("Bitte Sell-out-Report **und** Preisliste hochladen, um die Auswertung zu starten.")
