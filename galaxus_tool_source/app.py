# gal-app/galaxus_tool_source/app.py

import streamlit as st
import pandas as pd

# ───────── Passwortschutz ─────────
PW = "Nofava22caro!"
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pw = st.text_input("🔐 Passwort eingeben", type="password")
    if pw == PW:
        st.session_state.auth = True
        st.experimental_rerun()
    else:
        st.stop()

# ───────── Alias-Listen für Spaltennamen ─────────
ALIAS_NR    = ["Artikelnummer", "Artikelnr", "Artikel-Nr."]
ALIAS_EAN   = ["GTIN", "ean", "EAN"]
ALIAS_BEZ   = ["Bezeichnung", "Bez"]
ALIAS_CAT   = ["Zusatz"]
ALIAS_PREIS = ["NETTO NETTO", "Preis", "Verkaufspreis", "VK"]

def find_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Gibt den ersten in df.columns gefundenen Namen zurück."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte «{label}» fehlt – gesucht: {candidates}")

@st.cache_data(show_spinner="📥 Excel-Datei laden …")
def load_xlsx(uploaded: bytes) -> pd.DataFrame:
    return pd.read_excel(uploaded, engine="openpyxl")

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten finden
    s_nr  = find_col(sell,  ALIAS_NR,    "Hersteller-Nr.")
    s_ean = find_col(sell,  ALIAS_EAN,   "EAN/GTIN")
    p_nr  = find_col(price, ALIAS_NR,    "Artikelnr")
    p_ean = find_col(price, ALIAS_EAN,   "EAN/GTIN")
    p_bez = find_col(price, ALIAS_BEZ,   "Bezeichnung")
    p_cat = find_col(price, ALIAS_CAT,   "Kategorie")
    p_pr  = find_col(price, ALIAS_PREIS, "Preis")

    # 2) Preistabelle umbenennen auf Standard-Spalten
    price2 = price.rename(columns={
        p_nr:  "Artikelnr",
        p_ean: "EAN",
        p_bez: "Bez",
        p_cat: "Kategorie",
        p_pr:  "Preis",
    })

    # 3) Erstes Matching über Hersteller-Nr.
    merged = (
        sell.rename(columns={s_nr: "Hersteller-Nr.", s_ean:"EAN"})
        .merge(
            price2[["Artikelnr","EAN","Bez","Kategorie","Preis"]],
            left_on="Hersteller-Nr.", right_on="Artikelnr", how="left"
        )
    )

    # 4) Fehlende Preise per EAN auffüllen
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        fill = (
            merged[mask]
            .merge(
                price2.drop_duplicates("EAN")[["EAN","Bez","Kategorie","Preis"]],
                on="EAN", how="left", suffixes=("","_ean")
            )
        )
        # jetzt in merged kopieren
        merged.loc[mask, ["Bez","Kategorie","Preis"]] = fill[
            ["Bez_ean","Kategorie_ean","Preis_ean"]
        ].values

    return merged

@st.cache_data(show_spinner="🔢 Aggregation …")
def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tbl = (
        df.groupby("Artikelnr", as_index=False)
          .agg(
              Bezeichnung   = ("Bez",          "first"),
              Kategorie      = ("Kategorie",    "first"),
              Verkaufsmenge = ("Verkaufsmenge","sum"),
              Verkaufswert  = ("Verkaufswert", "sum"),
              Einkaufsmenge = ("Einkaufsmenge","sum"),
              Einkaufswert  = ("Einkaufswert", "sum"),
              Lagermenge    = ("Lagermenge",   "sum"),
              Lagerwert     = ("Lagerwert",    "sum"),
          )
    )
    totals = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals

# ───────── UI ─────────
st.title("📦 Galaxus Sell-out Aggregator")

col1, col2 = st.columns(2)
sell_upl  = col1.file_uploader("Sell-Out Report (.xlsx)", type="xlsx")
price_upl = col2.file_uploader("Preisliste (.xlsx)",   type="xlsx")

if sell_upl and price_upl:
    sell_df  = load_xlsx(sell_upl)
    price_df = load_xlsx(price_upl)

    # Debug: welche Spalten hat die PL wirklich?
    st.subheader("🔍 Spalten Deiner Preisliste")
    st.write(price_df.columns.tolist())

    # Anreichern und aggregieren
    enriched = enrich(sell_df, price_df)
    tbl, tot = aggregate(enriched)

    # Kennzahlen
    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)",    f"{tot['VK']:,.0f}")
    c2.metric("Einkaufswert (CHF)",    f"{tot['EK']:,.0f}")
    c3.metric("Lagerwert (CHF)",       f"{tot['LG']:,.0f}")

    # Ergebnis-Tabelle
    st.dataframe(tbl, use_container_width=True)
