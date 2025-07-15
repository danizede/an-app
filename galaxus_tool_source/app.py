import streamlit as st
import pandas as pd

# -------------------------------------------------
# 1) App-Header & File-Uploader
# -------------------------------------------------
st.title("📦 Galaxus Sell-out Aggregator")
st.write("Bitte lade Sell-out-Report und Preisliste hoch, um die Auswertung zu starten.")

sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)",    type="xlsx")

if not (sell_file and price_file):
    st.info("Warte auf beide Uploads …")
    st.stop()


# -------------------------------------------------
# 2) Excel laden (gecached)
# -------------------------------------------------
@st.cache_data(show_spinner="📥 Lade Excel …")
def load_xlsx(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, engine="openpyxl")


sell_df  = load_xlsx(sell_file)
price_df = load_xlsx(price_file)


# -------------------------------------------------
# 3) Matching & Anreicherung (gecached)
# -------------------------------------------------
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # Spalten der Preisliste auf Standard-Namen bringen
    price = price.rename(columns={
        "Artikelnummer": "ArtNr",
        "GTIN":           "GTIN",
        "Bezeichnung":    "Bezeichnung",
        "Zusatz":         "Kategorie",
        "NETTO NETTO":    "Preis",
    })

    # 1) Match nach Hersteller-Nr.
    merged = pd.merge(
        sell,
        price[["ArtNr","Bezeichnung","Kategorie","Preis"]],
        left_on="Hersteller-Nr.",
        right_on="ArtNr",
        how="left",
    )

    # 2) Fehlende Preise per GTIN nachtragen
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        df2 = pd.merge(
            merged[mask],
            price[["GTIN","Bezeichnung","Kategorie","Preis"]],
            left_on="EAN",
            right_on="GTIN",
            how="left",
        )
        for col in ["Bezeichnung","Kategorie","Preis"]:
            merged.loc[mask, col] = df2[col].values

    return merged


enriched = enrich(sell_df, price_df)


# -------------------------------------------------
# 4) Aggregation & Summen (gecached)
# -------------------------------------------------
@st.cache_data(show_spinner="📊 Berechne Kennzahlen …")
def compute_agg(df: pd.DataFrame):
    tbl = (
        df
        .groupby(["Hersteller-Nr.","Bezeichnung","Kategorie"], dropna=False)
        .agg(
            Einkaufsmenge   = ("Einkauf",   "sum"),
            Verkaufsmenge   = ("Verkauf",   "sum"),
            Bestand         = ("Bestand",   "sum"),
            Einkaufswert    = ("Preis",     lambda x: (x * df.loc[x.index,"Einkauf"]).sum()),
            Verkaufswert    = ("Preis",     lambda x: (x * df.loc[x.index,"Verkauf"]).sum()),
            Lagerwert       = ("Preis",     lambda x: (x * df.loc[x.index,"Bestand"]).sum()),
        )
        .reset_index()
    )

    totals = {
        "EK": tbl["Einkaufswert"].sum(),
        "VK": tbl["Verkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum(),
    }
    return tbl, totals


agg_tbl, totals = compute_agg(enriched)


# -------------------------------------------------
# 5) Ausgabe: Kennzahlen & Tabelle
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
c2.metric("Einkaufswert (CHF)", f"{totals['EK']:,.0f}")
c3.metric("Lagerwert (CHF)",     f"{totals['LG']:,.0f}")

st.markdown("### Detailauswertung pro Artikel")
st.dataframe(agg_tbl, use_container_width=True)
