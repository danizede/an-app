
import streamlit as st
import pandas as pd

st.set_page_config("Galaxus Aggregator", layout="wide")
st.title("📦 Galaxus Sell‑out Aggregator")

sell_file = st.file_uploader("Sell‑out‑Report (.xlsx)", type="xlsx")
pl_file   = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

if sell_file and pl_file:
    sell_df = pd.read_excel(sell_file)
    pl_df   = pd.read_excel(pl_file)

    need_s = ["Hersteller-Nr.", "Einkauf", "Verfügbar", "Verkauf"]
    need_p = ["Artikelnummer", "NETTO NETTO"]
    if not all(c in sell_df.columns for c in need_s):
        st.error(f"Sell‑out fehlt Spalten: {need_s}")
        st.stop()
    if not all(c in pl_df.columns for c in need_p):
        st.error(f"Preisliste fehlt Spalten: {need_p}")
        st.stop()

    price = pl_df.set_index("Artikelnummer")["NETTO NETTO"].to_dict()

    agg = (sell_df
           .groupby("Hersteller-Nr.", as_index=False)
           .agg(Einkaufsmenge=("Einkauf","sum"),
                Lagermenge   =("Verfügbar","last"),
                Verkaufsmenge=("Verkauf","sum")))

    agg["Preis"]        = agg["Hersteller-Nr."].map(price).fillna(0)
    agg["Einkaufswert"] = agg["Einkaufsmenge"] * agg["Preis"]
    agg["Verkaufswert"] = agg["Verkaufsmenge"] * agg["Preis"]
    agg["Lagerwert"]    = agg["Lagermenge"]    * agg["Preis"]

    k1,k2,k3 = st.columns(3)
    k1.metric("Verkaufswert (CHF)", f"{agg['Verkaufswert'].sum():,.0f}")
    k2.metric("Einkaufswert (CHF)", f"{agg['Einkaufswert'].sum():,.0f}")
    k3.metric("Lagerwert (CHF)",    f"{agg['Lagerwert'].sum():,.0f}")

    st.dataframe(
        agg[["Hersteller-Nr.","Einkaufsmenge","Einkaufswert",
             "Verkaufsmenge","Verkaufswert","Lagermenge","Lagerwert"]],
        use_container_width=True)
