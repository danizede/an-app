import streamlit as st
import pandas as pd

# ---------- Seiten-Layout ----------
st.set_page_config("Galaxus Sell-out Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

# ---------- Datei-Uploads ----------
sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
pl_file   = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

# ---------- sobald beide Dateien vorhanden ----------
if sell_file and pl_file:
    sell_df = pd.read_excel(sell_file)
    pl_df   = pd.read_excel(pl_file)

    # --- Pflicht­spalten prüfen ---
    need_s = ["Hersteller-Nr.", "Einkauf", "Verfügbar", "Verkauf"]
    need_p = ["Artikelnummer", "Bezeichnung", "NETTO NETTO"]
    miss_s = [c for c in need_s if c not in sell_df.columns]
    miss_p = [c for c in need_p if c not in pl_df.columns]
    if miss_s:
        st.error("Sell-out-Datei fehlt Spalten: " + ", ".join(miss_s))
        st.stop()
    if miss_p:
        st.error("Preisliste fehlt Spalten: " + ", ".join(miss_p))
        st.stop()

    # --- Key vereinheitlichen ---
    sell_df = sell_df.rename(columns={"Hersteller-Nr.": "Artikelnummer"})

    # --- Mengen aggregieren ---
    agg = (sell_df
           .groupby("Artikelnummer", as_index=False)
           .agg(Einkaufsmenge=("Einkauf",    "sum"),
                Lagermenge   =("Verfügbar", "last"),
                Verkaufsmenge=("Verkauf",    "sum")))

    # --- Preis & Beschreibung aus PL zuordnen ---
    price_dict = pl_df.set_index("Artikelnummer")["NETTO NETTO"].to_dict()
    descr_dict = pl_df.set_index("Artikelnummer")["Bezeichnung"].to_dict()

    agg["Preis"]       = agg["Artikelnummer"].map(price_dict).fillna(0)
    agg["Bezeichnung"] = agg["Artikelnummer"].map(descr_dict).fillna("—")

    # --- Wertspalten berechnen ---
    agg["Einkaufswert"] = agg["Einkaufsmenge"] * agg["Preis"]
    agg["Verkaufswert"] = agg["Verkaufsmenge"] * agg["Preis"]
    agg["Lagerwert"]    = agg["Lagermenge"]    * agg["Preis"]

    # --- KPIs ausgeben ---
    k1, k2, k3 = st.columns(3)
    k1.metric("Verkaufswert (CHF)", f"{agg['Verkaufswert'].sum():,.0f}")
    k2.metric("Einkaufswert (CHF)", f"{agg['Einkaufswert'].sum():,.0f}")
    k3.metric("Lagerwert (CHF)",    f"{agg['Lagerwert'].sum():,.0f}")

    # --- Tabelle anzeigen ---
    agg = agg.sort_values("Artikelnummer")
    st.dataframe(
        agg[["Artikelnummer", "Bezeichnung",
             "Einkaufsmenge", "Einkaufswert",
             "Verkaufsmenge", "Verkaufswert",
             "Lagermenge",    "Lagerwert"]],
        use_container_width=True
    )
else:
    st.info("Bitte Sell-out-Report **und** Preisliste hochladen, um die Auswertung zu starten.")
