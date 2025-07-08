# ──────────────────────────────────────────────────────────────
# Galaxus Sell-out Aggregator  –  Streamlit App
#  - Passwortschutz (Nofava22caro!)
#  - Matching nach Artikelnummer  ➜  Bezeichnung aus PL  (Spalte C)
#  - Berechnet Einkaufs-, Verkaufs- und Lagerwerte (CHF)
# ----------------------------------------------------------------
# Benötigte Pakete stehen in requirements.txt:
#   streamlit>=1.35
#   pandas>=2.2
#   openpyxl
# ----------------------------------------------------------------

import streamlit as st
import pandas as pd

# ---------- PASSWORTSCHUTZ ------------------------------------
CORRECT_PW = {"Nofava22caro!"}          # hier gewünschte Passwörter eintragen

pw = st.text_input("🔒 Passwort eingeben", type="password")
if pw not in CORRECT_PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()                           # Rest der App wird nicht ausgeführt
# --------------------------------------------------------------

# ---------- Seiteneinstellungen --------------------------------
st.set_page_config("Galaxus Sell-out Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")
st.caption("Upload Sell-out-Report **und** Preisliste → Kennzahlen & Tabelle erscheinen")

# ---------- Datei-Uploads --------------------------------------
sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
pl_file   = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

# ---------- Verarbeitung ---------------------------------------
if sell_file and pl_file:

    # Dateien einlesen
    sell_df = pd.read_excel(sell_file)
    pl_df   = pd.read_excel(pl_file)

    # Pflichtspalten prüfen
    need_s = ["Hersteller-Nr.", "Einkauf", "Verfügbar", "Verkauf"]
    need_p = ["Artikelnummer", "Bezeichnung", "NETTO NETTO"]
    miss_s = [c for c in need_s if c not in sell_df.columns]
    miss_p = [c for c in need_p if c not in pl_df.columns]

    if miss_s:
        st.error("❌ Sell-out-Datei fehlt Spalte(n): " + ", ".join(miss_s))
        st.stop()
    if miss_p:
        st.error("❌ Preisliste fehlt Spalte(n): " + ", ".join(miss_p))
        st.stop()

    # Hersteller-Nr. ➜ Artikelnummer umbenennen
    sell_df = sell_df.rename(columns={"Hersteller-Nr.": "Artikelnummer"})

    # Mengen aggregieren
    agg = (sell_df
           .groupby("Artikelnummer", as_index=False)
           .agg(Einkaufsmenge=("Einkauf",    "sum"),
                Lagermenge   =("Verfügbar", "last"),
                Verkaufsmenge=("Verkauf",    "sum")))

    # Preis & Bezeichnung aus PL mappen
    price_dict = pl_df.set_index("Artikelnummer")["NETTO NETTO"].to_dict()
    descr_dict = pl_df.set_index("Artikelnummer")["Bezeichnung"].to_dict()

    agg["Preis"]       = agg["Artikelnummer"].map(price_dict).fillna(0)
    agg["Bezeichnung"] = agg["Artikelnummer"].map(descr_dict).fillna("—")

    # Wertspalten berechnen
    agg["Einkaufswert"] = agg["Einkaufsmenge"] * agg["Preis"]
    agg["Verkaufswert"] = agg["Verkaufsmenge"] * agg["Preis"]
    agg["Lagerwert"]    = agg["Lagermenge"]    * agg["Preis"]

    # ---------- Kennzahlen-Kacheln ------------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Verkaufswert (CHF)", f"{agg['Verkaufswert'].sum():,.0f}")
    k2.metric("Einkaufswert (CHF)", f"{agg['Einkaufswert'].sum():,.0f}")
    k3.metric("Lagerwert (CHF)",    f"{agg['Lagerwert'].sum():,.0f}")

    st.markdown("---")

    # ---------- Ergebnis-Tabelle --------------------------------
    agg = agg.sort_values("Artikelnummer")
    st.dataframe(
        agg[["Artikelnummer", "Bezeichnung",
             "Einkaufsmenge", "Einkaufswert",
             "Verkaufsmenge", "Verkaufswert",
             "Lagermenge",    "Lagerwert"]],
        use_container_width=True
    )

else:
    st.info("⬆️ Bitte beide Dateien hochladen, um die Auswertung zu starten.")
