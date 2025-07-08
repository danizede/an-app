# ───────────────────────── app.py ─────────────────────────
# Galaxus Sell-out Aggregator  •  Streamlit App
# - Passwortschutz (Nofava22caro!)
# - Matching nach Artikelnummer ➊  →  EAN/GTIN ➋  →  erste 2 Wörter ➌
# - Bezeichnung kommt immer aus Spalte C der PL
# - Berechnet Einkaufs-, Verkaufs-, Lagerwerte (CHF)
# Benötigte Pakete stehen in requirements.txt:
#   streamlit>=1.35
#   pandas>=2.2
#   openpyxl
# ───────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from datetime import datetime

# ───────── PASSWORTSCHUTZ ────────────────────────────────
CORRECT_PW = {"Nofava22caro!"}

pw = st.text_input("🔒 Passwort eingeben", type="password")
if pw not in CORRECT_PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

# ───────── Pfade & Ordner (persistente Uploads) ──────────
DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SELL_PATH  = DATA_DIR / "last_sellout.xlsx"
PRICE_PATH = DATA_DIR / "last_price.xlsx"

# ───────── Helper: erste 2 Wörter normalisieren ─────────
def first_two(text: str) -> str:
    if pd.isna(text):
        return ""
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", str(text).lower())
    return " ".join(tokens[:2])

# ───────── UI: Überschrift ───────────────────────────────
st.title("📦 Galaxus Sell-out Aggregator")

# ------------------------------------------------------------------
# 1) SELL-OUT UPLOAD  (gespeichert als last_sellout.xlsx)
# ------------------------------------------------------------------
st.subheader("Sell-out-Report (.xlsx)")
up_sell = st.file_uploader(
    "Drag and drop file here", type="xlsx", key="sell"
)
if up_sell:                     # neues Upload ⇒ überschreiben
    SELL_PATH.write_bytes(up_sell.getbuffer())
    st.success("Sell-out gespeichert ✅")

if SELL_PATH.exists():
    sell_df = pd.read_excel(SELL_PATH)
else:
    st.info("Bitte Sell-out-Report hochladen, um fortzufahren.")
    st.stop()

# ------------------------------------------------------------------
# 2) PREISLISTE UPLOAD  (gespeichert als last_price.xlsx)
# ------------------------------------------------------------------
st.subheader("Preisliste (.xlsx)")
up_price = st.file_uploader(
    "Drag and drop file here", type="xlsx", key="price"
)
if up_price:
    PRICE_PATH.write_bytes(up_price.getbuffer())
    st.success("Preisliste gespeichert ✅")

if PRICE_PATH.exists():
    price_df = pd.read_excel(PRICE_PATH)
else:
    st.info("Bitte Preisliste hochladen, um die Auswertung zu starten.")
    st.stop()

# ───────── Preisliste aufbereiten ─────────────────────────
pl = price_df.rename(
    columns={
        "Artikelnummer": "ArtNr",
        "GTIN": "EAN",
        "Bezeichnung": "Bez",
        "NETTO NETTO": "Preis",
        "Zusatz": "Kategorie",
    }
)[["ArtNr", "EAN", "Bez", "Kategorie", "Preis"]].copy()

pl["tkn"] = pl["Bez"].apply(first_two)

# ───────── Sell-out aufbereiten ────────────────────────────
so = sell_df.rename(
    columns={
        "Hersteller-Nr.": "ArtNr",
        "EAN": "EAN",
        "Einkauf": "Einkauf",
        "Verfügbar": "Lager",
        "Verkauf": "Verkauf",
        "Sell-Out bis": "Datum",
    }
)[["ArtNr", "EAN", "Einkauf", "Lager", "Verkauf", "Datum"]].copy()

# Datum (US- oder ISO-Format) -> datetime
so["Datum"] = pd.to_datetime(so["Datum"], errors="coerce")

# ───────── Matching ➊ ArtNr ───────────────────────────────
merged = so.merge(
    pl[["ArtNr", "Bez", "Kategorie", "Preis"]],
    on="ArtNr",
    how="left",
    suffixes=("", "_pl"),
)

# ───────── Matching ➋ EAN für fehlende Preise ─────────────
mask_missing = merged["Preis"].isna() & merged["EAN"].notna()
if mask_missing.any():
    merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = (
        merged[mask_missing]
        .merge(
            pl[["EAN", "Bez", "Kategorie", "Preis"]],
            on="EAN",
            how="left",
        )[["Bez", "Kategorie", "Preis"]]
        .values
    )

# ───────── Matching ➌ erste 2 Wörter ──────────────────────
merged["tkn"] = merged["Bez"].apply(first_two)
price_by_tkn = pl.drop_duplicates("tkn").set_index("tkn")[["Bez", "Kategorie", "Preis"]]

mask_missing2 = merged["Preis"].isna() & merged["tkn"].notna()
if mask_missing2.any():
    merged.loc[mask_missing2, ["Bez", "Kategorie", "Preis"]] = (
        merged.loc[mask_missing2, "tkn"].map(price_by_tkn).apply(pd.Series).values
    )

# Fallback: Bez aus Sell-out, falls leer
merged["Bez"] = merged["Bez"].fillna(merged["ArtNr"])

# ───────── Aggregation (gesamter Zeitraum) ─────────────────
agg = (
    merged.groupby(["ArtNr", "Bez"], as_index=False)
    .agg(
        Einkaufsmenge=("Einkauf", "sum"),
        Verkaufsmenge=("Verkauf", "sum"),
        Lagermenge=("Lager", "last"),
        Preis=("Preis", "first"),
    )
)

agg["Einkaufswert"] = agg["Einkaufsmenge"] * agg["Preis"]
agg["Verkaufswert"] = agg["Verkaufsmenge"] * agg["Preis"]
agg["Lagerwert"]    = agg["Lagermenge"]   * agg["Preis"]

tot_verk  = int(agg["Verkaufswert"].sum())
tot_eink  = int(agg["Einkaufswert"].sum())
tot_lager = int(agg["Lagerwert"].sum())

# ───────── KPI-Kacheln ─────────────────────────────────────
st.markdown("### Gesamtwerte")
k1, k2, k3 = st.columns(3)
k1.metric("Verkaufswert (CHF)", f"{tot_verk:,.0f}")
k2.metric("Einkaufswert (CHF)", f"{tot_eink:,.0f}")
k3.metric("Lagerwert (CHF)",    f"{tot_lager:,.0f}")

st.markdown("---")

# ───────── Detailtabelle ───────────────────────────────────
st.dataframe(
    agg[[
        "ArtNr", "Bez",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge", "Lagerwert",
    ]],
    hide_index=True,
)
# ───────────────────────────────────────────────────────────
