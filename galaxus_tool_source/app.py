# ──────────────────────────────────────────────────────────────────────────────
#  Galaxus Sell-out Aggregator – Streamlit App
#
#  Features
#  ──────────────────────────────────────────────────────────────────────────────
#  • Passworthschutz  (PW = Nofava22caro!)
#  • Merken der zuletzt hochgeladenen Dateien (werden serverseitig gespeichert)
#  • Matching-Logik
#      1. Artikelnummer → exakter Join
#      2. GTIN/EAN       → exakter Join
#      3. Fallback:      → erste 2 Wörter aus Produktname vs. Bezeichnung
#  • Spalten aus PL:
#        C = Bezeichnung   → Bez
#        D = Zusatz        → Kategorie
#        F = NETTO NETTO   → Preis
#  • Berechnet Ein­kaufs-, Verkaufs- und Lagermengen/-werte (CHF)
# ──────────────────────────────────────────────────────────────────────────────
#  requirements.txt
#  ──────────────────────────────────────────────────────────────────────────────
#  streamlit>=1.35
#  pandas>=2.3
#  openpyxl
#  python-dateutil
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import streamlit as st

# ──────────────── Konfiguration ───────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SELL_FILE  = DATA_DIR / "last_sellout.xlsx"
PRICE_FILE = DATA_DIR / "last_prices.xlsx"

CORRECT_PW = {"Nofava22caro!"}   #  ← gewünschte Passwörter hier eintragen
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _read_excel(path: Path) -> pd.DataFrame:
    """Read an Excel file with openpyxl engine."""
    return pd.read_excel(path)


def first_two(text: str) -> str:
    if pd.isna(text):
        return ""
    tokens = re.findall(r"[A-Za-z0-9äöüÄÖÜß]+", str(text).lower())
    return " ".join(tokens[:2])


# ──────────────── PASSWORTSCHUTZ ──────────────────────────────────────────────
pw = st.text_input("🔐 Passwort eingeben", type="password")
if pw not in CORRECT_PW:
    st.stop()

# ──────────────── Layout ──────────────────────────────────────────────────────
st.title("📦 Galaxus Sell-out Aggregator")

# --- Upload widgets -----------------------------------------------------------
sell_file = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)", type="xlsx")

# --- Fallback auf gespeicherte Dateien ---------------------------------------
if sell_file is None and SELL_FILE.exists():
    with open(SELL_FILE, "rb") as f:
        sell_file = f
        st.info("▶ Verwende zuletzt gespeicherten Sell-out-Report.")

if price_file is None and PRICE_FILE.exists():
    with open(PRICE_FILE, "rb") as f:
        price_file = f
        st.info("▶ Verwende zuletzt gespeicherte Preisliste.")

if (sell_file is None) or (price_file is None):
    st.warning("Bitte Sell-out-Report **und** Preisliste hochladen, um die "
               "Auswertung zu starten.")
    st.stop()

# --- Dateien auf Disk zwischenspeichern --------------------------------------
if isinstance(sell_file, st.runtime.uploaded_file_manager.UploadedFile):
    SELL_FILE.write_bytes(sell_file.getvalue())
if isinstance(price_file, st.runtime.uploaded_file_manager.UploadedFile):
    PRICE_FILE.write_bytes(price_file.getvalue())

# --- Daten einlesen -----------------------------------------------------------
sell_df  = _read_excel(SELL_FILE)
price_df = _read_excel(PRICE_FILE)

# Umbenennen / Spalten normalisieren
sell_df = sell_df.rename(columns={
    "Hersteller-Nr.": "Artikelnummer",
    "Verfügbar":      "Lager",
    "Verkauf":        "Verkauf",
    "Einkauf":        "Einkauf",
    "EAN":            "GTIN",
})

price_df = price_df.rename(columns={
    "Artikelnummer": "Artikelnummer",
    "GTIN":          "GTIN",
    price_df.columns[2]: "Bez",          # Spalte C
    price_df.columns[3]: "Kategorie",    # Spalte D
    price_df.columns[5]: "Preis",        # Spalte F
})

# --- Matching 1: Artikelnummer -----------------------------------------------
merged = sell_df.merge(
    price_df[["Artikelnummer", "Bez", "Kategorie", "Preis"]],
    on="Artikelnummer",
    how="left",
    suffixes=("", "_pl"),
)

# --- Matching 2: GTIN ---------------------------------------------------------
mask_missing = merged["Preis"].isna() & merged["GTIN"].notna()
if mask_missing.any():
    merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = (
        merged[mask_missing]
        .merge(
            price_df[["GTIN", "Bez", "Kategorie", "Preis"]],
            on="GTIN",
            how="left",
            suffixes=("", "_pl"),
        )[["Bez", "Kategorie", "Preis"]]
        .values
    )

# --- Matching 3: erste 2 Wörter ----------------------------------------------
price_df["tok"] = price_df["Bez"].apply(first_two)
merged["tok"]   = merged["Produktname"].apply(first_two)

tok_map = price_df.drop_duplicates("tok").set_index("tok")[["Bez", "Kategorie", "Preis"]]
mask_missing = merged["Preis"].isna() & merged["tok"].notna()
if mask_missing.any():
    merged.loc[mask_missing, ["Bez", "Kategorie", "Preis"]] = (
        merged.loc[mask_missing, "tok"].map(tok_map).apply(pd.Series).values
    )

merged["Bez"] = merged["Bez"].fillna(merged["Produktname"])

# --- Aggregation --------------------------------------------------------------
agg = (
    merged.groupby(["Artikelnummer", "Bez"], as_index=False)
    .agg(
        Einkaufsmenge=("Einkauf", "sum"),
        Verkaufsmenge=("Verkauf", "sum"),
        Lagermenge=("Lager", "last"),
        Kategorie=("Kategorie", "first"),
        Preis=("Preis", "first"),
    )
)

agg["Einkaufswert"] = agg["Einkaufsmenge"] * agg["Preis"]
agg["Verkaufswert"] = agg["Verkaufsmenge"] * agg["Preis"]
agg["Lagerwert"]    = agg["Lagermenge"]    * agg["Preis"]

tot_verkauf = agg["Verkaufswert"].sum()
tot_einkauf = agg["Einkaufswert"].sum()
tot_lager   = agg["Lagerwert"].sum()

# --- Ausgabe ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Verkaufswert (CHF)", f"{tot_verkauf:,.0f}")
c2.metric("Einkaufswert (CHF)", f"{tot_einkauf:,.0f}")
c3.metric("Lagerwert (CHF)",    f"{tot_lager:,.0f}")

st.dataframe(
    agg[
        [
            "Artikelnummer", "Bez", "Kategorie",
            "Einkaufsmenge", "Einkaufswert",
            "Verkaufsmenge", "Verkaufswert",
            "Lagermenge",    "Lagerwert",
        ]
    ],
    hide_index=True,
    height=600,
)

st.caption(
    f"Letzter Upload Sell-out: **{datetime.fromtimestamp(SELL_FILE.stat().st_mtime).strftime('%d.%m.%Y %H:%M')}** &nbsp;•&nbsp; "
    f"Preisliste: **{datetime.fromtimestamp(PRICE_FILE.stat().st_mtime).strftime('%d.%m.%Y %H:%M')}**"
)
