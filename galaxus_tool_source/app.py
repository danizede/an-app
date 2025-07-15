import streamlit as st
import pandas as pd

# ---------------- Passwortschutz ----------------
PW = "Nofava22caro!"
pw = st.text_input("🔐 Passwort eingeben", type="password")
if pw != PW:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

# ---------- Alias-Listen für Spaltennamen ----------
ALIAS_NR     = ["Artikelnr", "Artikelnr.", "Artikelnummer", "Art.-Nr.", "Artikelnumm"] 
ALIAS_EAN    = ["EAN", "GTIN"]
ALIAS_BEZ    = ["Bezeichnung", "Bez"]
ALIAS_CAT    = ["Kategorie", "Warengruppe", "Zusatz"]
ALIAS_PREIS  = ["Preis", "Verkaufspreis", "VK", "NETTO NETTO", "Netto"]

def find_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Sucht in df.columns nach einem der Einträge in candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Spalte «{label}» fehlt – gesucht: {candidates}")

# ---------------- Lade-Funktion ----------------
@st.cache_data
def load_xlsx(bin_data: bytes) -> pd.DataFrame:
    return pd.read_excel(bin_data, engine="openpyxl")

# ------------- Matching & Anreicherung -------------
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Spalten in der Preis-Liste umbenennen
    nr_col   = find_col(price, ALIAS_NR,    "Artikelnr")
    ean_col  = find_col(price, ALIAS_EAN,   "EAN/GTIN")
    bez_col  = find_col(price, ALIAS_BEZ,   "Bezeichnung")
    cat_col  = find_col(price, ALIAS_CAT,   "Kategorie/Zusatz")
    pr_col   = find_col(price, ALIAS_PREIS, "Preis")
    price = price.rename(columns={
        nr_col: "Artikelnr",
        ean_col: "EAN",
        bez_col: "Bezeichnung",
        cat_col: "Kategorie",
        pr_col: "Preis"
    })

    # 2) Spalten im Sell-Out-Report umbenennen
    nr_s     = find_col(sell, ALIAS_NR,  "Artikelnr")
    ean_s    = find_col(sell, ALIAS_EAN, "EAN/GTIN")
    sell = sell.rename(columns={nr_s:"Artikelnr", ean_s:"EAN"})

    # 3) Erstes Matching über Hersteller-Nr. (Artikelnr)
    merged = sell.merge(
        price[["Artikelnr","Bezeichnung","Kategorie","Preis"]],
        on="Artikelnr", how="left"
    )

    # 4) Fehlende Preise per EAN nachtragen
    mask = merged["Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        df2 = ( merged[mask]
                .merge(
                    price.drop_duplicates("EAN")[["EAN","Bezeichnung","Kategorie","Preis"]],
                    on="EAN", how="left"
                )
              )
        # Werte ersetzen
        merged.loc[mask, ["Bezeichnung","Kategorie","Preis"]] = df2[["Bezeichnung","Kategorie","Preis"]].values

    return merged

# ---------------- Aggregation ----------------
@st.cache_data
def aggregate(df: pd.DataFrame):
    tbl = df.groupby("Artikelnr", dropna=False).agg(
        Verkaufsmenge = ("Verkauf",   "sum"),
        Verkaufswert  = ("Verkaufswert","sum"),
        Einkaufsm     = ("Einkauf",   "sum"),
        Einkaufswert  = ("Einkaufswert","sum"),
        Lagermenge    = ("Verfügbar", "sum"),
        Lagerwert     = ("Lagerwert", "sum")
    ).reset_index()
    tot = {
        "VK": tbl["Verkaufswert"].sum(),
        "EK": tbl["Einkaufswert"].sum(),
        "LG": tbl["Lagerwert"].sum()
    }
    return tbl, tot

# ---------------- UI ----------------
st.title("📦 Galaxus Sell-out Aggregator")

sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)",     type="xlsx")

if sell_file and price_file:
    sell_df  = load_xlsx(sell_file)
    price_df = load_xlsx(price_file)

    enriched = enrich(sell_df, price_df)
    data, totals = aggregate(enriched)

    c1, c2, c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)",    f"{totals['VK']:,.0f}")
    c2.metric("Einkaufswert (CHF)",    f"{totals['EK']:,.0f}")
    c3.metric("Lagerwert (CHF)",       f"{totals['LG']:,.0f}")

    st.dataframe(data, use_container_width=True)
else:
    st.info("Bitte sowohl Sell-out-Report als auch Preisliste hochladen, um die Auswertung zu starten.")
