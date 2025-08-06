import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# --------------------------------------------------
# Hilfsfunktion zum adaptiven Finden einer Spalte
# --------------------------------------------------
def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Sucht in df.columns nach der ersten Spalte, die in candidates auftaucht.
    Liefert den gefundenen Namen oder wirft einen verständlichen KeyError.
    """
    for c in candidates:
        if c in df.columns:
            return c
    avail = df.columns.tolist()
    raise KeyError(f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
                   f"Verfügbare Spalten: {avail}")

# --------------------------------------------------
# Anreicherungsfunktion
# --------------------------------------------------
@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    # --- 1) Spalten in Sell-Out Report finden ---
    s_nr   = find_column(sell_df, ["Artikelnummer","ArtNr","Hersteller-Nr."], "Artikelnummer")
    s_ean  = find_column(sell_df, ["EAN","GTIN"], "EAN")
    s_qty  = find_column(sell_df, ["Verkauf","Sell-Out Menge","Menge"], "Verkaufsmenge")
    s_stock= find_column(sell_df, ["Verfügbar","Bestand","Lagerbestand"], "Lagermenge")
    s_name = find_column(sell_df, ["Produktname","Bezeichnung","Name"], "Bezeichnung")
    s_cat  = find_column(sell_df, ["Warengruppe","Kategorie","Zusatz"], "Kategorie")

    # --- 2) Spalten in Preisliste finden ---
    p_nr    = find_column(price_df, ["Produkt ID","ArtNr","Hersteller-Nr."], "Artikelnummer")
    p_ean   = find_column(price_df, ["EAN","GTIN"], "EAN")
    p_buy   = find_column(price_df, ["Einkauf","EK","Einkaufspreis"], "EK-Preis")
    p_sell  = find_column(price_df, ["Verkauf","VK","Netto"], "VK-Preis")
    p_stock = find_column(price_df, ["Verfügbar","Bestand"], "Lagermenge")
    p_name  = find_column(price_df, ["Produktname","Bezeichnung","Name"], "Bezeichnung")
    p_cat   = find_column(price_df, ["Warengruppe","Kategorie","Zusatz"], "Kategorie")

    # Umbenennen, damit wir später einheitliche Spaltennamen haben
    pr = price_df.rename(columns={
        p_nr:    "ArtNr",
        p_ean:   "EAN",
        p_buy:   "EK_Preis",
        p_sell:  "VK_Preis",
        p_stock: "Lagermenge_PL",
        p_name:  "Bezeichnung_PL",
        p_cat:   "Kategorie_PL",
    })

    sd = sell_df.rename(columns={
        s_nr:   "ArtNr",
        s_ean:  "EAN",
        s_qty:  "Verkaufsmenge",
        s_stock:"Lagermenge_SO",
        s_name: "Bezeichnung_SO",
        s_cat:  "Kategorie_SO",
    })

    # ------------- Merge auf Hersteller-Nr. / ArtNr ----------------
    merged = sd.merge(
        pr[["ArtNr","EAN","EK_Preis","VK_Preis","Lagermenge_PL","Bezeichnung_PL","Kategorie_PL"]],
        on="ArtNr",
        how="left",
        suffixes=("_SO", "_PL")
    )

    # ------------- Falls per EAN noch fehlend ----------------------
    mask = merged["VK_Preis"].isna() & merged["EAN"].notna()
    if mask.any():
        fallback = merged[mask].merge(
            pr.drop_duplicates("EAN"),
            on="EAN", how="left",
            suffixes=("_SO", "_PL")
        )
        for col in ["EK_Preis","VK_Preis","Lagermenge_PL","Bezeichnung_PL","Kategorie_PL"]:
            merged.loc[mask, col] = fallback[col].values

    # ------------- Fallback: Fuzzy-Join auf Bezeichnung (erste 2 Worte) ----------
    # Damit es hier nicht ewig braucht, zuerst Token-Kolumne in pr
    pr["tok2"] = pr["Bezeichnung_PL"].astype(str).str.lower().str.split().str[:2].str.join(" ")
    sd["tok2"] = sd["Bezeichnung_SO"].astype(str).str.lower().str.split().str[:2].str.join(" ")
    mask2 = merged["VK_Preis"].isna() & sd["tok2"].notna()
    if mask2.any():
        fb2 = sd[mask2].merge(
            pr.drop_duplicates("tok2"),
            on="tok2", how="left",
            suffixes=("_SO", "_PL")
        )
        for col in ["EK_Preis","VK_Preis","Lagermenge_PL","Bezeichnung_PL","Kategorie_PL"]:
            merged.loc[mask2, col] = fb2[col].values

    # Jetzt sind alle Price-Spalten gefüllt – falls nicht, wird NaN bleiben

    # ------------- Endgültige Spalten setzen -----------------------
    merged["Einkaufsmenge"]  = merged["Verkaufsmenge"]  # hier ggf. anders, falls PL eine eigene Einkaufs-Spalte hat
    merged["Einkaufswert"]   = merged["Einkaufsmenge"] * merged["EK_Preis"]
    merged["Verkaufswert"]   = merged["Verkaufsmenge"] * merged["VK_Preis"]
    merged["Lagermenge"]     = merged["Lagermenge_PL"]
    merged["Lagerwert"]      = merged["Lagermenge"] * merged["VK_Preis"]

    # Aufräumen & zurückliefern
    return merged[[
        "ArtNr", "Bezeichnung_SO", "Kategorie_SO",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge",   "Lagerwert"
    ]].rename(columns={
        "ArtNr": "Artikelnummer",
        "Bezeichnung_SO": "Bezeichnung",
        "Kategorie_SO":   "Kategorie"
    })

# --------------------------------------------------
# Visualisierung und Aggregation
# --------------------------------------------------
@st.cache_data(show_spinner="⚙️ Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Gruppieren nach Artikel
    agg = (
        df.groupby(["Artikelnummer","Bezeichnung","Kategorie"], dropna=False)
          .agg(
            Einkaufsmenge = ("Einkaufsmenge", "sum"),
            Einkaufswert  = ("Einkaufswert",  "sum"),
            Verkaufsmenge = ("Verkaufsmenge", "sum"),
            Verkaufswert  = ("Verkaufswert",  "sum"),
            Lagermenge    = ("Lagermenge",    "sum"),
            Lagerwert     = ("Lagerwert",     "sum"),
          )
          .reset_index()
    )
    totals = {
        "EK_Menge":   agg["Einkaufsmenge"].sum(),
        "EK_Wert":    agg["Einkaufswert"].sum(),
        "Verk_Menge": agg["Verkaufsmenge"].sum(),
        "Verk_Wert":  agg["Verkaufswert"].sum(),
        "Lager_Menge":agg["Lagermenge"].sum(),
        "Lager_Wert": agg["Lagerwert"].sum(),
    }
    return agg, totals

# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------
def main():
    st.title("📦 Galaxus Sell-out Aggregator")

    # Upload-Widgets nebeneinander
    col1, col2 = st.columns(2)
    with col1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)",   type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen, um fortzufahren.")
        return

    # Einlesen
    try:
        sell_df  = pd.read_excel(sell_file)
        price_df = pd.read_excel(price_file)
    except Exception as e:
        st.error(f"Fehler beim Einlesen der Excel-Dateien:\n{e}")
        return

    # Anreichern & Aggregieren
    try:
        enriched = enrich(sell_df, price_df)
        agg_tbl, totals = compute_agg(enriched)
    except KeyError as ke:
        st.error(ke)
        return
    except Exception as e:
        st.error(f"Unerwarteter Fehler:\n{e}")
        return

    # Kennzahlen
    c1, c2, c3 = st.columns(3)
    c1.metric("🛒 Gesamt-Einkaufswert", f"{totals['EK_Wert']:,.0f} CHF")
    c2.metric("💰 Gesamt-Verkaufswert", f"{totals['Verk_Wert']:,.0f} CHF")
    c3.metric("📦 Gesamt-Lagerwert",     f"{totals['Lager_Wert']:,.0f} CHF")

    # Breite Tabelle ohne Scrollen
    st.markdown("### Detail-Tabelle")
    st.dataframe(agg_tbl, use_container_width=True)

if __name__ == "__main__":
    main()
