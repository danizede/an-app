import streamlit as st
import pandas as pd

# ------------------------------ Alias-Definitionen ------------------------------

ALIAS_NR    = ["Artikelnr", "Artikelnummer", "Hersteller-Nr.", "Produkt ID"]
ALIAS_EAN   = ["EAN", "GTIN"]
ALIAS_NAME  = ["Bezeichnung", "Bez", "Produktname", "Name"]
ALIAS_CAT   = ["Kategorie", "Warengruppe", "Zusatz"]
ALIAS_AVAIL = ["Verfügbar", "Bestand", "Startbestand"]
ALIAS_SOLD  = ["Verkauf", "Verkauft", "Sell-Out von"]
ALIAS_PRICE = ["Einkauf", "Preis", "Netto", "VK", "Verkaufspreis", "Einkaufspreis"]

# ------------------------------ Hilfsfunktionen ------------------------------

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """Findet in df.columns das erste Element aus candidates, sonst stoppt mit Fehlermeldung."""
    for c in candidates:
        if c in df.columns:
            return c
    st.error(
        f'Spalte für "{purpose}" fehlt – gesucht unter {candidates}.\n'
        f'Verfügbare Spalten: {list(df.columns)}'
    )
    st.stop()

@st.cache_data
def load_data(sell_file: bytes, price_file: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt die Excel-Dateien in DataFrames."""
    sell  = pd.read_excel(sell_file)
    price = pd.read_excel(price_file)
    return sell, price

@st.cache_data
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Matcht Preise in die Sell-Out-Daten und rechnet Werte aus."""
    # 1) Spalten erkennen
    c_nr    = find_column(price, ALIAS_NR,    "Artikelnr in PL")
    c_ean   = find_column(price, ALIAS_EAN,   "EAN in PL")
    c_name  = find_column(price, ALIAS_NAME,  "Bezeichnung in PL")
    c_cat   = find_column(price, ALIAS_CAT,   "Kategorie in PL")
    c_price = find_column(price, ALIAS_PRICE, "Preis in PL")

    s_nr    = find_column(sell,  ALIAS_NR,    "Artikelnr im Sell-Report")
    s_ean   = find_column(sell,  ALIAS_EAN,   "EAN im Sell-Report")
    s_name  = find_column(sell,  ALIAS_NAME,  "Bezeichnung im Sell-Report")
    s_cat   = find_column(sell,  ALIAS_CAT,   "Kategorie im Sell-Report")
    s_avail = find_column(sell,  ALIAS_AVAIL, "Lagerbestand im Sell-Report")
    s_sold  = find_column(sell,  ALIAS_SOLD,  "Verkaufsmenge im Sell-Report")

    # 2) Numerische Felder parsen (Apostrophe/Leerzeichen entfernen, Komma→Punkt)
    def to_num(s):
        return pd.to_numeric(
            s
            .astype(str)
            .str.replace("'",    "", regex=False)
            .str.replace(" ",    "", regex=False)
            .str.replace(",",    ".", regex=False),
            errors="coerce"
        ).fillna(0)

    sell[s_avail] = to_num(sell[s_avail])
    sell[s_sold]  = to_num(sell[s_sold])
    price[c_price] = to_num(price[c_price])

    # 3) PL-Daten auf Mindestspalten reduzieren & dupl. EAN entfernen
    price = price[[c_nr, c_ean, c_name, c_cat, c_price]] \
                .drop_duplicates(subset=[c_ean])

    # 4) 1. Match über Artikelnr
    merged = sell.merge(
        price,
        left_on = s_nr,
        right_on= c_nr,
        how     = "left",
        suffixes=("", "_pr")
    )

    # 5) 2. Match über EAN für die, die noch keinen Preis haben
    mask = merged[c_price].isna() & merged[s_ean].notna()
    if mask.any():
        tmp = (
            merged[mask]
            .merge(price, left_on=s_ean, right_on=c_ean, how="left")
        )
        for col in (c_name, c_cat, c_price):
            merged.loc[mask, col] = tmp[col].values

    # 6) 3. Fuzzy auf ersten zwei Wörtern der Bezeichnung
    def first_two(w: str) -> str:
        return " ".join(str(w).split()[:2])

    price["tkn"]  = price[c_name].apply(first_two)
    merged["tkn"] = merged[s_name].apply(first_two)

    mask = merged[c_price].isna() & merged["tkn"].notna()
    if mask.any():
        tmp = merged[mask].merge(price, on="tkn", how="left")
        for col in (c_name, c_cat, c_price):
            merged.loc[mask, col] = tmp[col].values

    # 7) Umbenennen in konsistente Spaltennamen
    merged = merged.rename(columns={
        s_nr    : "Artikelnr",
        s_ean   : "EAN",
        s_name  : "Bezeichnung",
        s_cat   : "Kategorie",
        s_avail : "Lagermenge",
        s_sold  : "Verkaufsmenge",
        c_price : "Stückpreis"
    })

    # 8) Werte berechnen
    merged["Einkaufswert"]  = merged["Lagermenge"]    * merged["Stückpreis"]
    merged["Verkaufswert"]  = merged["Verkaufsmenge"] * merged["Stückpreis"]
    merged["Lagerwert"]     = merged["Lagermenge"]    * merged["Stückpreis"]

    return merged

@st.cache_data
def compute_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Gruppiert nach Artikel und berechnet alle Kennzahlen."""
    return (
        df
        .groupby(["Artikelnr","Bezeichnung","Kategorie"], as_index=False)
        .agg(
            Einkaufsmenge = pd.NamedAgg("Lagermenge",    "sum"),
            Einkaufswert  = pd.NamedAgg("Einkaufswert",  "sum"),
            Verkaufsmenge = pd.NamedAgg("Verkaufsmenge", "sum"),
            Verkaufswert  = pd.NamedAgg("Verkaufswert",  "sum"),
            Lagermenge    = pd.NamedAgg("Lagermenge",    "sum"),
            Lagerwert     = pd.NamedAgg("Lagerwert",     "sum"),
        )
    )

# ------------------------------ Hauptprogramm ------------------------------

def main():
    st.set_page_config(layout="wide")  # volle Breite ausnutzen
    st.title("📦 Galaxus Sell-out Aggregator")

    col1, col2 = st.columns(2)
    with col1:
        sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", key="sell", type="xlsx")
    with col2:
        price_file = st.file_uploader("Preisliste (.xlsx)",  key="price", type="xlsx")

    if not sell_file or not price_file:
        st.info("Bitte beide Dateien hochladen, um fortzufahren.")
        return

    sell_df, price_df = load_data(sell_file, price_file)

    # Matching & Anreicherung
    enriched = enrich(sell_df, price_df)

    # Aggregation
    agg = compute_agg(enriched)

    # Ergebnis anzeigen
    st.markdown("### Ergebnis-Tabelle")
    st.dataframe(agg, use_container_width=True)

if __name__ == "__main__":
    main()
