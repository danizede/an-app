import streamlit as st
import pandas as pd
from thefuzz import process

# ------------------------------------------------------------------------------
# Hilfsfunktion: flexible Spalten­suche
# ------------------------------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    """
    Durchsucht df.columns nach einer der Kandidaten (Groß-/Kleinschreibung ignored).
    Gibt den gefundenen Spaltennamen zurück oder wirft KeyError mit Klartext.
    """
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\n"
                   f"Verfügbare Spalten: {list(df.columns)}")

# ------------------------------------------------------------------------------
# Daten einlesen (Streamlit-Datei-Uploader + Cache)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="⏳ Laden der Excel-Dateien …")
def load_excel(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return None
    return pd.read_excel(uploaded)

# ------------------------------------------------------------------------------
# Matching & Anreicherung
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="🔗 Matching & Anreicherung …")
def enrich(sell: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    # 1) Spaltennamen finden
    col_nr    = find_column(price, ["Artikelnr", "Hersteller-Nr.", "ArtNr"],      "Artikelnr")
    col_ean   = find_column(price, ["EAN", "GTIN"],                                "EAN/GTIN")
    col_bez   = find_column(price, ["Bezeichnung", "Bez", "Name"],                 "Bezeichnung")
    col_cat   = find_column(price, ["Kategorie", "Zusatz", "Warengruppe"],         "Kategorie")
    col_price = find_column(price, ["Preis", "EK", "Netto Netto", "NETTO NETTO"],  "Preis")
    col_qty   = find_column(sell,  ["Verkauf", "Menge", "Stück"],                  "Verkauf")
    col_nr_s  = find_column(sell,  ["Hersteller-Nr.", "Artikelnr", "ArtNr"],        "Artikelnr (Sell-Report)")
    col_ean_s = find_column(sell,  ["EAN", "GTIN"],                                "EAN/GTIN (Sell-Report)")

    # 2) Grund-Join auf Artikelnr.
    merged = sell.merge(
        price[[col_nr, col_bez, col_cat, col_price]],
        left_on=col_nr_s, right_on=col_nr, how="left", suffixes=("", "_pr1")
    )

    # 3) Falls kein Preis über Artikelnr, per EAN nachtragen
    mask_ean = merged[col_price].isna() & merged[col_ean_s].notna()
    if mask_ean.any():
        df2 = (
            merged[mask_ean]
            .merge(
                price[[col_ean, col_bez, col_cat, col_price]],
                left_on=col_ean_s, right_on=col_ean, how="left", suffixes=("", "_pr2")
            )
        )
        for c in [col_bez, col_cat, col_price]:
            merged.loc[mask_ean, c] = df2[c].values

    # 4) (Optional) Fuzzy-Matching auf Bezeichnung, wenn Preis weiterhin fehlt
    #    Hier ein sehr einfaches Beispiel, kann bei Bedarf ausgebaut werden:
    mask_fuzzy = merged[col_price].isna()
    if mask_fuzzy.any():
        # Index aller PL-Bezeichnungen
        choices = price[col_bez].dropna().unique().tolist()
        def _match(name):
            match, score = process.extractOne(str(name), choices)
            return match if score >= 80 else None

        merged.loc[mask_fuzzy, col_bez] = merged.loc[mask_fuzzy, col_bez].apply(_match)
        # Preis & Kategorie aus PL ziehen
        df3 = merged[mask_fuzzy].merge(
            price[[col_bez, col_cat, col_price]],
            on=col_bez, how="left", suffixes=("", "_pr3")
        )
        for c in [col_cat, col_price]:
            merged.loc[mask_fuzzy, c] = df3[c].values

    # 5) Wenn danach noch NaN im Preis, setzen wir 0
    merged[col_price] = merged[col_price].fillna(0)

    return merged

# ------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="🔢 Aggregation …")
def compute_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Gruppiert nach Artikel-Nummer, Bezeichnung, Kategorie, Preis.
    Liefert Detail-Tabelle und ein Dict mit den Gesamtsummen.
    """
    col_nr    = df.filter(regex="Artikeln?r?").columns[0]
    col_bez   = df.filter(regex="Bezeichnung|Bez|Name").columns[0]
    col_cat   = df.filter(regex="Kategorie|Zusatz|Warengruppe").columns[0]
    col_price = df.filter(regex="Preis|EK|Netto").columns[0]
    col_qty   = df.filter(regex="Verkauf|Menge|Stück").columns[0]
    col_stock = df.filter(regex="Bestand|Lagerbestand").columns[0] if df.filter(regex="Bestand|Lagerbestand").any().any() else None

    agg = df.groupby([col_nr, col_bez, col_cat, col_price], dropna=False).agg(
        Verkauf      = (col_qty, "sum"),
        Verkaufswert = (col_qty, lambda s: (s * df.loc[s.index, col_price]).sum()),
        Lagerbestand = (col_stock, "sum") if col_stock else pd.NamedAgg(column=col_qty, aggfunc=lambda s: 0),
        Lagerwert    = (col_stock, lambda s: (s * df.loc[s.index, col_price]).sum()) if col_stock else pd.NamedAgg(column=col_price, aggfunc=lambda s: 0),
    ).reset_index()

    totals = {
        "VK": agg["Verkaufswert"].sum(),
        "LG": agg["Lagerwert"].sum(),
    }
    return agg, totals

# ------------------------------------------------------------------------------
# Streamlit-UI
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Galaxus Sell-out Aggregator", layout="wide")

st.title("📦 Galaxus Sell-out Aggregator")

# **1. Datei-Uploader**
sell_file  = st.file_uploader("Sell-out-Report (.xlsx)", type="xlsx")
price_file = st.file_uploader("Preisliste (.xlsx)",      type="xlsx")

if not sell_file or not price_file:
    st.info("Bitte Sell-out-Report und Preisliste hochladen, um zu starten.")
    st.stop()

# **2. Daten laden**
sell_df  = load_excel(sell_file)
price_df = load_excel(price_file)

# **3. Matching & Enrichment**
try:
    enriched = enrich(sell_df, price_df)
except KeyError as e:
    st.error(f"Spaltenfehler: {e}")
    st.stop()

# **4. Aggregation**
agg_tbl, totals = compute_agg(enriched)

# **5. Kennzahlen anzeigen**
c1, c2 = st.columns(2)
c1.metric("🔹 Gesamt Verkaufswert (CHF)", f"{totals['VK']:,.0f}")
c2.metric("🔹 Gesamt Lagerwert (CHF)",    f"{totals['LG']:,.0f}")

# **6. Detail-Tabelle**
st.dataframe(agg_tbl, use_container_width=True)
