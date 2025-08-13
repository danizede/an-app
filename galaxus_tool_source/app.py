"""
Galaxus Sell‑out Aggregator
===========================

Dieses Script implementiert einen Streamlit basierten
Sell‑out Aggregator für Galaxus. Es liest wöchentlich
gelieferte Verkaufsreports sowie Preislisten ein, normalisiert
die Spalten, führt einen robusten Match der Produkte durch und
berechnet Einkaufs‑, Verkaufs‑ und Lagerwerte. Die Darstellung
erfolgt in tabellarischer Form mit gerundeten Zahlen und
Tausendertrennzeichen. Zusätzlich kann eine Periode über
Datumsfilter ausgewählt werden, wodurch sich die Darstellung
der Verkaufs‑ und Einkaufszahlen dynamisch anpasst.

Wichtige Anpassungen gegenüber der Ursprungsversion
--------------------------------------------------

* **Bugfix**: In der ursprünglichen Funktion `style_numeric` wurde
  innerhalb eines Generator‑Ausdrucks fälschlicherweise die
  Variable `c` in der Filterbedingung verwendet. Da sich `c` erst im
  übergeordneten Kontext definiert, führte dies zu einem
  ``NameError`` bzw. ``cannot access free variable 'c'``. Der
  Generator filtert jetzt korrekt anhand der Variablen `col`.

* **Datumsfilter**: Verkaufsdaten können jetzt optional nach einer
  ausgewählten Periode gefiltert werden. Wird eine Datums-Spalte
  gefunden, so liest der Nutzer mittels `st.date_input` ein
  Start‑ und Enddatum ein. Nur Daten innerhalb dieses
  Zeitraums werden anschließend für die Berechnungen
  herangezogen. Die aktuelle Lagermenge aus der Preisliste
  bleibt davon unberührt.

"""

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Anzeige-Helfer (Runden + Tausender)
# =========================
THOUSANDS_SEP = "'"
NUM_COLS_DEFAULT = [
    "Einkaufsmenge", "Einkaufswert",
    "Verkaufsmenge", "Verkaufswert",
    "Lagermenge",   "Lagerwert",
]

def _fmt_thousands(x, sep=THOUSANDS_SEP):
    """Rundet Zahlen und fügt Tausendertrennzeichen ein."""
    if pd.isna(x):
        return ""
    try:
        return f"{int(round(float(x))):,}".replace(",", sep)
    except Exception:
        # Fallback: gebe die Zeichenkette unverändert zurück
        return str(x)

def style_numeric(df: pd.DataFrame, num_cols=NUM_COLS_DEFAULT, sep=THOUSANDS_SEP):
    """
    Formatiert numerische Spalten eines DataFrames: rundet auf ganze
    Zahlen, konvertiert zu Integer‑dtype und setzt ein geeignetes
    Format zur Anzeige von Tausenderpunkten. Die Funktion
    liefert sowohl den manipulierten DataFrame als auch einen
    Styler zurück.
    """
    out = df.copy()
    # Ursprünglicher Fehler: `c` wurde in der Filterbedingung des
    # Generator-Ausdrucks referenziert, bevor es definiert war. Die
    # folgende Zeile verwendet nun korrekt `col` in der Bedingung.
    for c in (col for col in num_cols if col in out.columns):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    # Für jede vorhandene numerische Spalte wird ein Formatierungs-
    # lambda zugewiesen. Dieses lambda nutzt den Tausendertrenner
    # über das Default‑Argument `s`, sodass es nicht erneut an den
    # Abschluss der Schleife gebunden wird.
    fmt = {c: (lambda v, s=sep: _fmt_thousands(v, s)) for c in num_cols if c in out.columns}
    styler = out.style.format(fmt)
    return out, styler

# =========================
# Utilities
# =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisiert Spaltennamen: entfernt diakritische Zeichen,
    vereinheitlicht Leerzeichen und trimmt führende und
    abschließende Leerzeichen.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .map(lambda c: unicodedata.normalize("NFKC", str(c)))
        .map(lambda c: re.sub(r"\s+", " ", c).strip())
    )
    return df

def normalize_key(s: str) -> str:
    """Erzeugt einen normalisierten Schlüssel aus Strings."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True) -> str | None:
    """
    Sucht in einem DataFrame `df` nach einer Spalte, die zu einer der
    angegebenen Varianten in `candidates` passt. Falls keine
    Übereinstimmung gefunden wird und `required=True` gesetzt ist,
    wird ein KeyError ausgelöst, andernfalls None zurückgegeben.
    """
    cols = list(df.columns)
    # 1. Direkte Übereinstimmung
    for cand in candidates:
        if cand in cols:
            return cand
    # 2. Normalisierte Suche (ohne Leer- und Sonderzeichen)
    canon = {re.sub(r"[\s\-_/]+", "", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-_/]+", "", cand).lower()
        if key in canon:
            return canon[key]
    if required:
        raise KeyError(
            f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}"
        )
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Konvertiert eine Spalte zu numerischen Werten. Erlaubt Formate mit
    Tausendertrennzeichen, Leerzeichen oder Komma als Dezimaltrenner.
    """
    if s.dtype.kind in ("i", "u", "f"):
        return s
    def _clean(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        # Ersetze verschiedene Tausendertrennzeichen
        x = x.replace("’", "").replace("'", "").replace(" ", "")
        # Ersetze deutsches Komma mit Punkt
        x = x.replace(",", ".")
        # Entferne zusätzliche Punkte (nur der letzte Punkt darf Dezimalpunkt sein)
        if x.count(".") > 1:
            parts = x.split(".")
            x = "".join(parts[:-1]) + "." + parts[-1]
        try:
            return float(x)
        except Exception:
            return np.nan
    return s.map(_clean)

def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Versucht, eine Spalte in ein Datumsformat zu konvertieren.
    """
    # Bereits vorhandene Datums-Typen unverändert lassen
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

# =========================
# Parsing – Preislisten
# =========================
PRICE_COL_CANDIDATES = [
    "Preis", "VK", "Netto", "NETTO", "Einkaufspreis", "Verkaufspreis", "NETTO NETTO", "Einkauf"
]
BUY_PRICE_CANDIDATES = ["Einkaufspreis", "Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis", "VK", "Preis"]

ARTNR_CANDIDATES = [
    "Artikelnummer", "Artikelnr", "ArtikelNr", "Artikel-Nr.",
    "Hersteller-Nr.", "Produkt ID", "ProdNr", "ArtNr", "ArtikelNr.", "Artikel"
]
EAN_CANDIDATES = ["EAN", "GTIN", "BarCode", "Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung", "Produktname", "Name", "Titel", "Artikelname"]
CAT_CANDIDATES = ["Kategorie", "Warengruppe", "Zusatz"]
STOCK_CANDIDATES = ["Bestand", "Verfügbar", "Lagerbestand"]

# Neu: mögliche Spalten, in denen eine Farbangabe enthalten sein könnte. Diese
# Liste wird verwendet, um Produkte mit identischer Bezeichnung, aber
# unterschiedlicher Farbvariante zu differenzieren.
COLOR_CANDIDATES = ["Farbe", "Color", "Colour", "Farben", "Zusatz"]

def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet die Preisliste für den Abgleich vor. Es werden relevante
    Spalten identifiziert, normalisiert und soweit möglich numerisch
    interpretiert. EAN, Bezeichnung und Artikelnummer werden als
    Schlüssel zur Identifikation verwendet.
    """
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES, "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat   = find_column(df, CAT_CANDIDATES, "Kategorie", required=False)
    col_stock = find_column(df, STOCK_CANDIDATES, "Bestand/Lager", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_color = find_column(df, COLOR_CANDIDATES, "Farbe", required=False)
    col_any   = None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]   = df[col_art].astype(str)
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)
    out["EAN"] = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"] = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    # Kategorie: NaN durch leere Zeichenkette ersetzen
    out["Kategorie"] = df[col_cat].fillna("").astype(str) if col_cat else ""
    # Farbe: Farbinformation extrahieren, falls vorhanden
    if col_color:
        out["Farbe"] = df[col_color].fillna("").astype(str)
        # Leerzeichen und 'nan' entfernen
        out["Farbe"] = out["Farbe"].replace(["nan", "None"], "")
    else:
        out["Farbe"] = ""
    # Lagerbestand: wenn vorhanden, numerisch interpretieren, sonst null
    if col_stock:
        out["Lagermenge"] = parse_number_series(df[col_stock]).fillna(0).astype("Int64")
    else:
        out["Lagermenge"] = pd.Series([0]*len(out), dtype="Int64")
    # Preise
    if col_buy:
        out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell:
        out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        price_any = parse_number_series(df[col_any])
        out["Einkaufspreis"] = price_any
        out["Verkaufspreis"] = price_any
    if "Einkaufspreis" not in out:
        out["Einkaufspreis"] = out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out:
        out["Verkaufspreis"] = out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))
    # Ableitung des angezeigten Produktnamens: Basis + Farbzusatz (falls vorhanden)
    def build_display(row):
        base = row["Bezeichnung"].strip()
        color = str(row["Farbe"]).strip()
        if color:
            return f"{base} ({color})"
        # Wenn keine Farbe vorhanden ist, gib den Basisnamen zurück
        return base
    out["DisplayName"] = out.apply(build_display, axis=1)
    return out

# =========================
# Parsing – Sell-out-Report
# =========================
NAME_CANDIDATES_SO = ["Bezeichnung", "Name", "Artikelname", "Bezeichnung_Sales", "Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty", "Verkauf", "Verkaufte Menge", "Menge verkauft", "Absatz", "Stück", "Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf", "Einkaufsmenge", "Menge Einkauf"]
DATE_CANDIDATES = ["Datum", "Date", "Verkaufsdatum", "Bestelldatum", "Belegdatum"]

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereitet den Sell‑out Report vor. Zusätzliche zum ursprünglichen
    Verhalten wird hier versucht, eine Datums-Spalte zu
    identifizieren und zu normalisieren. Die Mengenangaben werden
    numerisch interpretiert.
    """
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES, "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung", required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)
    col_date  = find_column(df, DATE_CANDIDATES, "Datum", required=False)

    out = pd.DataFrame()
    out["ArtikelNr"] = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"] = out["ArtikelNr"].map(normalize_key)
    out["EAN"] = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"] = out["EAN"].map(lambda x: re.sub(r"[^0-9]+", "", str(x)))
    out["Bezeichnung"] = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    if col_buy:
        out["Einkaufsmenge"] = parse_number_series(df[col_buy]).fillna(0).astype("Int64")
    else:
        out["Einkaufsmenge"] = pd.Series([0]*len(out), dtype="Int64")
    # Datumsspalte: falls vorhanden, nach Datentyp konvertieren
    if col_date:
        dates = parse_date_series(df[col_date])
        out["Datum"] = dates
    return out

# =========================
# Merge & Berechnung (Series- statt ndarray-Fallback)
# =========================
@st.cache_data(show_spinner=False)
def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Führt den eigentlichen Abgleich zwischen Verkaufsdaten und
    Preisliste durch. Es werden zunächst die Schlüssel
    ``ArtikelNr_key`` zusammengeführt. Für alle nicht zuordenbaren
    Datensätze wird ein Fallback via EAN oder normalisierte
    Bezeichnung ausgeführt. Danach werden Einkaufs‑, Verkaufs‑ und
    Lagerwerte berechnet und aggregiert.
    """
    merged = sell_df.merge(
        price_df,
        on=["ArtikelNr_key"],
        how="left",
        suffixes=("", "_pl"),
    )
    # --- Fallback via EAN ---
    mask_need = merged["Verkaufspreis"].isna() & merged["EAN_key"].astype(bool)
    if mask_need.any():
        tmp = merged.loc[mask_need, ["EAN_key"]].merge(
            price_df[[
                "EAN_key", "Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie",
                "ArtikelNr", "Farbe", "DisplayName"
            ]],
            on="EAN_key", how="left"
        )
        idx = merged.index[mask_need]
        tmp.index = idx  # Align
        for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr", "Farbe", "DisplayName"]:
            merged.loc[idx, col] = merged.loc[idx, col].fillna(tmp[col])
    # --- Fallback via normalisierte Bezeichnung ---
    mask_need = merged["Verkaufspreis"].isna()
    if mask_need.any():
        name_map = price_df.drop_duplicates("Bezeichnung_key").set_index("Bezeichnung_key")
        idx = merged.index[mask_need]
        keys = merged.loc[idx, "Bezeichnung_key"]
        for i, k in zip(idx, keys):
            if k in name_map.index:
                row = name_map.loc[k]
                for col in ["Einkaufspreis", "Verkaufspreis", "Lagermenge", "Bezeichnung", "Kategorie", "ArtikelNr", "Farbe", "DisplayName"]:
                    if pd.isna(merged.at[i, col]) or col in ("ArtikelNr", "Bezeichnung", "Kategorie", "Farbe", "DisplayName"):
                        merged.at[i, col] = row.get(col, merged.at[i, col])
    # sauber numerisch
    for pcol in ["Einkaufspreis", "Verkaufspreis"]:
        merged[pcol] = pd.to_numeric(merged[pcol], errors="coerce")
    # **Fix**: Keine Series.replace mit Series als Ersatz – nur leere NAs auffüllen
    merged["Kategorie"] = merged["Kategorie"].fillna("")
    merged["Bezeichnung"] = merged["Bezeichnung"].fillna("")
    # Werte berechnen
    merged["Einkaufswert"] = (merged["Einkaufsmenge"].astype("Int64").fillna(0) * merged["Einkaufspreis"].fillna(0)).astype(float)
    merged["Verkaufswert"] = (merged["Verkaufsmenge"].astype("Int64").fillna(0) * merged["Verkaufspreis"].fillna(0)).astype(float)
    merged["Lagerwert"]    = (merged["Lagermenge"].astype("Int64").fillna(0)    * merged["Verkaufspreis"].fillna(0)).astype(float)
    display_cols = [
        "ArtikelNr", "DisplayName", "Kategorie",
        "Einkaufsmenge", "Einkaufswert",
        "Verkaufsmenge", "Verkaufswert",
        "Lagermenge", "Lagerwert",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    detail = merged[display_cols].copy()
    # Aggregation: gruppiere nach Artikelnummer, DisplayName und Kategorie
    totals = (
        detail.groupby(["ArtikelNr", "DisplayName", "Kategorie"], dropna=False, as_index=False)
              .agg({
                    "Einkaufsmenge": "sum",
                    "Einkaufswert": "sum",
                    "Verkaufsmenge": "sum",
                    "Verkaufswert": "sum",
                    "Lagermenge": "sum",
                    "Lagerwert": "sum",
              })
    )
    return detail, totals

# =========================
# UI
# =========================
st.set_page_config(page_title="Galaxus Sell‑out Aggregator", layout="wide")
st.title("📦 Galaxus Sell‑out Aggregator")
st.caption(
    "Lädt Preislisten & Sell-out-Daten, matcht sie robust und berechnet Einkaufs-/Verkaufs-/Lagerwerte. "
    "Anzeige ist gerundet mit Tausendertrennzeichen."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="sell")
    if "sell_last" in st.session_state and st.session_state["sell_last"]:
        st.text(st.session_state["sell_last"]["name"])
with col2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="price")
    if "price_last" in st.session_state and st.session_state["price_last"]:
        st.text(st.session_state["price_last"]["name"])

if sell_file and price_file:
    try:
        st.session_state["sell_last"]  = {"name": sell_file.name}
        st.session_state["price_last"] = {"name": price_file.name}
        # Dateien lesen
        raw_sell  = pd.read_excel(sell_file)
        raw_price = pd.read_excel(price_file)
        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)
        # Periodenfilter: Auswahl nach Kalenderwoche (KW/Jahr) via Dropdown
        if "Datum" in sell_df.columns and not sell_df["Datum"].isna().all():
            iso = sell_df["Datum"].dt.isocalendar()
            sell_df["KW"] = iso["week"]
            sell_df["KW_Year"] = iso["year"]
            sell_df["Period"] = sell_df["KW"].astype(str) + "/" + sell_df["KW_Year"].astype(str)
            periods = sorted(sell_df["Period"].dropna().unique().tolist())
            options = ["Alle"] + periods
            st.subheader("Periode wählen")
            selected_period = st.selectbox(
                "Wähle eine Kalenderwoche (KW/Jahr) oder 'Alle'",
                options=options,
                index=0,
                help="Zeigt die verfügbaren Kalenderwochen basierend auf den Datumswerten des Sell-out Reports."
            )
            if selected_period == "Alle":
                filtered_sell_df = sell_df.copy()
            else:
                filtered_sell_df = sell_df[sell_df["Period"] == selected_period].copy()
        else:
            filtered_sell_df = sell_df
        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, totals = enrich_and_merge(filtered_sell_df, price_df)
        # Anzeige nur der Summen pro Artikel (aggregiert)
        st.subheader("Summen pro Artikel")
        t_rounded, t_styler = style_numeric(totals)
        st.dataframe(t_styler, use_container_width=True)
        # Downloadoption für Summen
        st.download_button(
            "⬇️ Summen (CSV)",
            data=t_rounded.to_csv(index=False).encode("utf-8"),
            file_name="summen.csv",
            mime="text/csv",
        )
    except KeyError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
else:
    st.info("Bitte beide Dateien hochladen (Sell-out & Preisliste).")
