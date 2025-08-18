# app.py — Galaxus Sellout Analyse
# Struktur: Einkauf • Verkauf • Lager (Lager = letzter Stand aus Sell-out Spalte G je Artikel, bestimmt über neueste Periode I/J)
# - Robustes Einlesen
# - Matching per ArtikelNr_key (ArtNr normalisiert)
# - Zeitfilter wirkt nur auf Umsätze (nicht auf letzten Lagerstand)
# - Overflow-Schutz (Clipping/NaN-Handling)
# - Wochen-Chart mit Hover-Highlight
# - Kontrolltabelle: zeigt, aus welcher G/I/J-Zeile der letzte Lagerstand stammt

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Galaxus Sellout Analyse", layout="wide")

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

# =========================
# Konstanten & Anzeige
# =========================
THOUSANDS_SEP = "'"
MAX_QTY, MAX_PRICE = 1_000_000, 1_000_000

NUM_COLS_DEFAULT = [
    "Einkaufsmenge","Einkaufswert",
    "Verkaufsmenge","Verkaufswert",
    "Lagermenge","Lagerwert"
]

def _fmt_thousands(x, sep=THOUSANDS_SEP):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}".replace(",", sep)
    except Exception: return str(x)

def style_numeric(df: pd.DataFrame, num_cols=NUM_COLS_DEFAULT, sep=THOUSANDS_SEP):
    out = df.copy()
    present = [c for c in num_cols if c in out.columns]
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    fmt = {c: (lambda v, s=sep: _fmt_thousands(v, s)) for c in present}
    return out, out.style.format(fmt)

# =========================
# Einlesen (robust gegen Kopfzeilen)
# =========================
def read_excel_flat(upload) -> pd.DataFrame:
    raw = pd.read_excel(upload, header=None, dtype=object)
    if raw.empty: return pd.DataFrame()
    header_idx = int(raw.notna().mean(axis=1).idxmax())
    headers = raw.iloc[header_idx].fillna("").astype(str).tolist()
    headers = [re.sub(r"\s+"," ",h).strip() for h in headers]
    n = raw.shape[1]
    if len(headers) < n:
        headers += [f"col_{i}" for i in range(len(headers), n)]
    else:
        headers = headers[:n]
    df = raw.iloc[header_idx+1:].reset_index(drop=True)
    df.columns = headers
    # doppelte Spalten entschärfen
    seen, newcols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c]+=1; newcols.append(f"{c}.{seen[c]}")
        else:
            seen[c]=0; newcols.append(c)
    df.columns = newcols
    return df

# =========================
# Utilities
# =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
                  .map(lambda c: unicodedata.normalize("NFKC", str(c)))
                  .map(lambda c: re.sub(r"\s+"," ", c).strip()))
    return df

def normalize_key(s: str) -> str:
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    return re.sub(r"[^a-z0-9]+","", s)

def find_column(df: pd.DataFrame, candidates, purpose: str, required=True) -> str|None:
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols: return cand
    canon = {re.sub(r"[\s\-_/\.]+","", c).lower(): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-_/\.]+","", cand).lower()
        if key in canon: return canon[key]
    if required:
        raise KeyError(f"Spalte für «{purpose}» fehlt – gesucht unter {candidates}.\nVerfügbare Spalten: {cols}")
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in ("i","u","f"): return s
    def _clean(x):
        if pd.isna(x): return np.nan
        x=str(x).strip().replace("’","").replace("'","").replace(" ","").replace(",",".")
        if x.count(".")>1:
            parts=x.split("."); x="".join(parts[:-1])+"."+parts[-1]
        try: return float(x)
        except Exception: return np.nan
    return s.map(_clean)

def parse_date_series_us(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64): return s
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
    nums = pd.to_numeric(s, errors="coerce")
    dt2 = pd.to_datetime(nums, origin="1899-12-30", unit="d", errors="coerce")
    return dt1.combine_first(dt2)

def sanitize_numbers(qty: pd.Series, price: pd.Series) -> tuple[pd.Series,pd.Series]:
    q = pd.to_numeric(qty, errors="coerce").astype("float64").clip(lower=0, upper=MAX_QTY)
    p = pd.to_numeric(price, errors="coerce").astype("float64").clip(lower=0, upper=MAX_PRICE)
    return q, p

# =========================
# Feld-Kandidaten
# =========================
ARTNR_CANDIDATES = ["Artikelnummer","Artikelnr","ArtikelNr","Artikel-Nr.","Hersteller-Nr.","Produkt ID","ProdNr","ArtNr","ArtikelNr.","Artikel"]
EAN_CANDIDATES  = ["EAN","GTIN","BarCode","Barcode"]
NAME_CANDIDATES_PL = ["Bezeichnung","Produktname","Name","Titel","Artikelname"]
CAT_CANDIDATES  = ["Kategorie","Warengruppe","Zusatz"]

NAME_CANDIDATES_SO   = ["Bezeichnung","Name","Artikelname","Bezeichnung_Sales","Produktname"]
SALES_QTY_CANDIDATES = ["SalesQty","Verkauf","Verkaufte Menge","Menge verkauft","Absatz","Stück","Menge"]
BUY_QTY_CANDIDATES   = ["Einkauf","Einkaufsmenge","Menge Einkauf"]

DATE_START_CANDS     = ["Start","Startdatum","Start Date","Anfangs datum","Anfangsdatum","Von","Period Start"]
DATE_END_CANDS       = ["Ende","Enddatum","End Date","Bis","Period End"]

# WICHTIG: Lagermenge im Sell-out (Spalte G)
STOCK_SO_CANDIDATES  = ["Lagermenge","Lagerbestand","Bestand"]

# Preise Preisliste
BUY_PRICE_CANDIDATES  = ["Einkaufspreis","Einkauf"]
SELL_PRICE_CANDIDATES = ["Verkaufspreis","VK","Preis"]
PRICE_COL_CANDIDATES  = ["Preis","VK","Netto","NETTO","NETTO NETTO","Einkaufspreis","Verkaufspreis","Einkauf"]

# =========================
# Parsing – Preislisten
# =========================
def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES, "Artikelnummer")
    col_ean   = find_column(df, EAN_CANDIDATES,  "EAN/GTIN", required=False)
    col_name  = find_column(df, NAME_CANDIDATES_PL, "Bezeichnung")
    col_cat   = find_column(df, CAT_CANDIDATES,  "Kategorie", required=False)
    col_buy   = find_column(df, BUY_PRICE_CANDIDATES,  "Einkaufspreis", required=False)
    col_sell  = find_column(df, SELL_PRICE_CANDIDATES, "Verkaufspreis", required=False)
    col_any=None
    if not col_sell and not col_buy:
        col_any = find_column(df, PRICE_COL_CANDIDATES, "Preis", required=True)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str)
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","",str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str)
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)
    out["Kategorie"]       = df[col_cat].astype(str) if col_cat else ""

    if col_buy:  out["Einkaufspreis"] = parse_number_series(df[col_buy])
    if col_sell: out["Verkaufspreis"] = parse_number_series(df[col_sell])
    if not col_buy and not col_sell and col_any:
        anyp = parse_number_series(df[col_any]); out["Einkaufspreis"]=anyp; out["Verkaufspreis"]=anyp
    if "Einkaufspreis" not in out: out["Einkaufspreis"]=out.get("Verkaufspreis", pd.Series([np.nan]*len(out)))
    if "Verkaufspreis" not in out: out["Verkaufspreis"]=out.get("Einkaufspreis", pd.Series([np.nan]*len(out)))

    out = out.assign(_have=out["Verkaufspreis"].notna()).sort_values(["ArtikelNr_key","_have"], ascending=[True,False])
    out = out.drop_duplicates(subset=["ArtikelNr_key"], keep="first").drop(columns=["_have"])
    return out

# =========================
# Parsing – Sell-out (G/I/J)
# =========================
def _fallback_col_by_index(df: pd.DataFrame, idx0: int) -> str|None:
    try: return df.columns[idx0]
    except: return None

def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    col_art   = find_column(df, ARTNR_CANDIDATES,   "Artikelnummer", required=False)
    col_ean   = find_column(df, EAN_CANDIDATES,     "EAN/GTIN",      required=False)
    col_name  = find_column(df, NAME_CANDIDATES_SO, "Bezeichnung",   required=False)
    col_sales = find_column(df, SALES_QTY_CANDIDATES, "Verkaufsmenge", required=True)
    col_buy   = find_column(df, BUY_QTY_CANDIDATES,   "Einkaufsmenge", required=False)

    # Spalte G
    col_stock_so = find_column(df, STOCK_SO_CANDIDATES, "Lagermenge (Sell-out: Spalte G)", required=False)
    if not col_stock_so and df.shape[1] >= 7:
        col_stock_so = _fallback_col_by_index(df, 6)

    # I/J
    col_start = find_column(df, DATE_START_CANDS, "Startdatum (Spalte I)", required=False)
    col_end   = find_column(df, DATE_END_CANDS,   "Enddatum (Spalte J)",   required=False)
    if not col_start and df.shape[1] >= 9:  col_start = _fallback_col_by_index(df, 8)
    if not col_end   and df.shape[1] >= 10: col_end   = _fallback_col_by_index(df, 9)

    out = pd.DataFrame()
    out["ArtikelNr"]       = df[col_art].astype(str) if col_art else ""
    out["ArtikelNr_key"]   = out["ArtikelNr"].map(normalize_key)
    out["EAN"]             = df[col_ean].astype(str) if col_ean else ""
    out["EAN_key"]         = out["EAN"].map(lambda x: re.sub(r"[^0-9]+","",str(x)))
    out["Bezeichnung"]     = df[col_name].astype(str) if col_name else ""
    out["Bezeichnung_key"] = out["Bezeichnung"].map(normalize_key)

    out["Verkaufsmenge"] = parse_number_series(df[col_sales]).fillna(0).astype("Int64")
    out["Einkaufsmenge"] = parse_number_series(df[col_buy]).fillna(0).astype("Int64") if col_buy else pd.Series([0]*len(df), dtype="Int64")

    # Lagermenge nur aus Sell-out (G) – clippen
    if col_stock_so:
        out["SellLagermenge"] = (
            parse_number_series(df[col_stock_so])
            .astype(float)
            .clip(lower=0, upper=MAX_QTY)
        )

    # Datumsperioden (I/J)
    if col_start: out["StartDatum"] = parse_date_series_us(df[col_start])
    if col_end:   out["EndDatum"]   = parse_date_series_us(df[col_end])
    if "StartDatum" in out and "EndDatum" in out:
        out.loc[out["EndDatum"].isna(),"EndDatum"] = out.loc[out["EndDatum"].isna(),"StartDatum"]
    return out

# =========================
# Merge & Werte
# =========================
@st.cache_data(show_spinner=False)
def enrich_and_merge(filtered_sell_df: pd.DataFrame, price_df: pd.DataFrame, latest_stock_baseline_df: pd.DataFrame|None=None):
    """
    - Lager (Menge/Wert): ausschliesslich aus Sell-out G, je Artikel jüngste I/J-Zeile aus latest_stock_baseline_df (ungefiltert)
    - Umsätze/Einkäufe: aus filtered_sell_df mit Preisen aus Preisliste
    """
    sell_for_stock = latest_stock_baseline_df if latest_stock_baseline_df is not None else filtered_sell_df

    # Merge für Umsätze/Einkäufe
    merged = filtered_sell_df.merge(price_df[["ArtikelNr_key","ArtikelNr","Bezeichnung","Kategorie","Einkaufspreis","Verkaufspreis"]],
                                    on="ArtikelNr_key", how="left", suffixes=("", "_pl"))

    # Umsatz/Einkauf (overflow-sicher)
    q_buy,p_buy   = sanitize_numbers(merged.get("Einkaufsmenge",0), merged.get("Einkaufspreis",np.nan))
    q_sell,p_sell = sanitize_numbers(merged.get("Verkaufsmenge",0), merged.get("Verkaufspreis",np.nan))
    with np.errstate(over="ignore", invalid="ignore"):
        merged["Einkaufswert"] = (q_buy.fillna(0)*p_buy.fillna(0)).astype("float64")
        merged["Verkaufswert"] = (q_sell.fillna(0)*p_sell.fillna(0)).astype("float64")
    merged["Einkaufswert"].mask(~np.isfinite(merged["Einkaufswert"]), 0.0, inplace=True)
    merged["Verkaufswert"].mask(~np.isfinite(merged["Verkaufswert"]), 0.0, inplace=True)

    # === Letzter Lagerstand je Artikel (aus ungefiltertem Sell-out) ===
    stock_df = sell_for_stock.copy()

    def _row_date(df):
        if ("EndDatum" in df.columns) and ("StartDatum" in df.columns):
            d = df["EndDatum"].fillna(df["StartDatum"])
        elif "StartDatum" in df.columns:
            d = df["StartDatum"]
        elif "EndDatum" in df.columns:
            d = df["EndDatum"]
        else:
            d = pd.NaT
        return pd.to_datetime(d, errors="coerce")

    stock_df["_rowdate"] = _row_date(stock_df)

    if "SellLagermenge" in stock_df.columns:
        stock_df = stock_df[~stock_df["SellLagermenge"].isna()].copy()
        stock_df["SellLagermenge"] = pd.to_numeric(stock_df["SellLagermenge"], errors="coerce").clip(0, MAX_QTY)
    else:
        stock_df = stock_df.iloc[0:0].copy()
        stock_df["SellLagermenge"] = np.nan

    stock_df = stock_df.sort_values(["ArtikelNr_key","_rowdate"], ascending=[True, True])
    last_rows = stock_df.groupby("ArtikelNr_key", as_index=False).tail(1)

    latest_qty_map  = last_rows.set_index("ArtikelNr_key")["SellLagermenge"].to_dict()
    price_map       = pd.to_numeric(price_df.drop_duplicates("ArtikelNr_key").set_index("ArtikelNr_key")["Verkaufspreis"], errors="coerce").clip(0, MAX_PRICE).to_dict()

    # auf merged projizieren
    merged["Lagermenge_latest"]    = pd.to_numeric(merged["ArtikelNr_key"].map(latest_qty_map), errors="coerce").fillna(0.0).astype("float64").clip(0, MAX_QTY)
    merged["Verkaufspreis_latest"] = pd.to_numeric(merged["ArtikelNr_key"].map(price_map), errors="coerce").fillna(0.0).astype("float64").clip(0, MAX_PRICE)
    with np.errstate(over="ignore", invalid="ignore"):
        merged["Lagerwert_latest"] = (merged["Lagermenge_latest"] * merged["Verkaufspreis_latest"]).astype("float64")
    merged["Lagerwert_latest"].mask(~np.isfinite(merged["Lagerwert_latest"]), 0.0, inplace=True)

    # ---- Tabellenaufbau (alte Struktur) ----
    # Detail (Zeilenebene, aber Lager = letzter Stand je Artikel)
    detail = merged[["ArtikelNr","Bezeichnung","Kategorie","Einkaufsmenge","Einkaufswert","Verkaufsmenge","Verkaufswert"]].copy()
    detail.rename(columns={"Bezeichnung":"Bezeichnung_anzeige"}, inplace=True)
    detail["Lagermenge"] = merged["Lagermenge_latest"]
    detail["Lagerwert"]  = merged["Lagerwert_latest"]

    # Summen pro Artikel – Einkauf/Verkauf summieren, Lager = letzter Stand (nicht summieren)
    summary = (detail.groupby(["ArtikelNr","Bezeichnung_anzeige","Kategorie"], as_index=False)
               .agg({
                   "Einkaufsmenge":"sum",
                   "Einkaufswert":"sum",
                   "Verkaufsmenge":"sum",
                   "Verkaufswert":"sum",
                   "Lagermenge":"max",
                   "Lagerwert":"max",
               }))

    # Kontrolltabelle: aus welcher G/I/J-Zeile stammt der Lagerstand?
    latest_stock_view = (last_rows[["ArtikelNr_key","ArtikelNr","Bezeichnung","SellLagermenge","StartDatum","EndDatum"]]
                         .rename(columns={"SellLagermenge":"Lagermenge (G)","StartDatum":"Start (I)","EndDatum":"Ende (J)"}))
    latest_stock_view["Preis (PL)"] = latest_stock_view["ArtikelNr_key"].map(price_map).astype(float)
    latest_stock_view["Lagerwert"]  = latest_stock_view["Lagermenge (G)"].astype(float) * latest_stock_view["Preis (PL)"].astype(float)

    # Zeitquelle fürs Chart
    ts_source = pd.DataFrame()
    if "StartDatum" in merged.columns:
        ts_source = merged[["StartDatum","Kategorie","Verkaufswert"]].copy()
        ts_source["Kategorie"] = ts_source["Kategorie"].fillna("— ohne Kategorie —").replace({"":"— ohne Kategorie —"})

    return detail, summary, ts_source, latest_stock_view

# =========================
# UI
# =========================
st.title("📊 Galaxus Sellout Analyse (Einkauf • Verkauf • Lager)")
st.caption(
    "Lager basiert ausschliesslich auf Sell-out Spalte G; letzter Stand je Artikel per jüngster Periode I/J. "
    "Zeitfilter wirkt nur auf Umsätze/Einkäufe."
)

c1,c2 = st.columns(2)
with c1:
    st.subheader("Sell-out-Report (.xlsx)")
    sell_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="sell")
    if "sell_last" in st.session_state and st.session_state["sell_last"]:
        st.text(f"Letzter Sell-out: {st.session_state['sell_last']['name']}")
with c2:
    st.subheader("Preisliste (.xlsx)")
    price_file = st.file_uploader("Drag & drop oder Datei wählen", type=["xlsx"], key="price")
    if "price_last" in st.session_state and st.session_state["price_last"]:
        st.text(f"Letzte Preisliste: {st.session_state['price_last']['name']}")

if sell_file and price_file:
    try:
        st.session_state["sell_last"]  = {"name": sell_file.name}
        st.session_state["price_last"] = {"name": price_file.name}
        raw_sell  = read_excel_flat(sell_file)
        raw_price = read_excel_flat(price_file)

        with st.spinner("📖 Lese & prüfe Spalten…"):
            sell_df  = prepare_sell_df(raw_sell)
            price_df = prepare_price_df(raw_price)

        # ===== Zeitraumfilter (beeinflusst NUR Umsätze/Einkäufe) =====
        filtered_sell_df = sell_df
        if {"StartDatum","EndDatum"}.issubset(sell_df.columns) and not sell_df["StartDatum"].isna().all():
            st.subheader("Periode wählen")
            min_date = sell_df["StartDatum"].min().date()
            max_date = (sell_df["EndDatum"].dropna().max()
                        if "EndDatum" in sell_df else sell_df["StartDatum"].max()).date()

            if "date_range" not in st.session_state:
                st.session_state["date_range"] = (min_date, max_date)

            col_range, col_btn = st.columns([3,1])
            with col_range:
                date_value = st.date_input(
                    "Zeitraum (DD.MM.YYYY)",
                    value=st.session_state["date_range"],
                    min_value=min_date,
                    max_value=max_date,
                    format="DD.MM.YYYY",
                )
            with col_btn:
                st.write(""); st.write("")
                if st.button("Gesamten Zeitraum"):
                    st.session_state["date_range"] = (min_date, max_date)
                    st.experimental_rerun()

            start_date, end_date = (date_value if isinstance(date_value, tuple) else (date_value, date_value))
            st.session_state["date_range"] = (start_date, end_date)

            mask = ~((sell_df["EndDatum"].dt.date < start_date) |
                     (sell_df["StartDatum"].dt.date > end_date))
            filtered_sell_df = sell_df.loc[mask].copy()
        # =============================================================

        with st.spinner("🔗 Matche & berechne Werte…"):
            detail, summary, ts_source, latest_stock_view = enrich_and_merge(
                filtered_sell_df, price_df, latest_stock_baseline_df=sell_df
            )

        # ===== Chart: Wochen-Verlauf Verkauf =====
        st.markdown("### 📈 Verkaufsverlauf nach Kategorie (Woche)")
        if not ts_source.empty:
            ts = ts_source.dropna(subset=["StartDatum"]).copy()
            ts["Periode"] = ts["StartDatum"].dt.to_period("W").dt.start_time
            ts["Kategorie"] = ts["Kategorie"].astype("string").fillna("— ohne Kategorie —").replace({"":"— ohne Kategorie —"})

            all_cats = sorted(ts["Kategorie"].unique())
            sel_cats = st.multiselect("Kategorien filtern", options=all_cats, default=all_cats)
            if sel_cats:
                ts = ts[ts["Kategorie"].isin(sel_cats)]

            ts_agg = (ts.groupby(["Kategorie","Periode"], as_index=False)["Verkaufswert"]
                        .sum().rename(columns={"Verkaufswert":"Wert"}))
            ts_agg["Periode"] = pd.to_datetime(ts_agg["Periode"])
            ts_agg["Wert"] = pd.to_numeric(ts_agg["Wert"], errors="coerce").fillna(0.0)

            hover_cat = alt.selection_single(fields=["Kategorie"], on="mouseover", nearest=True, empty="none")
            hover_pt  = alt.selection_single(fields=["Periode","Kategorie"], on="mouseover", nearest=True, empty="none")

            base = alt.Chart(ts_agg)
            lines = (
                base.mark_line(point=alt.OverlayMarkDef(size=28), interpolate="linear")
                    .encode(
                        x=alt.X("Periode:T", title="Woche"),
                        y=alt.Y("Wert:Q", title="Verkaufswert (Summe pro Woche)", stack=None),
                        color=alt.Color("Kategorie:N", title="Kategorie"),
                        opacity=alt.condition(hover_cat, alt.value(1.0), alt.value(0.25)),
                        strokeWidth=alt.condition(hover_cat, alt.value(3), alt.value(1.5)),
                        tooltip=[alt.Tooltip("Periode:T", title="Woche"),
                                 alt.Tooltip("Kategorie:N", title="Kategorie"),
                                 alt.Tooltip("Wert:Q", title="Verkaufswert", format=",.0f")],
                    ).add_selection(hover_cat)
            )
            points = base.mark_point(size=70, opacity=0).encode(x="Periode:T", y="Wert:Q", color="Kategorie:N").add_selection(hover_pt)
            popup  = base.transform_filter(hover_pt).mark_text(align='left', dx=6, dy=-8, fontSize=12, fontWeight='bold')\
                        .encode(x="Periode:T", y="Wert:Q", text="Kategorie:N", color="Kategorie:N")
            endlab = (base.transform_window(row_number='row_number()', sort=[alt.SortField(field='Periode', order='descending')], groupby=['Kategorie'])
                          .transform_filter(alt.datum.row_number == 0)
                          .mark_text(align='left', dx=6, dy=-6, fontSize=11)
                          .encode(x='Periode:T', y='Wert:Q', text='Kategorie:N', color='Kategorie:N',
                                  opacity=alt.condition(hover_cat, alt.value(1.0), alt.value(0.6))))
            st.altair_chart((lines + points + popup + endlab).properties(height=380), use_container_width=True)
        else:
            st.info("Für den Verlauf werden gültige Startdaten benötigt.")

        # ===== Kontrolltabelle =====
        st.markdown("#### 🔍 Kontrolle: herangezogener letzter Lagerstand (aus Sell-out)")
        st.dataframe(latest_stock_view, use_container_width=True)
        st.download_button(
            "⬇️ Kontrolle (CSV)",
            data=latest_stock_view.to_csv(index=False).encode("utf-8"),
            file_name="kontrolle_letzter_lagerstand.csv",
            mime="text/csv"
        )

        # ===== Tabellen (alte Struktur) =====
        show_detail = st.checkbox("Detailtabelle anzeigen", value=False)
        if show_detail:
            st.subheader("Detailtabelle (Zeilen)")
            d_rounded, d_styler = style_numeric(detail)
            st.dataframe(d_styler, use_container_width=True)

        st.subheader("Summen pro Artikel – Einkauf • Verkauf • Lager (Lager = letzter Stand)")
        s_rounded, s_styler = style_numeric(summary)
        st.dataframe(s_styler, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Detail (CSV)",
                data=(detail if show_detail else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
                file_name="detail.csv", mime="text/csv", disabled=not show_detail
            )
        with dl2:
            st.download_button(
                "⬇️ Summen (CSV)",
                data=s_rounded.to_csv(index=False).encode("utf-8"),
                file_name="summen.csv", mime="text/csv"
            )

    except KeyError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
else:
    st.info("Bitte beide Dateien hochladen (Sell-out & Preisliste).")
