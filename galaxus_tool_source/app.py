import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Galaxus Sellout Analyse", layout="wide")
st.title("📊 Galaxus Sellout Analyse")

MAX_QTY = 1e9
MAX_PRICE = 1e9

# =========================
# Hilfsfunktionen
# =========================
def prepare_sell_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sell-out Report vorbereiten: Verkaufs-, Einkaufs- und Lagermengen erfassen"""
    df = df.copy()

    # Verkaufsmenge (Spalte H / Index 7)
    if "Verkaufsmenge" in df.columns:
        df["SalesQty"] = pd.to_numeric(df["Verkaufsmenge"], errors="coerce")
    elif df.shape[1] > 7:
        df["SalesQty"] = pd.to_numeric(df.iloc[:, 7], errors="coerce")
    else:
        df["SalesQty"] = 0

    # Einkaufsmenge (Spalte F / Index 5)
    if "Einkaufsmenge" in df.columns:
        df["PurchaseQty"] = pd.to_numeric(df["Einkaufsmenge"], errors="coerce")
    elif df.shape[1] > 5:
        df["PurchaseQty"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    else:
        df["PurchaseQty"] = 0

    # Lagermenge (Spalte G / Index 6)
    if any(x in df.columns for x in ["Lagermenge", "Lagerbestand", "Bestand"]):
        col = [x for x in df.columns if x in ["Lagermenge", "Lagerbestand", "Bestand"]][0]
        df["SellLagermenge"] = pd.to_numeric(df[col], errors="coerce")
    elif df.shape[1] > 6:
        df["SellLagermenge"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
    else:
        df["SellLagermenge"] = np.nan

    # Datums-Infos (Spalte I/J = Start/Enddatum)
    if "EndDatum" in df.columns:
        df["_rowdate"] = pd.to_datetime(df["EndDatum"], errors="coerce")
    elif df.shape[1] > 9:
        df["_rowdate"] = pd.to_datetime(df.iloc[:, 9], errors="coerce")
    elif "StartDatum" in df.columns:
        df["_rowdate"] = pd.to_datetime(df["StartDatum"], errors="coerce")
    elif df.shape[1] > 8:
        df["_rowdate"] = pd.to_datetime(df.iloc[:, 8], errors="coerce")
    else:
        df["_rowdate"] = pd.NaT

    return df


def enrich_and_merge(sell_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Merge: Verkäufe, Einkäufe und letzter Lagerstand"""
    merged = sell_df.merge(price_df, on="ArtikelNr", how="left")

    # Preise
    merged["VKPreis"] = pd.to_numeric(merged["VKPreis"], errors="coerce").fillna(0)
    merged["EKPreis"] = pd.to_numeric(merged["EKPreis"], errors="coerce").fillna(0)

    # Werte
    merged["SalesValue"] = merged["SalesQty"] * merged["VKPreis"]
    merged["PurchaseValue"] = merged["PurchaseQty"] * merged["EKPreis"]

    # ---------- Bestimmung "letzter Lagerstand" ----------
    stock_valid = merged.loc[~merged["SellLagermenge"].isna()].copy()
    if not stock_valid.empty:
        stock_valid = stock_valid.sort_values(["ArtikelNr", "_rowdate"])
        last_rows = stock_valid.groupby("ArtikelNr", as_index=False).tail(1)
        latest_qty_map = last_rows.set_index("ArtikelNr")["SellLagermenge"].to_dict()
    else:
        latest_qty_map = {}

    price_map = (
        pd.to_numeric(price_df.drop_duplicates("ArtikelNr").set_index("ArtikelNr")["VKPreis"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0, upper=MAX_PRICE)
        .to_dict()
    )

    merged["Lagermenge_latest"] = (
        pd.to_numeric(merged["ArtikelNr"].map(latest_qty_map), errors="coerce")
        .fillna(0.0).clip(lower=0, upper=MAX_QTY).astype("float64")
    )
    merged["Verkaufspreis_latest"] = (
        pd.to_numeric(merged["ArtikelNr"].map(price_map), errors="coerce")
        .fillna(merged["VKPreis"])
        .fillna(0.0).clip(lower=0, upper=MAX_PRICE).astype("float64")
    )

    merged["Lagerwert_latest"] = (merged["Lagermenge_latest"] * merged["Verkaufspreis_latest"]).astype("float64")

    # Aggregation pro Artikel
    summary = (
        merged.groupby(["ArtikelNr", "Produktname", "Kategorie"], as_index=False)
        .agg({
            "SalesQty": "sum",
            "SalesValue": "sum",
            "PurchaseQty": "sum",
            "PurchaseValue": "sum",
            "Lagermenge_latest": "max",
            "Lagerwert_latest": "max",
        })
    )

    summary.rename(columns={
        "SalesQty": "Verkäufe Menge",
        "SalesValue": "Verkäufe Wert",
        "PurchaseQty": "Einkäufe Menge",
        "PurchaseValue": "Einkäufe Wert",
        "Lagermenge_latest": "Lager Menge",
        "Lagerwert_latest": "Lager Wert"
    }, inplace=True)

    return summary


# =========================
# Demo-Daten (Platzhalter)
# =========================
sell_data = {
    "ArtikelNr": ["1001", "1001", "1002", "1002"],
    "Produktname": ["Albert little", "Albert little", "Roger big", "Roger big"],
    "Kategorie": ["Luftentfeuchter", "Luftentfeuchter", "Luftreiniger", "Luftreiniger"],
    "Verkaufsmenge": [5, 3, 2, 4],
    "Einkaufsmenge": [2, 1, 1, 2],
    "Lagermenge": [10, 7, 15, 12],   # Spalte G
    "EndDatum": ["2025-03-01", "2025-05-01", "2025-03-01", "2025-05-01"]
}
price_data = {
    "ArtikelNr": ["1001", "1002"],
    "Produktname": ["Albert little", "Roger big"],
    "Kategorie": ["Luftentfeuchter", "Luftreiniger"],
    "VKPreis": [299, 499],
    "EKPreis": [199, 349],
}

sell_df = prepare_sell_df(pd.DataFrame(sell_data))
price_df = pd.DataFrame(price_data)

summary = enrich_and_merge(sell_df, price_df)

# =========================
# Anzeige
# =========================
st.subheader("Summen pro Artikel")
st.dataframe(summary)

# Verkaufsverlauf nach Kategorie (Wochenbasis)
st.subheader("📈 Verkaufsverlauf nach Kategorie (Woche)")

ts = sell_df.dropna(subset=["_rowdate"]).copy()
ts["Periode"] = ts["_rowdate"].dt.to_period("W").dt.start_time

ts_agg = (
    ts.groupby(["Kategorie", "Periode"], as_index=False)["SalesQty"].sum().rename(columns={"SalesQty": "Wert"})
)
ts_agg["Periode"] = pd.to_datetime(ts_agg["Periode"])

hover = alt.selection_single(fields=["Kategorie"], on="mouseover", nearest=True, empty="none")

base = alt.Chart(ts_agg)

lines = (
    base.mark_line(point=True)
    .encode(
        x=alt.X("Periode:T", title="Woche"),
        y=alt.Y("Wert:Q", title="Verkäufe Menge (Summe pro Woche)"),
        color=alt.Color("Kategorie:N", title="Kategorie"),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.25)),
        strokeWidth=alt.condition(hover, alt.value(3), alt.value(1.5)),
        tooltip=[
            alt.Tooltip("Periode:T", title="Woche"),
            alt.Tooltip("Kategorie:N", title="Kategorie"),
            alt.Tooltip("Wert:Q", title="Verkäufe", format=",.0f"),
        ],
    )
    .add_selection(hover)
)

popups = (
    base.mark_text(align="left", dx=4, dy=-5, fontSize=11, fontWeight="bold")
    .transform_filter(hover)
    .encode(
        x="Periode:T",
        y="Wert:Q",
        text="Kategorie:N",
        color=alt.Color("Kategorie:N"),
    )
)

chart = (lines + popups).properties(height=400)
st.altair_chart(chart, use_container_width=True)
