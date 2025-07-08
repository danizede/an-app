# ──────────────────────────────────────────────────────────────
#  Galaxus Sell-out Aggregator – Streamlit
# ──────────────────────────────────────────────────────────────
import streamlit as st, pandas as pd
from io import BytesIO

# ---------- Passwort ----------------------------------------------------------
PW_OK = {"Nofava22caro!"}
if st.text_input("🔒 Passwort", type="password") not in PW_OK:
    st.warning("Bitte gültiges Passwort eingeben.")
    st.stop()

st.set_page_config(page_title="Galaxus Aggregator", layout="wide")
st.title("📦 Galaxus Sell-out Aggregator")

# ---------- Uploads (Merken in Session) --------------------------------------
ss = st.session_state
sell_up  = st.file_uploader("Sell-out-Report (.xlsx)",  type="xlsx", key="sell")
price_up = st.file_uploader("Preisliste (.xlsx)",       type="xlsx", key="price")

if sell_up:  ss["sell_bin"]  = sell_up.getvalue()
if price_up: ss["price_bin"] = price_up.getvalue()

def read(bin_: bytes) -> pd.DataFrame:   # helper für Excel-Bytes → DataFrame
    return pd.read_excel(BytesIO(bin_))

# ---------- Alias-Listen ------------------------------------------------------
AL_NR  = ["Artikelnr", "Artikelnummer", "Hersteller-Nr.", "Hersteller Nr"]
AL_EAN = ["EAN", "GTIN", "EAN Code", "EAN_Code"]
AL_BEZ = ["Bez", "Bezeichnung", "Produktname"]
AL_CAT = ["Kategorie", "Warengruppe", "Group", "Cat"]          #  ← erweitert
AL_PR  = ["Preis", "NETTO NETTO", "Netto", "VK"]

def pick(df, aliases, label):
    for c in aliases:
        if c in df.columns: return c
    # Fallback Kategorie: wenn Kategorie fehlt, erzeugen wir Dummy-Spalte
    if label == "Kategorie":
        df["Kategorie_dummy"] = "-"
        return "Kategorie_dummy"
    raise KeyError(f"Spalte für {label} fehlt – gesucht: {aliases}")

# ---------- Matching + Enrichment --------------------------------------------
@st.cache_data(show_spinner="🔗 Preise zuordnen …")
def enrich(sell_bin, price_bin):
    sell   = read(sell_bin)
    price  = read(price_bin)

    s_nr  = pick(sell , AL_NR , "Artikelnr")
    s_ean = pick(sell , AL_EAN, "EAN")
    p_nr  = pick(price, AL_NR , "Artikelnr")
    p_ean = pick(price, AL_EAN, "EAN")
    p_bez = pick(price, AL_BEZ, "Bez")
    p_cat = pick(price, AL_CAT, "Kategorie")
    p_pr  = pick(price, AL_PR , "Preis")

    price = price.rename(columns={p_nr:"nr", p_ean:"ean",
                                  p_bez:"Bez", p_cat:"Kategorie",
                                  p_pr:"Preis"})[["nr","ean","Bez","Kategorie","Preis"]]

    # --- 1) Artikelnr ---------------------------------------------------------
    merged = sell.merge(price.drop_duplicates("nr"),
                        left_on=s_nr, right_on="nr",
                        how="left", suffixes=("", "_p"))

    # --- 2) EAN ---------------------------------------------------------------
    m2 = merged["Preis"].isna() & merged[s_ean].notna()
    if m2.any():
        df2 = (merged[m2]
               .merge(price.drop_duplicates("ean"),
                      left_on=s_ean, right_on="ean", how="left")
               [["Bez","Kategorie","Preis"]])
        merged.loc[m2, ["Bez","Kategorie","Preis"]] = df2.values

    # --- 3) Fuzzy (erste 2 Wörter) -------------------------------------------
    def tkn(x): return " ".join(str(x).lower().split()[:2])
    price["tkn"] = price["Bez"].apply(tkn)
    merged["tkn"] = merged["Bez"].apply(tkn)
    m3 = merged["Preis"].isna()
    if m3.any():
        df3 = (merged[m3]
               .merge(price.drop_duplicates("tkn")[["tkn","Bez","Kategorie","Preis"]],
                      on="tkn", how="left")[["Bez","Kategorie","Preis"]])
        merged.loc[m3, ["Bez","Kategorie","Preis"]] = df3.values

    # CHF-Werte
    for col, qty in [("Einkaufswert","Einkauf"),
                     ("Verkaufswert","Verkauf"),
                     ("Lagerwert","Verfügbar")]:
        merged[col] = merged[qty] * merged["Preis"]

    return merged

# ---------- Aggregation -------------------------------------------------------
@st.cache_data(show_spinner="📊 Aggregiere …")
def aggregate(df):
    tbl = (df.groupby(["Hersteller-Nr.", "Bez"], as_index=False)
             .agg(Einkauf       =("Einkauf","sum"),
                  Verkaufsmenge =("Verkauf","sum"),
                  Verfügbar     =("Verfügbar","sum"),
                  Einkaufswert  =("Einkaufswert","sum"),
                  Verkaufswert  =("Verkaufswert","sum"),
                  Lagerwert     =("Lagerwert","sum")))
    tot = {"VK":tbl["Verkaufswert"].sum(),
           "EK":tbl["Einkaufswert"].sum(),
           "LG":tbl["Lagerwert"].sum()}
    return tbl, tot

# ---------- Anzeige -----------------------------------------------------------
if "sell_bin" in ss and "price_bin" in ss:
    data, total = aggregate(enrich(ss["sell_bin"], ss["price_bin"]))

    c1,c2,c3 = st.columns(3)
    c1.metric("Verkaufswert (CHF)", f"{total['VK']:,.0f}".replace(","," "))
    c2.metric("Einkaufswert (CHF)", f"{total['EK']:,.0f}".replace(","," "))
    c3.metric("Lagerwert (CHF)",    f"{total['LG']:,.0f}".replace(","," "))

    st.dataframe(data, use_container_width=True)
else:
    st.info("Bitte Sell-out-Report **und** Preisliste hochladen.")
