"""
Indonesia's Renewable Energy Transition: An Exploratory Data Analysis of
Progress and Challenges  |  SDG 7: Affordable and Clean Energy

Streamlit dashboard untuk hasil EDA 
Sumber data: Handbook of Energy and Economic Statistics of Indonesia 2024 (ESDM).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 1. KONFIGURASI HALAMAN & KONSTANTA
st.set_page_config(
    page_title="Indonesia Renewable Energy Transition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "cleaned_dataset.csv"
TEMPLATE = "plotly_white"

C = {
    "coal": "#2C3E50", "gas": "#E67E22", "hsd": "#7F8C8D", "fo": "#C0392B",
    "bio": "#27AE60", "on": "#2980B9", "off": "#27AE60", "own": "#2C3E50",
    "buy": "#3498DB", "up": "#E74C3C", "down": "#27AE60",
    "gdp": "#9B59B6", "energy": "#E67E22", "co2": "#E74C3C",
}

FUELS = {
    "coal_mtoe": ("Coal (Batu Bara)", C["coal"]),
    "gas_mtoe": ("Gas", C["gas"]),
    "hsd_mtoe": ("HSD (Diesel)", C["hsd"]),
    "fo_mtoe": ("Fuel Oil", C["fo"]),
    "biofame_mtoe": ("Biofame (Biofuel)", C["bio"]),
}

INDICATORS = {
    "coal_share": "Coal share in PP fuel mix (%)",
    "re_share": "Biofame / renewable share (%)",
    "total_mtoe": "Power plant fuel use (MTOE)",
    "coal_mt": "Coal consumed by CFPP (Mt)",
    "on_twh": "On-Grid generation (TWh)",
    "off_twh": "Off-Grid generation (TWh)",
    "own_twh": "PLN own generation (TWh)",
    "buy_twh": "IPP purchases (TWh)",
    "purchase_pct": "IPP share of On-Grid (%)",
    "co2_mt": "Energy-sector emissions (Mt CO₂e)",
    "energy_mboe": "Final energy consumption (Million BOE)",
    "gdp": "Real GDP, 2010 prices (Trillion IDR)",
    "energy_intensity": "Energy intensity (Million BOE / Trillion IDR)",
}


# 2. LOAD & PREPARE DATA 
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baca dataset gabungan, bersihkan duplikat, dan hitung kolom turunan.

    Returns
    -------
    raw : DataFrame hasil merge asli (untuk preview & download)
    df  : raw + kolom turunan yang dipakai untuk visualisasi
    """
    raw = pd.read_csv(DATA_PATH, sep=";")
    raw.columns = [str(c).replace("\n", " ").strip() for c in raw.columns]
    raw = raw.drop_duplicates(subset="Year").sort_values("Year").reset_index(drop=True)

    df = raw.copy()
    df["coal_mt"] = df["coal_ton"] / 1e6
    df["on_twh"] = df["Total_OnGrid"] / 1e3
    df["off_twh"] = df["Total_Off_Grid"].where(df["Total_Off_Grid"] > 0)  # data baru ada sejak 2018
    df["own_twh"] = df["Own_Sub_Total"] / 1e3
    df["buy_twh"] = df["Purchase_Sub_Total"] / 1e3
    df["offgrid_pct"] = (df["Total_Off_Grid"] / df["Grand_Total"] * 100).where(df["Total_Off_Grid"] > 0)
    df["purchase_pct"] = df["Purchase_Sub_Total"] / df["Total_OnGrid"] * 100
    df["coal_yoy"] = df["coal_mt"].pct_change() * 100
    df["co2_mt"] = df["Energy"] / 1e3  # Gg -> Mt
    df["energy_mboe"] = df["Final Energy Consumption (Thousand BOE)"] / 1e3
    df["gdp"] = df["GDP at 2010 Constant Price (Trillion Rupiahs)"]
    df["energy_intensity"] = df["energy_mboe"] / df["gdp"] * 1000
    return raw, df


def linear_fit(x: pd.Series, y: pd.Series):
    """Regresi linear sederhana. Return (slope, intercept, r2) atau None bila data kurang."""
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return None
    slope, intercept = np.polyfit(x[m], y[m], 1)
    r2 = np.corrcoef(x[m], y[m])[0, 1] ** 2
    return slope, intercept, r2


def fmt_delta(new: float, old: float, unit: str = "", digits: int = 1) -> str:
    return f"{new - old:+.{digits}f}{unit}"


# 3. KOMPONEN VISUALISASI (satu fungsi per grafik EDA)
def style(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE, height=height, hovermode="x unified",
        margin=dict(t=90, b=110, l=40, r=40),
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            yref="container", y=0, yanchor="bottom",
            font=dict(size=12), tracegroupgap=4,
        ),
    )
    if fig.layout.title.text:
        fig.update_layout(title=dict(
            x=0, xanchor="left", y=0.96, yanchor="top",
            font=dict(size=17), pad=dict(t=0, b=10),
        ))
    fig.update_annotations(font_size=14)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

def viz1_fuel_mix(d: pd.DataFrame, fuels: list[str]) -> go.Figure:
    fig = go.Figure()
    for col in fuels:
        label, color = FUELS[col]
        fig.add_trace(go.Scatter(
            x=d["Year"], y=d[col].fillna(0), name=label, mode="lines",
            stackgroup="one", line=dict(width=0.5, color=color), fillcolor=color,
        ))
    total = d[fuels].fillna(0).sum(axis=1)
    if "coal_mtoe" in fuels:
        coal_pct = d["coal_mtoe"] / total * 100
        fig.add_trace(go.Scatter(
            x=d["Year"], y=total, mode="text", text=[f"{v:.0f}%" for v in coal_pct],
            textposition="top center", showlegend=False, hoverinfo="skip",
            textfont=dict(size=11, color=C["coal"]),
        ))
    fig.update_layout(title="Power Plant Fuel Mix (MTOE) — labels show coal share",
                      yaxis_title="MTOE", xaxis=dict(dtick=1))
    return style(fig)


def viz2_coal_vs_bio(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=d["Year"], y=d["re_share"], name="Biofame share (%)",
                         marker_color=C["bio"], opacity=0.7,
                         text=[f"{v:.2f}%" if v > 0 else "" for v in d["re_share"]],
                         textposition="outside"), secondary_y=True)
    fig.add_trace(go.Scatter(x=d["Year"], y=d["coal_share"], name="Coal share (%)",
                             mode="lines+markers+text", line=dict(color=C["coal"], width=3),
                             text=[f"{v:.0f}%" for v in d["coal_share"]],
                             textposition="top center"), secondary_y=False)
    fig.update_yaxes(title_text="Coal share (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Biofame share (%)", range=[0, 2.0], secondary_y=True, showgrid=False)
    fig.update_layout(title="Coal Share (line) vs Biofame Share (bar) in the Power Plant Fuel Mix",
                      xaxis=dict(dtick=1))
    return style(fig)


def viz3_grid(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "On-Grid vs Off-Grid generation (TWh)", "Off-Grid share of grand total (%)"))
    fig.add_trace(go.Bar(x=d["Year"], y=d["on_twh"], name="On-Grid", marker_color=C["on"]), 1, 1)
    fig.add_trace(go.Bar(x=d["Year"], y=d["off_twh"], name="Off-Grid", marker_color=C["off"]), 1, 1)
    fig.add_trace(go.Bar(x=d["Year"], y=d["offgrid_pct"], name="Off-Grid share (%)",
                         marker_color=C["off"], showlegend=False,
                         text=[f"{v:.1f}%" if pd.notna(v) else "" for v in d["offgrid_pct"]],
                         textposition="outside"), 1, 2)
    fig.update_layout(barmode="group")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title_text="TWh", row=1, col=1)
    fig.update_yaxes(title_text="% of grand total", row=1, col=2)
    return style(fig)


def viz4_own_vs_purchase(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "PLN own generation vs IPP purchases (TWh)", "IPP purchases as % of total On-Grid"))
    fig.add_trace(go.Bar(x=d["Year"], y=d["own_twh"], name="PLN Own", marker_color=C["own"]), 1, 1)
    fig.add_trace(go.Bar(x=d["Year"], y=d["buy_twh"], name="Purchase (IPP)", marker_color=C["buy"]), 1, 1)
    fig.add_trace(go.Scatter(x=d["Year"], y=d["purchase_pct"], name="IPP share (%)",
                             mode="lines+markers+text", line=dict(color=C["buy"], width=3),
                             fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
                             text=[f"{v:.1f}%" for v in d["purchase_pct"]],
                             textposition="top center", showlegend=False), 1, 2)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title_text="TWh", row=1, col=1)
    fig.update_yaxes(title_text="% of On-Grid", range=[0, 55], row=1, col=2)
    return style(fig)


def viz5_coal_yoy(d: pd.DataFrame) -> go.Figure:
    d = d.dropna(subset=["coal_yoy"])
    colors = [C["up"] if v >= 0 else C["down"] for v in d["coal_yoy"]]
    fig = go.Figure(go.Bar(
        x=d["Year"], y=d["coal_yoy"], marker_color=colors,
        text=[f"{v:.1f}%" for v in d["coal_yoy"]], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Year-over-Year Growth in Coal Consumption at CFPP (%)",
                      yaxis_title="YoY growth (%)", xaxis=dict(dtick=1), showlegend=False)
    return style(fig)


def viz6_re_vs_co2(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d["Year"], y=d["re_share"], name="EBT share (%)",
                             mode="lines+markers", line=dict(color=C["bio"], width=3),
                             fill="tozeroy", fillcolor="rgba(39,174,96,0.12)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["Year"], y=d["co2_mt"], name="Energy CO₂e emissions (Mt)",
                             mode="lines+markers", line=dict(color=C["co2"], width=3, dash="dash"),
                             marker=dict(symbol="square")), secondary_y=True)
    fig.update_yaxes(title_text="EBT share (%)", secondary_y=False)
    fig.update_yaxes(title_text="Emissions (Mt CO₂e)", secondary_y=True, showgrid=False)
    fig.update_layout(title="Renewable (EBT) Share vs Energy-Sector Emissions", xaxis=dict(dtick=1))
    return style(fig)


def viz7_energy_vs_gdp(d: pd.DataFrame) -> tuple[go.Figure, float | None]:
    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {}]],
                        subplot_titles=("Final energy consumption vs real GDP", "Scatter: energy vs GDP"))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["energy_mboe"], name="Final energy (Million BOE)",
                             mode="lines+markers", line=dict(color=C["energy"], width=3)),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=d["Year"], y=d["gdp"], name="Real GDP (Trillion IDR, 2010)",
                             mode="lines+markers", line=dict(color=C["gdp"], width=3, dash="dash"),
                             marker=dict(symbol="square")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=d["gdp"], y=d["energy_mboe"], mode="markers+text", name="Year",
        text=d["Year"].astype(str), textposition="top right", showlegend=False,
        marker=dict(size=13, color=d["Year"], colorscale="Plasma",
                    colorbar=dict(title="Year", len=0.8), line=dict(color="white", width=1)),
        hovertemplate="Year %{text}<br>GDP %{x:,.0f}<br>Energy %{y:,.0f}<extra></extra>",
    ), row=1, col=2)
    fit = linear_fit(d["gdp"], d["energy_mboe"])
    r2 = None
    if fit:
        slope, intercept, r2 = fit
        xs = np.linspace(d["gdp"].min(), d["gdp"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                 name=f"Trend (R²={r2:.2f})", line=dict(color="gray", dash="dash")),
                      row=1, col=2)
    fig.update_yaxes(title_text="Million BOE", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Trillion IDR", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Real GDP (Trillion IDR, 2010)", row=1, col=2)
    fig.update_yaxes(title_text="Final energy (Million BOE)", row=1, col=2)
    fig.update_xaxes(dtick=1, row=1, col=1)
    fig.update_layout(hovermode="closest")
    return style(fig, 500), r2


# 4. SIDEBAR: NAVIGASI + FILTER GLOBAL
raw, df = load_data()
y_min, y_max = int(df["Year"].min()), int(df["Year"].max())

with st.sidebar:
    st.title("⚡ Energy Transition")
    st.caption("Indonesia · SDG 7 · 2014–2024")
    page = st.radio(
        "Navigation",
        ["Overview & Dataset", "Exploratory Data Analysis", "Interactive Insights"],
        label_visibility="collapsed",
    )
    st.divider()
    st.subheader("Filters")
    year_range = st.slider("Year range", y_min, y_max, (y_min, y_max))
    st.caption("The year filter applies to every page.")

dff = df[df["Year"].between(*year_range)].reset_index(drop=True)
raw_f = raw[raw["Year"].between(*year_range)].reset_index(drop=True)

if dff.empty:
    st.warning("No data for the selected range.")
    st.stop()

first, last = dff.iloc[0], dff.iloc[-1]


# 5. HALAMAN 1: OVERVIEW & DATASET
def page_overview():
    st.title("Indonesia's Renewable Energy Transition")
    st.markdown(
        "**An Exploratory Data Analysis of Progress and Challenges** — focused on "
        "**SDG 7: Affordable and Clean Energy**."
    )
    st.write(
        "This dashboard turns our EDA into an interactive app. It analyses how Indonesia's "
        "power-plant fuel mix, grid structure, emissions and economic growth evolved from "
        "2014 to 2024, and asks whether the country is on track toward a cleaner energy system. "
        "Data come from the *Handbook of Energy and Economic Statistics of Indonesia 2024* (ESDM)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(raw_f)}", help="One row per year after removing duplicates")
    c2.metric("Total columns", f"{raw.shape[1]}", help="Columns in the merged cleaned dataset")
    c3.metric("Period", f"{int(first['Year'])}–{int(last['Year'])}")
    c4.metric("Missing cells", f"{raw_f.isna().mean().mean() * 100:.1f}%",
              help="Mostly renewable categories not recorded before 2018")

    st.subheader("Key indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Coal share ({int(last['Year'])})", f"{last['coal_share']:.1f}%",
              fmt_delta(last["coal_share"], first["coal_share"], " pp"), delta_color="inverse")
    k2.metric(f"Biofame share ({int(last['Year'])})", f"{last['re_share']:.2f}%",
              fmt_delta(last["re_share"], first["re_share"], " pp", 2))
    k3.metric(f"Emissions ({int(last['Year'])})", f"{last['co2_mt']:.0f} Mt",
              fmt_delta(last["co2_mt"], first["co2_mt"], " Mt", 0), delta_color="inverse")
    k4.metric(f"IPP share of On-Grid ({int(last['Year'])})", f"{last['purchase_pct']:.1f}%",
              fmt_delta(last["purchase_pct"], first["purchase_pct"], " pp"))
    st.caption(f"Deltas compare {int(last['Year'])} with {int(first['Year'])} (start of the selected range).")

    st.subheader("Project at a glance")
    a, b = st.columns(2)
    with a:
        st.markdown(
            "**Research questions**\n"
            "- How has renewable energy grown over time?\n"
            "- How has the power-plant energy mix changed?\n"
            "- How do renewables relate to CO₂ emissions and GDP?\n"
            "- What challenges slow Indonesia's transition?"
        )
    with b:
        st.markdown(
            "**Method**\n"
            "1. Data understanding\n2. Data cleaning\n3. Data preparation (merge by year)\n"
            "4. Exploratory Data Analysis\n5. Insight generation & SDG 7 evaluation"
        )

    st.subheader("Dataset")
    with st.expander("Preview the dataset", expanded=False):
        show_all = st.toggle("Show all columns", value=False,
                             help="Off: key columns only. On: all merged columns.")
        if show_all:
            table = raw_f
        else:
            table = dff[["Year", "coal_share", "re_share", "on_twh", "off_twh", "own_twh",
                         "buy_twh", "coal_mt", "co2_mt", "energy_mboe", "gdp"]].rename(columns=INDICATORS | {"Year": "Year"})
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button("⬇️ Download filtered data (CSV)", raw_f.to_csv(index=False).encode("utf-8"),
                           file_name=f"indonesia_energy_{year_range[0]}_{year_range[1]}.csv",
                           mime="text/csv")
    with st.expander("Summary statistics"):
        st.dataframe(dff[list(INDICATORS)].rename(columns=INDICATORS).describe().T.round(2),
                     width="stretch")
    with st.expander("Data notes & limitations"):
        st.markdown(
            "- Renewable (biofame) share is 0% for 2014–2017, so renewable-vs-emissions analysis is "
            "meaningful only for 2018–2024.\n"
            "- Off-Grid generation data is available only from 2018 onward.\n"
            "- 2020 is a COVID-19 anomaly; the drop in coal use and emissions came from reduced "
            "activity, not from energy-transition policy.\n"
            "- Emissions use the *Energy* sector line of the GHG inventory (Gg CO₂e converted to Mt)."
        )


# 6. HALAMAN 2: EXPLORATORY DATA ANALYSIS (7 tab)
def page_eda():
    st.title("Exploratory Data Analysis")
    st.caption(f"Showing {int(first['Year'])}–{int(last['Year'])}. Hover, zoom and click legend items to explore.")

    tabs = st.tabs([
        "1 · Fuel Mix", "2 · Coal vs Biofame", "3 · On vs Off-Grid", "4 · PLN vs IPP",
        "5 · Coal YoY", "6 · Renewables vs CO₂", "7 · Energy vs GDP",
    ])

    # --- Tab 1
    with tabs[0]:
        st.subheader("Power Plant Fuel Mix")
        chosen = st.multiselect("Fuels to include", list(FUELS), default=list(FUELS),
                                format_func=lambda c: FUELS[c][0])
        if chosen:
            st.plotly_chart(viz1_fuel_mix(dff, chosen), width="stretch")
        else:
            st.info("Select at least one fuel.")
        st.markdown(
            f"Coal's share of power-plant fuel rose from **{first['coal_share']:.0f}%** "
            f"({int(first['Year'])}) to **{last['coal_share']:.0f}%** ({int(last['Year'])}). "
            "The sharpest build-up came before 2020; the 2020 dip reflects COVID-19 restrictions. "
            "Policy remains focused on cheap, readily available supply (energy security and equity) "
            "rather than on decarbonising the mix."
        )

    # --- Tab 2
    with tabs[1]:
        st.subheader("Coal vs Biofame Shares")
        st.plotly_chart(viz2_coal_vs_bio(dff), width="stretch")
        st.markdown(
            "Coal stays between roughly **59% and 78%** of the mix while biofame never exceeds "
            f"about **1.5%** (**{last['re_share']:.2f}%** in {int(last['Year'])}). Renewable growth "
            "has not yet been large enough to change the structure of the energy mix."
        )

    # --- Tab 3
    with tabs[2]:
        st.subheader("On-Grid vs Off-Grid Generation")
        st.plotly_chart(viz3_grid(dff), width="stretch")
        st.markdown(
            "On-Grid generation grew steadily with demand. Off-Grid data exist only from 2018 "
            "(shown as blank before that) and the Off-Grid share rose from about **5.9% to 7.4%**, "
            "a gradual step toward **SDG Target 7.1** (universal access to modern energy)."
        )
        st.caption("Values come from the power-plant production table (TWh).")

    # --- Tab 4
    with tabs[3]:
        st.subheader("PLN Own Generation vs IPP Purchases")
        st.plotly_chart(viz4_own_vs_purchase(dff), width="stretch")
        st.markdown(
            f"IPP purchases reached **{last['purchase_pct']:.1f}%** of On-Grid supply in "
            f"{int(last['Year'])} (from **{first['purchase_pct']:.1f}%** in {int(first['Year'])}), "
            "while PLN's own generation stayed fairly flat. Private investment is an opportunity for "
            "clean energy, but regulation must steer it toward renewables rather than more fossil capacity."
        )

    # --- Tab 5
    with tabs[4]:
        st.subheader("Year-over-Year Growth of Coal Consumption")
        st.plotly_chart(viz5_coal_yoy(dff), width="stretch")
        st.markdown(
            "Growth was positive in most years, peaking at **10.8% in 2019**. Declines in "
            "**2020 (-1.3%)** and **2023 (-0.9%)** were short-lived: coal use rebounded by "
            "**3.6% in 2024**. The 2023 decline should be read cautiously — no causal link to "
            "renewables or efficiency is established by this data."
        )

    # --- Tab 6
    with tabs[5]:
        st.subheader("Renewable Share vs Carbon Emissions")
        st.plotly_chart(viz6_re_vs_co2(dff), width="stretch")
        st.markdown(
            "The renewable share moved from **0.72% (2018)** to **1.45% (2024)** while emissions "
            "climbed from **614 Mt** to **791 Mt CO₂e**. The largest jump came between 2021 and 2022 "
            "(about +116 Mt), pointing to a post-pandemic recovery still driven by fossil fuels."
        )
        st.caption("Renewable share is 0% before 2018, so the relationship is only meaningful for 2018–2024.")

    # --- Tab 7
    with tabs[6]:
        st.subheader("Energy Consumption vs Economic Growth")
        fig, r2 = viz7_energy_vs_gdp(dff)
        st.plotly_chart(fig, width="stretch")
        if r2 is not None:
            st.metric("R² (selected range)", f"{r2:.2f}",
                      help="Full 2014–2024 range gives R² = 0.86 in the report.")
        else:
            st.info("Select at least 3 years to compute R².")
        st.markdown(
            "GDP and final energy consumption move together (both dipped in 2020 and rose again "
            "from 2021), so growth is still tightly tied to energy use — and, because supply is "
            "mostly fossil, to emissions. Decoupling will need renewables and efficiency gains."
        )
        st.caption("Correlation does not prove causation; with ~11 yearly points treat R² as descriptive.")


# 7. HALAMAN 3: INTERACTIVE INSIGHTS 
def page_insights():
    st.title("Interactive Insights")
    st.caption("Pick indicators, compare years and export what you see.")

    labels = list(INDICATORS.values())
    inv = {v: k for k, v in INDICATORS.items()}

    with st.sidebar:
        st.subheader("Insight filters")
        chosen_labels = st.multiselect("Indicators to plot", labels,
                                       default=[INDICATORS["coal_share"], INDICATORS["co2_mt"]])
        normalise = st.checkbox("Index to 100 at start year", value=True,
                                help="Makes indicators with different units comparable.")

    st.subheader("A. Compare indicators over time")
    if not chosen_labels:
        st.info("Choose at least one indicator in the sidebar.")
    else:
        fig = go.Figure()
        for lab in chosen_labels:
            s = dff[inv[lab]]
            base = s.dropna().iloc[0] if s.notna().any() else np.nan
            y = s / base * 100 if normalise and pd.notna(base) and base != 0 else s
            fig.add_trace(go.Scatter(x=dff["Year"], y=y, name=lab, mode="lines+markers"))
        fig.update_layout(yaxis_title="Index (start = 100)" if normalise else "Value",
                          xaxis=dict(dtick=1))
        st.plotly_chart(style(fig, 440), width="stretch")

    st.subheader("B. Relationship explorer")
    cx, cy = st.columns(2)
    x_lab = cx.selectbox("X axis", labels, index=labels.index(INDICATORS["gdp"]))
    y_lab = cy.selectbox("Y axis", labels, index=labels.index(INDICATORS["energy_mboe"]))
    x, y = dff[inv[x_lab]], dff[inv[y_lab]]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers+text", text=dff["Year"].astype(str), textposition="top right",
        marker=dict(size=13, color=dff["Year"], colorscale="Plasma", showscale=True,
                    colorbar=dict(title="Year"), line=dict(color="white", width=1)),
        hovertemplate="Year %{text}<br>X %{x:,.2f}<br>Y %{y:,.2f}<extra></extra>",
    ))
    fit = linear_fit(x, y)
    if fit:
        slope, intercept, r2 = fit
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                 line=dict(color="gray", dash="dash"), name=f"Trend (R²={r2:.2f})"))
    fig.update_layout(xaxis_title=x_lab, yaxis_title=y_lab, hovermode="closest")
    st.plotly_chart(style(fig, 440), width="stretch")
    if fit:
        st.info(f"R² = **{fit[2]:.2f}** over {int(first['Year'])}–{int(last['Year'])}. "
                "Strong association does not imply causation, and n is small.")
    else:
        st.info("Not enough overlapping data points in this range to fit a trend.")

    st.subheader("C. Compare two years")
    years = dff["Year"].astype(int).tolist()
    ca, cb = st.columns(2)
    ya = ca.selectbox("Year A", years, index=0)
    yb = cb.selectbox("Year B", years, index=len(years) - 1)
    ra, rb = dff[dff["Year"] == ya].iloc[0], dff[dff["Year"] == yb].iloc[0]
    comp = pd.DataFrame({
        "Indicator": labels,
        str(ya): [ra[inv[l]] for l in labels],
        str(yb): [rb[inv[l]] for l in labels],
    })
    comp["Change"] = comp[str(yb)] - comp[str(ya)]
    comp["Change (%)"] = np.where(comp[str(ya)].fillna(0) != 0, comp["Change"] / comp[str(ya)] * 100, np.nan)
    st.dataframe(comp.round(2), width="stretch", hide_index=True)

    st.subheader("D. Filtered data")
    cols = [inv[l] for l in labels]
    out = dff[["Year"] + cols].rename(columns=INDICATORS)
    with st.expander("Show filtered table", expanded=False):
        st.dataframe(out.round(3), width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download indicators (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"indicators_{year_range[0]}_{year_range[1]}.csv", mime="text/csv")
    d2.download_button("⬇️ Download full merged data (CSV)", raw_f.to_csv(index=False).encode("utf-8"),
                       file_name=f"full_dataset_{year_range[0]}_{year_range[1]}.csv", mime="text/csv")

    st.subheader("What the selected range says")
    st.success(
        f"Between **{int(first['Year'])}** and **{int(last['Year'])}**, coal's share of power-plant fuel changed by "
        f"**{last['coal_share'] - first['coal_share']:+.1f} pp**, biofame by **{last['re_share'] - first['re_share']:+.2f} pp**, "
        f"while energy-sector emissions changed by **{(last['co2_mt'] / first['co2_mt'] - 1) * 100:+.0f}%** and real GDP by "
        f"**{(last['gdp'] / first['gdp'] - 1) * 100:+.0f}%**."
    )


# 8. ROUTER & FOOTER
{
    "Overview & Dataset": page_overview,
    "Exploratory Data Analysis": page_eda,
    "Interactive Insights": page_insights,
}[page]()

st.sidebar.divider()
st.sidebar.caption(
    "Source: ESDM, *Handbook of Energy and Economic Statistics of Indonesia 2024*.  \n"
)
