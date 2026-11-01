"""
System UI — FloodGuard Bangladesh (redesigned).

Modern, citizen-first, map-centric flood risk forecasting web app.
Consumes the FastAPI inference API at http://localhost:8000.

Usage:
    streamlit run system/ui/app.py --server.port 8501
"""
from __future__ import annotations

import html as ihtml
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

API_BASE = "http://localhost:8000"
ROOT = Path(r"D:\Flood predictor")

APP_NAME = "FloodGuard Bangladesh"
APP_TAGLINE = "Flood risk forecasting for Bangladesh — powered by D16"

# ---------------------------------------------------------------------------
# Design system tokens (from spec)
# ---------------------------------------------------------------------------

FG_NAVY = "#102A43"
FG_TEAL = "#087E8B"
FG_BG = "#F5F8FA"
FG_CARD = "#FFFFFF"
FG_BORDER = "#D9E2EC"
FG_RADIUS = "14px"
FG_SHADOW = "0 4px 16px rgba(16,42,67,0.06)"

BAND_COLOR = {
    "low": "#218739",
    "moderate": "#8A6200",
    "high": "#B54708",
    "very_high": "#B42318",
    "unknown": "#52616B",
    "no_data": "#52616B",
}

BAND_LABEL = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "very_high": "Very high",
    "unknown": "Unknown / No data",
    "no_data": "Unknown / No data",
}

BAND_ACTION = {
    "low": "Routine monitoring",
    "moderate": "Stay informed and monitor updates",
    "high": "Prepare and follow local advisories",
    "very_high": "Urgently check official warnings and local guidance",
    "unknown": "Prediction unavailable",
    "no_data": "Prediction unavailable",
}

DISCLAIMER = (
    "Research prototype — not an official government warning system. "
    "Not for life-safety decisions. Always follow official FFWC advisories."
)

# ---------------------------------------------------------------------------
# Design system CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --fg-navy: {FG_NAVY};
            --fg-teal: {FG_TEAL};
            --fg-bg: {FG_BG};
            --fg-card: {FG_CARD};
            --fg-border: {FG_BORDER};
            --fg-radius: {FG_RADIUS};
            --fg-shadow: {FG_SHADOW};
        }}
        .stApp {{
            background: var(--fg-bg);
        }}
        /* Hide Streamlit chrome */
        [data-testid="stSidebarNav"] {{ display: none; }}
        header[data-testid="stHeader"] {{ display: none; }}
        [data-testid="stDecoration"] {{ display: none; }}
        .stApp > header {{ display: none; }}
        .stApp .block-container {{
            padding-top: 0;
            max-width: 1200px;
        }}
        /* Navigation bar */
        .fg-nav {{
            background: var(--fg-navy);
            color: #fff;
            padding: 16px 24px;
            border-radius: var(--fg-radius) 0 0 var(--fg-radius);
            margin-bottom: 20px;
        }}
        .fg-nav-title {{
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 0;
        }}
        .fg-nav-tagline {{
            font-size: 13px;
            opacity: 0.85;
            margin-top: 2px;
        }}
        .fg-status {{
            margin-top: 8px;
            font-size: 12px;
            opacity: 0.9;
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .fg-status-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .fg-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        .fg-dot--ok {{ background: #218739; }}
        .fg-dot--warn {{ background: #B54708; }}
        .fg-dot--err {{ background: #B42318; }}
        /* Section headings */
        .fg-section {{
            font-size: 20px;
            font-weight: 600;
            color: var(--fg-navy);
            margin: 24px 0 12px 0;
            letter-spacing: -0.3px;
        }}
        .fg-subtitle {{
            font-size: 14px;
            color: #52616B;
            margin: 0 0 16px 0;
        }}
        /* Cards */
        .fg-card {{
            background: var(--fg-card);
            border: 1px solid var(--fg-border);
            border-radius: var(--fg-radius);
            padding: 20px 24px;
            box-shadow: var(--fg-shadow);
            margin-bottom: 16px;
        }}
        .fg-card-header {{
            font-size: 13px;
            font-weight: 600;
            color: #52616B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        .fg-card-title {{
            font-size: 20px;
            font-weight: 600;
            color: var(--fg-navy);
            margin: 0 0 8px 0;
        }}
        /* Metric cards */
        .fg-metric {{
            background: var(--fg-card);
            border: 1px solid var(--fg-border);
            border-radius: var(--fg-radius);
            padding: 18px 20px;
            box-shadow: var(--fg-shadow);
            height: 100%;
        }}
        .fg-metric-label {{
            font-size: 12px;
            font-weight: 500;
            color: #52616B;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }}
        .fg-metric-value {{
            font-size: 30px;
            font-weight: 700;
            color: var(--fg-navy);
            line-height: 1.1;
        }}
        .fg-metric-value--teal {{ color: var(--fg-teal); }}
        .fg-metric-value--red {{ color: #B42318; }}
        /* Risk card */
        .fg-risk-card {{
            background: var(--fg-card);
            border: 1px solid var(--fg-border);
            border-radius: var(--fg-radius);
            padding: 24px;
            box-shadow: var(--fg-shadow);
        }}
        .fg-risk-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 16px;
        }}
        .fg-station-name {{
            font-size: 24px;
            font-weight: 700;
            color: var(--fg-navy);
            margin: 0;
        }}
        .fg-station-meta {{
            font-size: 13px;
            color: #52616B;
            margin: 2px 0;
        }}
        .fg-band-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            color: #fff;
        }}
        .fg-probability {{
            font-size: 56px;
            font-weight: 700;
            color: var(--fg-navy);
            line-height: 1;
            margin: 8px 0;
        }}
        .fg-action {{
            font-size: 15px;
            color: var(--fg-navy);
            font-weight: 500;
        }}
        .fg-action--em {{ color: #B42318; font-weight: 600; }}
        .fg-meta {{
            font-size: 12px;
            color: #52616B;
            margin-top: 12px;
        }}
        /* Horizon chips */
        .fg-chip-group {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .fg-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            border: 1px solid var(--fg-border);
            border-radius: 10px;
            background: #fff;
            font-size: 13px;
            font-weight: 500;
            color: #52616B;
            cursor: pointer;
            transition: all 0.18s;
        }}
        .fg-chip--active {{
            border-color: var(--fg-teal);
            background: #e3f4f6;
            color: var(--fg-teal);
        }}
        .fg-chip-label {{
            font-size: 11px;
            opacity: 0.7;
        }}
        /* Buttons */
        .stButton > button {{
            background: var(--fg-teal);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            min-height: 42px;
            transition: all 0.18s ease;
        }}
        .stButton > button:hover {{
            background: #066a77;
            transform: translateY(-1px);
        }}
        /* Disclaimer */
        .fg-disclaimer {{
            font-size: 12px;
            color: #52616B;
            margin-top: 24px;
            border-top: 1px solid var(--fg-border);
            padding-top: 16px;
        }}
        /* Dataframe styling */
        .stDataFrame {{ border-radius: var(--fg-radius); }}
        /* Empty state */
        .fg-empty {{
            text-align: center;
            padding: 48px 24px;
            color: #52616B;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        st.warning(f"API {path} returned {r.status_code}")
    except Exception as exc:
        st.error(f"Cannot reach API at {API_BASE}. Is the server running? ({exc})")
    return None

# ---------------------------------------------------------------------------
# Station master (names + IDs)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    p = ROOT / "system" / "data" / "raw" / "ffwc" / "ffwc_stations.csv"
    if not p.exists():
        return pd.DataFrame(columns=["station_id", "name", "district", "division", "river", "danger_level"])
    st_df = pd.read_csv(p)
    st_df = st_df.rename(columns={
        "id": "station_id",
        "long": "lon",
        "dangerlevel": "danger_level",
    })
    keep = ["station_id", "name", "district", "division", "river", "danger_level"]
    keep = [c for c in keep if c in st_df.columns]
    return st_df[keep].dropna(subset=["station_id", "name"])

# ---------------------------------------------------------------------------
# Status strip
# ---------------------------------------------------------------------------

def render_status_strip() -> dict | None:
    health = api_get("/health")
    if not health:
        st.markdown(
            '<div class="fg-status">'
            '<span class="fg-dot fg-dot--err"></span>'
            '<span style="color:#B42318;">API unreachable</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return None
    ok = health.get("status") == "ok" and health.get("model_loaded")
    feat = health.get("feature_store_dates", 0)
    dot_cls = "fg-dot--ok" if ok else "fg-dot--warn"
    dot_txt = "System online" if ok else "System degraded"
    dot_color = "#218739" if ok else "#B54708"
    st.markdown(
        f'<div class="fg-status">'
        f'<span class="fg-dot {dot_cls}"></span>'
        f'<span style="color:{dot_color};font-weight:600;">{dot_txt}</span>'
        f'<span style="color:#52616B;">• {feat} feature dates</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    return health

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f"""
        <div class="fg-nav">
            <h1 class="fg-nav-title">🌊 {APP_NAME}</h1>
            <p class="fg-nav-tagline">{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def render_disclaimer() -> None:
    st.markdown(
        f"""
        <div class="fg-disclaimer">
            <strong>Important safety disclaimer:</strong> {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Citizen view
# ---------------------------------------------------------------------------

def citizen_view() -> None:
    stations = load_stations()
    if stations.empty:
        st.error("Station data not available. Run ingestion first.")
        return

    dates_data = api_get("/dates")
    if dates_data is None:
        return
    available_dates = dates_data.get("dates", [])
    if not available_dates:
        st.warning("No feature data in the store yet.")
        return

    min_available = available_dates[0]
    max_available = available_dates[-1]

    # Forecast controls card
    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-title">Check your local flood risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="fg-subtitle">Choose a station and forecast horizon to understand the predicted risk.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1.5, 1.5])
    with col1:
        target_date = st.date_input(
            "Forecast date",
            value=date.today(),
            min_value=date.fromisoformat(min_available),
            max_value=date.today() + timedelta(days=7),
            label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="fg-card-header">FORECAST HORIZON</div>', unsafe_allow_html=True)
        horizon = st.segmented_control(
            "Horizon",
            options=["24h", "48h", "72h"],
            default="24h",
            label_visibility="collapsed",
        )
    with col3:
        st.markdown('<div class="fg-card-header">LATEST DATA</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:13px;font-weight:600;color:var(--fg-navy);">{max_available}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Station search card
    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-title">Station</div>', unsafe_allow_html=True)
    station_search = st.text_input(
        "Search station",
        placeholder="Type station name, district, division, or river…",
        key="citizen_station_search",
        label_visibility="collapsed",
    )

    stations_sorted = stations.sort_values("name")
    if station_search:
        mask = (
            stations_sorted["name"].str.contains(station_search, case=False, na=False)
            | stations_sorted["district"].str.contains(station_search, case=False, na=False)
            | stations_sorted["division"].str.contains(station_search, case=False, na=False)
            | stations_sorted["river"].str.contains(station_search, case=False, na=False)
        )
        stations_sorted = stations_sorted[mask]

    if stations_sorted.empty:
        st.warning(f"No stations match '{station_search}'")
        st.markdown('</div>', unsafe_allow_html=True)
        render_disclaimer()
        return

    label_to_row = {
        f"{r['name']} — {r.get('district','')}, {r.get('division','')}"
        + f"  (ID: {int(r['station_id'])})": r
        for _, r in stations_sorted.iterrows()
    }
    chosen_label = st.selectbox(
        f"Matching stations ({len(stations_sorted)} found)",
        sorted(label_to_row.keys()),
        key="citizen_station_select",
        label_visibility="collapsed",
    )
    chosen = label_to_row[chosen_label]
    sid = int(chosen["station_id"])
    st.markdown('</div>', unsafe_allow_html=True)

    target_str = target_date.isoformat()
    if target_str not in available_dates:
        st.info(
            f"No feature data for {target_str}. Available range: "
            f"{min_available} → {max_available}."
        )
        render_disclaimer()
        return

    # Fetch prediction
    with st.spinner("Computing forecast…"):
        preds = api_get(
            f"/predict/all?date={target_str}&horizon={horizon}"
        )

    if preds is None:
        render_disclaimer()
        return

    df = pd.DataFrame(preds)
    row = df[df["station_id"] == sid]

    if row.empty or row.iloc[0]["probability"] is None:
        st.markdown('<div class="fg-empty">', unsafe_allow_html=True)
        st.markdown("### Prediction unavailable")
        st.markdown(
            f"No valid forecast for **{ihtml.escape(str(chosen['name']))}** on {target_str}. "
            "The station may not have a recent sensor reading."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        render_disclaimer()
        return

    r = row.iloc[0]
    prob = float(r["probability"])
    band = r["band"] or "no_data"
    action = r["action"] or "N/A"

    # Risk card
    st.markdown('<div class="fg-risk-card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="fg-station-name">{ihtml.escape(str(chosen["name"]))}</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="fg-station-meta">'
        f'{ihtml.escape(str(chosen.get("district","")))}, '
        f'{ihtml.escape(str(chosen.get("division","")))} · '
        f'{ihtml.escape(str(chosen.get("river","")))} · '
        f'Station ID: {sid}</div>',
        unsafe_allow_html=True,
    )

    band_color = BAND_COLOR.get(band, BAND_COLOR["no_data"])
    band_label = BAND_LABEL.get(band, "Unknown / No data")
    st.markdown(
        f'<span class="fg-band-badge" style="background-color:{band_color};">{band_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="fg-probability">{prob:.1%}</div>', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-header" style="margin-top:16px;">FLOOD PROBABILITY</div>', unsafe_allow_html=True)

    action_color = "#B42318" if band in ("high", "very_high") else "var(--fg-navy)"
    st.markdown(
        f'<div class="fg-action" style="color:{action_color};">{str(action)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="fg-meta">'
        f'Date: {target_str}  |  Horizon: {horizon}  |  '
        f'Danger level: {chosen.get("danger_level","—")}m</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    render_disclaimer()

# ---------------------------------------------------------------------------
# Admin view helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v):.1%}"

def _fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{int(v)}"

# ---------------------------------------------------------------------------
# Admin view
# ---------------------------------------------------------------------------

def admin_view() -> None:
    stations = load_stations()
    dates_data = api_get("/dates")
    if dates_data is None or stations.empty:
        return
    available_dates = dates_data.get("dates", [])
    if not available_dates:
        st.warning("No feature data yet.")
        return

    # Admin controls
    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-title">National Flood Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="fg-subtitle">Forecast monitoring across Bangladesh</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1.5])
    with col1:
        target_date = st.date_input(
            "Forecast date",
            value=date.today(),
            key="admin_date",
            label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="fg-card-header">FORECAST HORIZON</div>', unsafe_allow_html=True)
        horizon = st.segmented_control(
            "Horizon",
            options=["24h", "48h", "72h"],
            default="24h",
            key="admin_horizon",
            label_visibility="collapsed",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    target_str = target_date.isoformat()
    if target_str not in available_dates:
        st.info(
            f"No feature data for {target_str}. Range: "
            f"{available_dates[0]} → {available_dates[-1]}"
        )
        render_disclaimer()
        return

    preds = api_get(
        f"/predict/all?date={target_str}&horizon={horizon}"
    )
    if preds is None:
        render_disclaimer()
        return

    df = pd.DataFrame(preds)
    valid = df[df["probability"].notna()].copy()
    if valid.empty:
        st.markdown('<div class="fg-empty">', unsafe_allow_html=True)
        st.markdown("### No predictions available")
        st.markdown(f"No valid forecasts for {target_str} at {horizon} horizon.")
        st.markdown('</div>', unsafe_allow_html=True)
        render_disclaimer()
        return

    # Merge station names
    valid = valid.merge(stations, on="station_id", how="left")

    # Summary metrics
    st.markdown('<div class="fg-section">Summary</div>', unsafe_allow_html=True)
    n_stations = len(valid)
    max_prob = valid["probability"].max()
    avg_prob = valid["probability"].mean()
    high_risk = valid[valid["band"].isin(["high", "very_high"])]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="fg-metric">', unsafe_allow_html=True)
        st.markdown('<div class="fg-metric-label">Stations reporting</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fg-metric-value">{_fmt_int(n_stations)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="fg-metric">', unsafe_allow_html=True)
        st.markdown('<div class="fg-metric-label">Maximum probability</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fg-metric-value fg-metric-value--teal">{_fmt_pct(max_prob)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="fg-metric">', unsafe_allow_html=True)
        st.markdown('<div class="fg-metric-label">Average probability</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fg-metric-value">{_fmt_pct(avg_prob)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="fg-metric">', unsafe_allow_html=True)
        st.markdown('<div class="fg-metric-label">High / very high</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fg-metric-value fg-metric-value--red">{_fmt_int(len(high_risk))}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bangladesh choropleth map
    st.markdown('<div class="fg-section">Bangladesh flood risk map</div>', unsafe_allow_html=True)
    st.markdown('<div class="fg-subtitle">Division and district boundaries colored by worst valid station risk in the selected forecast.</div>', unsafe_allow_html=True)
    map_level = st.radio("Map level", ["District (ADM2)", "Division (ADM1)"], horizontal=True, key="map_level", label_visibility="collapsed")
    level = "adm2" if map_level == "District (ADM2)" else "adm1"
    fig = build_choropleth(valid, stations, level)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"choropleth_{level}")
    else:
        st.info("Map boundaries not available.")

    # Risk band distribution chart
    st.markdown('<div class="fg-section">Risk band distribution</div>', unsafe_allow_html=True)
    band_order = ["very_high", "high", "moderate", "low", "unknown", "no_data"]
    band_labels = {
        "very_high": "Very high",
        "high": "High",
        "moderate": "Moderate",
        "low": "Low",
        "unknown": "Unknown",
        "no_data": "No data",
    }
    counts = valid["band"].value_counts().reindex(band_order).fillna(0).astype(int)
    chart_df = pd.DataFrame({
        "band": [band_labels[b] for b in band_order],
        "count": counts.values,
    })
    st.bar_chart(chart_df.set_index("band"), color="#087E8B")

    # Affected stations table
    st.markdown('<div class="fg-section">Affected stations (High / Very high)</div>', unsafe_allow_html=True)
    affected = high_risk.sort_values("probability", ascending=False)
    if not affected.empty:
        show = affected[["name", "district", "division", "river", "probability", "band", "action"]].copy()
        show["probability"] = show["probability"].round(4)
        show.columns = ["Station", "District", "Division", "River", "Probability", "Band", "Action"]
        st.dataframe(show, use_container_width=True, hide_index=True)
        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download affected areas CSV",
            data=csv,
            file_name=f"floodguard_affected_{target_str}_{horizon}.csv",
            mime="text/csv",
            key="admin_download_csv",
        )
    else:
        st.info("No high-risk stations for this forecast.")

    # Full table
    with st.expander("All stations"):
        show_all = valid.copy()
        show_all["probability"] = show_all["probability"].round(4)
        show_all = show_all.sort_values("station_id")
        st.dataframe(
            show_all[["station_id", "name", "probability", "band", "action"]],
            use_container_width=True,
            hide_index=True,
        )

    render_disclaimer()

# ---------------------------------------------------------------------------
# Choropleth map (admin view)
# ---------------------------------------------------------------------------

# Division/district name normalization (from system/map/config.py logic)
DIVISION_NORM = {
    "Chattogram": "Chittagong",
    "Chittagong": "Chittagong",
    "Barishal": "Barisal",
    "Sylhet": "Sylhet",
    "Dhaka": "Dhaka",
    "Rajshahi": "Rajshani",
    "Khulna": "Khulna",
    "Rangpur": "Rangpur",
    "Mymensingh": "Mymensingh",
}

DISTRICT_NORM = {
    "Barishal": "Barisal",
    "Brahmmonbaria": "Brahamanbaria",
    "Chapai Nawabganj": "Nawabganj",
    "Chattogram": "Chittagong",
    "Chittagong": "Chittagong",
    "Kishorganj": "Kishoreganj",
    "Netrokona": "Netrakona",
}

BAND_SEVERITY = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "very_high": 3,
    "unknown": -1,
    "no_data": -1,
}


def _norm_name(name, norm_map):
    if not name:
        return name
    return norm_map.get(str(name).strip(), str(name).strip())


def _load_geojson(level: str):
    """Load simplified admin boundary GeoJSON."""
    fname = {
        "adm1": "geoBoundaries-BGD-ADM1_simplified.geojson",
        "adm2": "geoBoundaries-BGD-ADM2_simplified.geojson",
    }[level]
    p = ROOT / "data" / "raw" / "geo" / "admin" / f"BGD_{level.upper()}" / fname
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_choropleth(predictions: pd.DataFrame, stations: pd.DataFrame, level: str):
    """Build a Plotly choropleth figure for admin level.

    Joins predictions to station admin info, aggregates worst band per polygon,
    and colors each polygon by its worst valid station risk.
    """
    if level == "adm1":
        admin_col = "division_norm"
        norm_map = DIVISION_NORM
        geojson = _load_geojson("adm1")
    else:
        admin_col = "district_norm"
        norm_map = DISTRICT_NORM
        geojson = _load_geojson("adm2")

    if geojson is None:
        return None

    # Join predictions to station admin info
    st_map = stations.copy()
    st_map["division_norm"] = st_map["division"].apply(lambda x: _norm_name(x, DIVISION_NORM))
    st_map["district_norm"] = st_map["district"].apply(lambda x: _norm_name(x, DISTRICT_NORM))

    df = predictions.merge(st_map, on="station_id", how="inner")
    if df.empty:
        return None

    # Aggregate worst band per admin polygon
    def worst_band(series):
        if len(series) == 0:
            return "unknown"
        sev = series.map(BAND_SEVERITY).fillna(-1)
        worst_idx = sev.idxmax()
        return series.loc[worst_idx]

    agg = (
        df.groupby(admin_col)
        .agg(
            worst_band=("band", worst_band),
            max_probability=("probability", "max"),
            station_count=("station_id", "nunique"),
        )
        .reset_index()
    )

    agg_lookup = {}
    for _, r in agg.iterrows():
        agg_lookup[r[admin_col]] = {
            "worst_band": r["worst_band"],
            "max_probability": float(r["max_probability"]),
            "station_count": int(r["station_count"]),
        }

    # Build Plotly choropleth
    feature_ids = []
    feature_names = []
    feature_bands = []
    feature_probs = []
    feature_counts = []

    for feature in geojson["features"]:
        shape_name = feature["properties"].get("shapeName", "")
        fid = shape_name
        feature_ids.append(fid)
        feature_names.append(shape_name)
        if fid in agg_lookup:
            info = agg_lookup[fid]
            feature_bands.append(info["worst_band"])
            feature_probs.append(info["max_probability"])
            feature_counts.append(info["station_count"])
        else:
            feature_bands.append("no_data")
            feature_probs.append(None)
            feature_counts.append(0)

    band_severity_map = {
        "no_data": -1,
        "unknown": -1,
        "low": 0,
        "moderate": 1,
        "high": 2,
        "very_high": 3,
    }
    severity_values = [band_severity_map.get(b, -1) for b in feature_bands]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=feature_ids,
            z=severity_values,
            featureidkey="properties.shapeName",
            colorscale=[
                [0, "#D9E2EC"],   # no data / unknown
                [0.25, "#218739"], # low
                [0.5, "#8A6200"],  # moderate
                [0.75, "#B54708"], # high
                [1.0, "#B42318"],  # very_high
            ],
            zmin=-1,
            zmax=3,
            text=[
                f"{name}<br>Band: {BAND_LABEL.get(b, 'Unknown')}<br>"
                f"Max probability: {('—' if p is None else f'{p:.1%}')}<br>"
                f"Stations: {c}"
                for name, b, p, c in zip(feature_names, feature_bands, feature_probs, feature_counts)
            ],
            hoverinfo="text",
            showscale=False,
            marker_line_color="#FFFFFF",
            marker_line_width=1,
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # Sidebar: view selector + status
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:16px 0;">
                <h2 style="font-size:20px;font-weight:700;color:var(--fg-navy);margin:0 0 4px 0;">{APP_NAME}</h2>
                <p style="font-size:12px;color:#52616B;margin:0;">{APP_TAGLINE}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        view = st.radio("View", ["Citizen", "Admin"], index=0, key="main_view")
        st.markdown("---")
        st.markdown('<div class="fg-card-header">SYSTEM STATUS</div>', unsafe_allow_html=True)
        render_status_strip()
        st.markdown("---")
        st.markdown('<div class="fg-disclaimer" style="font-size:11px;">' + DISCLAIMER.replace("**", "") + '</div>', unsafe_allow_html=True)

    # Main content
    render_header()
    render_status_strip()

    if view == "Citizen":
        citizen_view()
    else:
        admin_view()

if __name__ == "__main__":
    main()


