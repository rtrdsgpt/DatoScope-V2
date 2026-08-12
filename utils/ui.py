"""
Shared UI helpers and theme for the DatoScope Streamlit app.
Supports light (default) and dark modes, toggled from the sidebar.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


PALETTE = px.colors.qualitative.Bold

# ---------------------------------------------------------------------------
# Plotly theme helpers
# ---------------------------------------------------------------------------

def get_plotly_theme() -> dict:
    dark = st.session_state.get("theme", "light") == "dark"
    if dark:
        return dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.6)",
            font=dict(family="DM Sans", color="#e2e8f0"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
    else:
        return dict(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(241,245,249,0.6)",
            font=dict(family="DM Sans", color="#1e293b"),
            margin=dict(l=10, r=10, t=40, b=10),
        )


# Backwards-compatible proxy: pages that do fig.update_layout(**PLOTLY_THEME)
# will pick up the current theme at call time.
class _LazyThemeProxy(dict):
    def __getitem__(self, key):
        return get_plotly_theme()[key]
    def items(self):
        return get_plotly_theme().items()
    def __iter__(self):
        return iter(get_plotly_theme())
    def keys(self):
        return get_plotly_theme().keys()
    def values(self):
        return get_plotly_theme().values()
    def __bool__(self):
        return True


PLOTLY_THEME = _LazyThemeProxy()

# ---------------------------------------------------------------------------
# CSS – light and dark variable sets
# ---------------------------------------------------------------------------

_LIGHT_VARS = """
  :root {
    --bg: #f8fafc; --surface: #ffffff; --card: #f1f5f9; --border: #cbd5e1;
    --accent: #0284c7; --accent2: #7c3aed; --green: #059669; --amber: #d97706;
    --red: #dc2626; --text: #1e293b; --muted: #64748b;
  }
"""

_DARK_VARS = """
  :root {
    --bg: #0a0e1a; --surface: #111827; --card: #1a2235; --border: #1e3a5f;
    --accent: #00d4ff; --accent2: #7c3aed; --green: #10b981; --amber: #f59e0b;
    --red: #ef4444; --text: #e2e8f0; --muted: #64748b;
  }
"""

_SHARED_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important; color: var(--text); font-family: 'DM Sans', sans-serif;
  }
  [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
  div[data-testid="metric-container"] {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 12px; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 1.4rem;
  }
  .stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff !important;
    border: none; border-radius: 8px; font-weight: 600; font-family: 'DM Sans', sans-serif; padding: 10px 24px;
  }
  .stSelectbox > div > div, .stMultiSelect > div > div, .stSlider > div,
  .stNumberInput > div, .stRadio > div {
    background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important;
  }
  h1,h2,h3 { font-family: 'Space Mono', monospace; }
  .ds-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 20px 24px; margin-bottom: 16px;
  }
  .ds-badge {
    display:inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px;
    font-weight: 600; letter-spacing: .5px;
  }
  .ds-badge-cyan { background: #00d4ff22; color: var(--accent); border: 1px solid #00d4ff44; }
  .ds-badge-green { background: #10b98122; color: var(--green); border: 1px solid #10b98144; }
  .ds-badge-amber { background: #f59e0b22; color: var(--amber); border: 1px solid #f59e0b44; }
  .ds-badge-red { background: #ef444422; color: var(--red); border: 1px solid #ef444444; }
  .winner-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(2,132,199,0.08));
    border: 1px solid var(--green); border-radius: 12px; padding: 16px 20px; text-align: center;
  }
  .cleaned-preview-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(124,58,237,0.07));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 14px; padding: 14px 18px; margin: 10px 0 14px;
    animation: fadeUp .6s ease-out;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  hr { border-color: var(--border); }
  .theme-toggle-btn > button {
    background: var(--card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 20px !important;
    font-size: 12px !important; padding: 4px 14px !important; font-weight: 500 !important;
    background-image: none !important;
  }
"""


def _build_css(dark: bool) -> str:
    vars_block = _DARK_VARS if dark else _LIGHT_VARS
    return f"<style>\n{vars_block}\n{_SHARED_CSS}\n</style>"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

def setup_page(title: str = "DatoScope") -> None:
    st.set_page_config(
        page_title=title,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    dark = st.session_state["theme"] == "dark"
    st.markdown(_build_css(dark), unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Compact light/dark toggle rendered in the sidebar."""
    dark = st.session_state.get("theme", "light") == "dark"
    label = "☀️ Light mode" if dark else "🌙 Dark mode"
    col, _ = st.columns([1, 1])
    with col:
        st.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
        if st.button(label, key="__theme_toggle__"):
            st.session_state["theme"] = "light" if dark else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar branding
# ---------------------------------------------------------------------------

def render_sidebar_brand() -> None:
    st.markdown(
        """
    <div style='padding:8px 0 16px'>
      <div style='font-family:Space Mono,monospace;font-size:1.5rem;color:var(--accent);font-weight:700;letter-spacing:-1px;'>Dataset Processing</div>
      <div style='color:var(--muted);font-size:12px;margin-top:4px;'>Interactive ML workflow</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    render_theme_toggle()
    st.markdown("<hr style='margin:8px 0 16px'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared UI components
# ---------------------------------------------------------------------------

def render_metadata_panel(meta: dict, train_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    source = meta.get("source", "uploaded")
    split_method = meta.get("split_method", "Auto split from train file")
    target_col = meta.get("target_column", "Not selected yet")
    feature_count = meta.get("n_features", train_df.shape[1] if train_df is not None else 0)
    st.markdown("#### Dataset Metadata")
    st.markdown(
        f"""
        <div class="ds-card">
          <b>Source</b>: <span class="ds-badge ds-badge-cyan">{source.upper()}</span><br>
          <b>Train rows</b>: <code>{len(train_df) if train_df is not None else 0}</code><br>
          <b>Test rows</b>: <code>{len(test_df) if test_df is not None else 0}</code><br>
          <b>Feature count</b>: <code>{feature_count}</code><br>
          <b>Target column</b>: <code>{target_col}</code><br>
          <b>Split method</b>: <code>{split_method}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_preview_metrics(df: pd.DataFrame, label: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} rows", f"{df.shape[0]:,}")
    c2.metric(f"{label} cols", f"{df.shape[1]}")
    c3.metric(f"{label} missing", f"{df.isnull().sum().sum():,}")
    c4.metric(f"{label} duplicates", f"{df.duplicated().sum():,}")


def show_dataset_block(title: str, df: pd.DataFrame) -> None:
    st.markdown(f"#### {title}")
    left, right = st.columns(2)
    with left:
        st.dataframe(df.head(200), use_container_width=True, height=320)
    with right:
        summary = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non-null": df.notnull().sum(),
                "missing%": (df.isnull().mean() * 100).round(2),
                "unique": df.nunique(),
            }
        )
        st.dataframe(summary, use_container_width=True, height=320)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0,212,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
