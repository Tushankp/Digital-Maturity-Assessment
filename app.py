# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO

st.set_page_config(layout="wide", page_title="Digital Maturity Dashboard")

st.title("Digital Maturity Assessment — Interactive Dashboard")
st.markdown(
    "Upload a dataset with columns: `organization_id`, `dimension`, `sub_area`, `value`, `weight` "
    "or use the sample dataset provided."
)

# -----------------------
# Utilities & Defaults
# -----------------------
SAMPLE_PATH = "/mnt/data/digital_maturity_big_dataset.csv"

MATURITY_BANDS = {
    "Initial": (1.0, 2.0),
    "Developing": (2.0, 3.0),
    "Established": (3.0, 4.0),
    "Advanced": (4.0, 5.0),
}

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Loaded uploaded dataset.")
            return df
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            return None
    else:
        # Fallback to bundled sample dataset if present
        try:
            df = pd.read_csv(SAMPLE_PATH)
            st.info(f"Using bundled sample dataset at `{SAMPLE_PATH}`.")
            return df
        except FileNotFoundError:
            st.error("No sample dataset found. Please upload a CSV file.")
            return None

@st.cache_data
def compute_scores(df):
    # Ensure proper dtypes
    df = df.copy()
    # If organization_id missing, create
    if 'organization_id' not in df.columns:
        df['organization_id'] = 1

    # cast
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(1).astype(float)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0).astype(float)

    # Weighted score per row (sub-area)
    df['weighted_score'] = df['value'] * df['weight']

    # Dimension-level aggregation per organization (sum weighted scores for dimension)
    dim_scores = (
        df
        .groupby(['organization_id', 'dimension'], as_index=False)
        .agg(weighted_score=('weighted_score', 'sum'))
    )

    # Overall maturity per org (sum of dimension weighted scores). With weights summing to 1, range is 1..5
    overall = dim_scores.groupby('organization_id', as_index=False).agg(overall_maturity=('weighted_score', 'sum'))

    # Merge to get a table for display
    merged = dim_scores.merge(overall, on='organization_id', how='left')

    # Rank orgs
    overall['rank'] = overall['overall_maturity'].rank(ascending=False, method='min').astype(int)
    overall = overall.sort_values('overall_maturity', ascending=False).reset_index(drop=True)

    return df, dim_scores, overall, merged

def maturity_label(score):
    # score expected between 1 and 5
    for label, (lo, hi) in MATURITY_BANDS.items():
        # inclusive on low, exclusive on high (except last band)
        if label != "Advanced":
            if lo <= score < hi:
                return label
        else:
            if lo <= score <= hi:
                return label
    return "Unknown"

def priority_label(score):
    # Using normalized 1-5 scale: define thresholds (~ scaled similar to notebook)
    if score < 2.0:
        return "Critical"
    elif score < 3.5:
        return "Improve"
    else:
        return "Healthy"

def priority_color(score):
    pl = priority_label(score)
    if pl == "Critical":
        return "red"
    elif pl == "Improve":
        return "orange"
    else:
        return "green"

def recommend_actions(score, dimension, sub_area=""):
    pl = priority_label(score)
    if pl == "Critical":
        return f"Immediate investment in {sub_area or dimension}. Establish foundational capabilities in {dimension}."
    elif pl == "Improve":
        return f"Strengthen processes and tools in {sub_area or dimension}. Plan upgrades within 6–12 months."
    else:
        return f"Maintain and optimize {sub_area or dimension}. Consider innovation pilots."

def df_to_bytes_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# -----------------------
# Sidebar: upload & filters
# -----------------------
with st.sidebar:
    st.header("Data input / Filters")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    sample_preview = st.checkbox("Show raw dataset preview", value=False)
    df_raw = load_data(uploaded_file)

    org_list_limit = st.number_input("Max organizations to show in heatmap", min_value=10, max_value=1000, value=200, step=10)

# If no data, stop
if df_raw is None:
    st.stop()

# Optionally show raw data
if sample_preview:
    st.subheader("Raw dataset (sample)")
    st.dataframe(df_raw.head(200))

# -----------------------
# Compute scores and tables
# -----------------------
df, dim_scores, overall, merged = compute_scores(df_raw)

# Add labels
overall['maturity_label'] = overall['overall_maturity'].apply(maturity_label)
overall['priority_label'] = overall['overall_maturity'].apply(priority_label)
overall['priority_color'] = overall['overall_maturity'].apply(priority_color)

st.sidebar.markdown("## Quick stats")
st.sidebar.metric("Organizations", int(overall.shape[0]))
st.sidebar.metric("Dimensions", int(df['dimension'].nunique()))

# -----------------------
# Main dashboard layout
# -----------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Overall Rankings")
    st.dataframe(overall.sort_values('overall_maturity', ascending=False).reset_index(drop=True).head(50))

    st.markdown("---")
    st.subheader("Digital Maturity Heatmap (organizations × dimensions)")

    # Pivot for heatmap
    pivot = dim_scores.pivot(index='organization_id', columns='dimension', values='weighted_score').fillna(0)
    # Limit number of orgs displayed
    if org_list_limit and pivot.shape[0] > org_list_limit:
        pivot_display = pivot.head(org_list_limit)
    else:
        pivot_display = pivot

    fig_heat = px.imshow(
        pivot_display.values,
        labels=dict(x="Dimension", y="Organization", color="Weighted Score"),
        x=pivot_display.columns,
        y=pivot_display.index.astype(str),
        aspect="auto",
        color_continuous_scale='Blues',
        origin='lower'
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

with right_col:
    st.subheader("Select Organization")
    org_options = sorted(df['organization_id'].unique().tolist())
    selected_org = st.selectbox("Organization ID", org_options, index=0)

    # Show summary for selected org
    st.markdown("### Org Summary")
    org_overall = overall[overall['organization_id'] == selected_org].squeeze()
    if not org_overall.empty:
        st.metric("Overall Maturity", f"{org_overall['overall_maturity']:.2f}", delta=None)
        st.write("Maturity Level:", org_overall['maturity_label'])
        st.write("Priority:", org_overall['priority_label'])
    else:
        st.info("No overall score found for selected org.")

    # Radar chart for selected org
    st.markdown("### Radar Chart (dimension scores)")
    org_dim = dim_scores[dim_scores['organization_id'] == selected_org].copy()
    if org_dim.empty:
        st.info("No dimension scores found for this organization.")
    else:
        # ensure consistent order
        dims = org_dim['dimension'].tolist()
        scores = org_dim['weighted_score'].tolist()
        # Repeat first to close the radar
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dims + [dims[0]],
            fill='toself'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=450)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### Dimension contributions & priorities")
    org_contrib = org_dim.copy()
    total = org_contrib['weighted_score'].sum() if not org_contrib.empty else 1
    org_contrib['contribution_pct'] = org_contrib['weighted_score'] / total * 100
    org_contrib['priority_label'] = org_contrib['weighted_score'].apply(priority_label)
    org_contrib['recommended_action'] = org_contrib.apply(
        lambda r: recommend_actions(r['weighted_score'], r['dimension'], r.get('sub_area', "")), axis=1
    )

    st.dataframe(org_contrib[['dimension', 'weighted_score', 'contribution_pct', 'priority_label', 'recommended_action']].sort_values('weighted_score', ascending=False))

# -----------------------
# Global analytics & charts
# -----------------------
st.markdown("---")
st.subheader("Global Analytics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Maturity distribution (overall)**")
    fig_hist = px.histogram(overall, x='overall_maturity', nbins=20, title="Distribution of Overall Maturity")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("**Top/Bottom Organizations**")
    top_n = st.slider("Top / Bottom N", min_value=1, max_value=50, value=10)
    top_table = overall.head(top_n)
    bottom_table = overall.tail(top_n)
    st.write("Top organizations")
    st.dataframe(top_table)
    st.write("Bottom organizations")
    st.dataframe(bottom_table)

# -----------------------
# Roadmap export and recommendations
# -----------------------
st.markdown("---")
st.subheader("Generate Roadmap (CSV)")

# Build roadmap per organization: dimension-level priority + action
roadmap = df.copy()
roadmap['priority_label'] = roadmap['weighted_score'].apply(priority_label)
roadmap['recommended_action'] = roadmap.apply(lambda r: recommend_actions(r['weighted_score'], r['dimension'], r['sub_area']), axis=1)

# Merge with overall for convenience
roadmap = roadmap.merge(overall[['organization_id', 'overall_maturity', 'maturity_label', 'priority_label']], on='organization_id', how='left', suffixes=('', '_overall'))

# Allow user to download single org or all
download_option = st.radio("Download", ("Selected organization", "All organizations"))
if download_option == "Selected organization":
    download_df = roadmap[roadmap['organization_id'] == selected_org]
    st.write(f"Roadmap preview for org {selected_org}")
    st.dataframe(download_df[['organization_id', 'dimension', 'sub_area', 'weighted_score', 'priority_label', 'recommended_action']])
else:
    download_df = roadmap
    st.write("Roadmap preview (first 50 rows)")
    st.dataframe(download_df.head(50))

csv_bytes = df_to_bytes_csv(download_df)
st.download_button("Download roadmap CSV", data=csv_bytes, file_name="digital_maturity_roadmap.csv", mime="text/csv")

st.markdown("---")
st.info("Notes: If your input survey is in wide format (one row per org with Q1..Qn), transform it to long format "
        "(organization_id, dimension, sub_area, value, weight) before uploading. The notebook/script you provided contains "
        "helpers for this transformation and served as the basis for this dashboard. :contentReference[oaicite:1]{index=1}")

