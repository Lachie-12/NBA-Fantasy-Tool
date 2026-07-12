import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="NBA Fantasy Rankings")

# --- Trim Streamlit's default top padding so the header sits higher ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- 1. Header, top-left ---
st.markdown("### NBA Fantasy Rankings")

# --- Load data first, so filters below can be built from it ---
df = pd.read_csv("zscore_rankings_2025-26.csv")

# --- 2. Filters row ---
filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

with filter_col1:
    time_period = st.selectbox("Time Period", ["Season", "Last 30 Days", "Last 14 Days", "Last 7 Days"])

with filter_col2:
    punt = st.selectbox("Punt", ["None", "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG%", "FT%"])

with filter_col3:
    teams = ["All Teams"] + sorted(df["TEAM_ABBREVIATION"].unique().tolist())
    team = st.selectbox("Team", teams)

with filter_col4:
    position = st.selectbox("Position", ["All Positions", "PG", "SG", "SF", "PF", "C"])

with filter_col5:
    num_players = st.selectbox("Number of Players", ["150", "200", "All Players"])

# --- Apply filters to a working copy of the data ---
filtered_df = df.copy()

# Team filter
if team != "All Teams":
    filtered_df = filtered_df[filtered_df["TEAM_ABBREVIATION"] == team]

# Number of Players filter (applied last, after other filters, since it's a row-count cap)
if num_players != "All Players":
    filtered_df = filtered_df.head(int(num_players))

# --- 3. Table filling remaining space ---
st.dataframe(filtered_df, use_container_width=True, height=650, hide_index=True)