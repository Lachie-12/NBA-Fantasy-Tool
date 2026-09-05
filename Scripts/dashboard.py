import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from zscore_engine import build_rankings
import pull_player_stats

st.set_page_config(layout="wide", page_title="NBA Fantasy Rankings")

# --- Trim Streamlit's default padding so content uses more of the screen ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Data paths ---
RAW_STATS_PATH = Path(__file__).parent / "player_stats_2025-26.csv"

# How old the CSV is allowed to get before we bother re-pulling on launch.
# Tweak this freely -- lower = fresher data but more NBA.com calls,
# higher = fewer calls but a longer window where you could be stale.
STALE_AFTER_HOURS = 6

# The Punt dropdown shows friendly labels; the engine's ALL_CATS uses a
# couple of different internal names (FG_PCT / FT_PCT instead of FG% / FT%).
# Everything else already matches, but we map all of them explicitly so this
# doesn't silently break if either side's naming changes later.
PUNT_LABEL_TO_CAT = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "FG3M": "FG3M",
    "TOV": "TOV",
    "FG%": "FG_PCT",
    "FT%": "FT_PCT",
}


def _is_stale(path: Path, hours: float) -> bool:
    """No file at all counts as stale (forces a first-time pull)."""
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > hours


# --- Cached data loading and ranking computation ---
# Streamlit reruns this whole script on every widget interaction, so without
# caching, changing an unrelated filter (like Number of Players) would
# needlessly redo the full z-score calculation. Splitting this into two
# cached steps means:
#   - the CSV is only ever read from disk once per app launch
#   - the refresh-on-launch check below therefore only fires once per
#     launch too, NOT once per dropdown click, since load_raw_stats()
#     itself is cached and won't re-run just because a widget changed
#   - rankings are only recomputed when the punt selection actually changes
@st.cache_data
def load_raw_stats() -> pd.DataFrame:
    if _is_stale(RAW_STATS_PATH, STALE_AFTER_HOURS):
        try:
            with st.spinner("Refreshing player stats from NBA.com..."):
                pull_player_stats.main()
        except Exception as e:
            # Don't crash the dashboard over a failed refresh -- fall back
            # to whatever's already on disk (if anything) and surface a
            # visible warning instead. The "last updated" line further
            # down does the rest of the job of making staleness obvious.
            if RAW_STATS_PATH.exists():
                st.warning(f"Couldn't refresh data from NBA.com ({e}). Showing last saved data instead.")
            else:
                st.error(f"No cached data exists and the refresh failed: {e}")
                st.stop()
    return pd.read_csv(RAW_STATS_PATH)


@st.cache_data
def compute_rankings(punt_category: str | None) -> pd.DataFrame:
    raw = load_raw_stats()
    weights = {punt_category: 0} if punt_category else None
    # total_col fixed at "TOTAL_Z" regardless of punt, so the dashboard
    # never has to branch on column naming depending on punt state.
    return build_rankings(raw, weights=weights, total_col="TOTAL_Z")


raw_stats = load_raw_stats()

# --- 1. Header, top-left, with a subtle last-updated indicator directly beneath it ---
last_updated = datetime.fromtimestamp(RAW_STATS_PATH.stat().st_mtime).strftime("%d %b %Y, %I:%M %p")
st.markdown(f"""
    <h3 style="margin-bottom: 0.1rem;">NBA Fantasy Rankings</h3>
    <p style="font-size: 10px; color: #b0b0b0; margin-top: 0; margin-bottom: 0.3rem;">Data last updated: {last_updated}</p>
""", unsafe_allow_html=True)    

# --- 2. Filters row ---
filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

with filter_col1:
    time_period = st.selectbox("Time Period", ["Season", "Last 30 Days", "Last 14 Days", "Last 7 Days"])

with filter_col2:
    punt = st.selectbox("Punt", ["None", "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG%", "FT%"])

with filter_col4:
    position = st.selectbox("Position", ["All Positions", "PG", "SG", "SF", "PF", "C"])

with filter_col5:
    num_players = st.selectbox("Number of Players", ["150", "200", "All Players"])

# --- Resolve punt selection to an engine category, then compute rankings ---
punt_category = PUNT_LABEL_TO_CAT.get(punt) if punt != "None" else None
zscores = compute_rankings(punt_category)

# --- Merge raw stats with the live-computed z-scores on PLAYER_ID ---
merged = pd.merge(
    raw_stats,
    zscores,
    on="PLAYER_ID",
    suffixes=("_RAW", "_Z")
)

# --- Build the column order: Rank, Player, Team, GP, raw stats, z-scores ---
display_df = merged[[
    "RANK", "PLAYER_NAME_RAW", "TEAM_ABBREVIATION_RAW", "GP_RAW",
    "PTS_RAW", "REB_RAW", "AST_RAW", "STL_RAW", "BLK_RAW", "FG3M_RAW", "TOV_RAW", "FG_PCT_RAW", "FT_PCT_RAW",
    "PTS_Z", "REB_Z", "AST_Z", "STL_Z", "BLK_Z", "FG3M_Z", "TOV_Z", "FG_PCT_Z", "FT_PCT_Z",
    "TOTAL_Z"
]].rename(columns={
    "PLAYER_NAME_RAW": "PLAYER_NAME",
    "TEAM_ABBREVIATION_RAW": "TEAM",
    "GP_RAW": "GP",
    "PTS_RAW": "PTS",
    "REB_RAW": "REB",
    "AST_RAW": "AST",
    "STL_RAW": "STL",
    "BLK_RAW": "BLK",
    "FG3M_RAW": "3s",
    "TOV_RAW": "TO",
    "FG_PCT_RAW": "FG%",
    "FT_PCT_RAW": "FT%",
    "FG3M_Z": "3s_Z",
    "TOV_Z": "TO_Z",
    "FG_PCT_Z": "FG%_Z",
    "FT_PCT_Z": "FT%_Z",
})

# Keep default sort by TOTAL_Z (RANK already reflects this order)
display_df = display_df.sort_values("RANK")

with filter_col3:
    teams = ["All Teams"] + sorted(display_df["TEAM"].unique().tolist())
    team = st.selectbox("Team", teams)

# --- Apply filters to a working copy of the data ---
filtered_df = display_df.copy()

# Team filter
if team != "All Teams":
    filtered_df = filtered_df[filtered_df["TEAM"] == team]

# Number of Players filter (applied last, after other filters, since it's a row-count cap)
if num_players != "All Players":
    filtered_df = filtered_df.head(int(num_players))

# --- Z-score color tiers (calculated on the top 200 players by rank, not the full pool) ---
Z_COLUMNS = ["PTS_Z", "REB_Z", "AST_Z", "STL_Z", "BLK_Z", "3s_Z", "TO_Z", "FG%_Z", "FT%_Z", "TOTAL_Z"]

# Relevant player pool for percentile calculations
color_pool = display_df.sort_values("RANK").head(200)
def tier_color(pct):
    if pd.isna(pct):
        return ""  # player outside top 200 - no color
    elif pct >= 0.90:
        return "background-color: #1e7d32; color: white"   # Elite
    elif pct >= 0.70:
        return "background-color: #a5d6a7"                  # Good
    elif pct >= 0.30:
        return ""                                            # Neutral - no color
    elif pct >= 0.10:
        return "background-color: #ef9a9a"                  # Bad
    else:
        return "background-color: #c62828; color: white"   # Very Bad

# Percentile rank (0 to 1), calculated only within the top-200 pool
pool_percentiles = color_pool[Z_COLUMNS].rank(pct=True)

# Reindex back to the full display_df shape so lookups by row index still work,
# players outside the top 200 will get NaN (no color)
percentiles = pool_percentiles.reindex(display_df.index)

def style_row(row):
    styles = []
    for col in filtered_df.columns:
        if col in Z_COLUMNS:
            pct = percentiles.loc[row.name, col]
            styles.append(tier_color(pct))
        else:
            styles.append("")
    return styles

styled = filtered_df.style.apply(style_row, axis=1)

# Round all numeric columns to 2 decimal places for display
numeric_cols = filtered_df.select_dtypes(include="number").columns
styled = styled.format(precision=2, subset=numeric_cols)

# --- 3. Table filling remaining space ---
st.dataframe(styled, use_container_width=True, height=650, hide_index=True)