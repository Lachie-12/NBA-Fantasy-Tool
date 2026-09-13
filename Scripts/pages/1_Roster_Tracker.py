"""
Roster Tracker
--------------
Lets you assign players to your 10 fantasy teams, mark players as IR
(excluded from team totals), and compare each team's combined raw stat
output in a League Totals table.

Design notes (the "why"):

- TEAM_SLOT (1-10) is the stable identity for a team, exactly the same
  role PLAYER_ID plays for a player elsewhere in this project (see
  zscore_engine.py / dashboard.py). TEAM_NAME is just a label you can
  rename freely -- nothing else keys off it, so renaming a team never
  breaks a join or loses a roster.

- This page reads the SAME player_stats_2025-26.csv the main rankings
  dashboard uses. No new data pull, no separate refresh logic -- a
  roster is just a layer of "which team is this player assigned to"
  sitting on top of the existing stat pull.

- Team totals in the League Totals tab are RAW stat sums (matches what
  actually decides a head-to-head matchup), not z-scores. FG%/FT% are
  volume-weighted (sum of makes / sum of attempts across the roster),
  not a naive average of percentages -- same reasoning as
  zscore_engine.py's compute_pct_impact(): a bench player shooting 1/1
  shouldn't count for as much as a starter shooting 8/16.

- IR players stay visible in their team's table (greyed out) but are
  excluded from that team's contribution to League Totals. Streamlit's
  editable table widget (data_editor) can't apply per-row styling --
  only the read-only st.dataframe can, via the same pandas Styler
  approach the main dashboard already uses for its green/red tiering.
  So this page shows the roster as a styled read-only table, with IR
  toggling and removing handled by small multiselect controls
  underneath, rather than editing directly inside the table.

- MATCH UP TAB: shows a head-to-head category comparison between "My
  Team" and whoever you're facing that week, reusing
  compute_team_totals() rather than duplicating any stat logic -- it's
  the exact same League Totals numbers, just filtered to two teams and
  scored category-by-category.

  Two settings with different lifespans get stored in a new
  matchup_settings.csv, same small-file pattern as teams.csv /
  roster_assignments.csv: MY_TEAM_SLOT is set once and holds for the
  whole season (you are always Team 3, say), while OPPONENT_SLOT
  changes weekly and is only remembered until you next change it.
  There is deliberately no "automated schedule" here -- this is a
  hobby tool with no connection to whatever platform (ESPN, etc.) your
  real league runs on, so who you're facing is something you tell it,
  not something it looks up.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Roster Tracker")

# --- Data paths (Scripts/ folder -- one level up from this pages/ file,
# same folder the main dashboard.py and its CSVs live in) ---
DATA_DIR = Path(__file__).parent.parent
RAW_STATS_PATH = DATA_DIR / "player_stats_2025-26.csv"
TEAMS_PATH = DATA_DIR / "teams.csv"
ROSTER_PATH = DATA_DIR / "roster_assignments.csv"
MATCHUP_PATH = DATA_DIR / "matchup_settings.csv"

NUM_TEAMS = 10

RAW_DISPLAY_COLS = [
    "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN",
    "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG_PCT", "FT_PCT",
]

# The 9 head-to-head categories, in display order. "lower_is_better" flips
# the winner comparison for TOV -- everywhere else, higher wins.
MATCHUP_CATS = [
    ("PTS", "PTS", False),
    ("REB", "REB", False),
    ("AST", "AST", False),
    ("STL", "STL", False),
    ("BLK", "BLK", False),
    ("FG3M", "3s", False),
    ("TOV", "TO", True),
    ("FG_PCT", "FG%", False),
    ("FT_PCT", "FT%", False),
]


# ---------------------------------------------------------------------------
# Data loading / saving
# ---------------------------------------------------------------------------

@st.cache_data
def load_raw_stats() -> pd.DataFrame:
    return pd.read_csv(RAW_STATS_PATH)


def load_teams() -> pd.DataFrame:
    if TEAMS_PATH.exists():
        return pd.read_csv(TEAMS_PATH)
    teams = pd.DataFrame({
        "TEAM_SLOT": range(1, NUM_TEAMS + 1),
        "TEAM_NAME": [f"Team {i}" for i in range(1, NUM_TEAMS + 1)],
    })
    teams.to_csv(TEAMS_PATH, index=False)
    return teams


def save_teams(teams: pd.DataFrame) -> None:
    teams.to_csv(TEAMS_PATH, index=False)


def load_roster() -> pd.DataFrame:
    if ROSTER_PATH.exists():
        roster = pd.read_csv(ROSTER_PATH)
        roster["IS_IR"] = roster["IS_IR"].astype(bool)
        return roster
    roster = pd.DataFrame(columns=["TEAM_SLOT", "PLAYER_ID", "IS_IR"])
    roster.to_csv(ROSTER_PATH, index=False)
    return roster


def save_roster(roster: pd.DataFrame) -> None:
    roster.to_csv(ROSTER_PATH, index=False)


def load_matchup_settings() -> dict:
    """MY_TEAM_SLOT persists all season; OPPONENT_SLOT persists week to
    week. Both are None until first set. Stored as a single-row CSV so
    the on-disk shape stays consistent with teams.csv / roster
    assignments even though there's only ever one row."""
    if MATCHUP_PATH.exists():
        df = pd.read_csv(MATCHUP_PATH)
        row = df.iloc[0]
        my_team = None if pd.isna(row["MY_TEAM_SLOT"]) else int(row["MY_TEAM_SLOT"])
        opponent = None if pd.isna(row["OPPONENT_SLOT"]) else int(row["OPPONENT_SLOT"])
        return {"MY_TEAM_SLOT": my_team, "OPPONENT_SLOT": opponent}
    settings = pd.DataFrame([{"MY_TEAM_SLOT": pd.NA, "OPPONENT_SLOT": pd.NA}])
    settings.to_csv(MATCHUP_PATH, index=False)
    return {"MY_TEAM_SLOT": None, "OPPONENT_SLOT": None}


def save_matchup_settings(my_team_slot, opponent_slot) -> None:
    settings = pd.DataFrame([{"MY_TEAM_SLOT": my_team_slot, "OPPONENT_SLOT": opponent_slot}])
    settings.to_csv(MATCHUP_PATH, index=False)


# ---------------------------------------------------------------------------
# Pure logic -- deliberately kept Streamlit-free so it can be unit tested
# on its own (see test_roster_logic.py), same "sandbox test before
# delivering" approach used for the rest of this project.
# ---------------------------------------------------------------------------

def get_available_players(raw_stats: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Players not currently assigned to ANY team -- a player can only be
    rostered once across the whole league, same as a real fantasy league."""
    rostered_ids = set(roster["PLAYER_ID"])
    return raw_stats[~raw_stats["PLAYER_ID"].isin(rostered_ids)].sort_values("PLAYER_NAME")


def compute_team_totals(raw_stats: pd.DataFrame, roster: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per team: raw stat sums across ACTIVE (non-IR) rostered
    players. FG%/FT% are volume-weighted (sum makes / sum attempts),
    not averaged -- see module docstring."""
    active = roster[~roster["IS_IR"]]
    merged = active.merge(raw_stats, on="PLAYER_ID", how="left")

    sum_cols = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FGM", "FGA", "FTM", "FTA"]
    totals = merged.groupby("TEAM_SLOT")[sum_cols].sum()

    totals["FG_PCT"] = totals["FGM"] / totals["FGA"]
    totals["FT_PCT"] = totals["FTM"] / totals["FTA"]
    totals = totals.drop(columns=["FGM", "FGA", "FTM", "FTA"]).reset_index()

    totals = teams.merge(totals, on="TEAM_SLOT", how="left")

    # Teams with nobody rostered yet (or nobody off IR) get NaN sums --
    # show as 0 rather than a blank row.
    numeric_cols = [c for c in totals.columns if c not in ("TEAM_SLOT", "TEAM_NAME")]
    totals[numeric_cols] = totals[numeric_cols].fillna(0)

    return totals.round(3)


def build_matchup_comparison(totals: pd.DataFrame, my_slot: int, opp_slot: int) -> pd.DataFrame:
    """One row per category, one column per team, plus a WINNER column.
    Pulls both rows straight out of the already-computed League Totals
    (compute_team_totals) -- no separate stat aggregation here, so
    Match Up can never disagree with League Totals about a team's own
    numbers."""
    my_row = totals[totals["TEAM_SLOT"] == my_slot].iloc[0]
    opp_row = totals[totals["TEAM_SLOT"] == opp_slot].iloc[0]

    records = []
    for raw_col, label, lower_is_better in MATCHUP_CATS:
        my_val = my_row[raw_col]
        opp_val = opp_row[raw_col]
        if my_val == opp_val:
            winner = "Tie"
        elif (my_val < opp_val) == lower_is_better:
            winner = my_row["TEAM_NAME"]
        else:
            winner = opp_row["TEAM_NAME"]
        records.append({
            "Category": label,
            my_row["TEAM_NAME"]: my_val,
            opp_row["TEAM_NAME"]: opp_val,
            "Winner": winner,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

raw_stats = load_raw_stats()
teams = load_teams()
roster = load_roster()

st.markdown("### Roster Tracker")

teams_sorted = teams.sort_values("TEAM_SLOT").reset_index(drop=True)
tab_labels = teams_sorted["TEAM_NAME"].tolist() + ["League Totals", "Match Up"]
tabs = st.tabs(tab_labels)

for i, team_row in teams_sorted.iterrows():
    slot = team_row["TEAM_SLOT"]
    with tabs[i]:
        # --- Editable team name ---
        new_name = st.text_input("Team Name", value=team_row["TEAM_NAME"], key=f"name_{slot}")
        if new_name and new_name != team_row["TEAM_NAME"]:
            teams.loc[teams["TEAM_SLOT"] == slot, "TEAM_NAME"] = new_name
            save_teams(teams)
            st.rerun()

        # --- Add a player ---
        available = get_available_players(raw_stats, roster)
        if not available.empty:
            option_labels = (
                available["PLAYER_NAME"] + " (" + available["TEAM_ABBREVIATION"] + ")"
            ).tolist()
            id_lookup = dict(zip(option_labels, available["PLAYER_ID"]))

            add_col, btn_col = st.columns([4, 1])
            with add_col:
                pick = st.selectbox(
                    "Add a player", option_labels, index=None,
                    placeholder="Search for a player...", key=f"add_{slot}",
                )
            with btn_col:
                st.write("")
                if st.button("Add", key=f"addbtn_{slot}") and pick:
                    new_row = pd.DataFrame([{
                        "TEAM_SLOT": slot, "PLAYER_ID": id_lookup[pick], "IS_IR": False,
                    }])
                    roster = pd.concat([roster, new_row], ignore_index=True)
                    save_roster(roster)
                    st.rerun()
        else:
            st.caption("Every player is currently rostered on some team.")

        # --- This team's roster ---
        team_roster = roster[roster["TEAM_SLOT"] == slot]
        if team_roster.empty:
            st.caption("No players added yet.")
            continue

        team_stats = team_roster.merge(raw_stats, on="PLAYER_ID", how="left")
        team_stats = team_stats[["PLAYER_ID", "IS_IR"] + RAW_DISPLAY_COLS]
        team_stats = team_stats.rename(columns={
            "TEAM_ABBREVIATION": "TEAM", "FG3M": "3s", "TOV": "TO",
            "FG_PCT": "FG%", "FT_PCT": "FT%", "IS_IR": "IR",
        })

        def grey_if_ir(row):
            return ["opacity: 0.45"] * len(row) if row["IR"] else [""] * len(row)

        display_cols = [c for c in team_stats.columns if c != "PLAYER_ID"]
        styled = team_stats[display_cols].style.apply(grey_if_ir, axis=1)
        numeric_cols = team_stats[display_cols].select_dtypes(include="number").columns
        styled = styled.format(precision=2, subset=numeric_cols)
        st.dataframe(styled, hide_index=True, width="stretch")

        names = team_stats["PLAYER_NAME"].tolist()
        ir_col, remove_col = st.columns(2)
        with ir_col:
            ir_selection = st.multiselect(
                "On IR", names,
                default=team_stats[team_stats["IR"]]["PLAYER_NAME"].tolist(),
                key=f"ir_{slot}",
            )
        with remove_col:
            to_remove = st.multiselect("Remove from team", names, key=f"remove_{slot}")

        current_ir = set(team_stats[team_stats["IR"]]["PLAYER_NAME"])
        if set(ir_selection) != current_ir or to_remove:
            name_to_id = dict(zip(team_stats["PLAYER_NAME"], team_stats["PLAYER_ID"]))
            ir_ids = {name_to_id[n] for n in ir_selection}

            slot_mask = roster["TEAM_SLOT"] == slot
            roster.loc[slot_mask, "IS_IR"] = roster.loc[slot_mask, "PLAYER_ID"].isin(ir_ids)

            if to_remove:
                remove_ids = {name_to_id[n] for n in to_remove}
                roster = roster[~(slot_mask & roster["PLAYER_ID"].isin(remove_ids))]

            save_roster(roster)
            st.rerun()

# --- League Totals tab ---
with tabs[-2]:
    totals = compute_team_totals(raw_stats, roster, teams)
    totals_display = totals.rename(columns={
        "TEAM_NAME": "Team", "FG3M": "3s", "TOV": "TO", "FG_PCT": "FG%", "FT_PCT": "FT%",
    }).drop(columns=["TEAM_SLOT"])
    st.dataframe(totals_display, hide_index=True, width="stretch")
    st.caption("Click a column header to sort. IR players are excluded from these totals.")

# --- Match Up tab ---
with tabs[-1]:
    matchup_settings = load_matchup_settings()
    team_options = dict(zip(teams_sorted["TEAM_NAME"], teams_sorted["TEAM_SLOT"]))
    team_names = teams_sorted["TEAM_NAME"].tolist()

    # --- My Team: set once, holds all season. Shown as a locked caption
    # once set, with a small expander to change it -- deliberately NOT a
    # dropdown sitting in the main flow every visit, since re-picking it
    # weekly would be the wrong mental model (see module docstring). ---
    if matchup_settings["MY_TEAM_SLOT"] is None:
        st.info("Pick which team is yours. This is remembered for the rest of the season.")
        my_pick = st.selectbox("My Team", team_names, index=None, placeholder="Select your team...")
        if my_pick:
            save_matchup_settings(team_options[my_pick], matchup_settings["OPPONENT_SLOT"])
            st.rerun()
        st.stop()

    my_slot = matchup_settings["MY_TEAM_SLOT"]
    my_name = teams_sorted.loc[teams_sorted["TEAM_SLOT"] == my_slot, "TEAM_NAME"].iloc[0]

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.caption(f"My Team: **{my_name}**")
    with top_col2:
        with st.expander("Change"):
            new_my_pick = st.selectbox(
                "My Team", team_names,
                index=team_names.index(my_name), key="change_my_team",
            )
            if st.button("Save", key="save_my_team") and team_options[new_my_pick] != my_slot:
                save_matchup_settings(team_options[new_my_pick], matchup_settings["OPPONENT_SLOT"])
                st.rerun()

    # --- Opponent: changes weekly, defaults to whatever was last saved ---
    opponent_names = [n for n in team_names if n != my_name]
    default_opp_slot = matchup_settings["OPPONENT_SLOT"]
    default_opp_name = None
    if default_opp_slot is not None:
        matches = teams_sorted.loc[teams_sorted["TEAM_SLOT"] == default_opp_slot, "TEAM_NAME"]
        if not matches.empty:
            default_opp_name = matches.iloc[0]

    opp_pick = st.selectbox(
        "This Week's Opponent", opponent_names,
        index=opponent_names.index(default_opp_name) if default_opp_name in opponent_names else None,
        placeholder="Select this week's opponent...",
    )

    if opp_pick and team_options[opp_pick] != default_opp_slot:
        save_matchup_settings(my_slot, team_options[opp_pick])
        st.rerun()

    if not opp_pick:
        st.caption("Select an opponent to see the category comparison.")
        st.stop()

    # --- Comparison table ---
    totals = compute_team_totals(raw_stats, roster, teams)
    comparison = build_matchup_comparison(totals, my_slot, team_options[opp_pick])

    def highlight_winner(row):
        styles = [""] * len(row)
        if row["Winner"] == my_name:
            styles[list(row.index).index(my_name)] = "background-color: #1e7d32; color: white"
        elif row["Winner"] == opp_pick:
            styles[list(row.index).index(opp_pick)] = "background-color: #1e7d32; color: white"
        return styles

    styled_comparison = comparison.style.apply(highlight_winner, axis=1)
    numeric_cols = comparison.select_dtypes(include="number").columns
    styled_comparison = styled_comparison.format(precision=2, subset=numeric_cols)
    st.dataframe(styled_comparison, hide_index=True, width="stretch")

    my_cat_wins = (comparison["Winner"] == my_name).sum()
    opp_cat_wins = (comparison["Winner"] == opp_pick).sum()
    st.caption(f"Projected category count: **{my_name} {my_cat_wins} - {opp_cat_wins} {opp_pick}** "
               f"(based on current season totals, not adjusted for the matchup week specifically).")