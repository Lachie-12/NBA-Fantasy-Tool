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

NUM_TEAMS = 10

RAW_DISPLAY_COLS = [
    "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN",
    "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG_PCT", "FT_PCT",
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

raw_stats = load_raw_stats()
teams = load_teams()
roster = load_roster()

st.markdown("### Roster Tracker")

teams_sorted = teams.sort_values("TEAM_SLOT").reset_index(drop=True)
tab_labels = teams_sorted["TEAM_NAME"].tolist() + ["League Totals"]
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
with tabs[-1]:
    totals = compute_team_totals(raw_stats, roster, teams)
    totals_display = totals.rename(columns={
        "TEAM_NAME": "Team", "FG3M": "3s", "TOV": "TO", "FG_PCT": "FG%", "FT_PCT": "FT%",
    }).drop(columns=["TEAM_SLOT"])
    st.dataframe(totals_display, hide_index=True, width="stretch")
    st.caption("Click a column header to sort. IR players are excluded from these totals.")
