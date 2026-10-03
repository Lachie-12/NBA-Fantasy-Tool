"""
Weekly Projection - Games Scheduled + Manual Overrides (Chunk 1)
-----------------------------------------------------------------
Answers one question for every player: "how many games will this
player play in the current fantasy week?" The Weekly Projection table
(built in later chunks) multiplies a player's season averages by this
number.

Where the number comes from (two layers):

  1. BASE_GAMES  - counted automatically from team_games_this_week.csv,
                   which pull_team_schedule.py produces. Every player on
                   a team gets that team's game count (a bye team = 0).
  2. OVERRIDE    - an optional number YOU type in for one player for one
                   week ("he's resting the second half of the back-to-
                   back, so call it 2 not 3"). Stored in
                   games_overrides.csv. If an override exists it wins;
                   if not, the base count is used.

Design notes (the "why"):

- Overrides live in their OWN file, never inside team_games_this_week.csv.
  pull_team_schedule.py rewrites that file from scratch every time it
  runs, so anything you'd typed into it would be wiped. Keeping them
  separate means refreshing the schedule can never destroy an override.

- Overrides are keyed by (WEEK_START_DATE, PLAYER_ID). PLAYER_ID is the
  same stable identity used everywhere else in the project. Including
  the week means an override you set for this week quietly stops
  applying next week - no need to remember to clear it. (Old rows just
  sit in the file harmlessly; they can be pruned later.)

- An override of 0 is a real, valid value ("he's out all week") and is
  NOT the same as "no override". That's why overrides are stored as
  rows that either exist or don't, rather than as a column that is
  blank for most players.

- Everything here is pure pandas with no Streamlit. That's deliberate:
  it can be tested on its own with made-up data (see
  test_weekly_projection.py), the same approach used for the logic in
  1_Roster_Tracker.py.

- The week is passed IN (as a date) rather than worked out in here. That
  keeps these functions deterministic and testable. The real app will
  get it from pull_team_schedule.get_current_week_bounds().

File shapes:
  team_games_this_week.csv : TEAM_ABBREVIATION, WEEK_START_DATE, GAME_DATE
  games_overrides.csv      : WEEK_START_DATE, PLAYER_ID, GAMES
"""

from datetime import date
from pathlib import Path

import pandas as pd

# --- Data paths (same folder as dashboard.py and the other CSVs) ---
DATA_DIR = Path(__file__).parent
TEAM_GAMES_PATH = DATA_DIR / "team_games_this_week.csv"
OVERRIDES_PATH = DATA_DIR / "games_overrides.csv"

TEAM_GAMES_COLUMNS = ["TEAM_ABBREVIATION", "WEEK_START_DATE", "GAME_DATE"]
OVERRIDE_COLUMNS = ["WEEK_START_DATE", "PLAYER_ID", "GAMES"]

DATE_FORMAT = "%Y-%m-%d"  # how dates are written to / read from the CSVs


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def _parse_dates(series: pd.Series) -> pd.Series:
    """Explicit ISO parse (not left to pandas to guess) so a date like
    2026-10-05 can never be misread as day/month swapped."""
    return pd.to_datetime(series, format=DATE_FORMAT).dt.date


def load_team_games_this_week(path: Path = TEAM_GAMES_PATH) -> pd.DataFrame:
    """Read the schedule CSV written by pull_team_schedule.py. If it
    doesn't exist yet, return an empty table with the right columns
    (this script never creates the file - the pull script owns it)."""
    if not path.exists():
        return pd.DataFrame({
            "TEAM_ABBREVIATION": pd.Series(dtype=str),
            "WEEK_START_DATE": pd.Series(dtype=object),
            "GAME_DATE": pd.Series(dtype=object),
        })
    df = pd.read_csv(path)
    df["WEEK_START_DATE"] = _parse_dates(df["WEEK_START_DATE"])
    df["GAME_DATE"] = _parse_dates(df["GAME_DATE"])
    return df[TEAM_GAMES_COLUMNS]


def _empty_overrides() -> pd.DataFrame:
    return pd.DataFrame({
        "WEEK_START_DATE": pd.Series(dtype=object),
        "PLAYER_ID": pd.Series(dtype="int64"),
        "GAMES": pd.Series(dtype="int64"),
    })


def load_games_overrides(path: Path = OVERRIDES_PATH) -> pd.DataFrame:
    """Read games_overrides.csv. A missing file is created empty on first
    use - same pattern as load_roster() in 1_Roster_Tracker.py."""
    if not path.exists():
        empty = _empty_overrides()
        save_games_overrides(empty, path)
        return empty

    df = pd.read_csv(path)
    if df.empty:
        # A header-only CSV reads back with every column as "object" type;
        # return the properly typed empty table instead.
        return _empty_overrides()

    df["WEEK_START_DATE"] = _parse_dates(df["WEEK_START_DATE"])
    df["PLAYER_ID"] = df["PLAYER_ID"].astype("int64")
    df["GAMES"] = df["GAMES"].astype("int64")
    return df[OVERRIDE_COLUMNS]


def save_games_overrides(overrides: pd.DataFrame, path: Path = OVERRIDES_PATH) -> None:
    """Write overrides to disk. Refuses nonsense values (negative or
    fractional game counts) rather than saving them and breaking the
    projection later."""
    if not overrides.empty:
        games = overrides["GAMES"]
        if (games < 0).any() or (games != games.round()).any():
            raise ValueError("GAMES overrides must be whole numbers, 0 or greater.")

    out = overrides[OVERRIDE_COLUMNS].copy()
    if not out.empty:
        out["WEEK_START_DATE"] = pd.to_datetime(out["WEEK_START_DATE"]).dt.strftime(DATE_FORMAT)
        out["GAMES"] = out["GAMES"].astype("int64")
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def schedule_week_matches(team_games: pd.DataFrame, week_start: date) -> bool:
    """True if the schedule file actually contains games for `week_start`.

    Guards against a stale file: if you open the app on Monday but last
    ran the schedule pull the previous week, every team would silently
    show 0 games. The UI can use this to show a 'refresh the schedule'
    warning instead of quietly displaying zeros."""
    return bool((team_games["WEEK_START_DATE"] == week_start).any())


def resolve_games_scheduled(
    players: pd.DataFrame,
    team_games: pd.DataFrame,
    overrides: pd.DataFrame,
    week_start: date,
) -> pd.DataFrame:
    """
    One row per player with the final games-this-week number.

    players    : needs PLAYER_ID and TEAM_ABBREVIATION (e.g. the season
                 stats table). One row per player.
    team_games : output of load_team_games_this_week().
    overrides  : output of load_games_overrides().
    week_start : the Monday of the fantasy week being projected.

    Returns PLAYER_ID, TEAM_ABBREVIATION plus:
      BASE_GAMES      games the team has scheduled (0 if none / bye)
      OVERRIDE_GAMES  your manual number, or <NA> if you haven't set one
      IS_OVERRIDDEN   True where an override applies
      GAMES_SCHEDULED the number to actually use (override if present,
                      otherwise base)
    """
    # 1. Base: count this week's rows per team. Each row is one game.
    this_week = team_games[team_games["WEEK_START_DATE"] == week_start]
    base = this_week.groupby("TEAM_ABBREVIATION").size().rename("BASE_GAMES").reset_index()

    result = players[["PLAYER_ID", "TEAM_ABBREVIATION"]].merge(
        base, on="TEAM_ABBREVIATION", how="left"
    )
    # A team with no rows (bye week, or a team code not in the schedule)
    # comes through the merge as NaN - that genuinely means 0 games.
    result["BASE_GAMES"] = result["BASE_GAMES"].fillna(0).astype(int)

    # 2. Overrides: only rows for THIS week count. If a player somehow has
    #    two rows for the week, the last one wins.
    week_overrides = (
        overrides[overrides["WEEK_START_DATE"] == week_start][["PLAYER_ID", "GAMES"]]
        .drop_duplicates(subset="PLAYER_ID", keep="last")
        .rename(columns={"GAMES": "OVERRIDE_GAMES"})
    )
    result = result.merge(week_overrides, on="PLAYER_ID", how="left")
    result["OVERRIDE_GAMES"] = result["OVERRIDE_GAMES"].astype("Int64")  # keeps blanks as <NA>, not 0

    # 3. Final number: override where one exists, otherwise the base count.
    result["IS_OVERRIDDEN"] = result["OVERRIDE_GAMES"].notna()
    result["GAMES_SCHEDULED"] = (
        result["BASE_GAMES"]
        .where(~result["IS_OVERRIDDEN"], result["OVERRIDE_GAMES"])
        .astype(int)
    )
    return result
