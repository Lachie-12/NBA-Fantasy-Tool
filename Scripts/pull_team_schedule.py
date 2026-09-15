"""
Phase 4 - Weekly Team Schedule Pull
------------------------------------
Pulls the NBA schedule for the CURRENT fantasy week (Monday-Sunday,
US/Eastern -- matches the ESPN league's real week boundary, confirmed
directly rather than derived from NBA data) and saves one row per team
per scheduled game. This feeds the Weekly Projection table on the
Match Up tab: season_avg x games_scheduled.

Why ScheduleLeagueV2 instead of LeagueGameLog?
LeagueGameLog (used by pull_game_logs.py) only returns games that have
ALREADY been played -- there's no way to ask it for future games.
ScheduleLeagueV2 returns the full season's schedule up front, past and
future alike, which is exactly what "games scheduled this week" needs.
It's a single call for the whole season (no monthly chunking needed
like pull_game_logs.py's first-run backfill) -- a schedule is a much
smaller payload than full box-score game logs, so it doesn't hit the
same timeout problem.

Why per-team-per-game rows instead of a single GAMES_SCHEDULED count?
Storing one row per (team, game date) instead of a flat integer keeps
the door open for a future feature we discussed but deliberately
scoped OUT of this one: showing "games already played (with actual
stats so far) + games remaining (projected)" mid-week, so an early
outlier game isn't silently smoothed away by a flat weekly average.
That feature can filter this same table by GAME_DATE <= today without
needing a different pull or a schema change. This script itself only
sums rows to a per-team count -- the date-level detail is pure
future-proofing, not used by the Weekly Projection feature yet.

Manual per-player overrides (a different concept: "I don't think this
guy plays both ends of that back-to-back") do NOT live in this file --
they belong in a separate games_overrides.csv keyed by
(WEEK_START_DATE, PLAYER_ID), specifically so that re-running this
script never has a chance to clobber them. This script has no
awareness that overrides exist at all.

Fantasy week is fixed Monday-Sunday, US/Eastern. This script always
pulls the CURRENT week relative to real time when it's run -- there's
no "look up a different week" option, since the Weekly Projection
feature only ever cares about "right now."
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

# --- Settings you may want to tweak ---
SEASON = "2025-26"
LEAGUE_ID = "00"  # NBA -- same convention nba_api uses elsewhere

FANTASY_TZ = ZoneInfo("America/New_York")  # ESPN week boundary reference

# The real 30 NBA franchises -- used to filter out preseason exhibition
# games and All-Star Weekend fixtures, which ScheduleLeagueV2 includes
# alongside real games but which use non-franchise tricodes (see
# build_week_schedule's docstring/comment for real examples pulled from
# this project's own data).
REAL_NBA_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}

# Same retry/backoff pattern as pull_player_stats.py -- stats.nba.com is
# flaky regardless of which endpoint you're hitting.
MAX_RETRIES = 3
REQUEST_TIMEOUT = 45
RETRY_BACKOFF_SECONDS = 5

# Only the columns this script actually needs out of ScheduleLeagueV2's
# much wider SeasonGames dataset (broadcaster info, arena details, etc.
# all dropped here -- not relevant to "how many games does this team
# have this week").
KEEP_COLUMNS = [
    "gameDate", "gameStatus", "postponedStatus",
    "homeTeam_teamTricode", "awayTeam_teamTricode",
]


def get_current_week_bounds(today: date | None = None) -> tuple[date, date]:
    """
    Monday-Sunday week containing `today` (defaults to real "now" in
    US/Eastern). `today` is exposed as a parameter purely so this can be
    unit-tested deterministically -- production calls always use the
    default (real "now").
    """
    if today is None:
        today = datetime.now(FANTASY_TZ).date()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)              # Sunday
    return week_start, week_end


def _fetch_full_schedule(season: str, timeout: int) -> pd.DataFrame:
    response = scheduleleaguev2.ScheduleLeagueV2(
        league_id=LEAGUE_ID,
        season=season,
        timeout=timeout,
    )
    df = response.get_data_frames()[0]  # SeasonGames dataset
    return df[KEEP_COLUMNS]


def pull_full_schedule(season: str = SEASON) -> pd.DataFrame:
    """Fetch the full-season schedule, past and future games alike."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _fetch_full_schedule(season, REQUEST_TIMEOUT)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"Attempt {attempt}/{MAX_RETRIES} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Attempt {attempt}/{MAX_RETRIES} failed ({e}). No retries left.")

    # All attempts exhausted -- raise so the caller (e.g. a future
    # dashboard refresh-on-launch step, same pattern as
    # pull_player_stats.py) can catch it and fall back to cached data.
    raise last_error


def build_week_schedule(full_schedule: pd.DataFrame, week_start: date, week_end: date) -> pd.DataFrame:
    """
    Filters the full schedule down to the given week and reshapes to
    one row per (team, game date) -- each game produces two rows (home
    team + away team), since a fantasy team only cares about ITS games,
    not both sides of every matchup.
    """
    df = full_schedule.copy()

    # gameDate comes back as "MM/DD/YYYY 00:00:00" (confirmed against a
    # real pull). Format given explicitly rather than left to pandas'
    # inference -- that format is genuinely ambiguous for any day <=12
    # (e.g. is "01/02/2026" Jan 2nd or Feb 1st?), so guessing wrong here
    # could silently shift games into the wrong week without ever
    # raising an error.
    df["GAME_DATE"] = pd.to_datetime(df["gameDate"], format="%m/%d/%Y %H:%M:%S").dt.date

    # Drop postponed games -- postponedStatus is the literal string "N"
    # for a normal game and "Y" when a game won't actually happen on this
    # date (confirmed against a real pull; NOT documented anywhere, and
    # NOT the blank/NaN convention used elsewhere in this project, e.g.
    # pull_game_logs.py). fillna("N") treats a genuinely missing value as
    # "not postponed" rather than crashing on it. Without this filter, a
    # rescheduled game could silently double-count (once on its original
    # date, once on the makeup date) if both happened to fall in-season.
    df = df[df["postponedStatus"].fillna("N").str.upper() != "Y"]

    # Drop preseason exhibition games (e.g. an NBA team playing an
    # international club during the October preseason window) and
    # All-Star Weekend fixtures (Rising Stars / Celebrity Game / All-Star
    # Game, which use temporary draft-style team codes instead of real
    # franchises) -- confirmed against a real pull, where these showed up
    # as tricodes like "MEL", "GUA", "HAP" in October and "TMC", "VIN",
    # "STR", "WLD", "STP", "AUS" in mid-February. Neither counts as a
    # real team fixture for fantasy purposes, so a row only survives if
    # BOTH sides are one of the real 30 NBA teams -- catches both cases
    # in one filter without needing to separately detect "preseason" vs
    # "All-Star" game types.
    before_count = len(df)
    df = df[
        df["homeTeam_teamTricode"].isin(REAL_NBA_TEAMS)
        & df["awayTeam_teamTricode"].isin(REAL_NBA_TEAMS)
    ]
    dropped = before_count - len(df)
    if dropped:
        print(f"  (dropped {dropped} non-regular-season rows: preseason exhibitions / All-Star weekend)")

    in_week = df[(df["GAME_DATE"] >= week_start) & (df["GAME_DATE"] <= week_end)]

    home_rows = in_week[["homeTeam_teamTricode", "GAME_DATE"]].rename(
        columns={"homeTeam_teamTricode": "TEAM_ABBREVIATION"}
    )
    away_rows = in_week[["awayTeam_teamTricode", "GAME_DATE"]].rename(
        columns={"awayTeam_teamTricode": "TEAM_ABBREVIATION"}
    )

    week_schedule = pd.concat([home_rows, away_rows], ignore_index=True)
    week_schedule.insert(1, "WEEK_START_DATE", week_start)
    week_schedule = week_schedule.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    return week_schedule


def main():
    week_start, week_end = get_current_week_bounds()
    print(f"Current fantasy week: {week_start} to {week_end} (Mon-Sun, US/Eastern)")

    print(f"Pulling {SEASON} full-season schedule from nba_api...")
    full_schedule = pull_full_schedule()
    print(f"Retrieved {len(full_schedule)} total scheduled games for the season.")

    week_schedule = build_week_schedule(full_schedule, week_start, week_end)
    print(f"{len(week_schedule)} team-game rows fall within the current week.")

    output_path = Path(__file__).parent / "team_games_this_week.csv"
    week_schedule.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    print("\nGames scheduled this week, by team:")
    counts = week_schedule.groupby("TEAM_ABBREVIATION").size().sort_values(ascending=False)
    print(counts.to_string())


if __name__ == "__main__":
    main()