"""
Phase 3 - Game Log Data Pull (incremental, chunked first pull)
------------------------------
Pulls every individual game played this season (2025-26) for every NBA
player, using nba_api's LeagueGameLog endpoint. This is the data
G-score needs that z-score doesn't -- see pull_player_stats.py for the
season-average side of the pipeline; this script exists purely to
capture the game-to-game variation that a season average erases.

Why chunk the first pull into months?
The very first run needs the whole season's games -- and by the time
you're running this, that's most of a full season's worth of data in
one response. In testing, requesting that as a single unbounded call
timed out twice in a row at exactly 90 seconds each time -- not random
flakiness, but a sign the response itself is too large for
stats.nba.com to build quickly. Splitting the first pull into one
request per calendar month keeps every individual request roughly the
same size as a normal incremental pull, which is fast and reliable, and
if one month's request fails, only that month needs retrying rather
than the whole season.

Why incremental after that?
Every run after the first only needs the games that happened since the
last save, by passing LeagueGameLog's date_from_nullable parameter.
That's already a small request on its own, so it doesn't need chunking
-- only the initial backfill does.

Why re-fetch the last saved date instead of the day after it?
If this script last ran mid-day, some games from that date (e.g. late
West Coast tip-offs) may not have been final yet. Re-requesting that
whole date and de-duplicating on (PLAYER_ID, GAME_ID) -- keeping the
newer copy -- means a partially-final day gets safely completed on the
next run, rather than silently missing games forever.

Why LeagueGameLog instead of PlayerGameLog per player?
PlayerGameLog (singular) is a per-player endpoint -- one API call per
player, which is what the old, deprecated nba_fantasy_starter.py did.
LeagueGameLog (plural / league-wide) returns one row per player per
game for the ENTIRE league in a single call, same efficiency principle
as pull_player_stats.py's LeagueDashPlayerStats choice.

What this script does NOT do yet:
It only pulls and saves the raw game-by-game rows. Turning these rows
into the actual G-score "tau" noise value -- at the game level, per our
discussion -- is the next step once this pull is confirmed working.
"""

import calendar
import time
from datetime import date
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog
import pandas as pd

# --- Settings you may want to tweak ---
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"

# Deliberately a bit earlier than the real 2025-26 tip-off (mid-to-late
# October). Requesting a date range before the season actually started
# just returns an empty chunk -- harmless -- so there's no need to get
# this exactly right, just early enough to not miss opening night.
SEASON_START_DATE = date(2025, 10, 1)

# Each monthly chunk is a similarly small request to a normal
# incremental pull, so it gets a similarly short timeout -- nowhere
# near the 90s the old single unbounded request needed (and still
# timed out at).
CHUNK_TIMEOUT = 60
INCREMENTAL_TIMEOUT = 45

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5  # base delay; doubles each retry (5s, 10s, 20s)

# Same 9-cat columns as pull_player_stats.py, plus the game-identifying
# columns we need to know WHICH game and WHEN -- essential this time,
# since the whole point is looking at variation game to game.
KEEP_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE",
    "MIN",
    "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV",
    "FGM", "FGA", "FG_PCT",
    "FTM", "FTA", "FT_PCT",
]


def _fetch_with_retries(fetch_fn, timeout: int):
    """
    Calls fetch_fn(timeout) and retries with exponential backoff on
    failure. Shared by every request this script makes -- a monthly
    backfill chunk or an incremental pull -- same retry logic either
    way, just called with a different timeout depending on request size.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_fn(timeout)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"    Attempt {attempt}/{MAX_RETRIES} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Attempt {attempt}/{MAX_RETRIES} failed ({e}). No retries left.")

    raise last_error


def _fetch_game_logs(season: str, date_from: str | None, date_to: str | None, timeout: int) -> pd.DataFrame:
    """One LeagueGameLog call. date_from/date_to are "MM/DD/YYYY"
    strings or None (None means "no bound on that side")."""
    kwargs = dict(
        season=season,
        season_type_all_star=SEASON_TYPE,
        player_or_team_abbreviation="P",  # "P" = player rows, not team rows
        timeout=timeout,
    )
    if date_from:
        kwargs["date_from_nullable"] = date_from
    if date_to:
        kwargs["date_to_nullable"] = date_to

    response = leaguegamelog.LeagueGameLog(**kwargs)
    df = response.get_data_frames()[0]
    return df[KEEP_COLUMNS]


def _month_chunks(start: date, end: date):
    """
    Yield (chunk_start, chunk_end) date pairs, one per calendar month,
    covering the range [start, end] inclusive. The final chunk is
    trimmed to `end` rather than running to the end of that calendar
    month, so we never request a date range that reaches into the
    future.
    """
    current = start
    while current <= end:
        days_in_month = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, days_in_month)
        chunk_end = min(month_end, end)
        yield current, chunk_end

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _pull_full_season_chunked(season: str) -> pd.DataFrame:
    """
    First-ever pull: request one calendar month at a time from
    SEASON_START_DATE through today, instead of one unbounded request
    for the whole season (see module docstring for why -- it reliably
    timed out at 90s in testing). Each chunk gets its own retries.
    """
    today = date.today()
    chunks = []

    for chunk_start, chunk_end in _month_chunks(SEASON_START_DATE, today):
        date_from = chunk_start.strftime("%m/%d/%Y")
        date_to = chunk_end.strftime("%m/%d/%Y")
        print(f"  Pulling {date_from} to {date_to}...")

        chunk_df = _fetch_with_retries(
            lambda t: _fetch_game_logs(season, date_from, date_to, t),
            timeout=CHUNK_TIMEOUT,
        )
        print(f"    -> {len(chunk_df)} rows")
        chunks.append(chunk_df)

    return pd.concat(chunks, ignore_index=True)


def main():
    output_path = Path(__file__).parent / f"game_logs_{SEASON}.csv"
    existing = pd.read_csv(output_path) if output_path.exists() else None

    if existing is not None and len(existing) > 0:
        # Incremental pull -- re-fetch from the latest saved date onward
        # (see module docstring for why it's that date, not date + 1).
        # Small enough on its own that it doesn't need chunking.
        latest_date = pd.to_datetime(existing["GAME_DATE"]).max().date()
        date_from = latest_date.strftime("%m/%d/%Y")
        print(f"Existing data found through {latest_date}. Pulling incrementally from {date_from}...")

        new_rows = _fetch_with_retries(
            lambda t: _fetch_game_logs(SEASON, date_from, None, t),
            timeout=INCREMENTAL_TIMEOUT,
        )
        print(f"Retrieved {len(new_rows)} game rows from this pull.")

        combined = pd.concat([existing, new_rows], ignore_index=True)
        # Keep the newer copy of any (player, game) that appears in both --
        # resolves the intentional date-of-overlap re-fetch above.
        combined = combined.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")

    else:
        print(f"No existing data found. Pulling full {SEASON} season history in monthly chunks (first run)...")
        combined = _pull_full_season_chunked(SEASON)
        print(f"Retrieved {len(combined)} total game rows across all chunks.")

    combined = combined.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    print(f"Total rows after merge: {len(combined)} ({combined['PLAYER_ID'].nunique()} players).")

    combined.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()