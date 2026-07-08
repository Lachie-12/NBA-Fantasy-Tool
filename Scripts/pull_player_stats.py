"""
Phase 1 - Data Pull
--------------------
Pulls current-season (2025-26) per-game stats for every NBA player who has
played, using the nba_api library. Saves the result as a CSV that we'll use
in the next step to build the z-score ranking engine.

Why LeagueDashPlayerStats?
This endpoint returns stats the NBA has already aggregated per player, so we
don't need to pull individual game logs and sum them ourselves. That keeps
this first step simple and fast. We'll likely need game logs later (Phase 2)
for schedule-adjusted projections and true weekly time windows, but for a
first "does the pipeline work" pass, this is the right level of complexity.
"""

from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

# --- Settings you may want to tweak ---
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"   # could also be "Playoffs" later, not relevant yet

# Columns we care about for a 9-cat league:
# - Counting stats: PTS, REB, AST, STL, BLK, FG3M, TOV
# - Percentage stats, kept as raw makes/attempts (not just the %) so we can
#   volume-weight them properly once we get to the ranking step.
# - GP and MIN so we can filter out small sample sizes / assess playing time.
KEEP_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GP", "MIN",
    "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV",
    "FGM", "FGA", "FG_PCT",
    "FTM", "FTA", "FT_PCT",
]


def pull_player_stats(season: str = SEASON) -> pd.DataFrame:
    """Fetch per-game player stats for the given season from nba_api."""
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=SEASON_TYPE,
        per_mode_detailed="PerGame",
    )
    df = response.get_data_frames()[0]
    return df[KEEP_COLUMNS]


def main():
    print(f"Pulling {SEASON} player stats from nba_api...")
    df = pull_player_stats()

    # Basic sanity filter: drop anyone who hasn't actually played a game yet
    # (e.g. injured all season so far, or on a roster but unused).
    df = df[df["GP"] > 0].copy()

    print(f"Retrieved {len(df)} players who have played at least 1 game.")

    # Save next to this script, regardless of what folder the terminal is
    # sitting in when you run it (fixes the working-directory gotcha).
    output_path = Path(__file__).parent / f"player_stats_{SEASON}.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
