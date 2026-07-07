"""
NBA Fantasy Starter Script
---------------------------
Pulls recent game logs for a list of players using nba_api,
then calculates standard fantasy points for each game.

Install first:
    pip install nba_api pandas

Note: nba_api calls NBA.com's stats endpoints directly, so you need
an internet connection when running this. No API key required.
"""

import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# ---------------------------------------------------------
# 1. Look up a player's NBA "player ID" from their name
# ---------------------------------------------------------
def get_player_id(full_name: str) -> int:
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"No player found for '{full_name}'")
    return matches[0]["id"]


# ---------------------------------------------------------
# 2. Pull a player's game log for a given season
# ---------------------------------------------------------
def get_game_log(player_id: int, season: str = "2025-26") -> pd.DataFrame:
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = log.get_data_frames()[0]
    return df


# ---------------------------------------------------------
# 3. Calculate fantasy points using a standard scoring formula
#    (this matches common formats like ESPN standard scoring —
#    adjust the weights to match your own league's rules)
# ---------------------------------------------------------
def calculate_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FANTASY_PTS"] = (
        df["PTS"] * 1.0
        + df["REB"] * 1.2
        + df["AST"] * 1.5
        + df["STL"] * 3.0
        + df["BLK"] * 3.0
        - df["TOV"] * 1.0
    )
    return df


# ---------------------------------------------------------
# 4. Pull data for multiple players and combine into one table
# ---------------------------------------------------------
def build_player_comparison(player_names: list[str], season: str = "2025-26") -> pd.DataFrame:
    all_rows = []

    for name in player_names:
        print(f"Fetching data for {name}...")
        player_id = get_player_id(name)
        log = get_game_log(player_id, season)
        log = calculate_fantasy_points(log)
        log["PLAYER_NAME"] = name
        all_rows.append(log)

        # Be polite to NBA.com's servers - avoid hammering the API
        time.sleep(0.6)

    combined = pd.concat(all_rows, ignore_index=True)
    return combined


# ---------------------------------------------------------
# 5. Example: rank players by average fantasy points over
#    their last N games
# ---------------------------------------------------------
def rank_by_recent_average(df: pd.DataFrame, last_n_games: int = 10) -> pd.DataFrame:
    recent = (
        df.sort_values("GAME_DATE", ascending=False)
        .groupby("PLAYER_NAME")
        .head(last_n_games)
    )
    ranking = (
        recent.groupby("PLAYER_NAME")["FANTASY_PTS"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"FANTASY_PTS": f"AVG_FANTASY_PTS_LAST_{last_n_games}"})
    )
    return ranking


if __name__ == "__main__":
    # Swap in whichever players you want to compare
    players_to_check = [
        "Nikola Jokic",
        "Luka Doncic",
        "Shai Gilgeous-Alexander",
    ]

    all_data = build_player_comparison(players_to_check)
    rankings = rank_by_recent_average(all_data, last_n_games=10)

    print("\nFantasy Ranking (last 10 games avg):")
    print(rankings.to_string(index=False))

    # Save raw game logs and rankings to CSV for later use
    all_data.to_csv("player_game_logs.csv", index=False)
    rankings.to_csv("fantasy_rankings.csv", index=False)
