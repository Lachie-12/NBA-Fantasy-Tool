"""
Phase 3b - G-score Noise (tau) Calculation
--------------------------------------------
Uses the game-by-game data from pull_game_logs.py to compute the real,
per-category "tau" (game-to-game noise) values the G-score formula
needs -- replacing the placeholder GSCORE_SHRINKAGE weights in
scoring_engine.py, which were borrowed from a different season and pool
(see that module's docstring for the full backstory).

Recap of the formula (full detail in scoring_engine.py's docstring):
    G = (player_avg - pool_avg) / sqrt(sigma_M**2 + KAPPA * tau_M**2)

sigma_M**2 is player-to-player variance -- the exact same thing
zscore() already divides by (squared), computed from season averages,
same as it's always been.

tau_M**2 is game-to-game noise. Per the decision to keep this at the
GAME level (not week level) -- matching the existing z-score pipeline's
per-game basis, and because a league with daily roster changes makes
any single game relevant -- tau_M is computed like this for a counting
stat:
    1. For each player, take the standard deviation of their own
       individual per-game values across the season.
    2. tau_M for that category = the root-mean-square of every
       qualifying player's own std dev.

Step 2 deliberately pools everyone's personal noise into ONE number per
category, not a personalized per-player figure -- this matches the
G-score paper's own simplifying assumption (tau_M(p) ~= tau_M), not a
shortcut unique to this script. A consistently-good shot-blocker and a
streaky one still get treated the same way; see the scoring_engine.py
discussion for why that's a real limitation of G-score itself, not
something this script introduces.

FG%/FT% use the same volume-weighted "impact" idea already used at the
season level in scoring_engine.py's compute_pct_impact() -- just
computed per game instead of per season, then treated exactly like a
counting stat from there. A 0-attempt game contributes exactly 0.

Why a stricter games-played floor than the rest of the pipeline?
Estimating a *variance* reliably needs more data points than
estimating a *mean* does. A player with 10 games (scoring_engine.py's
MIN_GP) gives a fine season average, but a genuinely unreliable read on
how much they personally bounce around game to game. MIN_GP_FOR_TAU
below is intentionally stricter.
"""

from pathlib import Path

import numpy as np
import pandas as pd

MIN_GP_FOR_TAU = 20  # stricter than scoring_engine.py's MIN_GP=10 -- see module docstring
KAPPA = 1.04          # same constant scoring_engine.py's docstring already cites

COUNTING_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV"]
ALL_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG_PCT", "FT_PCT"]


def compute_game_level_impact(df: pd.DataFrame, makes_col: str, attempts_col: str) -> pd.Series:
    """
    Per-game version of scoring_engine.py's compute_pct_impact(): each
    individual game's volume-weighted impact against the SEASON league
    average rate (not that single game's own tiny-sample rate, which
    would be noisy in a different, uninteresting way -- going 2-for-2
    one night isn't meaningfully "100%", it's a very small sample). A
    0-attempt game contributes exactly 0, with no NaNs to clean up.
    """
    league_pct = df[makes_col].sum() / df[attempts_col].sum()
    attempts = df[attempts_col].to_numpy(dtype=float)
    makes = df[makes_col].to_numpy(dtype=float)

    rate = np.divide(makes, attempts, out=np.zeros_like(makes), where=attempts > 0)
    impact = np.where(attempts > 0, attempts * (rate - league_pct), 0.0)
    return pd.Series(impact, index=df.index)


def compute_tau(game_logs: pd.DataFrame) -> dict:
    """
    Computes one tau_M value per category from individual game rows.
    Returns a dict keyed the same way as scoring_engine.py's ALL_CATS.
    """
    df = game_logs.copy()

    df["FG_IMPACT"] = compute_game_level_impact(df, "FGM", "FGA")
    df["FT_IMPACT"] = compute_game_level_impact(df, "FTM", "FTA")

    cat_columns = {
        "PTS": "PTS", "REB": "REB", "AST": "AST", "STL": "STL",
        "BLK": "BLK", "FG3M": "FG3M", "TOV": "TOV",
        "FG_PCT": "FG_IMPACT", "FT_PCT": "FT_IMPACT",
    }

    # Only players with enough games to give a reliable personal
    # variance estimate (see module docstring).
    games_played = df.groupby("PLAYER_ID").size()
    qualifying_players = games_played[games_played >= MIN_GP_FOR_TAU].index
    df = df[df["PLAYER_ID"].isin(qualifying_players)]

    tau = {}
    for cat, col in cat_columns.items():
        per_player_std = df.groupby("PLAYER_ID")[col].std()
        tau[cat] = float(np.sqrt((per_player_std ** 2).mean()))

    return tau


def compute_sigma_from_season_stats(season_stats: pd.DataFrame) -> dict:
    """
    Recomputes sigma_M (player-to-player standard deviation) from
    player_stats_2025-26.csv, using the EXACT SAME filtered pool
    (GP >= MIN_GP, MIN >= MIN_MPG) and impact calc that
    scoring_engine.py's build_rankings() uses for the real z-scores --
    so sigma here matches the sigma actually dividing every player's
    z-score, not a differently-filtered stand-in.
    """
    from scoring_engine import MIN_GP, MIN_MPG, compute_pct_impact

    pool = season_stats[(season_stats["GP"] >= MIN_GP) & (season_stats["MIN"] >= MIN_MPG)].copy()
    pool["FG_IMPACT"] = compute_pct_impact(pool, "FGM", "FGA")
    pool["FT_IMPACT"] = compute_pct_impact(pool, "FTM", "FTA")

    sigma = {}
    for cat in COUNTING_CATS:
        sigma[cat] = pool[cat].std()
    sigma["FG_PCT"] = pool["FG_IMPACT"].std()
    sigma["FT_PCT"] = pool["FT_IMPACT"].std()
    return sigma


def compute_shrinkage(sigma: dict, tau: dict, kappa: float = KAPPA) -> dict:
    """
    Converts sigma (season-average basis) and tau (game-to-game basis,
    computed above) into the same GSCORE_SHRINKAGE-style ratio
    scoring_engine.py already uses: denom_Z / denom_G -- how much of the
    plain z-score survives once game-to-game noise is priced in.
    """
    shrinkage = {}
    for cat in ALL_CATS:
        denom_z = sigma[cat]
        denom_g = (sigma[cat] ** 2 + kappa * tau[cat] ** 2) ** 0.5
        shrinkage[cat] = round(denom_z / denom_g, 3)
    return shrinkage


def main():
    game_logs_path = Path(__file__).parent / "game_logs_2025-26.csv"
    season_stats_path = Path(__file__).parent / "player_stats_2025-26.csv"

    game_logs = pd.read_csv(game_logs_path)
    season_stats = pd.read_csv(season_stats_path)

    print(f"Loaded {len(game_logs)} game rows, {game_logs['PLAYER_ID'].nunique()} players.")

    tau = compute_tau(game_logs)
    sigma = compute_sigma_from_season_stats(season_stats)
    shrinkage = compute_shrinkage(sigma, tau)

    from scoring_engine import GSCORE_SHRINKAGE as OLD_PLACEHOLDER

    print(f"\n{'Category':10s} {'sigma':>8s} {'tau':>8s} {'shrink':>8s}   (old placeholder)")
    for cat in ALL_CATS:
        print(f"{cat:10s} {sigma[cat]:8.3f} {tau[cat]:8.3f} {shrinkage[cat]:8.3f}   (was {OLD_PLACEHOLDER[cat]})")

    print("\nPaste this into scoring_engine.py's GSCORE_SHRINKAGE dict once you're happy with it:")
    print("GSCORE_SHRINKAGE = {")
    for cat in ALL_CATS:
        print(f'    "{cat}": {shrinkage[cat]},')
    print("}")


if __name__ == "__main__":
    main()