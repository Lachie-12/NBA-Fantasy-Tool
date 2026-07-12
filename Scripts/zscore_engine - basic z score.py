"""
Phase 1 - Z-Score Ranking Engine
---------------------------------
Takes the per-game stat CSV from pull_player_stats.py and ranks players
using a 9-category z-score model.

Design notes (the "why"):

- Counting stats (PTS, REB, AST, STL, BLK, FG3M) get a standard z-score:
  (player_value - pool_mean) / pool_std. This tells you how many standard
  deviations above/below average a player is in that category.

- Turnovers are a "bad" stat -- more TOV hurts you in a 9-cat league --
  so after computing the z-score normally, we flip its sign. That way a
  below-average (good, low) TOV total shows up as a positive contribution,
  just like every other category, and can be summed directly.

- FG% and FT% are NOT z-scored directly, on purpose. A guy shooting 100%
  on 1 shot a game isn't actually helping your percentages, but a naive
  z-score of raw FG_PCT would rank him as elite. Instead we calculate each
  player's "impact": how many percentage-points-worth of value their
  volume + efficiency adds or subtracts compared to a league-average
  shooter taking the same number of shots. THAT impact number is what
  gets z-scored. This is why pull_player_stats.py kept raw FGM/FGA/FTM/FTA
  instead of just saving the percentage -- this step needs them.

- A minimum games-played / minutes-per-game filter is applied before
  computing the pool's mean and standard deviation. Without it, a bunch
  of end-of-bench guys with tiny, noisy samples drag the mean/std around
  and distort everyone else's z-scores.
"""

from pathlib import Path
import pandas as pd

# --- Settings you may want to tweak ---
MIN_GP = 10        # games played
MIN_MPG = 10.0     # minutes per game

COUNTING_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]
NEGATIVE_CATS = ["TOV"]  # lower is better, so we invert the z-score


def compute_pct_impact(df: pd.DataFrame, makes_col: str, attempts_col: str) -> pd.Series:
    """
    Impact = (player's shooting % minus league average %) x their attempts.
    A high-volume, above-average shooter scores well here.
    A high-volume, below-average shooter scores badly (this is the guy who
    quietly wrecks your FG% every week).
    A low-volume shooter, good or bad, ends up near zero -- correctly,
    since they barely move your team percentage either way.
    """
    league_pct = df[makes_col].sum() / df[attempts_col].sum()
    return df[attempts_col] * (df[makes_col] / df[attempts_col] - league_pct)


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def build_rankings(df: pd.DataFrame) -> pd.DataFrame:
    pool = df[(df["GP"] >= MIN_GP) & (df["MIN"] >= MIN_MPG)].copy()

    z = pd.DataFrame(index=pool.index)

    for cat in COUNTING_CATS:
        z[cat] = zscore(pool[cat])

    for cat in NEGATIVE_CATS:
        z[cat] = -zscore(pool[cat])

    pool["FG_IMPACT"] = compute_pct_impact(pool, "FGM", "FGA")
    pool["FT_IMPACT"] = compute_pct_impact(pool, "FTM", "FTA")
    z["FG_PCT"] = zscore(pool["FG_IMPACT"])
    z["FT_PCT"] = zscore(pool["FT_IMPACT"])

    z["TOTAL_Z"] = z.sum(axis=1)

    result = pool[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]].join(z)
    result = result.sort_values("TOTAL_Z", ascending=False).reset_index(drop=True)
    result.insert(0, "RANK", result.index + 1)
    return result.round(2)


def main():
    input_path = Path(__file__).parent / "player_stats_2025-26.csv"
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} players.")
    rankings = build_rankings(df)
    print(f"{len(rankings)} players passed the GP >= {MIN_GP} / MIN >= {MIN_MPG} filter.")

    output_path = Path(__file__).parent / "zscore_rankings_2025-26.csv"
    rankings.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}\n")

    print("Top 15:")
    print(rankings.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
