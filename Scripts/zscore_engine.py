"""
Phase 2 - Z-Score Ranking Engine (with punt support)
-------------------------------------------------------
Takes the per-game stat CSV from pull_player_stats.py and ranks players
using a 9-category z-score model. Supports three scoring modes through
one shared mechanism: basic z-score, Durant-style single-category punt,
and (later) fully custom category weighting.

Design notes (the "why"):

- Counting stats (PTS, REB, AST, STL, BLK, FG3M) get a standard z-score:
  (player_value - pool_mean) / pool_std.

- Turnovers are a "bad" stat, so after computing the z-score normally we
  flip its sign -- a below-average (good, low) TOV total becomes a
  positive contribution, summable with everything else.

- FG% and FT% are NOT z-scored directly. Instead we calculate each
  player's volume-weighted "impact": attempts x (their% - league avg%).
  That impact number is what gets z-scored. See pull_player_stats.py for
  why raw FGM/FGA/FTM/FTA were kept instead of just the percentage.

- A minimum games-played / minutes-per-game filter is applied before
  computing the pool's mean and standard deviation, so small, noisy
  samples from end-of-bench players don't distort everyone else's z.

- PUNTING: a true fantasy "punt" is a team-build decision -- you choose
  in advance to concede one category in exchange for value everywhere
  else. Every player is then re-ranked on the SAME remaining 8
  categories. This is different from (and better than) auto-dropping
  each player's own individual weakest category, which would compare
  players on inconsistent category sets and doesn't correspond to any
  real roster you could build.

  Mechanically, a punt is just "give the punted category a weight of 0
  and re-sum." That's why basic z-score, a single-category punt, and a
  fully custom weighted score (a later phase) are all the SAME function
  underneath -- weighted_total() -- just called with different weight
  dictionaries. Basic = all weights 1. Punt FT% = {"FT_PCT": 0}. Custom
  = whatever weights you want per category.

  Note: this is a "hard" punt (the category counts for nothing). Real
  DURANT on Basketball Monster does a more sophisticated "soft" punt
  based on win-probability/variance math rather than a clean on/off
  switch. A hard punt is the standard, well-understood starting point
  and is what most human punt strategy approximates anyway -- soft
  punting is a reasonable future refinement, not a v1 requirement.
"""

from pathlib import Path
import pandas as pd

# --- Settings you may want to tweak ---
MIN_GP = 10        # games played
MIN_MPG = 10.0     # minutes per game

COUNTING_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]
NEGATIVE_CATS = ["TOV"]  # lower is better, so we invert the z-score
ALL_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG_PCT", "FT_PCT"]


def compute_pct_impact(df: pd.DataFrame, makes_col: str, attempts_col: str) -> pd.Series:
    """
    Impact = (player's shooting % minus league average %) x their attempts.
    High volume + above-average efficiency scores well. High volume +
    below-average efficiency scores badly. Low volume, good or bad, ends
    up near zero -- correctly, since it barely moves your team % either way.
    """
    league_pct = df[makes_col].sum() / df[attempts_col].sum()
    return df[attempts_col] * (df[makes_col] / df[attempts_col] - league_pct)


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def compute_category_zscores(pool: pd.DataFrame) -> pd.DataFrame:
    """Compute all 9 per-category z-scores for an already-filtered player pool."""
    z = pd.DataFrame(index=pool.index)

    for cat in COUNTING_CATS:
        z[cat] = zscore(pool[cat])

    for cat in NEGATIVE_CATS:
        z[cat] = -zscore(pool[cat])

    pool["FG_IMPACT"] = compute_pct_impact(pool, "FGM", "FGA")
    pool["FT_IMPACT"] = compute_pct_impact(pool, "FTM", "FTA")
    z["FG_PCT"] = zscore(pool["FG_IMPACT"])
    z["FT_PCT"] = zscore(pool["FT_IMPACT"])

    return z


def weighted_total(z: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """
    Sum category z-scores using a per-category weight (default 1.0 for
    any category not mentioned). weights={} or None -> basic z-score.
    weights={"FT_PCT": 0} -> punt FT%. weights={"TOV": 1.5} -> custom.
    """
    weights = weights or {}
    total = pd.Series(0.0, index=z.index)
    for cat in ALL_CATS:
        total += z[cat] * weights.get(cat, 1.0)
    return total


def build_rankings(df: pd.DataFrame, weights: dict | None = None, total_col: str = "TOTAL_Z") -> pd.DataFrame:
    """Filter the pool, compute category z-scores, and rank by weighted total."""
    pool = df[(df["GP"] >= MIN_GP) & (df["MIN"] >= MIN_MPG)].copy()

    z = compute_category_zscores(pool)
    z[total_col] = weighted_total(z, weights)

    result = pool[["PLAYER_ID","PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]].join(z)
    result = result.sort_values(total_col, ascending=False).reset_index(drop=True)
    result.insert(0, "RANK", result.index + 1)
    return result.round(2)


def build_punt_rankings(df: pd.DataFrame, punt_category: str) -> pd.DataFrame:
    """
    Convenience wrapper for a single-category punt (matches the Streamlit
    dashboard's current single-select Punt dropdown). Re-ranks players with
    the chosen category weighted to 0 -- a Durant-style punt build.
    """
    if punt_category not in ALL_CATS:
        raise ValueError(f"punt_category must be one of {ALL_CATS}, got {punt_category!r}")
    return build_rankings(df, weights={punt_category: 0}, total_col="DURANT_Z")


def main():
    input_path = Path(__file__).parent / "player_stats_2025-26.csv"
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} players.")
    rankings = build_rankings(df)
    print(f"{len(rankings)} players passed the GP >= {MIN_GP} / MIN >= {MIN_MPG} filter.")

    output_path = Path(__file__).parent / "zscore_rankings_2025-26.csv"
    rankings.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}\n")

    print("Top 15 (basic z-score):")
    print(rankings.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
    