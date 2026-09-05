"""
Phase 2 - Z-Score Ranking Engine (with punt support + G-score stub)
-------------------------------------------------------
Takes the per-game stat CSV from pull_player_stats.py and ranks players
using either a 9-category z-score model or a G-score model (a documented,
public approximation of what Basketball Monster's DURANT rankings are
trying to achieve). Supports punting and custom weighting on top of
either method through one shared mechanism.

--- Standard z-score (see original docstring further down for full detail) ---
Counting stats: (value - pool_mean) / pool_std, TOV inverted, FG%/FT% via
volume-weighted impact.

--- G-score / DURANT-inspired scoring ---
Standard z-score implicitly assumes every category is equally *reliable*
week to week. In reality, some categories (steals is the extreme case)
swing wildly from game to game even for good players, so "banking on"
that category is inherently less certain than banking on a steadier one
like assists -- no matter how many standard deviations above average a
player's season total is. G-score (Rosenof, 2023, arXiv:2307.02188 --
a public, peer-reviewed method in the same family as Basketball
Monster's proprietary DURANT) fixes this by adding a game-to-game
"noise" term to the z-score denominator:

    G = (player_avg - pool_avg) / sqrt(sigma_M**2 + KAPPA * tau_M**2)

sigma_M**2 is the ordinary player-to-player variance -- exactly what
zscore() already divides by (squared). tau_M**2 is the pool-level
game-to-game variance for that category. KAPPA is a small constant
(~1.04 for typical league sizes) that can just be hardcoded.

THE CATCH: tau_M**2 requires individual game logs (how much a player's
own output bounces around game to game), not season averages. Our
pipeline only pulls season *averages* right now (pull_player_stats.py
uses LeagueDashPlayerStats in PerGame mode), so real tau_M isn't
available yet.

PLACEHOLDER STRATEGY (until game logs are wired in): rather than
guessing at absolute noise numbers on a scale that might not match our
own sigma, we borrow the *relative* shrinkage each category received in
Rosenof's real-world 2022-23 NBA analysis (Table 8) -- i.e. what
fraction of its z-score weight each category kept once game-to-game
noise was accounted for. Steals kept only 44%, assists kept 75%, etc.
Applying that fraction directly to our own z-scores approximates the
*shape* of the G-score adjustment using real basketball variance
patterns, without needing game logs yet.

THIS IS A STAND-IN. It borrows relationships between categories from a
different season, pool, and format assumptions -- not truth computed
from this season's actual data. Swap it for the real formula above
once pull_player_stats.py (or a new sibling script) pulls game logs and
we can compute genuine per-category tau_M values.
"""

from pathlib import Path
import pandas as pd

# --- Settings you may want to tweak ---
MIN_GP = 10        # games played
MIN_MPG = 10.0     # minutes per game

COUNTING_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]
NEGATIVE_CATS = ["TOV"]  # lower is better, so we invert the z-score
ALL_CATS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG_PCT", "FT_PCT"]

# Placeholder G-score shrinkage factors -- see module docstring.
# Source: Rosenof (2023), "Static quantification of player value for
# fantasy basketball", Table 8 (real 2022-23 NBA season data).
# Value = G-score denominator's weight relative to z-score's (i.e. how
# much of the category's z-score value survives once game-to-game noise
# is priced in). Replace with computed values once game logs exist.
GSCORE_SHRINKAGE = {
    "PTS": 0.65,
    "REB": 0.69,
    "AST": 0.75,
    "STL": 0.44,
    "BLK": 0.68,
    "FG3M": 0.72,
    "TOV": 0.62,
    "FG_PCT": 0.56,
    "FT_PCT": 0.58,
}


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
    """Compute all 9 per-category z-scores (standard method) for an
    already-filtered player pool."""
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


def compute_gscore_zscores(pool: pd.DataFrame) -> pd.DataFrame:
    """
    G-score-style category values (placeholder tau, see module docstring).
    Same numerator as compute_category_zscores; each category's z-score
    is scaled down by its GSCORE_SHRINKAGE factor, pulling noisy
    categories (STL, FG%) toward zero relative to steady ones (AST).
    """
    z = pd.DataFrame(index=pool.index)

    for cat in COUNTING_CATS:
        z[cat] = zscore(pool[cat]) * GSCORE_SHRINKAGE[cat]

    for cat in NEGATIVE_CATS:
        z[cat] = -zscore(pool[cat]) * GSCORE_SHRINKAGE[cat]

    pool["FG_IMPACT"] = compute_pct_impact(pool, "FGM", "FGA")
    pool["FT_IMPACT"] = compute_pct_impact(pool, "FTM", "FTA")
    z["FG_PCT"] = zscore(pool["FG_IMPACT"]) * GSCORE_SHRINKAGE["FG_PCT"]
    z["FT_PCT"] = zscore(pool["FT_IMPACT"]) * GSCORE_SHRINKAGE["FT_PCT"]

    return z


def weighted_total(z: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """
    Sum category z-scores using a per-category weight (default 1.0 for
    any category not mentioned). weights={} or None -> basic total.
    weights={"FT_PCT": 0} -> punt FT%. weights={"TOV": 1.5} -> custom.
    Works identically whether z came from compute_category_zscores or
    compute_gscore_zscores -- punting and custom weighting compose with
    either scoring method for free.
    """
    weights = weights or {}
    total = pd.Series(0.0, index=z.index)
    for cat in ALL_CATS:
        total += z[cat] * weights.get(cat, 1.0)
    return total


def build_rankings(
    df: pd.DataFrame,
    weights: dict | None = None,
    total_col: str = "TOTAL_Z",
    method: str = "zscore",
) -> pd.DataFrame:
    """Filter the pool, compute category values with the chosen method,
    and rank by weighted total.

    method: "zscore" (standard) or "gscore" (DURANT-inspired, placeholder
    tau -- see module docstring).
    """
    pool = df[(df["GP"] >= MIN_GP) & (df["MIN"] >= MIN_MPG)].copy()

    if method == "zscore":
        z = compute_category_zscores(pool)
    elif method == "gscore":
        z = compute_gscore_zscores(pool)
    else:
        raise ValueError(f"method must be 'zscore' or 'gscore', got {method!r}")

    z[total_col] = weighted_total(z, weights)

    result = pool[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]].join(z)
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


def build_gscore_rankings(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Convenience wrapper for G-score rankings (placeholder tau)."""
    return build_rankings(df, weights=weights, total_col="GSCORE", method="gscore")


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

    gscore_rankings = build_gscore_rankings(df)
    print("\nTop 15 (G-score, placeholder tau):")
    print(gscore_rankings.head(15).to_string(index=False))


if __name__ == "__main__":
    main()