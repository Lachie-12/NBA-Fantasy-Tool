"""
Sandbox tests for weekly_projection.py (Chunk 1).

Run with:   py test_weekly_projection.py
Uses only made-up data and temporary folders - it never touches your
real CSVs.
"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

import weekly_projection as wp

WEEK = date(2026, 10, 5)        # a Monday
NEXT_WEEK = date(2026, 10, 12)


def make_players():
    return pd.DataFrame({
        "PLAYER_ID": [1, 2, 3, 4],
        "TEAM_ABBREVIATION": ["AAA", "AAA", "BBB", "ZZZ"],  # ZZZ is not in the schedule
    })


def make_team_games():
    # AAA plays 3 this week, BBB plays 4, CCC plays 2 (no players on it),
    # and AAA also has a game NEXT week that must not leak into this week.
    rows = []
    for d in (5, 7, 9):
        rows.append(("AAA", WEEK, date(2026, 10, d)))
    for d in (5, 6, 8, 10):
        rows.append(("BBB", WEEK, date(2026, 10, d)))
    for d in (6, 11):
        rows.append(("CCC", WEEK, date(2026, 10, d)))
    rows.append(("AAA", NEXT_WEEK, date(2026, 10, 13)))
    return pd.DataFrame(rows, columns=wp.TEAM_GAMES_COLUMNS)


def make_overrides(rows=()):
    df = pd.DataFrame(list(rows), columns=wp.OVERRIDE_COLUMNS)
    df["PLAYER_ID"] = df["PLAYER_ID"].astype("int64")
    df["GAMES"] = df["GAMES"].astype("int64")
    return df


def games_for(result, player_id):
    return int(result.loc[result["PLAYER_ID"] == player_id, "GAMES_SCHEDULED"].iloc[0])


# ---------------------------------------------------------------------------

def test_base_counts_with_no_overrides():
    r = wp.resolve_games_scheduled(make_players(), make_team_games(), make_overrides(), WEEK)
    assert games_for(r, 1) == 3 and games_for(r, 2) == 3   # AAA, ignoring next week's game
    assert games_for(r, 3) == 4                            # BBB
    assert games_for(r, 4) == 0                            # team missing from schedule -> 0
    assert not r["IS_OVERRIDDEN"].any()
    assert r["OVERRIDE_GAMES"].isna().all()


def test_override_replaces_base_for_that_player_only():
    ov = make_overrides([(WEEK, 1, 2)])
    r = wp.resolve_games_scheduled(make_players(), make_team_games(), ov, WEEK)
    assert games_for(r, 1) == 2          # overridden 3 -> 2
    assert games_for(r, 2) == 3          # teammate untouched
    assert r.loc[r["PLAYER_ID"] == 1, "BASE_GAMES"].iloc[0] == 3   # base still recorded
    assert list(r["IS_OVERRIDDEN"]) == [True, False, False, False]


def test_override_can_raise_games_above_base():
    ov = make_overrides([(WEEK, 4, 2)])   # player on a team the schedule doesn't know
    r = wp.resolve_games_scheduled(make_players(), make_team_games(), ov, WEEK)
    assert games_for(r, 4) == 2


def test_override_of_zero_is_respected_not_treated_as_blank():
    ov = make_overrides([(WEEK, 3, 0)])
    r = wp.resolve_games_scheduled(make_players(), make_team_games(), ov, WEEK)
    assert games_for(r, 3) == 0
    assert bool(r.loc[r["PLAYER_ID"] == 3, "IS_OVERRIDDEN"].iloc[0]) is True


def test_override_for_a_different_week_is_ignored():
    ov = make_overrides([(NEXT_WEEK, 1, 1)])
    r = wp.resolve_games_scheduled(make_players(), make_team_games(), ov, WEEK)
    assert games_for(r, 1) == 3
    assert not r["IS_OVERRIDDEN"].any()


def test_duplicate_override_rows_last_one_wins():
    ov = make_overrides([(WEEK, 1, 1), (WEEK, 1, 2)])
    r = wp.resolve_games_scheduled(make_players(), make_team_games(), ov, WEEK)
    assert games_for(r, 1) == 2
    assert len(r) == 4   # no extra rows created by the duplicate


def test_row_count_and_order_match_players():
    players = make_players()
    r = wp.resolve_games_scheduled(players, make_team_games(), make_overrides(), WEEK)
    assert list(r["PLAYER_ID"]) == list(players["PLAYER_ID"])


def test_stale_schedule_detection():
    tg = make_team_games()
    assert wp.schedule_week_matches(tg, WEEK) is True
    assert wp.schedule_week_matches(tg, date(2026, 11, 2)) is False
    empty = wp.load_team_games_this_week(Path("does_not_exist.csv"))
    assert wp.schedule_week_matches(empty, WEEK) is False


def test_overrides_file_roundtrip_and_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "games_overrides.csv"

        first = wp.load_games_overrides(path)          # missing -> created empty
        assert first.empty and path.exists()
        assert list(first.columns) == wp.OVERRIDE_COLUMNS

        again = wp.load_games_overrides(path)          # header-only file reads back clean
        assert again.empty and str(again["PLAYER_ID"].dtype) == "int64"

        ov = make_overrides([(WEEK, 1, 2), (WEEK, 3, 0)])
        wp.save_games_overrides(ov, path)
        loaded = wp.load_games_overrides(path)
        assert len(loaded) == 2
        assert loaded["WEEK_START_DATE"].iloc[0] == WEEK        # real date objects, not strings
        assert list(loaded["PLAYER_ID"]) == [1, 3]
        assert list(loaded["GAMES"]) == [2, 0]

        # and the loaded copy works straight through resolve
        r = wp.resolve_games_scheduled(make_players(), make_team_games(), loaded, WEEK)
        assert games_for(r, 1) == 2 and games_for(r, 3) == 0


def test_save_rejects_bad_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "o.csv"
        for bad in (-1, 2.5):
            ov = pd.DataFrame({"WEEK_START_DATE": [WEEK], "PLAYER_ID": [1], "GAMES": [bad]})
            try:
                wp.save_games_overrides(ov, path)
            except ValueError:
                continue
            raise AssertionError(f"GAMES={bad} should have been rejected")
        assert not path.exists()   # nothing half-written


def test_team_games_file_loads_dates_as_dates():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "team_games_this_week.csv"
        # same shape pull_team_schedule.py writes
        path.write_text(
            "TEAM_ABBREVIATION,WEEK_START_DATE,GAME_DATE\n"
            "AAA,2026-10-05,2026-10-05\n"
            "AAA,2026-10-05,2026-10-07\n"
        )
        tg = wp.load_team_games_this_week(path)
        assert tg["WEEK_START_DATE"].iloc[0] == WEEK
        assert tg["GAME_DATE"].iloc[1] == date(2026, 10, 7)
        r = wp.resolve_games_scheduled(make_players(), tg, make_overrides(), WEEK)
        assert games_for(r, 1) == 2


# ===========================================================================
# Chunk 2 - projection math
# ===========================================================================

STAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FGM", "FGA", "FTM", "FTA"]
CAT_COLS = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FG_PCT", "FT_PCT"]


def make_stats():
    """Five made-up players. Per-game numbers chosen so the expected
    weekly answers can be worked out by hand."""
    rows = [
        # id, name, team, PTS, REB, AST, STL, BLK, 3PM, TOV, FGM, FGA, FTM, FTA
        (1, "Alpha",  "AAA", 20, 5, 4, 1.0, 0.5, 2, 3, 8, 16, 4, 5),
        (2, "Bravo",  "AAA", 10, 3, 2, 0.5, 0.0, 1, 1, 4, 10, 2, 2),
        (3, "Charlie", "BBB", 15, 6, 3, 1.0, 1.0, 1, 2, 6, 12, 3, 4),
        (4, "Delta",  "BBB", 30, 9, 9, 2.0, 2.0, 4, 5, 12, 20, 6, 6),   # will be on IR
        (5, "Echo",   "ZZZ", 12, 4, 2, 1.0, 0.0, 1, 1, 5, 10, 2, 2),    # team not in schedule -> 0 games
    ]
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"] + STAT_COLS
    df = pd.DataFrame(rows, columns=cols)
    df["FG_PCT"] = df["FGM"] / df["FGA"]
    df["FT_PCT"] = df["FTM"] / df["FTA"]
    return df


def make_teams():
    return pd.DataFrame({"TEAM_SLOT": [1, 2, 3, 4], "TEAM_NAME": ["One", "Two", "Three", "Four"]})


def make_roster():
    # slot 1: Alpha + Bravo | slot 2: Charlie + Delta (IR) | slot 3: Echo only | slot 4: nobody
    return pd.DataFrame({
        "TEAM_SLOT": [1, 1, 2, 2, 3],
        "PLAYER_ID": [1, 2, 3, 4, 5],
        "IS_IR":     [False, False, False, True, False],
    })


def build(overrides=None, roster=None):
    stats = make_stats()
    ov = overrides if overrides is not None else make_overrides()
    games = wp.resolve_games_scheduled(stats, make_team_games(), ov, WEEK)
    proj = wp.compute_weekly_projection(stats, roster if roster is not None else make_roster(), games)
    totals = wp.compute_weekly_team_totals(proj, make_teams())
    return proj, totals


def row(df, key, value):
    return df[df[key] == value].iloc[0]


def test_projection_is_per_game_times_games():
    proj, _ = build(make_overrides([(WEEK, 2, 2)]))   # Bravo overridden 3 -> 2
    alpha = row(proj, "PLAYER_ID", 1)                 # AAA, 3 games
    assert alpha["GAMES_SCHEDULED"] == 3
    assert alpha["PTS"] == 60 and alpha["FGM"] == 24 and alpha["FGA"] == 48
    assert abs(alpha["STL"] - 3.0) < 1e-9
    bravo = row(proj, "PLAYER_ID", 2)
    assert bravo["GAMES_SCHEDULED"] == 2 and bool(bravo["IS_OVERRIDDEN"])
    assert bravo["PTS"] == 20 and bravo["FGA"] == 20
    charlie = row(proj, "PLAYER_ID", 3)               # BBB, 4 games
    assert charlie["PTS"] == 60 and charlie["REB"] == 24


def test_player_percentages_stay_as_season_percentages():
    proj, _ = build()
    assert abs(row(proj, "PLAYER_ID", 1)["FG_PCT"] - 0.5) < 1e-9   # not multiplied by games


def test_projection_keeps_ir_players_flagged():
    proj, _ = build()
    assert len(proj) == 5
    assert bool(row(proj, "PLAYER_ID", 4)["IS_IR"]) is True
    assert bool(row(proj, "PLAYER_ID", 1)["IS_IR"]) is False


def test_team_totals_hand_calculated():
    _, totals = build(make_overrides([(WEEK, 2, 2)]))
    t1 = row(totals, "TEAM_SLOT", 1)
    # Alpha 3 games + Bravo 2 games
    assert t1["PTS"] == 80                                  # 60 + 20
    assert abs(t1["STL"] - 4.0) < 1e-9                      # 3.0 + 1.0
    assert abs(t1["TOV"] - 11.0) < 1e-9                     # 9 + 2
    assert abs(t1["FG_PCT"] - round(32 / 68, 3)) < 1e-9     # (24+8)/(48+20)
    assert abs(t1["FT_PCT"] - round(16 / 19, 3)) < 1e-9     # (12+4)/(15+4)


def test_percentages_are_volume_weighted_not_averaged():
    _, totals = build(make_overrides([(WEEK, 2, 2)]))
    fg = row(totals, "TEAM_SLOT", 1)["FG_PCT"]
    naive_average = (0.5 + 0.4) / 2        # 0.45 - what a lazy average would give
    assert abs(fg - naive_average) > 0.01
    assert abs(fg - 0.471) < 1e-9


def test_ir_player_is_excluded_from_totals():
    _, totals = build()
    t2 = row(totals, "TEAM_SLOT", 2)
    assert t2["PTS"] == 60            # Charlie only (15 x 4); Delta on IR ignored
    assert abs(t2["FG_PCT"] - 0.5) < 1e-9


def test_team_whose_players_all_have_zero_games_is_zero_not_nan():
    _, totals = build()
    t3 = row(totals, "TEAM_SLOT", 3)   # Echo's team has no games
    for c in CAT_COLS:
        assert t3[c] == 0, c


def test_team_with_empty_roster_is_zero():
    _, totals = build()
    t4 = row(totals, "TEAM_SLOT", 4)
    for c in CAT_COLS:
        assert t4[c] == 0, c


def test_totals_shape_matches_season_totals_table():
    _, totals = build()
    # Exactly what compute_team_totals() in 1_Roster_Tracker.py returns, so
    # build_matchup_comparison() can use this table unchanged.
    assert list(totals.columns) == ["TEAM_SLOT", "TEAM_NAME"] + CAT_COLS
    assert list(totals["TEAM_SLOT"]) == [1, 2, 3, 4]
    assert not totals.isna().any().any()


def test_override_flows_through_to_team_totals():
    _, without = build()
    _, with_override = build(make_overrides([(WEEK, 2, 0)]))   # Bravo out all week
    assert row(without, "TEAM_SLOT", 1)["PTS"] == 90           # 60 + 30
    assert row(with_override, "TEAM_SLOT", 1)["PTS"] == 60     # Alpha only
    assert row(with_override, "TEAM_SLOT", 2)["PTS"] == 60     # other teams unchanged


def test_one_game_each_equals_plain_sum_of_per_game_stats():
    # Independent check: if everyone plays exactly 1 game, weekly totals
    # must equal the simple sum of per-game numbers (what the season
    # League Totals table shows).
    stats = make_stats()
    ov = make_overrides([(WEEK, pid, 1) for pid in stats["PLAYER_ID"]])
    _, totals = build(ov)
    for slot, ids in {1: [1, 2], 2: [3]}.items():     # slot 2 excludes IR player 4
        sub = stats[stats["PLAYER_ID"].isin(ids)]
        t = row(totals, "TEAM_SLOT", slot)
        for c in ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV"]:
            assert abs(t[c] - sub[c].sum()) < 1e-6, (slot, c)
        assert abs(t["FG_PCT"] - round(sub["FGM"].sum() / sub["FGA"].sum(), 3)) < 1e-9
        assert abs(t["FT_PCT"] - round(sub["FTM"].sum() / sub["FTA"].sum(), 3)) < 1e-9


def test_rostered_player_missing_from_stats_does_not_crash_or_count():
    roster = pd.concat(
        [make_roster(), pd.DataFrame({"TEAM_SLOT": [1], "PLAYER_ID": [999], "IS_IR": [False]})],
        ignore_index=True,
    )
    proj, totals = build(roster=roster)
    assert len(proj) == 6
    assert row(proj, "PLAYER_ID", 999)["GAMES_SCHEDULED"] == 0
    assert row(totals, "TEAM_SLOT", 1)["PTS"] == 90    # same as without the unknown player


def test_empty_roster_in_both_shapes_the_csv_can_produce():
    # A brand-new roster file reads back as float columns; a freshly
    # created one as object columns. Both must work.
    float_empty = pd.DataFrame({"TEAM_SLOT": pd.Series(dtype=float),
                                "PLAYER_ID": pd.Series(dtype=float),
                                "IS_IR": pd.Series(dtype=float)})
    object_empty = pd.DataFrame(columns=["TEAM_SLOT", "PLAYER_ID", "IS_IR"])
    for empty in (float_empty, object_empty):
        proj, totals = build(roster=empty)
        assert proj.empty
        assert len(totals) == 4
        assert (totals[CAT_COLS] == 0).all().all()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)