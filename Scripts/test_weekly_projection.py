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
