"""Own-seat results for this account. Public matches from other accounts are ignored."""
from __future__ import annotations

from self_jev.agx import request

COMPETITIONS = {
    "fps": "2a97f94f-816a-52b7-b4b8-5925db160245",
    "racing": "1eb91364-950b-5725-aa4b-3501b16860ec",
}


def own_matches(game: str) -> list[dict]:
    competition = COMPETITIONS[game]
    found = {}
    _status, _headers, matches = request("GET", "/api/v1/matches", {"game_id": game, "limit": 100})
    for item in matches.get("items") or []:
        if item.get("competition_id") != competition:
            continue
        row = _own_row(item)
        if row:
            found[row["match_id"]] = row
    try:
        _status, _headers, jobs = request("GET", "/api/v1/jobs", {"mine": "true", "game_id": game, "limit": 50})
    except Exception:
        jobs = {"items": []}
    for item in jobs.get("items") or []:
        if item.get("competition_id") not in (None, competition):
            continue
        if item.get("status") != "succeeded":
            continue
        match_id = str(item.get("match_id") or item.get("id") or "")
        if not match_id or match_id in found:
            continue
        found[match_id] = {
            "match_id": match_id,
            "status": item.get("status") or "",
            "finished_at": item.get("finished_at") or item.get("created_at") or "",
            "rank": None,
            "team": None,
            "round_wins": None,
            "winner": None,
            "worse": True,
            "source": "job",
        }
    return sorted(found.values(), key=lambda row: row.get("finished_at") or "", reverse=True)


def _own_row(match: dict) -> dict | None:
    mine = [seat for seat in match.get("participants") or [] if seat.get("is_mine")]
    if not mine or match.get("status") != "succeeded":
        return None
    seat = mine[0]
    result = match.get("result") or {}
    index = seat.get("seat")
    ranks = result.get("ranks") or []
    rank = ranks[index] if isinstance(index, int) and isinstance(ranks, list) and index < len(ranks) else None
    wins = result.get("round_wins")
    team = seat.get("team")
    worse = _worse(rank, wins, team)
    return {
        "match_id": match["id"],
        "status": match["status"],
        "finished_at": match.get("finished_at") or "",
        "rank": rank,
        "team": team,
        "round_wins": wins if isinstance(wins, list) else None,
        "winner": result.get("winner"),
        "worse": worse,
        "source": "match",
    }


def _worse(rank, wins, team) -> bool:
    if isinstance(wins, list) and len(wins) == 2 and isinstance(team, int) and team in (0, 1):
        return wins[team] < wins[1 - team]
    if isinstance(rank, (int, float)):
        return rank > 1
    return True
