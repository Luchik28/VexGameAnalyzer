"""Thin RobotEvents API v2 client (https://www.robotevents.com/api/v2).

Requires a bearer token in ROBOTEVENTS_TOKEN (request one free at
robotevents.com -> API). Rate limited to ~1 request/second, so we throttle
and paginate politely. All calls return parsed JSON dicts.
"""

import time

import requests

from vexga.config import ROBOTEVENTS_TOKEN

BASE = "https://www.robotevents.com/api/v2"
_MIN_INTERVAL = 1.1
_last_call = 0.0


class RobotEventsError(RuntimeError):
    pass


def _get(path: str, **params) -> dict:
    global _last_call
    if not ROBOTEVENTS_TOKEN:
        raise RobotEventsError(
            "ROBOTEVENTS_TOKEN is not set. Request a key at "
            "https://www.robotevents.com/api/v2 and put it in .env"
        )
    wait = _MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(
        f"{BASE}{path}",
        params=params,
        headers={"Authorization": f"Bearer {ROBOTEVENTS_TOKEN}"},
        timeout=30,
    )
    _last_call = time.time()
    if resp.status_code != 200:
        raise RobotEventsError(f"GET {path} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _paginate(path: str, **params) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        data = _get(path, page=page, per_page=250, **params)
        out.extend(data["data"])
        if data["meta"]["current_page"] >= data["meta"]["last_page"]:
            return out
        page += 1


def search_events(sku: str | None = None, name: str | None = None, season_id: int | None = None) -> list[dict]:
    """Find events. sku is the RE-V5RC-XX-XXXX code shown on robotevents.com."""
    params: dict = {}
    if sku:
        params["sku[]"] = sku
    if season_id:
        params["season[]"] = season_id
    events = _paginate("/events", **params)
    if name:
        events = [e for e in events if name.lower() in e["name"].lower()]
    return events


def event_divisions(event_id: int) -> list[dict]:
    return _get(f"/events/{event_id}")["divisions"]


def event_matches(event_id: int, division_id: int) -> list[dict]:
    """All matches in a division: teams per alliance, scheduled/actual times,
    and final scores."""
    return _paginate(f"/events/{event_id}/divisions/{division_id}/matches")


def event_teams(event_id: int) -> list[dict]:
    return _paginate(f"/events/{event_id}/teams")


def seasons(program: str = "V5RC") -> list[dict]:
    return _paginate("/seasons", **{"program[]": 1})  # program 1 == V5RC
