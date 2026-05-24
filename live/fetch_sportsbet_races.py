#!/usr/bin/env python3
"""Fetch Sportsbet all racing + per-race racecard-with-context JSON for a date."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
import urllib.error
import urllib.request
from typing import Iterable

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    requests = None

BASE = "https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing"

PRICE_FLUC_COUNT = 6


def get_json_stdlib(url: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from {url}") from exc


def get_json(session: "requests.Session | None", url: str, timeout: float = 30.0) -> dict:
    if requests is None or session is None:
        return get_json_stdlib(url, timeout=timeout)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from {url}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from {url}") from exc


def iter_races(all_racing: dict, allowed_class_id: int | None = 1) -> Iterable[dict]:
    for date_obj in all_racing.get("dates", []):
        for section in date_obj.get("sections", []):
            race_type = section.get("raceType")
            for meeting in section.get("meetings", []):
                class_id = meeting.get("classId")
                if allowed_class_id is not None and class_id != allowed_class_id:
                    continue
                meeting_id = meeting.get("id")
                meeting_name = meeting.get("name")
                for event in meeting.get("events", []):
                    event_id = event.get("id")
                    if event_id is None:
                        continue
                    yield {
                        "event_id": event_id,
                        "class_id": class_id,
                        "meeting_id": meeting_id,
                        "meeting_name": meeting_name,
                        "race_number": event.get("raceNumber"),
                        "race_name": event.get("name"),
                        "race_type": race_type,
                        "start_time": event.get("startTime"),
                    }


def _format_price(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_win_market(event: dict) -> dict | None:
    markets = event.get("markets") or []
    for market in markets:
        name = (market.get("name") or "").lower()
        if "win" in name and (market.get("selections") or []):
            return market
    for market in markets:
        if market.get("selections"):
            return market
    return None


def _format_start_time(start_time: float | int | None) -> str:
    if not isinstance(start_time, (int, float)):
        return "--:--"
    if start_time > 10_000_000_000:
        start_time /= 1000
    return time.strftime("%H:%M", time.localtime(start_time))


def _runner_move_data(sel: dict) -> dict:
    flucs = sel.get("recentOddsFluctuations") or []
    open_price = flucs[0] if flucs else None
    current_price = flucs[-1] if flucs else None
    pct_change = None
    direction = None

    if isinstance(open_price, (int, float)) and isinstance(current_price, (int, float)) and open_price:
        delta = ((current_price - open_price) / open_price) * 100
        pct_change = abs(delta)
        if current_price < open_price:
            direction = "down"
        elif current_price > open_price:
            direction = "up"

    return {
        "flucs": flucs,
        "open_price": open_price,
        "current_price": current_price,
        "pct_change": pct_change,
        "direction": direction,
    }


COL_W = 60
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _pad_to(s: str, width: int) -> str:
    return s + " " * max(0, width - _visible_len(s))


def _format_move_indicator(move: dict) -> str:
    pct_change = move["pct_change"]
    direction = move["direction"]
    if pct_change is None or direction is None:
        return f"{DIM}  —{RESET}"
    if direction == "down":
        return f"{GREEN}▼ -{pct_change:.1f}%{RESET}"
    return f"{RED}▲ +{pct_change:.1f}%{RESET}"


def _runner_row_compact(sel: dict) -> str:
    """Single runner line sized to fit COL_W (60 visible chars)."""
    runner_no = sel.get("runnerNumber")
    runner_name = (sel.get("name") or "")[:15]
    move = _runner_move_data(sel)
    flucs = move["flucs"]
    open_price = move["open_price"]
    current_price = move["current_price"]
    pct_change = move["pct_change"]

    last3 = flucs[-3:] if flucs else []
    last3 = [None] * max(0, 3 - len(last3)) + list(last3)
    flucs_str = " ".join(f"{_format_price(v):>5}" for v in last3)

    arrow = _format_move_indicator(move)
    star = f"{BOLD}★{RESET}" if isinstance(pct_change, (int, float)) and pct_change >= 15 else " "

    # visible: 1+3+1+15+1+5+2+5+1+5+1+5+2+5+1+7 = 60
    return (
        f"{star}{str(runner_no or ''):>2}. {runner_name:<15} "
        f"{_format_price(open_price):>5}  {flucs_str}  "
        f"{_format_price(current_price):>5} {arrow}"
    )


def _render_race_block(
    track: str,
    race_no: int | None,
    start_time: float | int | None,
    selections: list[dict],
    watch_mode: bool,
) -> list[str]:
    """Return a list of strings, each padded to COL_W visible chars."""
    w = COL_W
    bar = "═" * w
    thin = "─" * w

    title_text = f"  R{race_no or '?'}  {(track or 'UNKNOWN').upper()}"
    time_text = _format_start_time(start_time)
    gap = max(1, w - len(title_text) - len(time_text))
    header_line = f"{CYAN}{BOLD}{title_text}{' ' * gap}{time_text}{RESET}"

    col_labels = f" {'#':>2}  {'Name':<15} {'Open':>5}  {'F1':>5} {'F2':>5} {'F3':>5}  {'Cur':>5} Move"

    lines: list[str] = [
        f"{DIM}{bar}{RESET}",
        header_line,
        f"{DIM}{bar}{RESET}",
        f"{DIM}{col_labels}{RESET}",
        f"{DIM}{thin}{RESET}",
    ]

    for sel in selections:
        lines.append(_runner_row_compact(sel))

    if watch_mode:
        movers = []
        for sel in selections:
            move = _runner_move_data(sel)
            pct = move["pct_change"]
            d = move["direction"]
            if pct and d and pct > 0:
                a = f"{GREEN}▼{RESET}" if d == "down" else f"{RED}▲{RESET}"
                movers.append((pct, f"{(sel.get('name') or '')[:10]}{a}{pct:.0f}%"))
        movers.sort(key=lambda x: x[0], reverse=True)
        lines.append(f"{DIM}{thin}{RESET}")
        if len(movers) >= 2:
            lines.append(f" 📊 {' '.join(t for _, t in movers[:3])}")
        lines.append(f"{DIM}{thin}{RESET}")

    return [_pad_to(line, w) for line in lines]


def _print_side_by_side(left: list[str], right: list[str] | None) -> None:
    sep = f" {DIM}║{RESET} "
    blank = " " * COL_W
    n = max(len(left), len(right) if right else 0)
    for i in range(n):
        l_col = left[i] if i < len(left) else blank
        if right is not None:
            r_col = right[i] if i < len(right) else blank
            print(f"{l_col}{sep}{r_col}")
        else:
            print(l_col)


def _clear_screen() -> None:
    print("\033[2J\033[H", end="")

def _select_next_races(races: list[dict], now_ts: float, max_races: int) -> list[dict]:
    races_with_time = [r for r in races if isinstance(r.get("start_time"), (int, float))]
    races_with_time.sort(key=lambda r: r["start_time"])
    upcoming = [r for r in races_with_time if r["start_time"] >= now_ts]
    if max_races > 0:
        return upcoming[:max_races]
    return upcoming


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch all races for a date and then fetch RacecardWithContext for each race"
    )
    # default to today's date if not provided
    today = time.strftime("%Y-%m-%d")
    parser.add_argument("--date", default=today, help="Date in YYYY-MM-DD format")
    parser.add_argument(
        "--outdir",
        default="sportsbet_racing_json",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds between race calls",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=5,
        help="Maximum number of races to fetch/list (default: 5). Use 0 for no limit.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh the on-screen listing (disables JSON writes).",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=5.0,
        help="Refresh interval seconds when --watch is set (default: 5).",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        help="Only fetch this classId from AllRacing (default: 1). Use 0 to disable class filter.",
    )
    args = parser.parse_args()

    all_racing_url = f"{BASE}/AllRacing/{args.date}"

    outdir = pathlib.Path("/home/theo/perplex/x7/x8/data") / f"{args.outdir}_{args.date}"
    if not args.watch:
        outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session() if requests is not None else None
    try:
        while True:
            all_racing = get_json(session, all_racing_url)

            allowed_class_id = args.class_id if args.class_id > 0 else None
            races = list(iter_races(all_racing, allowed_class_id=allowed_class_id))
            if args.watch:
                races = _select_next_races(races, time.time(), args.max_races)
            elif args.max_races > 0:
                races = races[: args.max_races]

            if args.watch:
                _clear_screen()

            class_label = f"classId={allowed_class_id}" if allowed_class_id is not None else "all classIds"
            timestamp = time.strftime('%H:%M:%S') if args.watch else time.strftime('%H:%M')
            print(
                f"{BOLD}{YELLOW}⚡ SportsBet Racing Monitor — {args.date}   {timestamp}  ({len(races)} races){RESET}"
            )
            if not args.watch:
                print(f"Showing {len(races)} races ({class_label})")

            index = []
            failures = []
            race_blocks: list[list[str]] = []

            if not args.watch:
                all_racing_path = outdir / f"all_racing_{args.date}.json"
                all_racing_path.write_text(json.dumps(all_racing, ensure_ascii=False, indent=2))

            for i, race in enumerate(races, start=1):
                event_id = race["event_id"]
                class_id = race["class_id"]
                if class_id is None:
                    failures.append({**race, "error": "missing classId"})
                    if not args.watch:
                        print(f"[{i}/{len(races)}] skip event {event_id}: missing classId")
                    continue

                url = f"{BASE}/Events/{event_id}/RacecardWithContext?classId={class_id}"

                try:
                    data = get_json(session, url)
                    if not args.watch:
                        race_path = outdir / f"race_{event_id}_class_{class_id}.json"
                        race_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                        index.append({**race, "url": url, "file": str(race_path)})
                        print(f"{GREEN}✓ {RESET}[{i}/{len(races)}] ok event={event_id} classId={class_id}")

                    event = (data.get("racecardEvent") or {}) if isinstance(data, dict) else {}
                    track = event.get("competitionName") or race.get("meeting_name") or ""
                    race_no = event.get("raceNumber") or race.get("race_number")
                    start_time = event.get("startTime") or race.get("start_time")
                    market = _extract_win_market(event)
                    if market is not None:
                        selections = [sel for sel in (market.get("selections") or []) if not sel.get("isOut")]
                        block = _render_race_block(track, race_no, start_time, selections, args.watch)
                        race_blocks.append(block)
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    failures.append({**race, "url": url, "error": f"HTTP {status}"})
                    if not args.watch:
                        print(f"{RED}✗ {RESET}[{i}/{len(races)}] fail event={event_id} classId={class_id}: HTTP {status}")
                except Exception as exc:  # noqa: BLE001
                    failures.append({**race, "url": url, "error": str(exc)})
                    if not args.watch:
                        print(f"{RED}✗ {RESET}[{i}/{len(races)}] fail event={event_id} classId={class_id}: {exc}")

                if args.sleep > 0:
                    time.sleep(args.sleep)

            # print races side-by-side in pairs
            for j in range(0, len(race_blocks), 2):
                left = race_blocks[j]
                right = race_blocks[j + 1] if j + 1 < len(race_blocks) else None
                _print_side_by_side(left, right)
                print()

            if args.watch:
                if failures:
                    print("")
                    print(f"Failures: {len(failures)}")
                time.sleep(max(0.0, args.refresh_seconds))
                continue

            index_path = outdir / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "date": args.date,
                        "all_racing_url": all_racing_url,
                        "total_races": len(races),
                        "successful": len(index),
                        "failed": len(failures),
                        "class_filter": allowed_class_id,
                        "items": index,
                        "failures": failures,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            print(f"Wrote all racing JSON to: {all_racing_path}")
            print(f"Wrote index JSON to: {index_path}")
            print(f"Success={len(index)} Failed={len(failures)}")
            return 0 if not failures else 1
    finally:
        if session is not None:
            session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
