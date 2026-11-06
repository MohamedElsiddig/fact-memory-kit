"""Full-screen review UI (stdlib curses). Falls back to the line UI.

Keys: j/k or arrows move, a accept, r reject, s skip, g/G first/last, q quit.
"""
from __future__ import annotations

import curses

from . import tui

HELP = " j/k move   a accept   r reject   s skip   g/G top/bottom   q quit "


def _init_colors() -> None:
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
    except curses.error:
        pass


def _attr(pair: int) -> int:
    try:
        return curses.color_pair(pair)
    except curses.error:
        return 0


def _draw(stdscr, items, index: int, status: str) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    list_width = max(28, min(48, width // 2))
    title = f" factkit review — {len(items)} pending "
    stdscr.addnstr(0, 0, title.ljust(width - 1), width - 1, curses.A_REVERSE)

    for row, item in enumerate(items[: height - 3]):
        if row == index:
            attr = curses.A_REVERSE
        else:
            attr = _attr(1) if item["kind"] == "conflict" else 0
        label = f"{item['kind'][:4]:<4} {item['title']}"
        stdscr.addnstr(1 + row, 1, label[: list_width - 3], list_width - 3, attr)

    stdscr.vline(1, list_width, curses.ACS_VLINE, height - 3)

    if items:
        item = items[index]
        x = list_width + 2
        stdscr.addnstr(1, x, item["kind"].upper(), width - x - 1,
                       curses.A_BOLD | _attr(2))
        stdscr.addnstr(2, x, item["title"][: width - x - 1], width - x - 1)
        detail = str(item.get("detail") or "")
        line = 4
        for chunk in detail.split(" | "):
            for i in range(0, len(chunk), max(20, width - x - 2)):
                if line >= height - 3:
                    break
                stdscr.addnstr(line, x, chunk[i: i + width - x - 2], width - x - 2)
                line += 1
        if item.get("keep_id"):
            fact = item.get("_keep")
            if fact:
                stdscr.addnstr(line + 1, x, f"keep: {fact[: width - x - 8]}",
                               width - x - 2, _attr(3))
            retire = item.get("_retire")
            if retire:
                stdscr.addnstr(line + 2, x, f"drop: {retire[: width - x - 8]}",
                               width - x - 2, _attr(4))

    stdscr.addnstr(height - 2, 0, HELP.ljust(width - 1), width - 1, curses.A_REVERSE)
    if status:
        stdscr.addnstr(height - 1, 1, status[: width - 2], width - 2, _attr(3))
    stdscr.refresh()


def _run(stdscr, store) -> int:
    curses.curs_set(0)
    stdscr.keypad(True)
    _init_colors()
    items = tui.queue(store)
    for item in items:
        if item["kind"] == "conflict":
            keep = store.get(item["keep_id"])
            retire = store.get(item["retire_id"])
            item["_keep"] = keep["content"] if keep else "(missing)"
            item["_retire"] = retire["content"] if retire else "(missing)"
    index = 0
    status = ""

    while items:
        _draw(stdscr, items, index, status)
        key = stdscr.getch()
        if key in (ord("q"), 27):
            break
        if key in (ord("j"), curses.KEY_DOWN):
            index = min(len(items) - 1, index + 1)
        elif key in (ord("k"), curses.KEY_UP):
            index = max(0, index - 1)
        elif key == ord("g"):
            index = 0
        elif key == ord("G"):
            index = len(items) - 1
        elif key in (ord("a"), ord("r"), ord("s")):
            action = {ord("a"): "accept", ord("r"): "reject", ord("s"): "skip"}[key]
            item = items[index]
            try:
                tui.decide(store, item["kind"], item["id"], action)
                status = f"{action}ed {item['kind']} #{item['id']}"
            except Exception as exc:  # noqa: BLE001
                status = f"error: {exc}"
            items.pop(index)
            index = min(index, max(0, len(items) - 1))
    return 0


def interactive(store) -> int:
    try:
        return curses.wrapper(_run, store)
    except Exception:  # noqa: BLE001 - terminal cannot do curses
        return tui.interactive(store)
