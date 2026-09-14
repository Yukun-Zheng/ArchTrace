"""Stable URL query encoding for explorer state."""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode

from archtrace.query.models import ExplorerLevel, ExplorerState


def encode_explorer_state(state: ExplorerState) -> str:
    """Encode explorer presentation state without mutating ATIR."""

    pairs: list[tuple[str, str]] = [("view", state.view.value)]
    if state.selected_key:
        pairs.append(("selected", state.selected_key))
    if state.run_id:
        pairs.append(("run", state.run_id))
    for key in state.expanded_keys:
        pairs.append(("expanded", key))
    if state.search:
        pairs.append(("q", state.search))
    return urlencode(pairs)


def decode_explorer_state(query: str) -> ExplorerState:
    """Decode a URL query string into validated explorer state."""

    raw = query[1:] if query.startswith("?") else query
    values = parse_qs(raw, keep_blank_values=False)
    raw_view = _first(values, "view") or ExplorerLevel.PAPER.value
    try:
        view = ExplorerLevel(raw_view)
    except ValueError:
        view = ExplorerLevel.PAPER
    return ExplorerState(
        view=view,
        selected_key=_first(values, "selected"),
        run_id=_first(values, "run"),
        expanded_keys=values.get("expanded", []),
        search=_first(values, "q"),
    )


def _first(values: dict[str, list[str]], key: str) -> str | None:
    items = values.get(key)
    return items[0] if items else None
