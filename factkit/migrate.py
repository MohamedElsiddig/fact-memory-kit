"""Per-model embedding migration: plan, backfill, verify, roll back.

Switching embedding models is **additive**. New vectors are written beside the
old ones in ``fact_embeddings(fact_id, model, dims, vec, created_at)``; no row
belonging to another model is ever updated or deleted by :func:`run`. That is
what makes a switch reversible: if the new model retrieves worse, point
retrieval back at the old name and the old vectors are still there, byte for
byte.

Write discipline
----------------
* :func:`plan`, :func:`verify` and dry-run :func:`run` are read-only.
* :func:`run` inserts with ``INSERT OR IGNORE`` (append-only): an existing
  ``(fact_id, model)`` row keeps its original bytes and timestamp. Rows are
  committed one batch at a time, so an interrupted run resumes where it
  stopped and a re-run simply plans zero work.
* :func:`rollback` is the only function that deletes, only rows of the named
  model, and only behind the literal confirm token. It refuses to remove the
  last remaining model unless ``force=True``, because that would leave recall
  with no vectors at all.

``dims`` is always taken from the vector the endpoint returned, never assumed
from configuration, so a misconfigured client shows up as a verify-time dims
mismatch instead of silently poisoning the corpus.
"""
from __future__ import annotations

import math
import sqlite3
import time
from typing import Any, Callable, Iterable, Sequence

from . import vectors

DEFAULT_RATE = 8.0          # embeddings per second, conservative Pi-class guess
CONFIRM_TOKEN = "remove-model-vectors"
MAX_MODEL_LEN = 128


# ------------------------------------------------------------------ helpers
def _validate_model(model: str) -> str:
    """Return ``model`` unchanged, or raise ``ValueError``.

    The name is stored exactly as given: no case folding, no trimming. Only
    names that cannot be a usable key (blank) or that exceed the column budget
    are rejected.
    """
    if not isinstance(model, str):
        raise TypeError("model name must be a string")
    if not model.strip():
        raise ValueError("model name must not be empty")
    if len(model) > MAX_MODEL_LEN:
        raise ValueError(
            f"model name is {len(model)} chars; the limit is {MAX_MODEL_LEN}")
    return model


def _validate_batch(batch: int) -> int:
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    return batch


def _validate_rate(rate: float) -> float:
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("rate must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("rate must be a positive number")
    return value


def _has_table(store, name: str = "fact_embeddings") -> bool:
    row = _db(store).execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone()
    return row is not None


def _db(store):
    """The sqlite connection behind any store shape we support."""
    for name in ("db", "_conn", "conn"):
        conn = getattr(store, name, None)
        if isinstance(conn, sqlite3.Connection):
            return conn
    path = getattr(store, "db_path", None)
    if path:
        return sqlite3.connect(str(path))
    raise ValueError("store exposes no sqlite connection")


def _reject_readonly(store) -> None:
    if getattr(store, "readonly", False):
        raise ValueError("store is read-only; open it writable to migrate")


def _total_facts(store) -> int:
    """Total facts across the store shapes we support (bare, plugin, provider)."""
    for name in ("count_facts", "count"):
        fn = getattr(store, name, None)
        if callable(fn):
            try:
                return int(fn())
            except (sqlite3.Error, TypeError, ValueError):
                pass
    try:
        rows = store.read("SELECT COUNT(*) AS n FROM facts")
        return int(rows[0]["n"]) if rows else 0
    except Exception:  # noqa: BLE001 - a plan must not explode on shape drift
        return 0


def _missing_facts(store, model: str) -> list[dict]:
    """Facts without a vector for ``model``, oldest id first."""
    if not _has_table(store):
        try:
            return list(store.list_facts(limit=1_000_000))
        except (sqlite3.Error, TypeError):
            return []
    rows = _db(store).execute(
        "SELECT f.fact_id, f.content FROM facts f "
        "LEFT JOIN fact_embeddings e ON e.fact_id = f.fact_id AND e.model = ? "
        "WHERE e.fact_id IS NULL ORDER BY f.fact_id", (model,)).fetchall()
    return [{"fact_id": int(r[0]), "content": r[1]} for r in rows]


def _blank_content(fact: dict) -> bool:
    return not str(fact.get("content") or "").strip()


def _model_summary(store) -> dict[str, dict[str, Any]]:
    """``{model: {rows, facts, dims, duplicates}}`` for every stored model."""
    if not _has_table(store):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _db(store).execute(
            "SELECT model, COUNT(*) AS rows, COUNT(DISTINCT fact_id) AS facts "
            "FROM fact_embeddings GROUP BY model ORDER BY model"):
        out[str(row["model"])] = {
            "rows": int(row["rows"]),
            "facts": int(row["facts"]),
            "dims": [],
            "duplicates": int(row["rows"]) - int(row["facts"]),
            "unusable": 0,
        }
    for row in _db(store).execute(
            "SELECT model, dims, COUNT(*) AS n FROM fact_embeddings "
            "GROUP BY model, dims ORDER BY model, dims"):
        entry = out.setdefault(str(row["model"]),
                               {"rows": 0, "facts": 0, "dims": [],
                                "duplicates": 0, "unusable": 0})
        entry["dims"].append(int(row["dims"]))
    for row in _db(store).execute(
            "SELECT model, COUNT(*) AS n FROM fact_embeddings "
            "WHERE dims <= 0 OR vec IS NULL OR LENGTH(vec) != dims * 4 "
            "GROUP BY model"):
        entry = out.get(str(row["model"]))
        if entry is not None:
            entry["unusable"] = int(row["n"])
    return out


def _existing_dims(store, model: str) -> int | None:
    """The single dims value for ``model``, or ``None`` when absent/mixed."""
    if not _has_table(store):
        return None
    dims = [int(r[0]) for r in _db(store).execute(
        "SELECT DISTINCT dims FROM fact_embeddings WHERE model=?", (model,))]
    return dims[0] if len(dims) == 1 else None


def _pack_vector(vec: Sequence[float]) -> tuple[bytes, int]:
    """Validate a returned vector and pack it as float32."""
    if vec is None:
        raise ValueError("client returned no vector")
    try:
        values = [float(x) for x in vec]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector must be a sequence of numbers") from exc
    if not values:
        raise ValueError("vector is empty")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("vector contains non-finite values")
    return vectors.pack(values), len(values)


def _notify(callback: Callable[[dict], None] | None, event: dict) -> None:
    """Progress reporting must never break the migration itself."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - a UI callback is not a data path
        pass


# --------------------------------------------------------------------- plan
def plan(store, target_model: str, *, batch: int = 64,
         rate: float = DEFAULT_RATE) -> dict:
    """Describe what :func:`run` would do. Read-only; no client, no writes.

    ``blocked`` lists **data** reasons only — the caller owns endpoint health
    (``client.health()``) because an unreachable endpoint is not a property of
    the store.

    Returns ``target_model``, ``total_facts``, ``missing``, ``present``,
    ``batch``, ``rate``, ``api_calls``, ``estimated_s``, ``models``,
    ``counts_by_model``, ``blocked`` and ``ready``.
    """
    model = _validate_model(target_model)
    batch = _validate_batch(batch)
    rate = _validate_rate(rate)

    total = _total_facts(store)
    missing_rows = _missing_facts(store, model)
    missing = len(missing_rows)
    models = _model_summary(store)

    blocked: list[str] = []
    blanks = [f for f in missing_rows if _blank_content(f)]
    if blanks:
        blocked.append(f"{len(blanks)} fact(s) have empty content")
    target = models.get(model)
    if target:
        if target["duplicates"]:
            blocked.append(
                f"{target['duplicates']} duplicate row(s) already exist for {model!r}")
        if len(target["dims"]) > 1:
            blocked.append(
                f"existing {model!r} vectors disagree on dims: "
                + ", ".join(str(d) for d in target["dims"]))
        if target["unusable"]:
            blocked.append(
                f"{target['unusable']} existing {model!r} row(s) have unusable "
                "vectors; roll back that model and re-run")

    return {
        "target_model": model,
        "total_facts": total,
        "missing": missing,
        "present": max(total - missing, 0),
        "batch": batch,
        "rate": rate,
        "api_calls": math.ceil(missing / batch) if missing else 0,
        "estimated_s": round(missing / rate, 3),
        "models": models,
        "counts_by_model": {name: entry["facts"] for name, entry in models.items()},
        "blocked": blocked,
        "ready": not blocked,
    }


# ---------------------------------------------------------------------- run
def run(store, client, target_model: str, *, batch: int = 32,
        limit: int | None = None, dry_run: bool = True,
        on_progress: Callable[[dict], None] | None = None) -> dict:
    """Embed facts that lack a vector for ``target_model``, in batches.

    ``dry_run`` (the default) performs no writes and makes no client calls; it
    returns the same shape with every counter at zero so a caller can print a
    plan without a second code path.

    Each batch is one ``client.embed`` call and one committed transaction of
    ``INSERT OR IGNORE`` rows. A failing batch is recorded in ``failures`` and
    the remaining batches still run, so one bad chunk does not cost the whole
    migration. Facts whose content is blank are never sent to the endpoint and
    count as failed.

    ``on_progress`` is called once per batch with ``batch`` (1-based),
    ``batches``, ``planned``, ``processed``, ``embedded``, ``skipped`` and
    ``failed``.

    Returns ``target_model``, ``planned``, ``embedded``, ``skipped``,
    ``failed``, ``dims``, ``elapsed_s`` plus ``dry_run``, ``failures`` and
    ``ok`` (no failures).
    """
    started = time.monotonic()
    model = _validate_model(target_model)
    batch = _validate_batch(batch)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    if not dry_run:
        _reject_readonly(store)

    missing = _missing_facts(store, model)
    if limit is not None:
        missing = missing[:limit]
    planned = len(missing)

    failures: list[dict[str, Any]] = []
    todo: list[dict] = []
    for fact in missing:
        if _blank_content(fact):
            failures.append({"fact_id": fact["fact_id"],
                             "error": "empty content"})
        else:
            todo.append(fact)
    failed = len(failures)

    result: dict[str, Any] = {
        "target_model": model,
        "planned": planned,
        "embedded": 0,
        "skipped": 0,
        "failed": failed,
        "dims": _existing_dims(store, model),
        "elapsed_s": 0.0,
        "dry_run": bool(dry_run),
        "failures": failures,
    }
    if dry_run:
        result["ok"] = failed == 0
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        return result

    embedded = skipped = 0
    dims = result["dims"]
    batches = math.ceil(len(todo) / batch) if todo else 0
    processed = 0

    for index, start in enumerate(range(0, len(todo), batch), start=1):
        chunk = todo[start:start + batch]
        processed += len(chunk)
        try:
            vectors_out = client.embed([f["content"] for f in chunk])
            if len(vectors_out) != len(chunk):
                raise ValueError(
                    f"client returned {len(vectors_out)} vectors for "
                    f"{len(chunk)} inputs")
        except Exception as exc:  # noqa: BLE001 - one batch must not abort the run
            failed += len(chunk)
            failures.append({"fact_ids": [f["fact_id"] for f in chunk],
                             "error": str(exc)[:300]})
            _notify(on_progress, {
                "batch": index, "batches": batches, "planned": planned,
                "processed": processed, "embedded": embedded,
                "skipped": skipped, "failed": failed})
            continue

        rows: list[tuple[int, int, bytes]] = []
        for fact, vec in zip(chunk, vectors_out):
            try:
                blob, dim = _pack_vector(vec)
            except ValueError as exc:
                failed += 1
                failures.append({"fact_id": fact["fact_id"], "error": str(exc)})
                continue
            rows.append((int(fact["fact_id"]), dim, blob))

        batch_embedded = batch_skipped = 0
        new_dims: int | None = None
        try:
            with _db(store):  # commit per batch: resumable after a crash
                for fact_id, dim, blob in rows:
                    cur = _db(store).execute(
                        "INSERT OR IGNORE INTO fact_embeddings"
                        "(fact_id, model, dims, vec) VALUES (?,?,?,?)",
                        (fact_id, model, dim, blob))
                    if cur.rowcount:
                        batch_embedded += 1
                        if new_dims is None:
                            new_dims = dim
                    else:
                        batch_skipped += 1
        except sqlite3.Error as exc:
            failed += len(rows)
            failures.append({"fact_ids": [r[0] for r in rows],
                             "error": f"store write failed: {exc}"[:300]})
        else:
            embedded += batch_embedded
            skipped += batch_skipped
            if dims is None and new_dims is not None:
                dims = new_dims

        _notify(on_progress, {
            "batch": index, "batches": batches, "planned": planned,
            "processed": processed, "embedded": embedded,
            "skipped": skipped, "failed": failed})

    result.update({
        "embedded": embedded,
        "skipped": skipped,
        "failed": failed,
        "dims": dims,
        "failures": failures,
        "ok": failed == 0,
    })
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


# ------------------------------------------------------------------- verify
def verify(store, target_model: str) -> dict:
    """Audit coverage and integrity of ``target_model`` vectors.

    ``coverage`` counts facts holding at least one structurally valid row.
    A row is a dim mismatch when its ``dims`` is not positive, its BLOB length
    disagrees with ``dims``, or its ``dims`` differs from the model's dominant
    value (a corpus split between two dimensions cannot be searched as one).
    ``duplicates`` is the number of surplus rows over one per fact.

    ``ok`` requires coverage exactly 1.0, zero dim mismatches and zero
    duplicates. An empty store has nothing to cover and verifies as ok.
    """
    model = _validate_model(target_model)
    total = _total_facts(store)

    rows: list[sqlite3.Row] = []
    if _has_table(store):
        rows = list(_db(store).execute(
            "SELECT fact_id, dims, vec FROM fact_embeddings WHERE model=?",
            (model,)))

    counts: dict[int, int] = {}
    valid: list[tuple[int, int]] = []          # (fact_id, dims)
    for row in rows:
        dims = row["dims"]
        blob = row["vec"]
        ok_row = (isinstance(dims, int) and dims > 0
                  and isinstance(blob, (bytes, bytearray, memoryview))
                  and len(bytes(blob)) == dims * 4)
        if ok_row:
            counts[dims] = counts.get(dims, 0) + 1
            valid.append((int(row["fact_id"]), int(dims)))

    expected = max(counts, key=lambda d: (counts[d], -d)) if counts else None
    mismatches = len(rows) - sum(1 for _, d in valid if d == expected)
    covered = len({fid for fid, _ in valid})
    duplicates = len(rows) - len({int(r["fact_id"]) for r in rows})
    coverage = 1.0 if total == 0 else covered / total

    return {
        "target_model": model,
        "total_facts": total,
        "covered": covered,
        "missing": max(total - covered, 0),
        "coverage": coverage,
        "dims": sorted(counts),
        "expected_dims": expected,
        "dim_mismatches": mismatches,
        "duplicates": duplicates,
        "rows": len(rows),
        "ok": coverage == 1.0 and mismatches == 0 and duplicates == 0,
    }


# ----------------------------------------------------------------- rollback
def rollback(store, target_model: str, *, confirm_token: str,
             force: bool = False) -> dict:
    """Delete every vector for ``target_model``; nothing else.

    ``confirm_token`` must be the literal ``remove-model-vectors``. When the
    target is the only model left in the table the call refuses (that would
    leave recall with no vectors at all) unless ``force=True``.

    Returns ``target_model``, ``removed``, ``remaining_models``, ``forced``
    and ``ok``. Removing a model that is not present is a no-op, not an error.
    """
    model = _validate_model(target_model)
    if confirm_token != CONFIRM_TOKEN:
        raise ValueError(
            f"confirmation token must be {CONFIRM_TOKEN!r}")
    _reject_readonly(store)

    if not _has_table(store):
        return {"target_model": model, "removed": 0, "remaining_models": [],
                "forced": bool(force), "ok": True}

    present = {str(r[0]) for r in _db(store).execute(
        "SELECT DISTINCT model FROM fact_embeddings")}
    target_rows = int(_db(store).execute(
        "SELECT COUNT(*) FROM fact_embeddings WHERE model=?",
        (model,)).fetchone()[0])

    if target_rows and present == {model} and not force:
        raise ValueError(
            f"{model!r} is the only model with vectors; removing it would "
            "leave no recall. Pass force=True to do it anyway.")

    with _db(store):
        cur = _db(store).execute(
            "DELETE FROM fact_embeddings WHERE model=?", (model,))
        removed = max(cur.rowcount, 0)

    remaining = [str(r[0]) for r in _db(store).execute(
        "SELECT DISTINCT model FROM fact_embeddings ORDER BY model")]
    return {"target_model": model, "removed": removed,
            "remaining_models": remaining, "forced": bool(force), "ok": True}
