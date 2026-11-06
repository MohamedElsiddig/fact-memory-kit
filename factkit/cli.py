"""factkit command line."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import (bridge, bulk, capture, conflicts, entities, golden, metrics,
               router, ttl, tui)
from .config import Config
from .embed import EmbeddingClient, from_config
from .hybrid import HybridSearch
from .store import Store


# --------------------------------------------------------------------- helpers
def _store(args) -> Store:
    return Store(Config.load(getattr(args, "db", None)).db_path)


def _cfg(args) -> Config:
    return Config.load(getattr(args, "db", None))


def _embedder(cfg: Config) -> EmbeddingClient:
    return from_config(cfg)


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=1, ensure_ascii=False, default=str))


def index_embeddings(store: Store, cfg: Config, *, batch_report: bool = True,
                     only_missing: bool = True) -> dict:
    client = _embedder(cfg)
    model = client.model
    todo = store.facts_missing_embedding(model) if only_missing else store.list_facts(limit=1_000_000)
    if not todo:
        return {"model": model, "indexed": 0, "pending": 0}
    vectors = client.try_embed([f["content"] for f in todo])
    if vectors is None:
        raise SystemExit(
            f"embedding service unreachable at {cfg.embed_base_url} "
            f"(model {model}). Facts are unchanged; lexical search still works.")
    for fact, vec in zip(todo, vectors):
        store.put_embedding(fact["fact_id"], model, vec)
    store.db.commit()
    store.metric("index", "embedded", len(todo), model=model)
    store.db.commit()
    if batch_report:
        print(f"indexed {len(todo)} fact(s) with {model}")
    return {"model": model, "indexed": len(todo), "pending": len(todo)}


def _gate(store: Store, args, *, quiet: bool = False) -> int:
    path = Path(args.golden).expanduser() if getattr(args, "golden", None) else None
    cases, path, thresholds = golden.load_cases(path, store)
    search = HybridSearch(store, _embedder(_cfg(args)) if not args.no_vectors else None,
                          model=_cfg(args).embed_model).search
    result = golden.run(store, search, cases)
    golden.check(result, thresholds)
    if not getattr(args, "quiet", False):
        print(result.render(verbose=getattr(args, "verbose", False)))
        print(f"\ngolden file: {path} ({len(cases)} cases)")
    return 0 if result.ok else 1


def _fmt_fact(fact: dict) -> str:
    return (f"#{fact['fact_id']}  [{fact.get('category', 'general')}]  "
            f"trust {fact.get('trust_score', 0):.2f}  {fact.get('updated_at', '')}\n"
            f"  {fact['content']}")


# -------------------------------------------------------------------- commands
def cmd_init(args) -> int:
    cfg = _cfg(args)
    store = Store(cfg.db_path)
    total = store.count()
    print(f"database ready: {cfg.db_path} ({total} facts)")
    if args.demo and total == 0:
        demo = [
            ("Atlas Server (Debian, 192.0.2.10) - 16 GB, 4 cores at 2.4 GHz.", "entity", "node-a,specs"),
            ("Workstation (example-host, 192.0.2.30) - 8 cores, 32 GB RAM.", "entity", "workstation,specs"),
            ("The dashboard service runs in a container on node-a with host networking.", "tool", "dashboard,container"),
            ("Mail is sent through the CLI; piped messages need an explicit From header.", "tool", "email,cli"),
            ("User prefers short bullet answers over long prose.", "user_pref", "style"),
        ]
        for content, category, tags in demo:
            fact = store.add_fact(content, category=category, tags=tags,
                                  trust_score=0.9, source="demo")
            entities.link(store, fact["fact_id"], content)
        print(f"seeded {len(demo)} demo facts")
    return 0


def cmd_add(args) -> int:
    store, cfg = _store(args), _cfg(args)
    fact = store.add_fact(args.content, category=args.category, tags=args.tags,
                          trust_score=args.trust, source=args.source,
                          review_after=args.review_after, reason=args.reason or "")
    entities.link(store, fact["fact_id"], args.content)
    if not args.no_embed:
        try:
            client = _embedder(cfg)
            vec = client.embed_one(args.content)
            store.put_embedding(fact["fact_id"], client.model, vec)
            store.db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"warning: not embedded ({exc})", file=sys.stderr)
    print(_fmt_fact(store.get(fact["fact_id"])))
    return 0


def cmd_get(args) -> int:
    store = _store(args)
    fact = store.get(args.fact_id) if args.fact_id else None
    if fact is None:
        print(f"no fact #{args.fact_id}", file=sys.stderr)
        return 2
    print(_fmt_fact(fact))
    ents = store.fact_entities(fact["fact_id"])
    if ents:
        print("  entities: " + ", ".join(e["name"] for e in ents))
    return 0


def cmd_list(args) -> int:
    store = _store(args)
    for fact in store.list_facts(limit=args.limit, offset=args.offset,
                                 category=args.category):
        print(f"#{fact['fact_id']:>4}  {fact['category']:<10} {fact['content'][:100]}")
    return 0


def cmd_search(args) -> int:
    store, cfg = _store(args), _cfg(args)
    embedder = None if args.no_vectors else _embedder(cfg)
    search = HybridSearch(store, embedder, model=cfg.embed_model)
    hits = search.search(args.query, k=args.k, category=args.category,
                         use_vectors=not args.no_vectors)
    if args.json:
        _print_json([h.as_dict() for h in hits])
        return 0
    if not hits:
        print("no hits")
        return 0
    for hit in hits:
        flags = []
        if hit.bm25_rank:
            flags.append(f"bm25#{hit.bm25_rank}")
        if hit.vec_rank:
            flags.append(f"vec#{hit.vec_rank} {hit.vec_score:.3f}")
        print(f"#{hit.fact_id:>4}  {hit.score:.4f}  {' '.join(flags):<26} {hit.content[:90]}")
    return 0


def cmd_update(args) -> int:
    store = _store(args)
    fact = store.update_fact(args.fact_id, content=args.content,
                             category=args.category, tags=args.tags,
                             trust_score=args.trust, reason=args.reason or "")
    _print_json(fact)
    return 0


def cmd_remove(args) -> int:
    store = _store(args)
    ok = store.remove_fact(args.fact_id, reason=args.reason or "")
    print("removed" if ok else "not found")
    return 0 if ok else 2


def cmd_embed(args) -> int:
    store, cfg = _store(args), _cfg(args)
    client = _embedder(cfg)
    if args.health:
        _print_json(client.health())
        return 0
    result = index_embeddings(store, cfg, only_missing=not args.all)
    _print_json(result)
    if not args.no_gate:
        return _gate(store, args)
    return 0


def cmd_gate(args) -> int:
    return _gate(_store(args), args)


def cmd_golden_write(args) -> int:
    store = _store(args)
    cases = [golden.Case(**c) for c in golden.seed_cases(store, limit=args.limit)]
    path = golden.write_cases(cases, Path(args.out) if args.out else None)
    print(f"wrote {len(cases)} cases to {path}")
    return 0


def cmd_conflicts(args) -> int:
    store = _store(args)
    if args.action == "scan":
        found = conflicts.scan(store, overlap_min=args.overlap)
        print(f"{store.count()} facts -> {len(found)} conflict candidate(s)")
        for prop in found:
            print(f"KEEP #{prop.keep} <- retire #{prop.retire}  j={prop.jaccard} "
                  f"slots={','.join(prop.slots)}")
            for delta in prop.delta:
                print(f"   ! {delta}")
        return 0 if not found or not args.gate else 1
    if args.action == "list":
        rows = store.conflicts("pending")
        if args.json:
            _print_json(rows)
        else:
            for row in rows:
                print(f"[{row['id']}] keep #{row['keep_id']} retire #{row['retire_id']} "
                      f"j={row['jaccard']} {row['delta']}")
        return 0
    if args.action in ("apply", "reject"):
        if args.action == "apply":
            result = tui.decide(store, "conflict", args.id, "accept")
        else:
            result = tui.decide(store, "conflict", args.id, "reject")
        _print_json(result)
        return 0 if result.get("ok") else 2
    return 2


def cmd_ttl(args) -> int:
    store, cfg = _store(args), _cfg(args)
    if args.action == "scan":
        flags = ttl.scan(store, cfg.ttl_days, cfg.slot_ttl_days)
        print(f"{len(flags)} flag(s)")
        for flag in flags:
            print(f"#{flag['fact_id']:>4} {flag['kind']:<5} {flag['age_days']:>6.0f}d  "
                  f"{flag['detail'][:100]}")
        return 0
    if args.action == "list":
        rows = store.ttl_flags("pending")
        if args.json:
            _print_json(rows)
        else:
            for row in rows:
                print(f"[{row['id']}] #{row['fact_id']} {row['kind']} "
                      f"{row['age_days']:.0f}d {row['detail'][:90]}")
        return 0
    if args.action == "ack":
        ttl.acknowledge(store, args.id)
        print(f"acknowledged flag {args.id}")
        return 0
    return 2


def cmd_review(args) -> int:
    store = _store(args)
    if args.json:
        _print_json(tui.queue(store))
        return 0
    if args.action:
        _print_json(tui.decide(store, args.kind, args.id, args.action))
        return 0
    if args.curses or (sys.stdout.isatty() and not args.line):
        from . import tui_curses
        return tui_curses.interactive(store)
    return tui.interactive(store)


def cmd_capture(args) -> int:
    result = capture.suggest(args.text)
    _print_json(result.as_dict())
    return 0


def cmd_route(args) -> int:
    result = router.route(args.query, _cfg(args) if args.llm else None,
                          use_llm=args.llm)
    _print_json(result.as_dict())
    return 0


def cmd_bridge(args) -> int:
    store, cfg = _store(args), _cfg(args)
    client = _embedder(cfg)
    result = bridge.run(store, cfg, Path(args.path).expanduser(),
                        include_assistant=args.include_assistant,
                        embedder=client, model=client.model,
                        max_chunks=args.max_chunks, dry_run=args.dry_run,
                        session_id=args.session, sessions=args.sessions)
    _print_json({k: v for k, v in result.items() if k != "facts"})
    for fact in result["facts"]:
        print(f"  + [{fact['category']}] {fact['content'][:100]}")
    return 0


def cmd_props(args) -> int:
    store = _store(args)
    if args.action == "list":
        rows = store.proposals("pending")
        if args.json:
            _print_json(rows)
        else:
            for row in rows:
                print(f"[{row['id']}] [{row['category']}] {row['content'][:100]}")
        return 0
    if args.action == "apply":
        fact = bridge.apply(store, args.id)
        print(_fmt_fact(fact))
        return 0
    if args.action == "reject":
        store.resolve_proposal(args.id, "rejected")
        store.db.commit()
        print(f"rejected {args.id}")
        return 0
    return 2


def cmd_audit(args) -> int:
    store = _store(args)
    rows = store.audit_log(limit=args.limit, fact_id=args.fact_id)
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        print(f"{row['ts']}  {row['action']:<7} #{row['fact_id']}  "
              f"{row['actor']}  {row['reason'][:60]}")
    return 0


def cmd_export(args) -> int:
    store = _store(args)
    text = bulk.export_facts(store, None, fmt=args.format,
                             include_embeddings=args.with_embeddings)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_import(args) -> int:
    store = _store(args)
    if args.hermes:
        result = bulk.import_hermes(store, Path(args.hermes), dry_run=args.dry_run)
        _print_json({k: v for k, v in result.items() if k != "detail"})
        return 0
    if not args.path:
        print("error: give a path or --hermes <memory_store.db>", file=sys.stderr)
        return 2
    result = bulk.import_facts(store, Path(args.path), dry_run=args.dry_run,
                               default_category=args.category)
    _print_json({k: v for k, v in result.items() if k != "detail"})
    if args.validate_golden:
        print(json.dumps(bulk.validate_against_golden(
            Path(args.path), Path(args.validate_golden)), indent=1))
    return 0


def cmd_report(args) -> int:
    store = _store(args)
    data = metrics.summary(store, days=args.days)
    if args.json:
        _print_json(data)
    elif args.html:
        print(metrics.render_html(data))
    else:
        print(metrics.render_text(data))
    return 0


def cmd_dashboard(args) -> int:
    from .server import serve
    token = args.token or os.environ.get("FACTKIT_DASHBOARD_TOKEN", "")
    serve(_store(args), host=args.host, port=args.port, token=token)
    return 0


def cmd_entity(args) -> int:
    store = _store(args)
    if args.action == "list":
        for ent in store.entities():
            print(f"#{ent['entity_id']:>3}  {ent['name']:<36} {ent['n']} fact(s)")
        return 0
    if args.action == "relink":
        _print_json(entities.relink_all(store))
        return 0
    if args.action == "link":
        print(entities.link(store, args.fact_id))
        return 0
    if args.action == "alias":
        entity_id = entities.resolve(store, args.name)
        if entity_id is None:
            entity_id = store.ensure_entity(args.name)
        store.add_alias(args.alias, entity_id)
        store.db.commit()
        print(f"alias {args.alias} -> {args.name} (#{entity_id})")
        return 0
    return 2


def cmd_doctor(args) -> int:
    store, cfg = _store(args), _cfg(args)
    client = _embedder(cfg)
    print(f"database      {cfg.db_path}")
    print(f"facts         {store.count()}")
    print(f"embedding     {client.model} @ {cfg.embed_base_url}")
    print(f"health        {json.dumps(client.health())}")
    print(f"coverage      {json.dumps(store.embedding_models())}")
    return 0


# ----------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factkit",
        description="Local-first fact memory: hybrid search, golden gate, conflicts, TTL, audit, capture.")
    parser.add_argument("--db", help="database path (default ~/.factkit/facts.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create the database (optionally seed a demo)")
    p.add_argument("--demo", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="add a fact")
    p.add_argument("content")
    p.add_argument("-c", "--category", default="general")
    p.add_argument("-t", "--tags", default="")
    p.add_argument("--trust", type=float, default=0.5)
    p.add_argument("--source", default="manual")
    p.add_argument("--review-after", default=None)
    p.add_argument("--reason", default="")
    p.add_argument("--no-embed", action="store_true")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("get", help="show one fact")
    p.add_argument("fact_id", type=int)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("list", help="list facts")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--category")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("search", help="hybrid BM25 + embedding search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--category")
    p.add_argument("--no-vectors", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("update", help="update a fact")
    p.add_argument("fact_id", type=int)
    p.add_argument("--content")
    p.add_argument("--category")
    p.add_argument("--tags")
    p.add_argument("--trust", type=float)
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("remove", help="remove a fact")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("embed", help="index embeddings, then run the golden gate")
    p.add_argument("--all", action="store_true", help="re-embed everything")
    p.add_argument("--health", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--no-vectors", action="store_true")
    p.add_argument("--golden")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("gate", help="run the golden regression gate")
    p.add_argument("--golden")
    p.add_argument("--no-vectors", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("golden-write", help="(re)seed golden.json from the corpus")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--out")
    p.set_defaults(func=cmd_golden_write)

    p = sub.add_parser("conflicts", help="golden-rule stale-fact detection")
    p.add_argument("action", choices=["scan", "list", "apply", "reject"], nargs="?", default="scan")
    p.add_argument("id", type=int, nargs="?")
    p.add_argument("--overlap", type=float, default=conflicts.OVERLAP_MIN)
    p.add_argument("--gate", action="store_true", help="exit 1 when candidates exist")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_conflicts)

    p = sub.add_parser("ttl", help="flag aged or rotting facts")
    p.add_argument("action", choices=["scan", "list", "ack"], nargs="?", default="scan")
    p.add_argument("id", type=int, nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ttl)

    p = sub.add_parser("review", help="interactive review of all queues")
    p.add_argument("--json", action="store_true")
    p.add_argument("--curses", action="store_true", help="force the full-screen UI")
    p.add_argument("--line", action="store_true", help="force the simple line UI")
    p.add_argument("--kind", choices=["conflict", "proposal", "ttl"])
    p.add_argument("--id", type=int)
    p.add_argument("--action", choices=["accept", "reject", "skip"])
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("capture", help="should this text become a fact?")
    p.add_argument("text")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("route", help="which source should answer this query?")
    p.add_argument("query")
    p.add_argument("--llm", action="store_true")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("bridge", help="propose facts from a transcript file")
    p.add_argument("path")
    p.add_argument("--include-assistant", action="store_true")
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--session", help="Hermes session id (when path is a .db)")
    p.add_argument("--sessions", type=int, default=3,
                   help="recent Hermes sessions to read (default 3)")
    p.set_defaults(func=cmd_bridge)

    p = sub.add_parser("props", help="review bridge proposals")
    p.add_argument("action", choices=["list", "apply", "reject"], nargs="?", default="list")
    p.add_argument("id", type=int, nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_props)

    p = sub.add_parser("audit", help="append-only change log")
    p.add_argument("--fact-id", type=int)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("export", help="export JSON or CSV")
    p.add_argument("--format", choices=["json", "csv"], default="json")
    p.add_argument("--out")
    p.add_argument("--with-embeddings", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="import JSON/CSV (or --hermes) with dry-run")
    p.add_argument("path", nargs="?")
    p.add_argument("--hermes", metavar="MEMORY_DB",
                   help="import facts from a Hermes memory_store.db (read-only)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--category", default="general")
    p.add_argument("--validate-golden")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("report", help="metrics summary")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--json", action="store_true")
    p.add_argument("--html", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="serve the review dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", default="",
                   help="require this bearer token (or FACTKIT_DASHBOARD_TOKEN)")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("entity", help="entities and aliases")
    p.add_argument("action", choices=["list", "relink", "link", "alias"], nargs="?", default="list")
    p.add_argument("fact_id", type=int, nargs="?")
    p.add_argument("--name")
    p.add_argument("--alias")
    p.set_defaults(func=cmd_entity)

    p = sub.add_parser("sync", help="mirror a Hermes memory_store.db into factkit")
    p.add_argument("--hermes", required=True, metavar="MEMORY_DB")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prune", action="store_true", help="retire facts deleted upstream")
    p.add_argument("--no-embed", action="store_true")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("doctor", help="environment and embedding health")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_sync(args) -> int:
    store, cfg = _store(args), _cfg(args)
    result = bulk.sync_hermes(store, Path(args.hermes), dry_run=args.dry_run,
                              prune=args.prune)
    _print_json({k: v for k, v in result.items() if k != "detail"})
    if not args.dry_run and (result["added"] or result["updated"]) and not args.no_embed:
        try:
            index_embeddings(store, cfg)
        except SystemExit as exc:
            print(f"warning: {exc}", file=sys.stderr)
    return 0
