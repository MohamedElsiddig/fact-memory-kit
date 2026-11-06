"""Read/write dashboard over stdlib HTTP: metrics, queues and one-click apply.

Binds to loopback by default. Set ``FACTKIT_DASHBOARD_TOKEN`` (or ``--token``) to
require a bearer token; a browser can exchange ``/?token=...`` once for an
HttpOnly cookie.

The database is chosen explicitly (``--db`` / ``FACTKIT_DB``) and the live
Hermes store is always opened read-only; see :func:`factkit.config.resolve_dashboard_db`.
"""
from __future__ import annotations

import hmac
import html
import json
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import metrics, tui

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>factkit</title>
<style>
body{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;margin:2rem;color:#111;max-width:60rem}
h1{font-size:1.3rem} table{border-collapse:collapse;width:100%;margin:.5rem 0 2rem}
td,th{padding:.4rem .7rem;border-bottom:1px solid #e6e6e6;text-align:left;vertical-align:top}
button{font:inherit;padding:.15rem .6rem;margin-right:.3rem;cursor:pointer}
.muted{color:#666} code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}
.db{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;padding:.5rem .7rem;
    margin:.4rem 0 1.2rem;border-radius:4px;background:#f4f4f4;word-break:break-all}
.db.ro{background:#fff4e5;border-left:4px solid #b45309}
.db.rw{background:#eef7ee;border-left:4px solid #2f7d32}
</style></head><body>
<h1>factkit dashboard</h1>
<div id="db" class="db __DB_CLASS__" title="database and access mode">__DB_LABEL__</div>
<div id="summary" class="muted">loading…</div>
<h2>Conflicts <span class="muted">(newer supersedes older)</span></h2>
<table id="conflicts"></table>
<h2>Bridge proposals</h2><table id="proposals"></table>
<h2>TTL flags</h2><table id="ttl"></table>
<script>
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function act(kind, id, action) {
  await fetch(`/api/${kind}/${id}/${action}`, {method:'POST'});
  load();
}
async function load() {
  const s = await (await fetch('/api/summary')).json();
  if (s.database) {
    const el = document.getElementById('db');
    el.textContent = s.database.label;
    el.className = 'db ' + (s.database.readonly ? 'ro' : 'rw');
  }
  const ro = !!(s.database && s.database.readonly);
  document.getElementById('summary').textContent =
    `${s.facts} facts · ${s.entities} entities · ${(s.embeddings.coverage*100).toFixed(0)}% embedded · ` +
    `p95 ${s.retrieval.p95_ms||0} ms · ${s.backlog.conflicts_pending} conflicts · ${s.backlog.ttl_pending} TTL` +
    (ro ? ' · read-only (review actions disabled)' : '');
  const render = (rows, el, cols, kind) => {
    const t = document.getElementById(el);
    t.innerHTML = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') +
      (ro ? '' : '<th>action</th>') + '</tr>' +
      (rows.length ? rows.map(r => '<tr>' + cols.map(c => `<td>${esc(r[c])}</td>`).join('') +
        (ro ? '' : `<td><button onclick="act('${kind}',${r.id},'apply')">accept</button>` +
        `<button onclick="act('${kind}',${r.id},'reject')">reject</button></td>`) + '</tr>').join('')
        : `<tr><td colspan="${cols.length + (ro ? 0 : 1)}" class="muted">empty</td></tr>`);
  };
  render(await (await fetch('/api/conflicts')).json(), 'conflicts',
         ['keep_id','retire_id','delta','jaccard'], 'conflict');
  render(await (await fetch('/api/proposals')).json(), 'proposals',
         ['content','category','score'], 'proposal');
  render(await (await fetch('/api/ttl')).json(), 'ttl',
         ['fact_id','kind','detail','age_days'], 'ttl');
}
load();
</script></body></html>"""


def _selection(db):
    """Accept a DashboardDB, a Store or a path and return a DashboardDB."""
    from .config import DashboardDB, resolve_dashboard_db

    if isinstance(db, DashboardDB):
        return db
    path = getattr(db, "path", db)
    selection = resolve_dashboard_db(str(path))
    if getattr(db, "readonly", False) and not selection.readonly:
        selection = DashboardDB(path=selection.path, mode=selection.mode,
                                readonly=True, source=selection.source)
    return selection


def make_handler(db, token: str = ""):
    """One Store per request: sqlite connections are thread-affine, and the
    dashboard is threaded. WAL makes concurrent readers cheap.

    ``db`` may be a :class:`factkit.config.DashboardDB`, a path, or a Store.
    Read-only selections are opened with ``mode=ro`` and POSTs are refused.
    """
    selection = _selection(db)
    page = (PAGE.replace("__DB_LABEL__", html.escape(selection.label))
                .replace("__DB_CLASS__", "ro" if selection.readonly else "rw"))

    class Handler(BaseHTTPRequestHandler):
        server_version = "factkit/0.1"

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass

        # ------------------------------------------------------------------ io
        def _send(self, status: int, body: bytes, ctype: str = "application/json",
                  headers: list[tuple[str, str]] | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in headers or []:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200, headers=None):
            self._send(status,
                       json.dumps(payload, ensure_ascii=False, default=str).encode(),
                       headers=headers)

        # ----------------------------------------------------------------- auth
        def _supplied_token(self, query) -> str | None:
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                return auth[7:].strip()
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            if "fk_token" in cookie:
                return cookie["fk_token"].value
            values = query.get("token")
            return values[0] if values else None

        def _authorised(self, query) -> bool:
            if not token:
                return True
            supplied = self._supplied_token(query)
            return bool(supplied) and hmac.compare_digest(supplied, token)

        def _deny(self):
            self._send(401, json.dumps({"error": "unauthorized"}).encode(),
                       headers=[("WWW-Authenticate", 'Bearer realm="factkit"')])

        # ------------------------------------------------------------------ GET
        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            if not self._authorised(query):
                return self._deny()

            if path in ("/", "/index.html") and token and query.get("token"):
                # exchange the token for a cookie so the JS fetches authenticate
                return self._send(303, b"", "text/plain",
                                  headers=[("Location", "/"),
                                           ("Set-Cookie",
                                            f"fk_token={token}; HttpOnly; SameSite=Strict; Path=/")])
            if path in ("/", "/index.html"):
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if path == "/api/database":
                return self._json(selection.as_dict())
            try:
                with selection.open() as store:
                    if path == "/api/summary":
                        data = metrics.summary(store)
                        data["database"] = selection.as_dict()
                        return self._json(data)
                    if path == "/api/conflicts":
                        return self._json(store.conflicts("pending"))
                    if path == "/api/proposals":
                        return self._json(store.proposals("pending"))
                    if path == "/api/ttl":
                        return self._json(store.ttl_flags("pending"))
                    if path == "/api/facts":
                        limit = int((query.get("limit") or ["50"])[0])
                        return self._json(store.list_facts(limit=limit))
                    if path == "/api/audit":
                        return self._json(store.audit_log(limit=100))
                    if path == "/api/search":
                        from .hybrid import HybridSearch
                        q = (query.get("q") or [""])[0]
                        hits = HybridSearch(store).search(q, k=8, use_vectors=False)
                        return self._json([h.as_dict() for h in hits])
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc), "database": selection.as_dict()}, 500)
            return self._json({"error": "not found"}, 404)

        # ----------------------------------------------------------------- POST
        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if not self._authorised(parse_qs(url.query)):
                return self._deny()
            if selection.readonly:
                return self._json(
                    {"error": f"read-only: {selection.path} is served read-only",
                     "database": selection.as_dict()}, 403)
            parts = [p for p in url.path.split("/") if p]
            if len(parts) == 4 and parts[0] == "api":
                kind, item_id, action = parts[1], parts[2], parts[3]
                if not item_id.isdigit() or action not in ("apply", "reject"):
                    return self._json({"error": "bad request"}, 400)
                if kind not in ("conflict", "proposal", "ttl"):
                    return self._json({"error": "unknown kind"}, 404)
                try:
                    with selection.open() as store:
                        result = tui.decide(store, kind, int(item_id),
                                            "accept" if action == "apply" else "reject")
                    return self._json(result)
                except Exception as exc:  # noqa: BLE001
                    return self._json({"error": str(exc)}, 500)
            return self._json({"error": "not found"}, 404)

    return Handler


def serve(db, host: str = "127.0.0.1", port: int = 8765, token: str = "") -> None:
    selection = _selection(db)
    httpd = ThreadingHTTPServer((host, port), make_handler(selection, token))
    print(f"factkit dashboard on http://{host}:{port}"
          f" — {selection.label}"
          + (" (token required)" if token else ""))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
