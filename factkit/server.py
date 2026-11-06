"""Read/write dashboard over stdlib HTTP: metrics, queues and one-click apply.

Binds to loopback by default. Set ``FACTKIT_DASHBOARD_TOKEN`` (or ``--token``) to
require a bearer token; a browser can exchange ``/?token=...`` once for an
HttpOnly cookie.
"""
from __future__ import annotations

import hmac
import json
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import metrics, tui

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>factkit</title>
<style>
body{{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;margin:2rem;color:#111;max-width:60rem}}
h1{{font-size:1.3rem}} table{{border-collapse:collapse;width:100%;margin:.5rem 0 2rem}}
td,th{{padding:.4rem .7rem;border-bottom:1px solid #e6e6e6;text-align:left;vertical-align:top}}
button{{font:inherit;padding:.15rem .6rem;margin-right:.3rem;cursor:pointer}}
.muted{{color:#666}} code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}}
</style></head><body>
<h1>factkit dashboard</h1><div id="summary" class="muted">loading…</div>
<h2>Conflicts <span class="muted">(newer supersedes older)</span></h2>
<table id="conflicts"></table>
<h2>Bridge proposals</h2><table id="proposals"></table>
<h2>TTL flags</h2><table id="ttl"></table>
<script>
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function act(kind, id, action) {{
  await fetch(`/api/${{kind}}/${{id}}/${{action}}`, {{method:'POST'}});
  load();
}}
async function load() {{
  const s = await (await fetch('/api/summary')).json();
  document.getElementById('summary').textContent =
    `${{s.facts}} facts · ${{s.entities}} entities · ${{(s.embeddings.coverage*100).toFixed(0)}}% embedded · ` +
    `p95 ${{s.retrieval.p95_ms||0}} ms · ${{s.backlog.conflicts_pending}} conflicts · ${{s.backlog.ttl_pending}} TTL`;
  const render = (rows, el, cols, kind) => {{
    const t = document.getElementById(el);
    t.innerHTML = '<tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr>' +
      (rows.length ? rows.map(r => '<tr>' + cols.map(c => `<td>${{esc(r[c])}}</td>`).join('') +
        `<td><button onclick="act('${{kind}}',${{r.id}},'apply')">accept</button>` +
        `<button onclick="act('${{kind}}',${{r.id}},'reject')">reject</button></td></tr>`).join('')
        : '<tr><td colspan="6" class="muted">empty</td></tr>');
  }};
  render(await (await fetch('/api/conflicts')).json(), 'conflicts',
         ['keep_id','retire_id','delta','jaccard'], 'conflict');
  render(await (await fetch('/api/proposals')).json(), 'proposals',
         ['content','category','score'], 'proposal');
  render(await (await fetch('/api/ttl')).json(), 'ttl',
         ['fact_id','kind','detail','age_days'], 'ttl');
}}
load();
</script></body></html>"""


def make_handler(db_path, token: str = ""):
    """One Store per request: sqlite connections are thread-affine, and the
    dashboard is threaded. WAL makes concurrent readers cheap."""
    from .store import Store

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
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            try:
                with Store(db_path) as store:
                    if path == "/api/summary":
                        return self._json(metrics.summary(store))
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
                return self._json({"error": str(exc)}, 500)
            return self._json({"error": "not found"}, 404)

        # ----------------------------------------------------------------- POST
        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if not self._authorised(parse_qs(url.query)):
                return self._deny()
            parts = [p for p in url.path.split("/") if p]
            if len(parts) == 4 and parts[0] == "api":
                kind, item_id, action = parts[1], parts[2], parts[3]
                if not item_id.isdigit() or action not in ("apply", "reject"):
                    return self._json({"error": "bad request"}, 400)
                if kind not in ("conflict", "proposal", "ttl"):
                    return self._json({"error": "unknown kind"}, 404)
                try:
                    with Store(db_path) as store:
                        result = tui.decide(store, kind, int(item_id),
                                            "accept" if action == "apply" else "reject")
                    return self._json(result)
                except Exception as exc:  # noqa: BLE001
                    return self._json({"error": str(exc)}, 500)
            return self._json({"error": "not found"}, 404)

    return Handler


def serve(store, host: str = "127.0.0.1", port: int = 8765, token: str = "") -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(store.path, token))
    print(f"factkit dashboard on http://{host}:{port}"
          + (" (token required)" if token else ""))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
