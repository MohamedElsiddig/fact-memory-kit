#!/usr/bin/env python3
"""Build live_cases.json from a read-only copy of the live store."""
import sqlite3, tempfile, shutil, json, os, re, argparse
from pathlib import Path

def build_cases(src_db: Path, out_path: Path, target: int = 25):
    td = Path(tempfile.mkdtemp(prefix='fk-hrr-live-'))
    for sfx in ['', '-wal', '-shm']:
        p = Path(str(src_db)+sfx)
        if p.exists(): shutil.copy2(p, td/p.name)
    db = td/src_db.name
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    facts = cur.execute("""
        SELECT fact_id, category, content FROM facts
        WHERE length(content) >= 40
          AND content NOT LIKE '%VIN%'
          AND content NOT LIKE '%JM1BPAAL5M1328615%'
          AND content NOT LIKE '%password%'
          AND content NOT LIKE '%credential%'
          AND content NOT LIKE '%api_key%'
          AND content NOT LIKE '%token%'
          AND content NOT LIKE '%secret%'
        ORDER BY category, fact_id
    """).fetchall()
    cats = cur.execute("SELECT category FROM facts GROUP BY category ORDER BY count(*) DESC").fetchall()
    per_cat = max(1, target // len(cats))
    selected = []
    for cat_row in cats:
        cat = cat_row['category']
        cat_facts = [f for f in facts if f['category'] == cat]
        selected.extend(cat_facts[:per_cat])
    if len(selected) > target:
        selected = selected[:target]
    def make_query(content, category):
        c = content.strip()
        c = re.sub(r'^(Car|Laptop|User|Raspberry Pi|Transcriber|Dashboard|Proxy|Backup|Caddy|Voice|Hermes|When user|The user)\s*:?\s*', '', c, flags=re.I)
        first = re.split(r'[.!?]\s+', c)[0]
        words = first.split()[:8]
        phrase = ' '.join(words).rstrip(',:;')
        if category == 'entity':
            return f"what is {phrase}"
        elif category in ('user_pref', 'tool', 'project'):
            if any(w in first.lower() for w in ['port', 'listen', 'runs on', 'uses']):
                return f"which port does {phrase}"
            if any(w in first.lower() for w in ['memory', 'ram', 'gb', 'tb', 'storage', 'disk', 'ssd']):
                return f"how much memory does {phrase}"
            return f"what is {phrase}"
        return f"what is {phrase}"
    def overlap(q, c):
        qt = set(q.lower().split())
        ct = set(c.lower().split())
        return len(qt & ct) / max(len(qt), 1)
    cases = []
    for f in selected:
        q = make_query(f['content'], f['category'])
        if overlap(q, f['content']) > 0.8:
            q = f"tell me about fact {f['fact_id']}"
        cases.append({"query": q, "expect": [f['fact_id']], "note": f["category"]})
    ids = [c['expect'][0] for c in cases]
    valid = {r['fact_id'] for r in cur.execute(f"SELECT fact_id FROM facts WHERE fact_id IN ({','.join('?'*len(ids))})", ids).fetchall()}
    cases = [c for c in cases if c['expect'][0] in valid]
    seen = set()
    deduped = []
    for c in cases:
        if c['query'] not in seen:
            seen.add(c['query'])
            deduped.append(c)
    con.close()
    shutil.rmtree(td)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"cases": deduped}, indent=2))
    return deduped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target', type=int, default=25)
    args = ap.parse_args()
    build_cases(Path(args.db), Path(args.out), args.target)

if __name__ == '__main__':
    main()
