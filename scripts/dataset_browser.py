"""
Interactive browser for the downloaded FineWeb shards.

Run:
    uv run python -m scripts.dataset_browser
and open the printed URL.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from threading import Lock
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from nanochat.dataset import DATA_DIR, list_parquet_files

app = FastAPI(title="nanochat dataset browser", version="0.1")

_metadata_lock = Lock()
_shard_metadata: Dict[str, Dict[str, object]] = {}


def _build_metadata(path: str) -> Dict[str, object]:
    """Return on-disk metadata for a shard."""
    pf = pq.ParquetFile(path)
    stat = os.stat(path)
    return {
        "name": os.path.basename(path),
        "path": path,
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def refresh_metadata() -> Dict[str, Dict[str, object]]:
    """Re-scan DATA_DIR and update cached metadata."""
    available = list_parquet_files(DATA_DIR)
    refreshed: Dict[str, Dict[str, object]] = {}
    for path in available:
        refreshed[os.path.basename(path)] = _build_metadata(path)
    with _metadata_lock:
        _shard_metadata.clear()
        _shard_metadata.update(refreshed)
    return refreshed


def _ensure_metadata() -> Dict[str, Dict[str, object]]:
    with _metadata_lock:
        meta = dict(_shard_metadata)
    if meta:
        return meta
    return refresh_metadata()


def _get_shard(name: str) -> Dict[str, object]:
    meta = _ensure_metadata()
    if name not in meta:
        meta = refresh_metadata()
    shard = meta.get(name)
    if shard is None:
        raise HTTPException(status_code=404, detail=f"Shard {name} not found in {DATA_DIR}")
    return shard


def _read_rows(shard: Dict[str, object], offset: int, limit: int) -> Tuple[List[Dict[str, object]], int]:
    """Read a window of rows from a shard."""
    total_rows = int(shard["num_rows"])
    if offset >= total_rows:
        return [], total_rows
    remaining = limit
    results: List[Dict[str, object]] = []
    pf = pq.ParquetFile(str(shard["path"]))
    global_row = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_meta = pf.metadata.row_group(rg_idx)
        rg_rows = rg_meta.num_rows
        rg_end = global_row + rg_rows
        if offset >= rg_end:
            global_row = rg_end
            continue
        table = pf.read_row_group(rg_idx, columns=["text"])
        texts = table.column("text").to_pylist()
        start_in_rg = max(0, offset - global_row)
        for local_idx in range(start_in_rg, rg_rows):
            if remaining <= 0:
                return results, total_rows
            text = texts[local_idx] or ""
            results.append(
                {
                    "index": global_row + local_idx,
                    "row_group": rg_idx,
                    "text": text,
                    "preview": text[:160],
                    "token_count_hint": len(text.split()),
                }
            )
            remaining -= 1
        global_row = rg_end
        if remaining <= 0:
            break
    return results, total_rows


def _search_rows(
    shard: Dict[str, object], query: str, max_matches: int, case_sensitive: bool
) -> Tuple[List[Dict[str, object]], bool, int]:
    """Scan rows until matches are found."""
    if not query:
        raise HTTPException(status_code=400, detail="Query must be non-empty")
    pf = pq.ParquetFile(str(shard["path"]))
    matches: List[Dict[str, object]] = []
    q = query if case_sensitive else query.lower()
    global_row = 0
    scanned_rows = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["text"])
        texts = table.column("text").to_pylist()
        for local_idx, raw_text in enumerate(texts):
            haystack = raw_text or ""
            hay = haystack if case_sensitive else haystack.lower()
            scanned_rows += 1
            if q in hay:
                matches.append(
                    {
                        "index": global_row + local_idx,
                        "row_group": rg_idx,
                        "text": haystack,
                    }
                )
                if len(matches) >= max_matches:
                    return matches, False, scanned_rows
        global_row += len(texts)
    return matches, True, scanned_rows


def _serialize_meta(meta: Dict[str, object]) -> Dict[str, object]:
    return {
        "name": meta["name"],
        "num_rows": meta["num_rows"],
        "num_row_groups": meta["num_row_groups"],
        "size_bytes": meta["size_bytes"],
        "modified_at": datetime.fromtimestamp(float(meta["modified_at"])).isoformat(),
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(CLIENT_HTML)


@app.get("/api/shards")
async def list_shards():
    meta = _ensure_metadata()
    payload = [
        {
            **_serialize_meta(info),
            "size_mb": round(info["size_bytes"] / (1024 * 1024), 2),
        }
        for info in sorted(meta.values(), key=lambda m: m["name"])
    ]
    total_bytes = sum(item["size_bytes"] for item in meta.values())
    total_rows = sum(int(item["num_rows"]) for item in meta.values())
    return JSONResponse(
        {
            "count": len(payload),
            "total_rows": total_rows,
            "total_size_bytes": total_bytes,
            "shards": payload,
        }
    )


@app.post("/api/shards/refresh")
async def refresh_shards():
    refreshed = refresh_metadata()
    return JSONResponse({"count": len(refreshed), "shards": [_serialize_meta(m) for m in refreshed.values()]})


@app.get("/api/shards/{shard_name}/rows")
async def shard_rows(
    shard_name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
):
    shard = _get_shard(shard_name)
    rows, total = _read_rows(shard, offset, limit)
    return JSONResponse(
        {
            "shard": shard_name,
            "offset": offset,
            "limit": limit,
            "total_rows": total,
            "page": math.floor(offset / limit) if limit else 0,
            "rows": rows,
        }
    )


@app.get("/api/shards/{shard_name}/search")
async def shard_search(
    shard_name: str,
    query: str,
    max_matches: int = Query(25, ge=1, le=200),
    case_sensitive: bool = False,
):
    shard = _get_shard(shard_name)
    matches, exhausted, scanned = _search_rows(shard, query=query, max_matches=max_matches, case_sensitive=case_sensitive)
    return JSONResponse(
        {
            "shard": shard_name,
            "query": query,
            "matches": matches,
            "max_matches": max_matches,
            "case_sensitive": case_sensitive,
            "exhausted": exhausted,
            "scanned_rows": scanned,
        }
    )


CLIENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>nanochat dataset browser</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      margin: 0;
      padding: 0;
      background: #101418;
      color: #f0f4f8;
    }
    .page {
      display: flex;
      height: 100vh;
    }
    .sidebar {
      width: 320px;
      background: #0b1014;
      border-right: 1px solid rgba(255, 255, 255, 0.1);
      overflow-y: auto;
    }
    .sidebar h2 {
      margin: 16px;
      font-size: 1.1rem;
      color: #61dafb;
    }
    .sidebar ul {
      list-style: none;
      margin: 0;
      padding: 0;
    }
    .sidebar li {
      padding: 10px 16px;
      cursor: pointer;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar li.active {
      background: rgba(97, 218, 251, 0.15);
    }
    .content {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px;
      gap: 16px;
    }
    .controls, .search-bar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    label {
      font-size: 0.8rem;
      opacity: 0.8;
    }
    select, input, button {
      padding: 6px 8px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(255, 255, 255, 0.05);
      color: inherit;
    }
    button {
      cursor: pointer;
      font-weight: 600;
      border: 1px solid rgba(97, 218, 251, 0.7);
    }
    .entries {
      flex: 1;
      overflow-y: auto;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 12px;
      background: rgba(0, 0, 0, 0.25);
    }
    .entry {
      padding: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .entry:last-child {
      border-bottom: none;
    }
    .entry pre {
      white-space: pre-wrap;
      margin: 4px 0 0;
      font-size: 0.9rem;
      line-height: 1.4;
    }
    .stats {
      display: flex;
      gap: 24px;
      font-size: 0.9rem;
    }
    .search-results {
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 12px;
      background: rgba(18, 24, 30, 0.8);
    }
    .search-results h4 {
      margin: 0 0 8px;
    }
    .search-row {
      margin-bottom: 12px;
    }
    .search-row:last-child {
      margin-bottom: 0;
    }
    .meta {
      opacity: 0.7;
      font-size: 0.78rem;
    }
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <h2>Shards</h2>
      <button style="width:calc(100% - 32px); margin: 0 16px 8px;" onclick="refreshShards()">Refresh</button>
      <ul id="shard-list"></ul>
    </aside>
    <main class="content">
      <div class="stats" id="stats"></div>
      <section class="controls">
        <label>Rows per page
          <select id="page-size" onchange="changePageSize(this.value)">
            <option value="10">10</option>
            <option value="20" selected>20</option>
            <option value="50">50</option>
            <option value="100">100</option>
            <option value="200">200</option>
          </select>
        </label>
        <label>Page
          <input type="number" id="page-number" value="0" min="0" style="width:80px" onchange="jumpToPage()"/>
        </label>
        <label>Jump to row
          <input type="number" id="jump-row" min="0" style="width:120px" placeholder="row #" />
        </label>
        <button onclick="jumpToRow()">Go</button>
        <button onclick="prevPage()">Prev</button>
        <button onclick="nextPage()">Next</button>
      </section>
      <section class="entries" id="entries"></section>
      <section class="search-bar">
        <input id="search-query" placeholder="Search substring in shard..." style="flex:1"/>
        <label>Max matches
          <input type="number" id="search-limit" value="20" min="1" max="200" style="width:80px"/>
        </label>
        <label><input type="checkbox" id="case-sensitive"/>Case sensitive</label>
        <button onclick="runSearch()">Search</button>
      </section>
      <section class="search-results" id="search-results"></section>
    </main>
  </div>
  <script>
    const state = {
      shard: null,
      page: 0,
      limit: 20,
      totalRows: 0,
      shards: []
    };

    async function fetchJSON(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return res.json();
    }

    function formatBytes(bytes) {
      const units = ['B','KB','MB','GB','TB'];
      let i = 0;
      let value = bytes;
      while (value >= 1024 && i < units.length - 1) {
        value /= 1024;
        i += 1;
      }
      return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[i]}`;
    }

    async function loadShards() {
      try {
        const data = await fetchJSON('/api/shards');
        state.shards = data.shards;
        const stats = document.getElementById('stats');
        stats.textContent = `Shards: ${data.count} · Rows: ${data.total_rows.toLocaleString()} · Size: ${formatBytes(data.total_size_bytes)}`;
        const list = document.getElementById('shard-list');
        list.innerHTML = '';
        data.shards.forEach((shard) => {
          const item = document.createElement('li');
          item.textContent = `${shard.name} · ${formatBytes(shard.size_bytes)} · ${shard.num_rows.toLocaleString()} rows`;
          item.onclick = () => {
            document.querySelectorAll('.sidebar li').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
            state.shard = shard.name;
            state.page = 0;
            loadRows();
          };
          list.appendChild(item);
          if (!state.shard) {
            state.shard = shard.name;
            item.classList.add('active');
          }
        });
        if (state.shard) {
          loadRows();
        } else {
          document.getElementById('entries').textContent = 'No shards detected under ~/.cache/nanochat/base_data';
        }
      } catch (err) {
        alert(`Failed to load shards: ${err.message}`);
      }
    }

    async function refreshShards() {
      await fetchJSON('/api/shards/refresh', { method: 'POST' });
      await loadShards();
    }

    async function loadRows() {
      if (!state.shard) return;
      const offset = state.page * state.limit;
      try {
        const data = await fetchJSON(`/api/shards/${state.shard}/rows?offset=${offset}&limit=${state.limit}`);
        state.totalRows = data.total_rows;
        document.getElementById('page-number').value = state.page;
        const container = document.getElementById('entries');
        if (data.rows.length === 0) {
          container.textContent = 'No rows at this offset.';
          return;
        }
        container.innerHTML = '';
        data.rows.forEach((row) => {
          const div = document.createElement('div');
          div.className = 'entry';
          div.innerHTML = `
            <div class="meta">Row ${row.index.toLocaleString()} · Row group ${row.row_group} · Tokens≈${row.token_count_hint}</div>
            <pre>${row.text.replace(/</g, '&lt;')}</pre>
          `;
          container.appendChild(div);
        });
      } catch (err) {
        alert(`Failed to load rows: ${err.message}`);
      }
    }

    function changePageSize(value) {
      state.limit = Number(value);
      state.page = 0;
      loadRows();
    }

    function prevPage() {
      if (state.page === 0) return;
      state.page -= 1;
      loadRows();
    }

    function nextPage() {
      const lastPage = Math.floor((state.totalRows - 1) / state.limit);
      if (state.page >= lastPage) return;
      state.page += 1;
      loadRows();
    }

    function jumpToPage() {
      const desired = Number(document.getElementById('page-number').value || 0);
      state.page = Math.max(0, desired);
      loadRows();
    }

    function jumpToRow() {
      const row = Number(document.getElementById('jump-row').value);
      if (Number.isNaN(row) || row < 0) return;
      state.page = Math.floor(row / state.limit);
      loadRows();
    }

    async function runSearch() {
      if (!state.shard) return;
      const query = document.getElementById('search-query').value.trim();
      if (!query) {
        alert('Enter a search term');
        return;
      }
      const limit = Number(document.getElementById('search-limit').value || 20);
      const caseSensitive = document.getElementById('case-sensitive').checked;
      const resultsBox = document.getElementById('search-results');
      resultsBox.textContent = 'Searching...';
      try {
        const data = await fetchJSON(`/api/shards/${state.shard}/search?query=${encodeURIComponent(query)}&max_matches=${limit}&case_sensitive=${caseSensitive}`);
        if (data.matches.length === 0) {
          resultsBox.textContent = `No matches after scanning ${data.scanned_rows.toLocaleString()} rows.`;
          return;
        }
        resultsBox.innerHTML = `<h4>${data.matches.length} match(es) · scanned ${data.scanned_rows.toLocaleString()} rows${data.exhausted ? '' : ' (more available)'}</h4>`;
        data.matches.forEach((match) => {
          const div = document.createElement('div');
          div.className = 'search-row';
          div.innerHTML = `
            <div class="meta">Row ${match.index.toLocaleString()} · Row group ${match.row_group}</div>
            <pre>${match.text.replace(/</g, '&lt;')}</pre>
          `;
          div.onclick = () => {
            state.page = Math.floor(match.index / state.limit);
            document.getElementById('jump-row').value = match.index;
            loadRows();
          };
          resultsBox.appendChild(div);
        });
      } catch (err) {
        resultsBox.textContent = `Search failed: ${err.message}`;
      }
    }

    loadShards();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Serve an interactive browser for local dataset shards.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind (default: 8008)")
    args = parser.parse_args()
    print(f"Serving dataset browser on http://{args.host}:{args.port}")
    refresh_metadata()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
