from __future__ import annotations

import argparse
import json
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sts2sim.live_capture import (
    AUTO_BASE_URL,
    LiveApiClient,
    LiveApiConfig,
    LiveApiError,
    detect_live_bridge,
    normalize_live_snapshot,
)

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STS2 Live View</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #121417;
      --panel: #1b1f24;
      --panel-2: #222830;
      --line: #343b45;
      --text: #eef2f6;
      --muted: #a6b0bd;
      --good: #4fd18b;
      --bad: #ff6961;
      --warn: #f0c05a;
      --accent: #78a6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: #171a1f;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: 0; }
    h1 { font-size: 18px; }
    h2 { font-size: 15px; margin-bottom: 10px; }
    h3 { font-size: 13px; color: var(--muted); margin-bottom: 6px; }
    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 3px rgba(240, 192, 90, 0.16);
    }
    .dot.ok {
      background: var(--good);
      box-shadow: 0 0 0 3px rgba(79, 209, 139, 0.16);
    }
    .dot.err {
      background: var(--bad);
      box-shadow: 0 0 0 3px rgba(255, 105, 97, 0.16);
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 0.8fr) minmax(360px, 1.2fr);
      gap: 12px;
      padding: 12px;
      max-width: 1500px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .stack { display: grid; gap: 12px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #15191e;
      padding: 10px;
      min-height: 64px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric .value {
      font-size: 18px;
      overflow-wrap: anywhere;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      max-height: calc(100vh - 190px);
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      color: #d9e2ec;
    }
    .muted { color: var(--muted); }
    .error { color: var(--bad); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>STS2 Live View</h1>
      <div class="muted" id="bridge">Connecting...</div>
    </div>
    <div class="status">
      <span class="dot" id="dot"></span>
      <span id="status">Waiting for bridge</span>
      <button id="refresh" type="button">Refresh</button>
    </div>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>Current State</h2>
        <div class="grid" id="metrics"></div>
      </section>
      <section>
        <h2>Options / Actions</h2>
        <div class="chips" id="actions"></div>
      </section>
      <section>
        <h2>Simulator Snapshot</h2>
        <pre id="snapshot">{}</pre>
      </section>
    </div>
    <section>
      <h2>Raw Bridge Payload</h2>
      <pre id="raw">{}</pre>
    </section>
  </main>
  <script>
    const el = (id) => document.getElementById(id);
    const nice = (value) => {
      if (value === null || value === undefined || value === "") return "None";
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    };
    const metric = (label, value) => `
      <div class="metric">
        <div class="label">${label}</div>
        <div class="value">${nice(value)}</div>
      </div>`;
    const actionLabel = (action) => {
      if (typeof action === "string") return action;
      if (action && typeof action === "object") {
        return action.name || action.type || action.action_type || JSON.stringify(action);
      }
      return nice(action);
    };
    async function loadState() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
        el("dot").className = "dot ok";
        el("status").textContent = `Live, ${new Date(data.timestamp * 1000).toLocaleTimeString()}`;
        el("bridge").textContent = `${data.base_url} · ${data.health?.message || "bridge ready"}`;
        const state = data.state || {};
        const snapshot = data.snapshot || {};
        el("metrics").innerHTML = [
          metric("State", state.state_type || snapshot.phase || state.screen),
          metric("Message", state.message || state.room_type || state.screen),
          metric("Character", snapshot.character_id || state.character || state.player?.character),
          metric("Floor", snapshot.floor ?? state.run?.floor),
          metric("HP", snapshot.player_hp ?? state.player?.hp),
          metric("Gold", snapshot.gold ?? state.player?.gold),
          metric("Actions", data.action_count),
          metric("Updated", new Date(data.timestamp * 1000).toLocaleTimeString()),
        ].join("");
        const actions = data.actions || [];
        const menuOptions = Array.isArray(state.options) ? state.options : [];
        const visible = actions.length ? actions : menuOptions;
        el("actions").innerHTML = visible.length
          ? visible.map((item) => `<span class="chip">${actionLabel(item)}</span>`).join("")
          : `<span class="muted">No actions reported.</span>`;
        el("snapshot").textContent = JSON.stringify(snapshot, null, 2);
        el("raw").textContent = JSON.stringify(state, null, 2);
      } catch (err) {
        el("dot").className = "dot err";
        el("status").textContent = "Bridge not readable";
        el("bridge").textContent = err.message;
        el("metrics").innerHTML = "";
        el("actions").innerHTML = `<span class="error">${err.message}</span>`;
      }
    }
    el("refresh").addEventListener("click", loadState);
    loadState();
    setInterval(loadState, 1000);
  </script>
</body>
</html>
"""


class LiveViewServer(BaseHTTPRequestHandler):
    server_version = "STS2LiveView/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send_text(HTML, content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self._send_json(self._read_live_state())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_live_state(self) -> dict[str, Any]:
        live_server: ThreadingHTTPServer = self.server  # type: ignore[assignment]
        base_url = str(live_server.base_url)  # type: ignore[attr-defined]
        try:
            if base_url == AUTO_BASE_URL:
                base_url = detect_live_bridge().base_url
            client = LiveApiClient(LiveApiConfig(base_url=base_url))
            try:
                health = client.health()
                state = client.state()
                actions = client.actions(state)
            finally:
                client.close()
            return {
                "ok": True,
                "base_url": base_url,
                "timestamp": time.time(),
                "health": health,
                "state": state,
                "snapshot": normalize_live_snapshot(state),
                "actions": actions,
                "action_count": len(actions),
            }
        except (LiveApiError, OSError, RuntimeError) as exc:
            return {
                "ok": False,
                "base_url": base_url,
                "timestamp": time.time(),
                "error": str(exc),
            }

    def _send_json(self, payload: dict[str, Any]) -> None:
        status = HTTPStatus.OK if payload.get("ok", True) else HTTPStatus.BAD_GATEWAY
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, payload: str, *, content_type: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only STS2 live-state dashboard.")
    parser.add_argument("--base-url", default=AUTO_BASE_URL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), LiveViewServer)
    server.base_url = args.base_url  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    log_dir = Path("live_traces")
    log_dir.mkdir(exist_ok=True)
    print(f"STS2 live view running at {url}")
    print(f"Polling bridge: {args.base_url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping STS2 live view.")


if __name__ == "__main__":
    main()
