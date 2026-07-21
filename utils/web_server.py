"""
Web-based streaming server for the tyre counter.

Serves:
  /              → HTML page with the live video feed and stats
  /video_feed    → Raw MJPEG stream
  /stats         → JSON endpoint for live counter stats
  /zone_setup    → Interactive zone-setup page
  /zone_confirm  → POST endpoint to confirm zone positions

Frames are pushed into this module from the main processing loop via
``update_frame()`` and stats via ``update_stats()``.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Shared state (thread-safe via locks)
# ---------------------------------------------------------------------------
_frame_lock = threading.Lock()
_latest_frame: Optional[bytes] = None  # JPEG-encoded bytes

_stats_lock = threading.Lock()
_latest_stats: dict = {
    "current_count": 0,
    "entry_count": 0,
    "exit_count": 0,
    "fps": 0,
    "frame_idx": 0,
}

_zone_lock = threading.Lock()
_zone_confirmed: Optional[dict] = None  # set by POST /zone_confirm
_zone_frame: Optional[bytes] = None     # JPEG of the first frame for zone setup


def update_frame(frame) -> None:
    """Push an OpenCV BGR frame to the stream."""
    global _latest_frame
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _frame_lock:
        _latest_frame = jpeg.tobytes()


def update_stats(stats: dict) -> None:
    """Push latest counter stats."""
    global _latest_stats
    with _stats_lock:
        _latest_stats = dict(stats)


def set_zone_frame(frame) -> None:
    """Set the frozen frame used by the zone-setup page."""
    global _zone_frame
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with _zone_lock:
        _zone_frame = jpeg.tobytes()


def get_zone_result() -> Optional[dict]:
    """Return the confirmed zone ratios, or None if not yet confirmed."""
    with _zone_lock:
        return _zone_confirmed


def clear_zone_result() -> None:
    """Reset zone confirmation."""
    global _zone_confirmed
    with _zone_lock:
        _zone_confirmed = None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


def _generate_mjpeg():
    """Generator that yields MJPEG frames."""
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.03)  # ~30 fps max


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    with _stats_lock:
        return jsonify(_latest_stats)


@app.route("/zone_frame")
def zone_frame():
    """Serve the frozen frame for zone setup."""
    with _zone_lock:
        frame = _zone_frame
    if frame is None:
        return "", 204
    return Response(frame, mimetype="image/jpeg")


@app.route("/zone_confirm", methods=["POST"])
def zone_confirm():
    """Receive confirmed zone positions from the web UI."""
    global _zone_confirmed
    data = request.get_json()
    with _zone_lock:
        _zone_confirmed = data
    return jsonify({"status": "ok"})


@app.route("/zone_skip", methods=["POST"])
def zone_skip():
    """User chose to skip zone setup."""
    global _zone_confirmed
    with _zone_lock:
        _zone_confirmed = {"skipped": True}
    return jsonify({"status": "skipped"})


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

MAIN_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tyre Counter — Live</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --cyan: #58a6ff; --yellow: #d29922;
    --accent: #238636;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; height: 100vh;
    overflow: hidden;
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 16px; font-weight: 700; }
  header .dot { width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); display: inline-block;
    margin-right: 8px; animation: pulse 1.5s infinite; }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  .stats-bar {
    display: flex; gap: 24px; padding: 10px 20px;
    background: var(--surface); border-bottom: 1px solid var(--border);
  }
  .stat { text-align: center; }
  .stat .label { font-size: 11px; text-transform: uppercase;
    color: var(--muted); letter-spacing: 1px; margin-bottom: 2px; }
  .stat .value { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat .value.count { color: var(--green); }
  .stat .value.enter { color: var(--cyan); }
  .stat .value.exit  { color: var(--yellow); }
  .stat .value.fps   { color: var(--muted); font-size: 18px; }
  .video-wrap {
    flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 12px; overflow: hidden; background: #000;
  }
  .video-wrap img {
    max-width: 100%; max-height: 100%; object-fit: contain;
    border-radius: 6px; border: 1px solid var(--border);
  }
  footer {
    padding: 6px 20px; font-size: 11px; color: var(--muted);
    background: var(--surface); border-top: 1px solid var(--border);
    text-align: center;
  }
</style>
</head>
<body>
  <header>
    <div><span class="dot"></span><h1 style="display:inline">Tyre Counter</h1></div>
    <span style="font-size:12px;color:var(--muted)" id="frame-idx">Frame: 0</span>
  </header>
  <div class="stats-bar">
    <div class="stat"><div class="label">Count</div><div class="value count" id="s-count">0</div></div>
    <div class="stat"><div class="label">Entries</div><div class="value enter" id="s-enter">0</div></div>
    <div class="stat"><div class="label">Exits</div><div class="value exit" id="s-exit">0</div></div>
    <div class="stat"><div class="label">FPS</div><div class="value fps" id="s-fps">—</div></div>
  </div>
  <div class="video-wrap">
    <img src="/video_feed" alt="Live Feed">
  </div>
  <footer>Live MJPEG stream • Press Q in terminal to stop</footer>
<script>
  setInterval(async () => {
    try {
      const r = await fetch('/stats');
      const d = await r.json();
      document.getElementById('s-count').textContent = d.current_count;
      document.getElementById('s-enter').textContent = d.entry_count;
      document.getElementById('s-exit').textContent  = d.exit_count;
      document.getElementById('s-fps').textContent    = d.fps > 0 ? d.fps.toFixed(1) : '—';
      document.getElementById('frame-idx').textContent = 'Frame: ' + d.frame_idx;
    } catch(e) {}
  }, 500);
</script>
</body>
</html>"""


ZONE_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tyre Counter — Zone Setup</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --cyan: #58a6ff; --red: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  header {
    padding: 10px 20px; background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 15px; font-weight: 700; white-space: nowrap; }
  .step-pills { display: flex; gap: 8px; }
  .pill {
    padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600;
    border: 1px solid var(--border); color: var(--muted); cursor: pointer;
    transition: all 0.15s;
  }
  .pill.active-zone { border-color: var(--green); color: var(--green); background: rgba(63,185,80,.12); }
  .pill.active-line { border-color: var(--cyan);  color: var(--cyan);  background: rgba(88,166,255,.12); }
  .pill.done-zone   { border-color: var(--green); color: var(--green); opacity: 0.6; }
  .pill.done-line   { border-color: var(--cyan);  color: var(--cyan);  opacity: 0.6; }
  .toolbar {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 20px; background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .toolbar button {
    padding: 5px 16px; border-radius: 6px; border: 1px solid var(--border);
    font-family: inherit; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.15s;
  }
  .btn-zone    { background: rgba(63,185,80,.15);  color: var(--green); border-color: var(--green); }
  .btn-zone:hover { background: rgba(63,185,80,.30); }
  .btn-line    { background: rgba(88,166,255,.15); color: var(--cyan);  border-color: var(--cyan); }
  .btn-line:hover { background: rgba(88,166,255,.30); }
  .btn-confirm { background: #238636; color: #fff; border-color: #238636; }
  .btn-confirm:hover { background: #2ea043; }
  .btn-reset   { background: transparent; color: var(--muted); }
  .btn-reset:hover { background: #21262d; }
  .btn-skip    { background: transparent; color: var(--muted); margin-left: auto; font-size: 11px; }
  .hint {
    margin-left: auto; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums; max-width: 380px; text-align: right;
  }
  .canvas-wrap {
    flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 10px; overflow: hidden; background: #000; position: relative;
  }
  canvas { max-width: 100%; max-height: 100%; border-radius: 6px; border: 1px solid var(--border); }
  .status-msg {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
    background: rgba(35,134,54,0.95); color: #fff; padding: 16px 32px;
    border-radius: 10px; font-size: 18px; font-weight: 700;
    display: none; z-index: 99;
  }
</style>
</head>
<body>
  <header>
    <h1>🎯 Zone Setup</h1>
    <div class="step-pills">
      <div class="pill active-zone" id="pill-zone">① Draw Entry Zone</div>
      <div class="pill" id="pill-line">② Place Counting Line</div>
    </div>
  </header>
  <div class="toolbar">
    <button class="btn-zone"    id="btn-zone" onclick="setMode('zone')">🟩 Entry Zone (drag)</button>
    <button class="btn-line"    id="btn-line" onclick="setMode('line')">📏 Counting Line (click)</button>
    <button class="btn-reset"   onclick="resetAll()">↺ Reset</button>
    <button class="btn-confirm" onclick="confirmZones()">✓ Confirm &amp; Start</button>
    <button class="btn-skip"    onclick="skipZones()">Skip →</button>
    <div class="hint" id="hint">Draw a rectangle over the area where tyres enter/exit</div>
  </div>
  <div class="canvas-wrap">
    <canvas id="cv"></canvas>
  </div>
  <div class="status-msg" id="status-msg"></div>

<script>
const canvas = document.getElementById('cv');
const ctx    = canvas.getContext('2d');
const img    = new Image();

let imgW = 640, imgH = 480;
let mode = 'zone';   // 'zone' | 'line'

// Entry zone (pixel coords on the canvas image)
let zone = null;     // { x1,y1,x2,y2 }
let drag = null;     // { sx,sy }  start of drag

// Counting line (pixel y on canvas image)
let lineY = null;

// ── image load ───────────────────────────────────────────────────────────────
img.onload = () => {
  imgW = img.naturalWidth;
  imgH = img.naturalHeight;
  canvas.width  = imgW;
  canvas.height = imgH;
  draw();
};
img.src = '/zone_frame?' + Date.now();

// ── mode switch ───────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  const hints = {
    zone: 'Click and drag a rectangle over the entry area',
    line: 'Click anywhere inside the zone to place the counting line',
  };
  document.getElementById('hint').textContent = hints[m];

  document.getElementById('pill-zone').className = 'pill ' + (m==='zone' ? 'active-zone' : (zone ? 'done-zone' : ''));
  document.getElementById('pill-line').className = 'pill ' + (m==='line' ? 'active-line' : (lineY!==null ? 'done-line' : ''));

  canvas.style.cursor = m === 'zone' ? 'crosshair' : 'ns-resize';
}
setMode('zone');

// ── mouse ─────────────────────────────────────────────────────────────────────
function canvasXY(e) {
  const r  = canvas.getBoundingClientRect();
  const sx = imgW / r.width;
  const sy = imgH / r.height;
  return [ (e.clientX - r.left) * sx, (e.clientY - r.top) * sy ];
}

canvas.addEventListener('mousedown', e => {
  const [x, y] = canvasXY(e);
  if (mode === 'zone') {
    drag = { sx: x, sy: y };
    zone = { x1: x, y1: y, x2: x, y2: y };
  } else if (mode === 'line') {
    lineY = Math.max(0, Math.min(imgH - 1, y));
    draw();
  }
});

canvas.addEventListener('mousemove', e => {
  if (mode !== 'zone' || !drag) return;
  const [x, y] = canvasXY(e);
  zone = {
    x1: Math.min(drag.sx, x), y1: Math.min(drag.sy, y),
    x2: Math.max(drag.sx, x), y2: Math.max(drag.sy, y),
  };
  draw();
});

canvas.addEventListener('mouseup', e => {
  if (mode === 'zone' && drag) {
    drag = null;
    // auto-switch to line mode if zone is big enough
    if (zone && (zone.x2 - zone.x1) > 20 && (zone.y2 - zone.y1) > 20) {
      setMode('line');
      document.getElementById('hint').textContent = 'Now click inside the zone to place the counting line';
    }
  }
});

canvas.addEventListener('mouseleave', () => { drag = null; });

// ── draw ──────────────────────────────────────────────────────────────────────
function draw() {
  ctx.clearRect(0, 0, imgW, imgH);
  ctx.drawImage(img, 0, 0);

  if (zone) {
    const { x1, y1, x2, y2 } = zone;

    // tint
    ctx.fillStyle = 'rgba(0,200,100,0.18)';
    ctx.fillRect(x1, y1, x2-x1, y2-y1);

    // border
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // corner handles
    for (const [cx,cy] of [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]) {
      ctx.fillStyle = '#3fb950';
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI*2); ctx.fill();
    }

    // label
    ctx.fillStyle = '#3fb950';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.fillText('ENTRY ZONE', x1 + 8, y1 + 22);

    // ratio hint
    const rx1 = (x1/imgW).toFixed(3), ry1 = (y1/imgH).toFixed(3);
    const rx2 = (x2/imgW).toFixed(3), ry2 = (y2/imgH).toFixed(3);
    ctx.fillStyle = 'rgba(63,185,80,0.8)';
    ctx.font = '10px Inter, monospace';
    ctx.fillText(`(${rx1},${ry1}) → (${rx2},${ry2})`, x1 + 8, y1 + 40);
  }

  if (lineY !== null) {
    const lx1 = zone ? zone.x1 : 0;
    const lx2 = zone ? zone.x2 : imgW;
    const ly  = Math.max(zone ? zone.y1 : 0, Math.min(zone ? zone.y2 : imgH, lineY));

    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx1, ly); ctx.lineTo(lx2, ly); ctx.stroke();

    ctx.fillStyle = '#58a6ff';
    ctx.beginPath(); ctx.arc((lx1+lx2)/2, ly, 7, 0, Math.PI*2); ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(`COUNTING LINE  y=${(ly/imgH).toFixed(3)}`, lx1 + 10, ly - 8);
  }

  // pill update
  document.getElementById('pill-zone').className = 'pill ' +
    (mode==='zone' ? 'active-zone' : (zone ? 'done-zone' : ''));
  document.getElementById('pill-line').className = 'pill ' +
    (mode==='line' ? 'active-line' : (lineY!==null ? 'done-line' : ''));
}

// ── actions ───────────────────────────────────────────────────────────────────
function resetAll() {
  zone  = null;
  lineY = null;
  drag  = null;
  setMode('zone');
  draw();
}

async function confirmZones() {
  if (!zone || (zone.x2 - zone.x1) < 10 || (zone.y2 - zone.y1) < 10) {
    alert('Please draw the entry zone first (drag a rectangle).'); return;
  }
  if (lineY === null) {
    alert('Please place the counting line first (click inside the zone).'); return;
  }
  const ly = Math.max(zone.y1, Math.min(zone.y2, lineY));
  const data = {
    enabled: true,
    x_min: parseFloat((zone.x1 / imgW).toFixed(4)),
    x_max: parseFloat((zone.x2 / imgW).toFixed(4)),
    y_min: parseFloat((zone.y1 / imgH).toFixed(4)),
    y_max: parseFloat((zone.y2 / imgH).toFixed(4)),
    line_y: parseFloat((ly / imgH).toFixed(4)),
  };
  await fetch('/zone_confirm', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data),
  });
  const msg = document.getElementById('status-msg');
  msg.textContent = '✓ Zones saved — starting counting…';
  msg.style.display = 'block';
  setTimeout(() => { window.location.href = '/'; }, 1500);
}

async function skipZones() {
  await fetch('/zone_skip', { method: 'POST' });
  window.location.href = '/';
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(MAIN_PAGE)


@app.route("/zone_setup")
def zone_setup():
    return render_template_string(ZONE_PAGE)


# ---------------------------------------------------------------------------
# Server start helper
# ---------------------------------------------------------------------------
_server_thread: Optional[threading.Thread] = None


def start_server(port: int = 5050) -> None:
    """Start the Flask server in a background daemon thread."""
    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        return

    import logging as _logging
    # Suppress Flask request logs (very noisy with MJPEG)
    _logging.getLogger("werkzeug").setLevel(_logging.WARNING)

    def _run():
        app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()
    time.sleep(0.5)  # let the server bind
    print(f"[web_server] Serving on http://localhost:{port}")
    print(f"[web_server] Zone setup: http://localhost:{port}/zone_setup")
