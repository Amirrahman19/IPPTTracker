"""Live assessment dashboard: phone camera streams frames over WebSocket,
server runs MediaPipe pose + streaming rep counter, display page shows
big counter, skeleton overlay and scrolling angle graph.

Stdlib + WebSocket (no Flask/FastAPI/ffmpeg).
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..live import LiveSession

STATIC = Path(__file__).parent / "static"
WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_session = LiveSession()
_session_lock = threading.Lock()
_display_clients: set[socket.socket] = set()
_display_lock = threading.Lock()


# ── minimal WebSocket implementation ──────────────────────────────────────

def _ws_accept_key(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1(key.encode() + WS_MAGIC).digest()
    ).decode()


def _ws_handshake(handler: BaseHTTPRequestHandler):
    key = handler.headers.get("Sec-WebSocket-Key", "")
    handler.send_response(101)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", _ws_accept_key(key))
    handler.end_headers()


def _ws_recv(sock: socket.socket) -> tuple[int | None, bytes | None]:
    """Read one WebSocket frame. Returns (opcode, payload) or (None, None) on disconnect."""
    def _read(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    header = _read(2)
    if header is None or len(header) < 2:
        return None, None
    opcode = header[0] & 0x0F
    masked = (header[1] >> 7) & 1
    length = header[1] & 0x7F
    if length == 126:
        ext = _read(2)
        if ext is None:
            return None, None
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = _read(8)
        if ext is None:
            return None, None
        length = struct.unpack(">Q", ext)[0]
    mask = _read(4) if masked else b"\x00\x00\x00\x00"
    if mask is None:
        return None, None
    data = _read(length) if length > 0 else b""
    if data is None:
        return None, None
    if masked:
        data = bytearray(data)
        for i in range(len(data)):
            data[i] ^= mask[i % 4]
        data = bytes(data)
    return opcode, data


def _ws_send(sock: socket.socket, data: bytes | str, opcode: int | None = None):
    if isinstance(data, str):
        data = data.encode()
        opcode = opcode or 1
    else:
        opcode = opcode or 2
    header = bytearray()
    header.append(0x80 | opcode)
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack(">H", length))
    else:
        header.append(127)
        header.extend(struct.pack(">Q", length))
    sock.sendall(bytes(header) + data)


def _ws_close(sock: socket.socket):
    try:
        _ws_send(sock, b"", opcode=8)
    except OSError:
        pass


# ── WebSocket handlers ────────────────────────────────────────────────────

def _broadcast_to_displays(msg: str | bytes):
    with _display_lock:
        dead = set()
        for s in _display_clients:
            try:
                _ws_send(s, msg)
            except OSError:
                dead.add(s)
        _display_clients.difference_update(dead)


def _handle_camera_ws(sock: socket.socket):
    """Phone camera connection: receives JPEG frames, processes them, broadcasts results."""
    print("[camera] connected")
    frames = 0
    try:
        while True:
            opcode, data = _ws_recv(sock)
            if opcode is None or opcode == 8:
                break
            if opcode == 2 and data:
                try:
                    with _session_lock:
                        result = _session.process_frame(data)
                    msg = json.dumps(result, separators=(",", ":"))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    continue  # bad frame or a pose-estimation hiccup; keep the connection alive
                _broadcast_to_displays(data)  # raw JPEG frame, for the live video feed
                _broadcast_to_displays(msg)
                _ws_send(sock, msg)
                frames += 1
                if frames == 1:
                    print("[camera] first frame processed, state=" + result["state"])
    except (OSError, ConnectionError):
        pass
    finally:
        print(f"[camera] disconnected after {frames} frame(s)")
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _handle_display_ws(sock: socket.socket):
    """Display connection: receives control commands, gets broadcasts from camera processing."""
    print("[display] connected", f"({len(_display_clients) + 1} total)")
    with _display_lock:
        _display_clients.add(sock)
    try:
        while True:
            opcode, data = _ws_recv(sock)
            if opcode is None or opcode == 8:
                break
            if opcode == 1 and data:
                try:
                    cmd = json.loads(data)
                    with _session_lock:
                        if cmd.get("cmd") == "reset":
                            _session.reset()
                        elif cmd.get("cmd") == "set_exercise":
                            _session.set_exercise(cmd.get("exercise", "pushup"))
                except (json.JSONDecodeError, KeyError):
                    pass
    except (OSError, ConnectionError):
        pass
    finally:
        with _display_lock:
            _display_clients.discard(sock)
        print("[display] disconnected", f"({len(_display_clients)} left)")
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# ── HTTP handler ──────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "ippt-live"

    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str):
        """Serve a file with HTTP Range support (browsers need it to seek video)."""
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            try:
                a, _, b = rng[6:].partition("-")
                if a == "":
                    start = max(size - int(b), 0)
                else:
                    start = int(a)
                    end = min(int(b), size - 1) if b else size - 1
                if start > end or start >= size:
                    raise ValueError
                code = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # viewer seeked or closed the video mid-transfer

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def _is_local_viewer(self) -> bool:
        """True only for a direct loopback connection (not via a reverse proxy)."""
        if self.headers.get("X-Forwarded-For") or self.headers.get("Forwarded"):
            return False
        return self.client_address[0] in ("127.0.0.1", "::1")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")

            if path in ("/camera", "/camera.html"):
                return self._send(200, (STATIC / "camera.html").read_bytes(), "text/html; charset=utf-8")

            if path == "/api/status":
                with _session_lock:
                    body = {
                        "state": _session.state,
                        "exercise": _session.exercise,
                        "rep_count": _session.rep_count,
                    }
                # The LAN address is only useful (and only revealed) to someone
                # on the host machine itself, so they can hand it to a phone.
                # Remote visitors never see it; the page uses its own origin.
                if self._is_local_viewer():
                    body["camera_url"] = f"http://{_local_ip()}:{self.server.server_address[1]}/camera"
                return self._json(body)

            if path == "/api/recordings":
                from . import recordings
                return self._json(recordings.list_recordings())

            if path.startswith("/api/recordings/"):
                from . import recordings
                res = recordings.analyse(path[len("/api/recordings/"):])
                return self._json(res) if res is not None else self._json({"error": "not found"}, 404)

            if path.startswith("/media/"):
                from . import recordings
                f = recordings.media_path(path[len("/media/"):])
                if f is None:
                    return self._json({"error": "not found"}, 404)
                return self._send_file(f, "video/mp4")

            # WebSocket upgrade
            if self.headers.get("Upgrade", "").lower() == "websocket":
                _ws_handshake(self)
                # Flush any buffered response data
                self.wfile.flush()
                sock = self.request
                sock.settimeout(None)
                self.close_connection = True
                try:
                    if path == "/ws/camera":
                        _handle_camera_ws(sock)
                    elif path == "/ws/display":
                        _handle_display_ws(sock)
                except Exception:
                    # Connection is already a raw WebSocket stream at this point;
                    # an HTTP/JSON error response here would just corrupt it further.
                    import traceback
                    traceback.print_exc()
                return

            self._json({"error": "not found"}, 404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": "internal error"}, 500)  # details stay in server logs


# ── local IP detection ────────────────────────────────────────────────────

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def serve(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True) -> None:
    # Without this, prints from connection-handler threads can sit in a buffer
    # and never show up when stdout isn't a live terminal (e.g. piped/redirected).
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    ip = _local_ip()
    print(f"\n  IPPT Live Dashboard")
    print(f"  Display (PC):    http://127.0.0.1:{port}/")
    print(f"  Camera (phone):  http://{ip}:{port}/camera")
    print()
    print(f"  Phone camera setup (one-time):")
    print(f"    1. Open chrome://flags on your phone")
    print(f"    2. Search: 'insecure origins'")
    print(f"    3. Add: http://{ip}:{port}")
    print(f"    4. Relaunch Chrome")
    print(f"\n  Ctrl+C to stop\n")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(f"http://127.0.0.1:{port}/",)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
