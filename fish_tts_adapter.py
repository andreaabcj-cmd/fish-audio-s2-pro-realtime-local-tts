"""Minimal Windows adapter: audio.cpp (Fish Audio S2 Pro GGUF, streaming) -> raw 24 kHz PCM16 over plain HTTP.

Why it exists: audio.cpp streams Fish audio as SSE events carrying base64 PCM16 at 44.1 kHz. A voice assistant usually wants a
simple "POST text, read raw PCM as it arrives" endpoint at a fixed rate. This file does that with the standard library plus
numpy/scipy/requests, and manages the audio.cpp child process (start, wake/sleep to free VRAM, auto-kill with the parent).

  GET  /health            -> {"ready": true, ...}
  POST /speech {"text"}   -> streaming raw PCM16 mono 24 kHz (closing the connection stops generation)
  POST /voice  {"voice"}  -> switch the reference voice
  POST /sleep, /wake      -> stop / restart audio.cpp (frees ~5 GB of VRAM while asleep)

Setup (all paths via environment variables, defaults below):
  ACPP_BIN      path to audiocpp_server.exe built from a source tree that supports Fish streaming (NOT release 0.9.1)
  ACPP_MODEL    path to fish-audio-s2-pro-q4_k.gguf  (q6_k / q8_0 also work)
  ACPP_SPECS    path to <audio.cpp source>/model_specs   (kept from our setup; may be unnecessary with a new enough binary)
  VOICES_JSON   {"my_voice": {"ref": "C:/path/ref.wav", "text": "exact transcript of ref.wav"}}
Run: python fish_tts_adapter.py   (listens on 127.0.0.1:17863)
"""
import base64
import ctypes
import json
import math
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests
from scipy.signal import resample_poly

BIN = Path(os.getenv("ACPP_BIN", r"D:\audiocpp\bin\audiocpp_server.exe"))
MODEL = Path(os.getenv("ACPP_MODEL", r"D:\audiocpp\models\Fish-Audio-S2-Pro-GGUF\fish-audio-s2-pro-q4_k.gguf"))
SPECS = os.getenv("ACPP_SPECS", "")
VOICES = Path(os.getenv("VOICES_JSON", "voices.json"))
PORT, PORT_ACPP = int(os.getenv("PORT", "17863")), int(os.getenv("PORT_ACPP", "18190"))
URL_ACPP = f"http://127.0.0.1:{PORT_ACPP}"
OUT_SR, MODEL_SR = 24000, 44100          # Fish S2 Pro outputs 44.1 kHz
KNEE = 0.75                              # soft limiter knee

state = {"ready": False, "asleep": False, "voice": None, "error": None}
proc = {"p": None, "job": None}
lock_proc, lock_speech = threading.RLock(), threading.Lock()


def voices():
    return json.loads(VOICES.read_text(encoding="utf-8"))


def kill_with_parent():
    """Windows Job Object: if this adapter dies, audio.cpp (and its VRAM) dies with it."""
    k = ctypes.windll.kernel32
    h = k.CreateJobObjectW(None, None)

    class Basic(ctypes.Structure):
        _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("Flags", ctypes.c_uint32), ("c", ctypes.c_size_t),
                    ("d", ctypes.c_size_t), ("e", ctypes.c_uint32), ("f", ctypes.c_size_t), ("g", ctypes.c_uint32),
                    ("h", ctypes.c_uint32)]

    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in "abcdef"]

    class Ext(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", Io), ("p", ctypes.c_size_t), ("j", ctypes.c_size_t),
                    ("pm", ctypes.c_size_t), ("pk", ctypes.c_size_t)]
    info = Ext()
    info.Basic.Flags = 0x2000            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info))
    proc["job"] = h


def acpp_ready():
    try:
        return requests.get(URL_ACPP + "/health", timeout=1.5).ok
    except Exception:
        return False


def start_acpp(timeout=180):
    with lock_proc:
        if acpp_ready():                             # ours is up, or you started audio.cpp yourself on PORT_ACPP: reuse it
            return
        model = {"id": "fish", "family": "fish_audio", "task": "tts", "mode": "streaming", "path": str(MODEL),
                 "session_options": {"fish_audio.reference_cache_slots": "4"},
                 "voice_presets": {n: {"voice_ref": v["ref"], "reference_text": v["text"]} for n, v in voices().items()}}
        if SPECS:
            model["model_spec_override"] = SPECS
        cfg = Path("acpp_server.json")
        cfg.write_text(json.dumps({"host": "127.0.0.1", "port": PORT_ACPP, "backend": "cuda", "device": 0,
                                   "busy_timeout_ms": 600000, "models": [model]}, indent=1), encoding="utf-8")
        p = subprocess.Popen([str(BIN), "--config", str(cfg.resolve())], cwd=str(BIN.parent),
                             stdout=open("acpp.log", "w"), stderr=subprocess.STDOUT, creationflags=0x08000000)
        proc["p"] = p
        if proc["job"]:
            ctypes.windll.kernel32.AssignProcessToJobObject(proc["job"], int(p._handle))
        end = time.time() + timeout
        while time.time() < end and not acpp_ready():
            if p.poll() is not None:
                raise RuntimeError("audio.cpp exited at startup (see acpp.log)")
            time.sleep(0.3)
        for _ in sse("Hello.", state["voice"]):      # warm-up: loads the model lazily + caches the reference
            pass


def stop_acpp():
    with lock_proc:
        p, proc["p"] = proc["p"], None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(15)
            except Exception:
                p.kill()


def sse(text, voice):
    body = {"model": "fish", "input": text, "voice": voice, "response_format": "pcm", "stream_format": "sse"}
    r = requests.post(URL_ACPP + "/v1/audio/speech", json=body, stream=True, timeout=(5, 300),
                      headers={"Accept": "text/event-stream"})
    try:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("type") == "speech.audio.delta":
                yield np.frombuffer(base64.b64decode(ev["audio"]), dtype=np.int16)
            elif ev.get("type") == "error":
                raise RuntimeError(str(ev)[:300])
    finally:
        r.close()            # closing the connection makes audio.cpp stop generating (measured: next request starts in ~0.65 s)


class StreamResampler:
    """44.1 kHz -> 24 kHz in blocks with context on both sides: bit-identical to one-shot resample_poly (tested)."""

    def __init__(self):
        g = math.gcd(MODEL_SR, OUT_SR)
        self.up, self.down = OUT_SR // g, MODEL_SR // g
        self.unit, self.ctx = self.down * 20, self.down * 3
        self.pend, self.left = np.zeros(0, np.float32), 0

    def push(self, x):
        self.pend = np.concatenate([self.pend, x])
        out = []
        while len(self.pend) >= self.left + self.unit + self.ctx:
            y = resample_poly(self.pend[: self.left + self.unit + self.ctx], self.up, self.down)
            a = self.left * self.up // self.down
            out.append(y[a: a + self.unit * self.up // self.down])
            self.pend, self.left = self.pend[self.left + self.unit - self.ctx:], self.ctx
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def flush(self):
        rest = len(self.pend) - self.left
        if rest <= 0:
            return np.zeros(0, np.float32)
        y = resample_poly(np.concatenate([self.pend, np.zeros(self.ctx, np.float32)]), self.up, self.down)
        a = self.left * self.up // self.down
        return y[a: a + int(round(rest * self.up / self.down))]


def pcm16(x):
    a = np.abs(x)
    over = a > KNEE
    if over.any():                                   # soft limiter instead of hard clipping
        x = x.copy()
        x[over] = np.sign(x[over]) * (KNEE + (1 - KNEE) * np.tanh((a[over] - KNEE) / (1 - KNEE)))
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def speak(text):
    with lock_speech:
        if state["asleep"]:
            state["asleep"] = False
            start_acpp()
        rs = StreamResampler()
        for chunk in sse(text, state["voice"]):
            y = rs.push(chunk.astype(np.float32) / 32768.0)
            if len(y):
                yield pcm16(y)
        tail = rs.flush()
        if len(tail):
            yield pcm16(tail)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"                    # body ends when the connection closes -> trivial streaming

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply(200, dict(state))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/speech":
            if not state["ready"]:
                return self.reply(503, {"detail": "not ready"})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            gen = speak(str(body.get("text") or ""))
            try:
                for piece in gen:
                    self.wfile.write(piece)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                                  # client left (user interrupted): generator close stops audio.cpp
            finally:
                gen.close()
        elif self.path == "/voice" and body.get("voice") in voices():
            state["voice"] = body["voice"]
            self.reply(200, {"voice": state["voice"]})
        elif self.path == "/sleep":
            state["asleep"] = True
            stop_acpp()
            self.reply(200, {"asleep": True})
        elif self.path == "/wake":
            with lock_speech:
                state["asleep"] = False
                start_acpp()
            self.reply(200, {"asleep": False})
        else:
            self.reply(404, {"detail": "not found"})


def boot():
    try:
        state["voice"] = next(iter(voices()))
        kill_with_parent()
        start_acpp()
        state["ready"] = True
    except Exception as exc:
        state["error"] = repr(exc)


if __name__ == "__main__":
    threading.Thread(target=boot, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    finally:
        stop_acpp()
