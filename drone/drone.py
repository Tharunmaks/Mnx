#!/usr/bin/env python3
"""Mnx Phone Drone — runs in Termux on your Android phone.

It dials out to your Mnx Hive gateway and carries out commands from it:
screen control (tap, swipe, type, open apps, read the screen) through ADB
wireless debugging or Shizuku, and phone features (battery, notifications,
SMS, torch, volume, clipboard, speech, location) through Termux:API.

Setup (once):
  pkg install python android-tools termux-api
  pip install websockets            # optional: pkg install python-pillow (smaller screenshots)
  # Screen control, pick one:
  #  ADB:     Settings > Developer options > Wireless debugging > Pair device with pairing code
  #           adb pair localhost:<pair-port> <code>  then  adb connect localhost:<port>
  #  Shizuku: start Shizuku, export rish into Termux, then use --backend shizuku
Run:
  python drone.py --gateway http://<server>:8000 --token <access token>
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import shlex
import shutil
import socket
import struct
import sys
import uuid
import xml.etree.ElementTree as ET

try:
    import websockets
except ImportError:
    sys.exit("Install websockets first:  pip install websockets")

KEYS = {
    "home": 3, "back": 4, "call": 5, "endcall": 6, "up": 19, "down": 20, "left": 21, "right": 22,
    "volume_up": 24, "volume_down": 25, "power": 26, "camera": 27, "tab": 61, "space": 62,
    "enter": 66, "delete": 67, "menu": 82, "search": 84, "play_pause": 85, "mute": 164,
    "recents": 187, "paste": 279,
}
# Common built-in apps worth listing next to the ones you installed.
SYSTEM_APPS = (
    "com.android.chrome", "com.android.settings", "com.google.android.dialer", "com.android.dialer",
    "com.google.android.apps.messaging", "com.android.mms", "com.google.android.gm",
    "com.google.android.youtube", "com.google.android.apps.maps", "com.android.camera",
    "com.google.android.GoogleCamera", "com.google.android.apps.photos", "com.android.vending",
    "com.google.android.calendar", "com.google.android.contacts", "com.android.contacts",
    "com.google.android.deskclock", "com.google.android.calculator",
)
PKG_RE = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)+$")
NUMBER_RE = re.compile(r"^\+?[0-9 ]{3,20}$")
SCREEN_CMDS = {"screenshot", "tap", "long_press", "swipe", "text", "key", "open_app", "apps", "ui_tree", "tap_text", "screen_size"}
API_CMDS = {
    "battery": "termux-battery-status", "notifications": "termux-notification-list", "sms_list": "termux-sms-list",
    "sms_send": "termux-sms-send", "call": "termux-telephony-call", "torch": "termux-torch", "volume": "termux-volume",
    "clipboard_get": "termux-clipboard-get", "clipboard_set": "termux-clipboard-set", "tts": "termux-tts-speak",
    "vibrate": "termux-vibrate", "location": "termux-location", "toast": "termux-toast",
    "open_url": "termux-open-url", "wifi": "termux-wifi-connectioninfo",
}


class CmdError(RuntimeError):
    pass


async def run(argv: list[str], timeout: float = 20, data: bytes | None = None) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(data), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise CmdError(f"{argv[0]} timed out")
    if proc.returncode != 0:
        raise CmdError((err or out).decode(errors="replace").strip()[:300] or f"{argv[0]} failed")
    return out


class Drone:
    def __init__(self, args):
        self.args = args
        self.backend = args.backend
        self.serial = args.adb_serial
        self.capabilities: list[str] = []
        self.size = None  # real screen size (w, h)

    # ----- setup -----
    async def detect(self) -> None:
        caps = set()
        if self.backend == "auto":
            self.backend = "none"
            if shutil.which("adb"):
                serial = self.serial or await self._first_adb_device()
                if serial:
                    self.backend, self.serial = "adb", serial
            if self.backend == "none" and shutil.which("rish"):
                self.backend = "shizuku"
        elif self.backend == "adb" and not self.serial:
            self.serial = await self._first_adb_device()
            if not self.serial:
                raise SystemExit("No ADB device. Run 'adb connect localhost:<port>' first (Wireless debugging).")
        if self.backend != "none":
            try:
                await self.shell("echo ok")
                caps |= SCREEN_CMDS
            except CmdError as exc:
                print(f"[drone] Screen control unavailable: {exc}")
                self.backend = "none"
        for cmd, binary in API_CMDS.items():
            if shutil.which(binary):
                caps.add(cmd)
        caps -= set(filter(None, self.args.deny.split(",")))
        self.capabilities = sorted(caps)

    async def _first_adb_device(self) -> str | None:
        try:
            out = (await run(["adb", "devices"], 10)).decode()
        except (CmdError, FileNotFoundError):
            return None
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                return parts[0]
        return None

    # ----- shell on the phone with shell-user rights (ADB or Shizuku) -----
    async def shell(self, cmdline: str, timeout: float = 20) -> bytes:
        if self.backend == "adb":
            return await run(["adb", "-s", self.serial, "shell", cmdline], timeout)
        if self.backend == "shizuku":
            return await run(["rish", "-c", cmdline], timeout)
        raise CmdError("Screen control needs ADB wireless debugging or Shizuku")

    async def exec_out(self, cmdline: str, timeout: float = 20) -> bytes:
        if self.backend == "adb":
            return await run(["adb", "-s", self.serial, "exec-out", cmdline], timeout)
        return await self.shell(cmdline, timeout)

    # ----- commands -----
    async def handle(self, cmd: str, a: dict):
        if cmd not in self.capabilities:
            raise CmdError(f"'{cmd}' isn't available on this phone")
        fn = getattr(self, "c_" + cmd, None)
        if not fn:
            raise CmdError(f"Unknown command {cmd}")
        return await fn(a)

    async def c_screen_size(self, a):
        out = (await self.shell("wm size")).decode()
        m = re.findall(r"(\d+)x(\d+)", out)
        if not m:
            raise CmdError("Couldn't read screen size")
        w, h = map(int, m[-1])  # override size wins if set
        self.size = (w, h)
        return {"width": w, "height": h}

    async def c_screenshot(self, a):
        png = await self.exec_out("screencap -p", 25)
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            png = png.replace(b"\r\n", b"\n")  # old adb shells mangle line endings
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            raise CmdError("Screenshot failed (is the screen on?)")
        w, h = struct.unpack(">II", png[16:24])
        self.size = (w, h)
        width = int(a.get("width") or 540)
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
            img.thumbnail((width, width * 4))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=60)
            data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            data = "data:image/png;base64," + base64.b64encode(png).decode()
        return {"image": data, "width": w, "height": h}

    async def c_tap(self, a):
        await self.shell(f"input tap {int(a['x'])} {int(a['y'])}")
        return {"ok": True}

    async def c_long_press(self, a):
        x, y = int(a["x"]), int(a["y"])
        await self.shell(f"input swipe {x} {y} {x} {y} 700")
        return {"ok": True}

    async def c_swipe(self, a):
        ms = max(50, min(int(a.get("ms", 300)), 3000))
        await self.shell(f"input swipe {int(a['x1'])} {int(a['y1'])} {int(a['x2'])} {int(a['y2'])} {ms}")
        return {"ok": True}

    async def c_text(self, a):
        text = str(a.get("text", ""))[:2000]
        if not text:
            return {"ok": True}
        if text.isascii():
            # 'input text' treats %s as a space and needs shell quoting for the rest
            await self.shell("input text " + shlex.quote(text.replace("%", "\\%").replace(" ", "%s")))
        elif shutil.which("termux-clipboard-set"):
            await run(["termux-clipboard-set"], 10, text.encode())
            await self.shell(f"input keyevent {KEYS['paste']}")
        else:
            raise CmdError("Typing non-English text needs Termux:API (clipboard)")
        return {"ok": True}

    async def c_key(self, a):
        key = str(a.get("key", "")).lower()
        if key == "notifications":
            await self.shell("cmd statusbar expand-notifications")
        elif key in KEYS:
            await self.shell(f"input keyevent {KEYS[key]}")
        else:
            raise CmdError(f"Unknown key {key}")
        return {"ok": True}

    async def c_open_app(self, a):
        pkg = str(a.get("package", ""))
        if not PKG_RE.match(pkg):
            raise CmdError("Bad package name")
        await self.shell(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1")
        return {"ok": True}

    async def c_apps(self, a):
        third = (await self.shell("pm list packages -3")).decode()
        every = (await self.shell("pm list packages")).decode()
        pkgs = {l.split(":", 1)[1].strip() for l in third.splitlines() if ":" in l}
        pkgs |= {l.split(":", 1)[1].strip() for l in every.splitlines() if ":" in l and l.split(":", 1)[1].strip() in SYSTEM_APPS}
        return sorted(({"package": p, "label": label_for(p)} for p in pkgs), key=lambda x: x["label"].lower())

    async def c_ui_tree(self, a):
        await self.shell("uiautomator dump /sdcard/mnx_ui.xml", 25)
        xml = (await self.shell("cat /sdcard/mnx_ui.xml")).decode(errors="replace")
        return parse_ui(xml)

    async def c_tap_text(self, a):
        want = str(a.get("text", "")).lower().strip()
        if not want:
            raise CmdError("No text given")
        nodes = await self.c_ui_tree({})
        exact = [n for n in nodes if want in (n["text"].lower(), n["desc"].lower())]
        partial = [n for n in nodes if want in n["text"].lower() or want in n["desc"].lower()]
        for n in exact or partial:
            await self.shell(f"input tap {n['x']} {n['y']}")
            return {"tapped": n}
        raise CmdError(f"Nothing on screen says '{a.get('text')}'")

    # Termux:API
    async def _json(self, argv, timeout=20):
        out = (await run(argv, timeout)).decode().strip()
        try:
            return json.loads(out) if out else None
        except json.JSONDecodeError:
            return out

    async def c_battery(self, a): return await self._json(["termux-battery-status"])
    async def c_notifications(self, a): return await self._json(["termux-notification-list"])
    async def c_wifi(self, a): return await self._json(["termux-wifi-connectioninfo"])
    async def c_clipboard_get(self, a): return {"text": (await run(["termux-clipboard-get"])).decode()}

    async def c_sms_list(self, a):
        n = max(1, min(int(a.get("limit", 20)), 100))
        return await self._json(["termux-sms-list", "-l", str(n), "-t", "inbox"])

    async def c_sms_send(self, a):
        number, text = str(a.get("number", "")), str(a.get("text", ""))[:1000]
        if not NUMBER_RE.match(number) or not text:
            raise CmdError("Need a valid number and a message")
        await run(["termux-sms-send", "-n", number.replace(" ", ""), text])
        return {"sent": True}

    async def c_call(self, a):
        number = str(a.get("number", ""))
        if not NUMBER_RE.match(number):
            raise CmdError("Bad number")
        await run(["termux-telephony-call", number.replace(" ", "")])
        return {"calling": True}

    async def c_torch(self, a):
        await run(["termux-torch", "on" if a.get("on") else "off"])
        return {"on": bool(a.get("on"))}

    async def c_volume(self, a):
        if "stream" in a and "level" in a:
            stream = str(a["stream"])
            if stream not in ("alarm", "music", "notification", "ring", "system", "call"):
                raise CmdError("Bad stream")
            await run(["termux-volume", stream, str(int(a["level"]))])
        return await self._json(["termux-volume"])

    async def c_clipboard_set(self, a):
        await run(["termux-clipboard-set"], 10, str(a.get("text", "")).encode())
        return {"ok": True}

    async def c_tts(self, a):
        await run(["termux-tts-speak", str(a.get("text", ""))[:1000]], 60)
        return {"ok": True}

    async def c_vibrate(self, a):
        await run(["termux-vibrate", "-d", str(max(50, min(int(a.get("ms", 300)), 3000)))])
        return {"ok": True}

    async def c_location(self, a):
        return await self._json(["termux-location", "-p", "network", "-r", "once"], 45)

    async def c_toast(self, a):
        await run(["termux-toast", str(a.get("text", ""))[:300]])
        return {"ok": True}

    async def c_open_url(self, a):
        url = str(a.get("url", ""))
        if not re.match(r"^https?://", url):
            raise CmdError("Only http(s) links")
        await run(["termux-open-url", url])
        return {"ok": True}

    # ----- link to the gateway -----
    async def serve(self) -> None:
        base = self.args.gateway.rstrip("/")
        url = re.sub(r"^http", "ws", base) + "/ws/drone?token=" + self.args.token
        delay = 1
        while True:
            try:
                async with websockets.connect(url, max_size=16 * 1024 * 1024, ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "type": "hello", "device_id": self.args.device_id, "name": self.args.name,
                        "model": await getprop(self, "ro.product.model"),
                        "android": await getprop(self, "ro.build.version.release"),
                        "backend": self.backend, "capabilities": self.capabilities,
                    }))
                    print(f"[drone] Connected to {base} as '{self.args.name}' ({self.backend}); {len(self.capabilities)} commands")
                    delay = 1
                    async for raw in ws:
                        msg = json.loads(raw)
                        asyncio.create_task(self._reply(ws, msg))
            except (OSError, websockets.WebSocketException) as exc:
                code = getattr(getattr(exc, "rcvd", None), "code", None)
                if code == 4401:
                    sys.exit("[drone] Wrong access token.")
                print(f"[drone] Disconnected ({type(exc).__name__}); retrying in {delay}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def _reply(self, ws, msg: dict) -> None:
        rid, cmd = msg.get("id"), str(msg.get("cmd"))
        try:
            result = await self.handle(cmd, msg.get("args") or {})
            reply = {"type": "reply", "id": rid, "ok": True, "result": result}
        except Exception as exc:
            reply = {"type": "reply", "id": rid, "ok": False, "error": str(exc)}
        if self.args.verbose:
            print(f"[drone] {cmd} -> {'ok' if reply['ok'] else reply['error']}")
        await ws.send(json.dumps(reply))


async def getprop(d: Drone, name: str) -> str | None:
    try:
        if d.backend != "none":
            return (await d.shell(f"getprop {name}", 5)).decode().strip() or None
        return (await run(["getprop", name], 5)).decode().strip() or None
    except Exception:
        return None


def label_for(pkg: str) -> str:
    parts = [p for p in pkg.split(".") if p not in ("com", "org", "in", "net", "app", "android", "apps", "google", "mobile")]
    word = parts[-1] if parts else pkg
    return re.sub(r"[_-]+", " ", word).strip().title() or pkg


def parse_ui(xml: str) -> list[dict]:
    start = xml.find("<?xml")
    root = ET.fromstring(xml[start:] if start >= 0 else xml)
    out = []
    for n in root.iter("node"):
        text, desc = n.get("text", ""), n.get("content-desc", "")
        clickable = n.get("clickable") == "true"
        editable = n.get("class", "").endswith("EditText")
        if not (text or desc or clickable or editable):
            continue
        m = re.findall(r"\d+", n.get("bounds", ""))
        if len(m) != 4:
            continue
        x1, y1, x2, y2 = map(int, m)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({
            "text": text[:120], "desc": desc[:120], "id": n.get("resource-id", "").split("/")[-1],
            "cls": n.get("class", "").split(".")[-1], "clickable": clickable, "editable": editable,
            "x": (x1 + x2) // 2, "y": (y1 + y2) // 2, "bounds": [x1, y1, x2, y2],
        })
        if len(out) >= 300:
            break
    return out


def device_id() -> str:
    path = os.path.expanduser("~/.mnx_drone_id")
    try:
        return open(path).read().strip()
    except FileNotFoundError:
        did = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        with open(path, "w") as f:
            f.write(did)
        return did


def main() -> None:
    ap = argparse.ArgumentParser(description="Mnx Phone Drone")
    ap.add_argument("--gateway", default=os.getenv("MNX_GATEWAY", "http://127.0.0.1:8000"))
    ap.add_argument("--token", default=os.getenv("MNX_TOKEN", ""))
    ap.add_argument("--name", default="My phone")
    ap.add_argument("--backend", choices=["auto", "adb", "shizuku", "none"], default="auto")
    ap.add_argument("--adb-serial", default=None, help="e.g. localhost:37123 (default: first adb device)")
    ap.add_argument("--deny", default="", help="comma-separated commands this phone refuses, e.g. sms_send,call")
    ap.add_argument("--device-id", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not args.token:
        sys.exit("Pass --token (shown in the gateway log and data/token.txt)")
    args.device_id = args.device_id or device_id()

    drone = Drone(args)

    async def go():
        await drone.detect()
        print(f"[drone] Backend: {drone.backend}; commands: {', '.join(drone.capabilities) or 'none'}")
        await drone.serve()

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
