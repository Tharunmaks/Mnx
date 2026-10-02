# Mnx Hive

Web app for Mnx Hive: a Queen agent that plans and a swarm of Bees that do the work.
**No AI is wired in yet.** When you send a message, the Queen says the Mnx brain isn't
connected. Everything the Bees will use already works by hand: the Connector Hub,
the Browser Bee and the Phone Drone.

## Run it

```bash
pip install -r requirements.txt
uvicorn gateway.api:app --host 0.0.0.0 --port 8000
```

The server prints an **access token** on start (also saved in `data/token.txt`, or set
`MNX_TOKEN`). Open http://localhost:8000, and enter the token when asked (or in Settings).
Every API call and socket needs it, because the gateway can drive a browser and your phone.

For the Browser Bee, install Chromium once: `playwright install --with-deps chromium`
(or point `MNX_CHROMIUM_PATH` at an existing Chrome/Chromium).

| Page | What it is |
| --- | --- |
| `/` | Chat: left rail (new chat, history, MCP tools, Hive, settings, profile), wordmark + greeting, message box, live step tree, Bees panel |
| `/hive.html` | Hive: every Bee's state (busy / idle / scheduled / failed), add or remove Bees, activity log |
| `/connectors.html` | Connector Hub: one search across all six layers, connect MCP servers, Go/Ask/You lane per tool, try tools |
| `/browser.html` | Browser Bee: watch and drive the Hive's own Chromium (click on the picture, type, scroll, read the page) |
| `/phone.html` | Phone Drone: live phone screen (tap, swipe, long-press, type), open apps, read the screen, device controls, notifications and SMS |
| `/dev/gallery.html` | Every card and step type with placeholder data, to check the design without a model |

## Layout

```
gateway/api.py    FastAPI: serves web/, REST for Bees + tools, WebSocket /ws for live events
hive/core.py      Task: emit events, ask the user (connect / clarify / confirm) and wait for the tap
hive/queen.py     Plans and runs Bees — currently reports "brain not connected"
hive/brain.py     Where Mnx plugs in (MNX_BRAIN_URL + ask())
hive/connectors.py  Connector Hub: MCP registry search, MCP client, lanes, hourly health check
hive/browser.py   Browser Bee engine (Playwright); private/local addresses are blocked
hive/phone.py     Gateway side of the Phone Drone link
drone/drone.py    Termux agent for your phone (served at /drone.py)
web/              index.html, hive.html, css/, js/ — plain HTML/CSS/JS, no build step
data/             Created at runtime: token.txt, hive.json (Bees), connectors.json, vault.json (connector secrets, chmod 600)
```

Chat history is kept in the browser (localStorage) for now.

## "1 million connectors": how the layers add up

No single service has a million official connectors, so the Hub stacks six layers
(from the blueprint) and searches them all at once:

| Layer | Reach | In this app |
| --- | --- | --- |
| 1. Native Bees | ~20 key apps | Listed as planned; built one by one |
| 2. MCP registry | 18,000+ servers | Live search of registry.modelcontextprotocol.io; remote servers connect in one click |
| 3. Zapier MCP | 9,000+ apps | Add by URL with your Zapier MCP link |
| 4. Pipedream | 3,000+ APIs | Add by URL with your Pipedream MCP link |
| 5. Auto-connector | Any app with API docs | Needs the Mnx brain (Coder Bee) |
| 6. Browser + Phone | Any website, any Android app | Browser Bee and Phone Drone, working now |

Layer 6 is what takes the reach past a million: anything you can open in a browser or on
your phone. Registry servers that only run locally show as "Local only" (they need the
Docker sandbox, not built yet). New tools start in the **Ask** lane.

## Phone Drone (full phone control)

On the phone, in Termux (install Termux and Termux:API from F-Droid):

```bash
pkg install python android-tools termux-api python-pillow
pip install websockets
curl -O http://<server>:8000/drone.py
# screen control: Developer options > Wireless debugging > Pair device with pairing code
adb pair localhost:<pair-port> <code> && adb connect localhost:<port>
python drone.py --gateway http://<server>:8000 --token <token> --name "My phone"
```

The phone dials out to the gateway, so it works on mobile data. With ADB (or Shizuku,
`--backend shizuku`) it can screenshot, tap, swipe, long-press, type (including Tamil or
emoji, through the clipboard), press keys, open any app, list apps, and read the screen's
buttons and text (`ui_tree`, `tap_text`). With Termux:API it adds battery, notifications,
SMS read/send, calls, torch, volume, clipboard, speech, vibrate, toast, location and Wi-Fi.
SMS, calls and opening links always need your confirmation; `--deny sms_send,call`
blocks them on the phone itself. Wireless debugging turns off after a reboot, so pair again
then (or use Shizuku).

## Event protocol

Browser → gateway (WebSocket `/ws`):

```json
{"type": "user_message", "chat": "c_…", "task": "t_…", "text": "…"}
{"type": "action", "task": "t_…", "ref": "<card id>", "action": "<button id>", "values": {}}
{"type": "stop", "task": "t_…"}
```

Gateway → browser. Every event has `task`, `type`, `at`; most have `id`, `text`, `bee`.
Send the same `id` again to update a step; set `parent` to nest it under another (the "L" lines).

| `type` | Shows as | Extra fields |
| --- | --- | --- |
| `planning` `thinking` `searching` `reading` `web` `opening` `fetching` `writing` `running` `booking` `waiting` `memory` `done` `error` `step` | One line in the step tree | `status` (running/done/error/waiting), `detail` |
| `bee_created` | `[NAME BEE] created` chip | `bee` |
| `connect` | "Connect an app" card | `service`, `actions: [{id, label, style}]` |
| `clarify` | Form card | `fields: [{name, label, input: text/date/time/select/textarea, options, required}]` |
| `confirm` | Confirm card | `service`, `details: [[label, value]]`, `actions` |
| `result` | Success / failed card | `service`, `status`, `details`, `note` |
| `answer` / `answer_delta` | Final answer text (set / stream) | |
| `task_done` / `task_error` | Ends the task | |
| `bees_changed` `tools_changed` `bee_status` | Refreshes the Bees panel / tools | |

From Python, a Bee uses the `Task` it is given:

```python
step = await task.step("fetching", "Checking account", bee="Notes")
reply = await task.ask("connect", service="Drive", text="Sign in to continue")  # waits for the tap
await task.finish_step(step)
await task.emit("result", service="Drive", title="Saved", details=[["File", "notes.md"]])
await task.answer("Done.")
```

Bees can also use the hub, browser and phone directly from Python:

```python
from hive.connectors import hub
from hive.browser import session as browser
from hive import phone

await hub.call("notion", "search", {"query": "trip"})
await browser.do("goto", url="https://irctc.co.in"); await browser.do("elements")
await phone.get().call("tap_text", {"text": "Book"})
```

## Connecting Mnx later

1. Serve your model with an OpenAI-compatible server (e.g. llama.cpp `llama-server`).
2. In `hive/brain.py`, make `is_connected()` check it and `ask()` call it.
3. In `hive/queen.py`, replace the placeholder with real planning and Bee calls.
