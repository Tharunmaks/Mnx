# Mnx Hive

Web app for Mnx Hive: a Queen agent that plans and a swarm of Bees that do the work.
**No AI is wired in yet.** When you send a message, the Queen says the Mnx brain isn't
connected. Everything the Bees will use already works by hand: the Connector Hub,
the Browser Bee, the Phone Drone, and a **Cell** for every Bee (its own always-on
container with a terminal, browser storage, files and 24/7 jobs).

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
| `/cell.html?bee=<id>` | A Bee's Cell: terminal, browser (own saved logins), jobs (always-on / scheduled / manual) and files |
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
hive/cells.py     Cells: per-Bee container, terminal, jobs, cron scheduler
hive/browser.py   Browser engine (Playwright), one saved profile per Bee; private/local addresses are blocked
hive/phone.py     Gateway side of the Phone Drone link
drone/drone.py    Termux agent for your phone (served at /drone.py)
services/         systemd unit to run the gateway 24/7
web/              index.html, hive.html, css/, js/ — plain HTML/CSS/JS, no build step
data/             Created at runtime: token.txt, hive.json (Bees), connectors.json, vault.json (connector secrets, chmod 600),
                  cells/<bee>/ (work/ files, browser/ profile, jobs.json, logs/)
```

Chat history is kept in the browser (localStorage) for now.

## Cells: every Bee's own always-on workspace

Every Bee, including ones you add, gets a Cell on your server:

| Part | What it is |
| --- | --- |
| Container | Its own Docker container (`mnx-cell-<bee>`, default image `python:3.12-slim`, 1 GB RAM, 1 CPU) with a persistent `/work` folder. Restarts with Docker (`--restart unless-stopped`). |
| Terminal | A real shell in the container. It keeps running when you close the app; reopen the Cell and the scrollback is there. |
| Browser | Its own Chromium profile, so cookies, logins and site storage are kept across restarts. Idle browsers close after 15 min; their storage stays on disk. |
| Jobs | Commands that run inside the Cell: **always on** (restarted if they stop), **every N minutes / daily / cron**, or **manual**. Output is logged per job. |
| Files | Browse, upload, download and delete files in `/work`. |

Cells belong to the gateway, not the web page, so closing the app or turning your phone
off changes nothing. To survive server reboots, run the gateway as a service:
`sudo cp services/mnx-hive.service /etc/systemd/system/ && sudo systemctl enable --now mnx-hive`
(edit the user and paths in it first). After a restart, always-on jobs start again
automatically, and any copy left running inside a container is stopped first so nothing
runs twice. Deleting a Bee deletes its Cell: container, files, browser storage and jobs.

Settings (environment variables): `MNX_CELL_MODE` = `auto` (default: Docker if available),
`docker`, or `local` (no isolation: commands run as the gateway's own user, only for
testing); `MNX_CELL_IMAGE`, `MNX_CELL_MEMORY`, `MNX_CELL_CPUS`, `MNX_BROWSER_IDLE` (seconds).
For Docker mode the gateway's user needs Docker access (`sudo usermod -aG docker <user>`).

These are containers on your one server, not separate cloud machines, so all Cells share
its CPU, memory and disk. Watch the Storage chip on each Cell; on the Oracle free ARM VM
(24 GB RAM) a handful of busy Cells with browsers open is comfortable.

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
from hive.cells import cells
from hive import phone

await hub.call("notion", "search", {"query": "trip"})
cell = cells.get("watcher")                                   # a Bee's own Cell
await cell.exec("pip install requests && python check.py")    # runs in its container
await cell.browser.do("goto", url="https://irctc.co.in")      # its own browser profile
cell.add_job("Price check", "python check.py", "*/30 * * * *")
await phone.get().call("tap_text", {"text": "Book"})
```

## Connecting Mnx later

1. Serve your model with an OpenAI-compatible server (e.g. llama.cpp `llama-server`).
2. In `hive/brain.py`, make `is_connected()` check it and `ask()` call it.
3. In `hive/queen.py`, replace the placeholder with real planning and Bee calls.
