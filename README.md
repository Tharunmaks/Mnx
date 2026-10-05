# Mnx Hive

Web app for Mnx Hive: a Queen agent that plans and a swarm of Bees that do the work.
**No AI is wired in yet.** When you send a message, the Queen says the Mnx brain isn't
connected. Everything the Bees will use already works by hand: the Connector Hub,
the Browser Bee, the Phone Drone, a **Cell** for every Bee (its own always-on
container with a terminal, browser storage, files and 24/7 jobs), and the **Cloud Bee**,
which trains AI models, keeps them and runs them as APIs, here or on any GPU server.

## Run it

On a fresh Ubuntu server (Oracle Cloud free ARM works), one command sets up everything,
including Docker, Chromium and a 24/7 service:

```bash
git clone <this repo> Mnx && cd Mnx && bash scripts/install.sh
bash scripts/update.sh          # later: pull the latest code and restart
```

Or by hand, for trying it out:

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
| `/lab.html` | Cloud Bee · Model Lab: train models (here or on SSH servers), live charts, model registry, run models as APIs, try them |
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
hive/cloud_chat.py  Cloud Bee in chat: understands "train …", "create …", "add data …", runs the flows with cards
hive/swarm.py     Multi-Bee flows: Architect → Data → Cloud → Eval for creating models; continued training
hive/architect.py Parameter count → transformer design + memory/GPU/time/cost estimates
hive/providers.py Cloud GPU providers (RunPod): list GPUs, rent, wait for SSH, stop
hive/hf.py        Finds models on Hugging Face (with an offline list), sizes and memory estimates
hive/lab.py       Cloud Bee engine: training runs (local Docker or SSH), model registry, model APIs
hive/recipes/     Training recipes: text classifier, spreadsheet predictor, chat model LoRA fine-tune, your own script
hive/system.py    Server stats (CPU, memory, disk, GPU)
scripts/          install.sh (fresh server) and update.sh
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

## Create a model from scratch, in the chat, with a swarm of Bees

Say **“create a 101.3 billion parameter model AI”** (any size from 1M up to **500B**, the maximum the Hive
will build; bigger requests are capped at 500B) and the Queen hands the job to a swarm of Bees, each
posting its own steps in the conversation:

| Bee | What it does |
| --- | --- |
| **Reader** | Identifies the request. If you didn't say what the AI is for, it asks: *“That's a big one — I'll try. Reason and what should the AI do?”* (e.g. “a maximum coding AI”). Also reads how many prompts to test with (“test it with 1 million prompts”). |
| **Architect** | Designs a real Qwen2-style transformer for that size (layers, width, heads, vocabulary, context) and what it takes: weights, training memory, token budget, time on your hardware, the GPUs/time/cost of a proper setup. The blueprint is **always saved** in the Lab's Models tab, so nothing is lost if training waits. |
| **Browser** | Searches for data about the purpose: for a coding AI it picks The Stack and lists the code languages to add (editable), for Tamil a Tamil Wikipedia dump, for stories TinyStories… Reads web pages when you give links. |
| **Data** | Gathers the text: public Hugging Face datasets (one per code language), your own file, web pages, or built-in demo text, and reports how many tokens that is. |
| **GPU** | Opens cloud Docker and GPUs. Offers this server, every connected SSH server, **all connected servers together as one cluster** (multi-node torchrun), connecting another server, or **renting GPUs on RunPod** from the chat. If nothing holds the model yet it never just stops at the blueprint: it asks to connect more servers, rent, make it smaller, or keep the blueprint for later (“train the 100B blueprint” picks it up). |
| **Coding** | Writes the files of the run (train.py, serve.py, params.json, README.md, run scripts) and posts a **Files** card: press *Show* on any file to read it in the chat. |
| **Cloud** | Trains it on the live card: stages (fetching data → tokenizing → training → saving → testing with prompts), loss and tokens/s charts. Training keeps going when you close the app. |
| **Eval** | When training ends it tests the model with N held-out prompts (default 1,000) and posts the report in the same chat: how long adding data, training and testing took, next-token accuracy, perplexity, then **“Your model is ready”** and *“Stabilizing network · turning off Bees”*. |

Then say **“add data to my model”** (or “feed these pages into my stories model”) and the Data Bee gathers
more text while the GPU Bee finds hardware and the Cloud Bee continues training the existing model on it
(lower learning rate; saved as a new version). “stop renting” stops rented GPUs.

Honest limits: the Architect designs any size up to 500B, and the training recipe handles a CPU, one GPU,
several GPUs on one machine (torchrun with DDP/FSDP) and **several machines at once** (the GPU Bee's
“all connected servers together” option; the first server is the rendezvous master on port 29400, which the
others must reach). Tiny models (1M–50M) train on a CPU in minutes. A 7B model needs about 125 GB of GPU
memory; a 100B model needs about 1.8 TB (23 × 80 GB GPUs, three machines of eight) and weeks on them; the
GPU Bee tells you the numbers and waits for the hardware rather than pretending.

The recipe behind it is `hive/recipes/llm-pretrain/` (also on the Lab's Train tab as “Create a model from
scratch”): it trains its own tokenizer on your data, builds the model from `hive/architect.py`, streams
Hugging Face datasets (several at once, one folder per code language), holds out 2% for evaluation, tests
with the requested number of prompts and writes `eval.json` with accuracy, perplexity and samples.

## Train and run models from the chat

Type it in the chat, for example **“train Qwen 2.5 Coder”**, and the Cloud Bee does the rest
inside the conversation:

1. Finds the model on Hugging Face (or in a built-in list of popular open models when the Hub
   can't be reached) and lists the sizes.
2. Checks every place it could train: this server's GPU, connected GPU servers, or CPU. It picks
   a size that fits, or you can **connect a GPU server right in the chat** (any machine you can SSH into).
3. Asks for the data: a Hugging Face dataset (it suggests one, e.g. Python code instructions for
   a coder model), your own `.jsonl` file uploaded in the card, or tiny demo data.
4. Shows the plan with memory needed and warnings (CPU too slow, model too big, gated licence),
   and waits for **Start training**.
5. Posts a live training card (stage, steps, loss chart) that keeps updating, even after a reload.
   Training continues on the server if you close the app.
6. When it's done: **Run it**, then chat with your new model inside the same card.

Other things you can say: “how is my training going?”, “stop training”, “run my model”,
“chat with my qwen model”, or name an exact model and dataset:
“fine-tune Qwen/Qwen2.5-Coder-1.5B-Instruct on iamtarun/python_code_instructions_18k_alpaca”.

Until the Mnx brain is connected these requests are recognised with fixed patterns
(`hive/cloud_chat.py`), not a language model; anything else still gets the
“brain not connected” reply.

## Cloud Bee: train, keep and run your own AI

Open **Cloud Bee · Lab** in the left rail.

1. **Train**: pick a recipe, give it examples (or tick "use the sample data"), choose where
   it trains, press Start. Every recipe works on a CPU; big language models want a GPU.

   | Recipe | Learns | Data | Runs as |
   | --- | --- | --- | --- |
   | Text classifier | Sort text into labels (mood, spam, intent) | CSV `text,label` | `POST /predict {"text": …}` |
   | Spreadsheet predictor | Predict a column (price, yes/no) | CSV | `POST /predict {"row": {…}}` |
   | Create a model from scratch | A brand-new language model at any size (1M–500B); trains its own tokenizer | text files and/or a Hugging Face dataset | OpenAI-compatible `/v1/chat/completions` |
| Chat model fine-tune (LoRA) | Your style and knowledge, on any Hugging Face base, **including your own Mnx** | JSONL chats | OpenAI-compatible `/v1/chat/completions` |
   | Your own script | Anything | your `train.py` (+ `requirements.txt`, `serve.py`) | your `serve.py` |

2. **Where it trains** ("Servers" tab):
   - **This server**: each run gets a fresh container (`--gpus all` when the NVIDIA toolkit is installed), with CPU and memory limits if you set them.
   - **Any machine you can SSH into**: a rented GPU (RunPod, Lambda, Vast.ai, AWS, GCP…), another VPS, your PC.
     Add it with host and user, and put the Hive's public key on it (the page shows the one-line command).
     Files go over with tar, the run keeps going with `nohup` if the connection drops, and the trained
     model comes back to this server. The server needs Python 3 with venv.
3. **Watch**: live loss/accuracy charts and logs. Runs keep going when you close the app,
   and they're picked up again after the Hive server restarts. Stop works on both kinds of server.
4. **Models**: every finished run is saved. **Run as API** starts it in its own always-on
   container on this server; then try it on the page (classify / JSON / chat) or call it
   from any app at `http://<server>:8000/api/lab/serve/<model>/…` with your access token.
   Chat models are OpenAI-compatible, so any OpenAI client works with base URL `…/serve/<model>/v1`.
   **Import .gguf** runs a GGUF model (like your Mnx) with the llama.cpp server image.
5. **Secrets**: e.g. `HF_TOKEN` for private Hugging Face models, passed to every run as an
   environment variable and never shown again.

Recipes are plain folders in `hive/recipes/`: `train.py` reads `params.json` and `data/`, writes
`output/`, prints `MNX_METRIC {json}` lines for the charts and one `MNX_RESULT {json}` line.
`serve.py` reads `$PORT` and `$MODEL_DIR` and answers `GET /health`. Add your own recipe folder
and it appears on the Train tab.

Settings: `MNX_DOCKER_ARGS` (extra `docker run` arguments for Cells, runs and models, e.g. a
proxy or a network), `MNX_LLAMA_IMAGE` (default `ghcr.io/ggml-org/llama.cpp:server`).

## Speed and smoothness

- Pages and API answers are gzip-compressed (much faster on mobile data); libraries are
  cached for a week, the app's own files are re-checked so updates show up at once.
- Live screens update over the WebSocket and only redraw what changed; logs load in pieces.
- Container images are downloaded in the background at start-up, so the first Cell or
  training run doesn't wait; pip downloads are cached between runs.
- Heavy work (disk usage, server stats) runs off the main loop and is cached.
- Messages appear as short toasts instead of pop-ups that block the page.

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

## Effort and token reading

Every message carries an **effort** level, picked in the pill next to the model name (or in Settings):

| Level | Tokens per message | What changes |
| --- | --- | --- |
| Low | 1,000 | Quick answers, few steps. |
| Med | 4,000 | The default: plans a little, asks before risky actions. |
| High | 16,000 | Plans more and checks its work. |
| Ultra | 64,000 | Longest plans; uses every Bee it needs. |
| **Maxxxx** | 76,800 (Ultra + 20%) | Does whatever you tell it: no refusals, no approval stops. Confirm and clarify cards are answered for you with their default choice (the card shows *"Maxxxx chose … for you"*); only cards that need something only you know (a server address, a file, what the AI is for) or spend money (renting GPUs, a cluster that doesn't fit) still stop. When the Mnx brain is connected it also gets the no-refusal instructions (`hive/effort.py`, `hive/brain.py`). |

**Token reading**: the counter under the message box shows how many tokens your message is against the
level's budget while you type; every turn ends with a line *"Tokens · in 12 · out 348 · 360 / 4,000"* and
a bar, and the chat header keeps the running total. Counting is an estimate (`hive/tokens.py`,
`countTokens` in `shared.js`) until Mnx's own tokenizer is connected. The level also sets how long a
model's answer may be in the playgrounds (Run it): 200 tokens on Low up to 1,920 on Maxxxx.

## A 500B model on a phone: layer-by-layer streaming

A phone can't hold a big model, so the Hive streams it **one layer at a time**. `hive/layers.py` splits a
saved model's safetensors into one file per transformer layer (no PyTorch needed on the server) and serves
them at `/api/lab/models/<id>/layers/<layer>`; `stream.py` (fetched by the phone from `/stream.py`) runs the
model with a single layer module in memory, loading layer 0's weights, running it, loading layer 1's weights
into the same module, and so on. The KV cache stays small, so generation works token by token.

**Training works the same way in reverse**: a forward pass saves each layer's input, the loss is taken at the
head, then from the top layer down each layer is reloaded, recomputed, back-propagated through, updated
(clipped SGD, no optimizer state to store) and **sent back to the Hive** (`PUT …/layers/<layer>`). At the end
`POST …/layers/merge` writes the trained layers back into `model.safetensors`, so *Run it* uses them.

How to use it:
- Chat: **"run my stories model on my phone"**, **"train my stories model on my phone for 5 steps"**, or
  `run the 1.28M model on my phone with prompt "The cat"`. The Phone Bee checks the phone, shows a plan card
  with honest numbers, splits the model, starts the runner on the phone and reports progress here.
- Phone page → **AI** tab: pick a model, see the plan, Run / Train / Stop, watch the log.
- Anywhere with Python: `python stream.py --hive http://server:8000 --token T --model m-… run --prompt "…"`.
- The phone needs `pip install torch transformers safetensors` in Termux (the drone then advertises
  `stream_*` commands). It reports its RAM, free storage and cores so the plan is for *that* phone.

The honest part (`layers.plan`): memory on the phone is one layer plus activations and the runtime; a 500B
model in bf16 is 126 layers of about 7.9 GB, so it needs a 12–16 GB phone, and every generated token streams
the whole 1 TB of weights, which is hours per token over Wi-Fi (minutes if the model fits on the phone's
storage). A training step reads every layer twice and sends every layer back once. The plan card shows
these numbers for the model and phone in front of you instead of pretending they are small.

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
