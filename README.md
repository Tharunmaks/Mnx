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

Tests (no model files or GPU needed; the virtual LPU is checked against an independent numpy transformer):

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

For the Browser Bee, install Chromium once: `playwright install --with-deps chromium`
(or point `MNX_CHROMIUM_PATH` at an existing Chrome/Chromium).

| Page | What it is |
| --- | --- |
| `/` | Chat: left rail (new chat, history, MCP tools, Hive, settings, profile), wordmark + greeting, message box, live step tree, Bees panel |
| `/usage.html` | Usage and limits: tokens, server, each Bee's CPU / memory / disk, training, GPU rentals, with the limits |
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
hive/usage.py     Usage ledger (tokens per day, GPU rentals) and the measurements behind the Usage page
hive/usage_chat.py "show my usage"
hive/bees.py      Bee registry: Bees you make and Bees that Bees make, family limits, keys, computer sizes
hive/bee_chat.py  "create a bee called … that …", "give the … bee a large computer", "list my bees"
hive/budget.py    GPU Bee budget planner: "I have 1500 INR, which GPU should I rent?" with live or approximate prices
hive/lpu.py       Virtual LPU: cycle-accurate chip simulator (SRAM, matrix/vector units, links) + deterministic compiler + chip mesh
hive/lpu_lab.py   "run my model on the lpu / on 8 lpu chips": plan card with compiled cycles, simulation, chip statistics
hive/modes.py     Rush (every Bee together: fan-out commands, roll call, the board) and Peace (one chat per Bee)
hive/cloud.py     Real cloud computers per Bee: Hetzner / DigitalOcean, cheapest 8 GB / 40 GB server, SSH, delete on remove
hive/plan.py      Free / Plus plans (both ₹0); hive/plans.py applies a plan to every Bee, "subscribe me to plus"
hive/admin.py     Owner commands: set / list / unset credentials, owner token, user tokens (never in git)
hive/iotamine.py  Iotamine VPS API: list your VPSes, price and deploy one with the Hive's SSH key, add it to the Hive's cloud
hive/gpu_pool.py  The Hive's GPU: free Kaggle GPU (16 GB VRAM) jobs with the Bee's files, outputs back into its computer
hive/bots.py      Bots you make: personality, tone, starters, knowledge search, memory, voice, export/import, templates
hive/brain.py     The brain: any OpenAI-compatible server (Ollama, Groq, Gemini, OpenRouter, your Lab model)
web/bots.html     Make, edit, explore, remix, export and import bots · web/freecloud.html: Plus and the free cloud
hive/voice.py     Fish Audio: text to speech, speech to text, voice search; voice settings
hive/calls.py     Calls: "call me …", ringing in the app or a real phone call (Twilio), cards answered by voice
web/js/voice.js   The call screen, Talk mode, Fish voice playback and listening
hive/vortex.py    Vortex OS on every Bee's computer: Docker image, local Node process or SSH tunnel; accounts and ports
drone/drone.py    Termux agent for your phone (served at /drone.py)
services/         systemd unit to run the gateway 24/7
web/              index.html, hive.html, css/, js/ — plain HTML/CSS/JS, no build step
data/             Created at runtime: token.txt, hive.json (Bees), connectors.json, vault.json (connector secrets, chmod 600),
                  cells/<bee>/ (work/ files, browser/ profile, jobs.json, logs/)
```

Chat history is kept in the browser (localStorage) for now.

## Usage and limits

**Usage** in the sidebar (`/usage.html`, data from `GET /api/usage`) shows what has actually been used next to
the limit it runs under. Everything is measured; what can't be measured says "not measured" instead of a guess.

- **Tokens**: today, last 7 days, all time, a 14-day chart (with a table view), and for each effort level its limit
  per message, how many messages used it today and the biggest one as a percent of the limit. Every message is
  counted when it ends, saved per day in `data/usage.json` (120 days) and counted by the Hive's own token reading,
  within about 10-15% of a real tokenizer.
- **This server**: CPU, memory, disk, the Hive's data folder, and each GPU's memory and busy percent.
- **Bees**: how many of the 64, how deep the family goes, and per Bee its computer, CPU and memory from
  `docker stats` against its limit, disk, and jobs running. Without Docker only disk and jobs are measured.
- **Training and models**: runs by status, time spent training, models and their disk, models being served.
- **Rented GPUs**: machines billing now with hours and cost so far (hours x the price RunPod reported), and the
  hours and cost of every finished rental, kept after the machine is gone. The invoice on runpod.io is the authority.
- **Limits**: tokens per message, computer sizes, Bee limits and the upload size.
- **Plan** (Free / Plus, both free) and **Cloud computers**: each Bee's own cloud server with hours, price and cost so far.

In the chat: **"show my usage"**, **"how many tokens have I used"**, **"what are my limits"**.

## Bees that make Bees, each with its own computer

Anyone can make a Bee, and so can a Bee. In the chat:

- **"create a bee called Scout that watches prices"**: a card asks for the name, the one-line skill, the
  computer size and where it runs, then the Bee is ready with its own computer.
- **"tell the Coding bee to create a bee called Tester that runs the tests"**: the new Bee belongs to Coding.
- **"give the Scout bee a large computer"**, **"upgrade Scout to xl"**, **"move Scout to my RunPod server"**,
  **"give Scout the most powerful computer"** (pick a connected GPU server, the whole of this one, or rent a
  GPU machine on RunPod after a price confirmation), **"list my bees"** (the family tree), **"delete the Scout bee"**.

On the Hive page, **Add Bee** has *Made by*, *Computer* and *Runs on*; every card shows who made it, its
children and its computer, and **Computer** resizes or moves it.

**A Bee makes Bees from inside its own computer** with its own key (`MNX_BEE_KEY`, set in its Cell):

```bash
python3 .mnx/mnx-bee new "Deal Finder" "finds deals online" medium
python3 .mnx/mnx-bee children
```

The key only opens `/bee-api` (create and list its own children); it can't reach your `/api`. Limits stop
a runaway: 64 Bees in all, 4 generations, 12 children per Bee. Removing a Bee hands its children to its parent.

**Computers** (`hive/bees.py`, `hive/cells.py`): every Bee gets a **Standard** computer: 2 CPUs, 8 GB RAM and
40 GB disk. Other sizes: Small (1 CPU, 1 GB, 10 GB), Medium (2, 4, 20), Large (4, 16, 80), XL (8, 32, 160) and
Max (the whole machine). On this server CPU and memory are Docker limits (`--cpus`, `--memory`), kept within
what the server has; the disk is watched by the Hive, and a Bee whose files fill its disk can't start jobs until
some are deleted. A new size restarts the container and keeps its files. Without Docker (a phone, say) the Cell
is a folder and CPU and memory limits aren't enforced. On a cloud server (any SSH server in the Lab, including
rented machines) the Bee's terminal, commands, jobs and files run over SSH in `<server folder>/cells/<bee>`,
with the whole machine. A child gets a Standard computer on its parent's machine. Bees saved on the old Small
default move up to Standard; a size you chose yourself is kept.
Every computer runs Vortex OS by default (below); choose Plain Linux (`MNX_CELL_IMAGE`) per Bee instead.
Set `MNX_PUBLIC_URL` so Bees on remote servers can reach the Hive to make children.

## Rush and Peace: two ways to work with the Bees

The chat has a **Rush · Peace** switch at the top (`hive/modes.py`).

**Rush**: every Bee works together on one page. The Queen gets your message and hands it to all of them at
once. A board under the top bar has a tile for each Bee showing what it's doing right now, lit while it works.
- **"everyone run pip install numpy"**, **"all bees run: df -h"**: the same command on every Bee's own computer
  in parallel (8 at a time), after one confirmation. Each Bee's output appears under its name, with a summary.
- **"roll call"**, **"everyone report"**: each Bee reports its skill, computer, jobs and children.
- Everything else the Hive understands (train, create a model, create Bees, cloud computers …) runs with every
  Bee it needs. With the Mnx brain connected, the Queen splits any goal between the Bees and merges their work.

**Peace**: a separate chat with each Bee, like chatting with one bot. The rail lists the Bees with the last thing
said; pick one and it answers as itself, under its own badge.
- **"$ ls -la"**, **"run `df -h`"**: runs on that Bee's own computer.
- **"who are you"**, **"status"**: its skill, computer, jobs and children.
- **"create a bee called … that …"**: the new Bee is this Bee's child.
- Everything else the Hive understands, answered by this Bee. With the brain connected it talks in its own voice.

## Real cloud computers for every Bee

A Bee can have its own machine in the cloud (`hive/cloud.py`): the cheapest server type with at least 8 GB RAM,
40 GB disk and 2 CPUs, from **Hetzner Cloud** (`HCLOUD_TOKEN`, Singapore first, the nearest to India) or
**DigitalOcean** (`DIGITALOCEAN_TOKEN`, Bangalore first). Save the token in Cloud Bee · Lab → Secrets, then:
- **"give every bee a real cloud computer"**, **"give the Scout bee its own cloud computer"**, or *Its own cloud
  computer* in a Bee's Computer dialog, or *Runs on → Its own cloud computer* when creating a Bee.
- A card shows the machine, the price per month for all of them (and roughly in rupees) and asks first; it's
  real money on your account.
- The Hive starts each server with only its own SSH key allowed (no passwords), Docker and Node.js, waits for
  SSH, adds it as a Lab server and moves the Bee there (Vortex OS runs on it). If a server can't be reached, it
  is deleted at once so nothing bills for nothing.
- **"bring the bees back home"** or removing a Bee deletes its server, so billing stops. The Usage page lists
  every cloud computer with hours, price per month and cost so far.
- Provider keys (and the RunPod key) stay with the Hive: they are never passed to training code.

These paid cloud computers are optional. The free way is **Plus** and the Hive's free cloud, below.

## Plus: free 16 GB computers with a 16 GB GPU

Two plans, both **₹0** (`hive/plan.py`, `hive/plans.py`):

| Plan | Every Bee and bot gets |
| --- | --- |
| **Free** | Standard: 2 CPUs, 8 GB RAM, 40 GB disk |
| **Plus** | 4 CPUs, 16 GB RAM, 100 GB disk and a GPU with 16 GB VRAM |

Subscribe with one tap on the **Free cloud** page or the Usage page, or say **"subscribe me to plus"**. Every
Bee moves to the Plus computer, except Bees whose computer you chose yourself and the whole-machine Bees
(Cloud, GPU); **"switch back to free"** reverses it. A hosted Hive can lock the plan with `MNX_PLAN=free|plus`.

**Users never bring a GPU or a key: the Hive gives them.** It runs on the Hive's own free cloud, which the Hive's
owner sets up once on the **Free cloud** page (about 10 minutes, no payment):

1. **Free GPU, 16 GB VRAM** (`hive/gpu_pool.py`): Kaggle gives every account a free NVIDIA T4 or P100 with
   16 GB VRAM, about 30 hours a week. Paste your Kaggle user name and API token; the Hive installs nothing on
   the Bees. A Bee's GPU job is its script plus its files, run on that GPU; the output files come back into the
   Bee's computer under `gpu-out/<job>/`. From a Bee's terminal: `python3 .mnx/mnx-bee gpu train.py data.csv`;
   in its chat: **"run train.py on the gpu"**, **"gpu status"**. The Hive tracks the weekly free hours. When the
   Hive's own machine has an NVIDIA GPU, Plus Bees use it directly instead. Needs `pip install kaggle` on the Hive.
2. **Free servers** (`hive/cloud.py`): Oracle Cloud's Always Free machine (ARM, 4 CPUs, 24 GB RAM, 200 GB disk)
   costs nothing. Create it with the Hive's SSH key (shown on the page, with a Copy button) and paste its public
   IP; the Hive logs in, installs Docker and Node and puts Plus Bees there by itself, two 16 GB Bees per machine
   (memory limits are ceilings, so they share it well). Add more machines for more Bees. Until one is added,
   Plus Bees run on the Hive's own machine with 16 GB limits (on a phone those can't be enforced).

3. **Iotamine VPS** (`hive/iotamine.py`, iotamine.com, India): paste an Iotamine API key (scope “Read & write” to
   deploy). The Hive lists your VPSes; **Use for the Hive** adds one (it needs the Hive's SSH key in root's
   `authorized_keys`), and **Price and deploy** makes a new Ubuntu VPS with the Hive's key already on it, after
   showing the hourly and monthly price from Iotamine's own rates (the IP is a monthly charge). Upgrade a VPS on
   Iotamine (say 10 vCPUs, 20 GB) and the Hive places more Bees on it: two Plus Bees per 16 GB.

Kaggle's and Oracle's own rules apply: they're for your own work (no crypto mining; Kaggle's GPU has a weekly
limit and sessions of up to about 12 hours). The Kaggle and cloud keys are Hive-only: never passed to training code.

## Owner and users: only the owner enters credentials

Two kinds of access token (`gateway/auth.py`):
- **Owner** (you): the token the Hive prints on start (`python3 -m hive.admin token`). Only it can enter or see
  credentials (Iotamine, Kaggle, Hetzner, DigitalOcean, RunPod, Fish Audio, Twilio, the brain), add servers, rent or
  deploy machines, or read the Hive's SSH key. Those pages and settings are hidden from users, and the server answers
  users **403** if they try.
- **Users**: a token each (`python3 -m hive.admin user-token`; or `MNX_USER_TOKENS`). They chat, make bots, run their
  Bees, subscribe to Plus and use the GPU, and never see or add a key, an account or a server. The Free cloud page shows
  them only a status line.

Credentials never go into the code or git. Enter them once, on the Hive's own machine:

```bash
python3 -m hive.admin set IOTAMINE_API_KEY        # asks for the value, hidden as you type
python3 -m hive.admin set KAGGLE_USERNAME <name>
python3 -m hive.admin list                        # names only
```

They are kept in `data/lab/secrets.json` (owner-only file, `data/` is git-ignored) and are never passed to training code.

## Make your own bots

Not premade: you make them (`hive/bots.py`, the **Bots** page, `web/js/bots.js`). Ideas from Grok's companions
(a personality you talk to, its own voice, it remembers you), OpenAI's GPTs (instructions, knowledge files,
conversation starters, a store to share and remix) and creative muse bots.

- **Make one**: on the Bots page, or say **"make a bot called Maya that is a friendly Tamil tutor"** (a card lets
  you shape it). Give it an emoji, a one-line description, its **personality and instructions**, a **tone**
  (friendly, playful, professional, concise, caring), a **first message** and up to four **conversation starters**.
- **Knowledge**: add text files (txt, md, csv, json, code; 2 MB each, 20 per bot). It answers from them and
  names the file; without a brain it quotes the lines that match your question.
- **Memory**: **"remember that I'm vegetarian"**, **"what do you remember about me"**, **"forget that …"**,
  **"forget everything"**. With a brain it uses what it remembers in every answer.
- **Voice**: pick a Fish Audio voice for it; its replies, Talk mode and **"call me"** use that voice.
- **Its own computer**: a bot is a Bee, so "$ …" runs on its computer (switch it off for a chat-only bot).
- **Explore**: start from an idea (Buddy, Tutor, Code Buddy, Muse, Coach, Language Buddy, Interview Coach,
  Researcher) and make it yours; see bots shared on this Hive; **Remix** any bot (knowledge included);
  **Export** a bot as a file and **Import** it into any Mnx Hive.
- **Chat with it** in **Peace**: your bots are first in the list, each opens with its emoji, first message and
  starters. With a brain it talks in character and sees the last turns of the chat.

## The brain: what the Hive and your bots think with

**Settings → Brain** (`hive/brain.py`): any OpenAI-compatible chat server. Free choices: **Ollama** on any
computer (no key), your own model served from the Lab (llama.cpp), or the free tiers of **Groq**, **Google
Gemini** and **OpenRouter** (paste a key). **Test** asks it to say OK. The key stays on the server
(`data/brain.json`, mode 600). Without a brain everything that doesn't need one keeps working.

## Voice: Fish Audio, Talk mode and calls

The Bees speak with a [Fish Audio](https://fish.audio) voice and hear you with Fish Audio speech to text
(`hive/voice.py`, `hive/calls.py`, `web/js/voice.js`). In the chat's **Settings → Voice**: paste your Fish Audio
API key, search fish.audio voices and pick one, choose the model (`s1`, `s2-pro`, `s2.1-pro`) and language.
The key stays on the server (`data/voice.json`, mode 600); the page only gets the audio.

- **Speak replies aloud**: every answer is read out with the Fish voice.
- **Talk** (the microphone in the composer): a voice call with the Hive, or in Peace with that Bee. Say what you
  want; it's run as a Hive request and the answer is spoken. A Hive call works in Rush (every Bee), a Bee's call
  in that Bee's own chat.
- **"call me"**, **"call me in 10 minutes"**, **"call me at 9am and remind me to stretch"**, **"have the Cloud bee
  call me about my training"**, **"call me and tell me how my training is going"** (it finds out when you pick up
  and says it). Times use your time zone (Settings → Voice, India by default); scheduled calls survive a restart.
- The call rings on every open Hive page (and a phone running drone.py vibrates). Answer, and the Bee speaks
  with the Fish voice, listens (it pauses while it speaks, so it doesn't hear itself) and does what you say.
  Cards are read out: say **"yes"**, **"no"**, or a choice's name ("make it large"); cards that need typing stay
  on screen. **"Stop"** stops what it's doing, **"bye"** hangs up. You can also type on the call screen.
- **Real phone calls**: add your number and a Twilio account (SID, auth token, a Twilio number) and set
  `MNX_PUBLIC_URL` to the Hive's public https address. The Bee then calls your phone; its lines are the Fish
  Audio MP3s played into the call, and what you say is heard by Twilio's speech recognition. Twilio's webhooks are
  checked with its signature and each call's own secret. **"call me on my phone"** forces a phone call.
- Without a Fish Audio key, the browser's own voice and speech recognition stand in (Chrome on Android has both).

## Vortex OS on every Bee's computer

Each Bee's computer runs [Vortex OS](https://github.com/Tharunmaks/Vortex-os-), a web desktop with files,
terminal and apps, signed in as the Bee itself (`hive/vortex.py`). The desktop's home folder is the Bee's own
work folder, so files the Bee makes show up there, and `mnx-bee` works in its terminal.

- **Docker**: the Bee's container *is* Vortex OS (image `MNX_VORTEX_IMAGE`, built once from `MNX_VORTEX_REPO`),
  hardened like Vortex's own compose file: no capabilities, read-only root, its own user, the computer's limits.
  If the image can't be built, the computer starts as plain Linux and its Cell says why.
- **No Docker (a phone)**: one Node process per Bee from a checkout in `MNX_VORTEX_DIR` (default `vortex-os/`),
  cloned and installed the first time a desktop starts. Needs Node 20+, and `make`/`clang` for its terminal
  (Termux: `pkg install nodejs python make clang`).
- **Cloud servers**: Vortex runs on the server (Docker if it has it, else Node) and the Hive opens an SSH
  tunnel to it.

Desktops listen on `MNX_VORTEX_BIND` (default `127.0.0.1`; set `0.0.0.0` to open them from another device on
your network), one port per Bee from 8421. Each Bee's account and strong password are kept in
`data/vortex.json` (private, mode 600). Open it from the Cell's **Desktop** tab (Start, Open, Show/Copy
password, Stop), or say “open the scout bee's desktop”, “give the scout bee vortex os”, “switch scout to
plain linux”. The Add Bee and Computer dialogs have an Operating system choice.

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

**Which GPU should I rent?** Say your budget: “I have 1500 INR, which GPU should I rent?”, “₹2k to fine-tune a 7B
model”, “$20, what can I train?”, “with 1.5 lakh rupees can I pretrain a 1B model”. The GPU Bee (`hive/budget.py`)
converts the money (approximate rates for INR, EUR, GBP), prices every GPU (RunPod's live prices when
`RUNPOD_API_KEY` is saved, else a built-in table marked approximate) and shows, per GPU: hours bought (15 minutes
of each rental go to booting and downloads), the largest model you could pretrain from scratch in that time
(compute-optimal at 20 tokens per parameter, capped by memory), the largest model a LoRA fine-tune fits, and how
many tokens a LoRA fine-tune gets through. It recommends the cheapest per hour among the GPUs within 15% of the most
training compute (more hours leave room for mistakes), names a bigger-memory alternative, and for a named size says
what a proper pretraining run costs and how long it takes. For ₹1,500 (about $18) that is about 39 hours of an
RTX 4090 or 65 of an RTX A5000 at typical prices: enough to pretrain a model of roughly 270M parameters, or to LoRA
fine-tune up to about 8B.

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

## A virtual LPU: cycle-accurate chip + deterministic compiler

`hive/lpu.py` is a Groq-style Language Processing Unit written as software, and the compiler that schedules a
transformer onto it. Nothing in it is a guess at runtime:

- **Virtual silicon** (`Chip`): the only memory is SRAM, a banks × words 2-D array (230 MB of 16-bit words per
  chip, like the model's own bf16 weights; the units compute in 32-bit). No caches: every access takes a fixed number of cycles.
  Functional units are 2 matrix units (320×320 systolic arrays, weights stationary: fill, stream the rows through,
  drain), 4 vector units (320 lanes), one move port per unit (SRAM → the unit's input), a host link (32 GB/s)
  and chip-to-chip links (100 GB/s, fixed 450-cycle latency). A master tick loop advances one clock cycle per
  iteration: instructions that end write their result into SRAM, instructions that start check their inputs are
  there and their unit is free; if not, it stops with a `ScheduleError` (a compiler bug, never a stall). The loop
  can skip idle cycles; the state after every cycle is identical either way.
- **The brain** (`compile_model`): takes the model's shapes (Qwen2 / Llama decoder layers) and plans everything
  up front: which SRAM address holds which tensor from which cycle (a static allocator with time, so a region is
  reused only after its last reader has finished), and when every load, move, matmul, vector op, store, send and
  receive starts and ends, on which unit. Weights that fit stay **resident** (loaded once); otherwise they
  **stream** from the host every token, like the phone. The compiled program is replayed for every token.
- **Virtual network** (`Network`): chips step in lockstep on one global clock and talk over fixed-latency
  queues (no switches). A layer too big for one chip is spread over a group (whole heads and MLP rows per chip,
  partial sums combined with a ring all-reduce over the links); groups form pipeline stages. The layout uses the
  chip budget you give: `run my model on 8 lpu chips`.

It runs the real weights: the simulated chips produce the same text as the model (bit-exact on one chip against a
plain numpy forward pass before bf16 rounding; across chips the partial sums round differently, as on real hardware).

How to use it:
- Chat: **"run my stories model on the lpu"**, **"run stories on 8 lpu chips"**, or
  `run the 1.28M model on 4 lpu chips with prompt "The cat"`. The LPU Bee compiles the model and shows the plan
  (cycles per token, time on the chip, tokens/s, SRAM, resident or streamed, what the simulation costs here),
  then simulates token by token and reports the chip's numbers next to the answer (instructions, unit
  utilisation, bytes over the host link and the chip links, SRAM peak).
- Training: **"train my stories model on the lpu for 20 steps"** (or on 2 lpu chips). The compiler schedules a whole
  SGD step: the forward pass keeps each layer's input, cross-entropy at the head, then from the top layer down each
  layer is recomputed, back-propagated through (attention, RoPE, RMSNorm, SwiGLU, all as matrix- and vector-unit
  instructions), and every weight shard is updated in place; replicated-input gradients are all-reduced over the
  group, the gradient goes back down the pipeline over the links, and the embedding gradient goes to the host.
  Training uses 32-bit SRAM words (exact master weights) and plain SGD (lr 0.03, no optimizer state on the chip).
  Afterwards the trained layers are written back in the model's own dtype and merged, exactly like the phone's.
  Checked against an independent float64 numpy backward pass (itself checked by finite differences): the updated
  weights match to 1e-7 on 1 to 8 chips, resident or streamed.
- API: `GET /api/lab/models/<id>/lpu/plan?chips=8&seq=64` gives the same plan as JSON (`&train=1` for a training step).
- Phone: in the Phone page's AI tab choose **Virtual LPU** as the engine (Run and Train both use it), or say
  **"run my stories model on my phone on the lpu"** / **"train … on my phone on the lpu"**: the phone streams each
  layer from the Hive and runs it on the simulated chip (`stream.py run|train --engine lpu`, which fetches `/lpu.py`
  from the Hive). Only numpy is needed for the chip itself.
- Anywhere with Python: `lpu.plan(config, params, seq, chips)` for the numbers, `lpu.Model(...)` to run.

The honest part: the plan card is computed by compiling, not guessing. A 1.3B model does not fit one chip (a
layer's 134 MB plus activations is more than the SRAM) but fits 2 chips per layer; 24 chips keep it resident at
about 94 tokens/s on the chips, 8 chips stream it at about 21 tokens/s. A 500B model needs 64 chips per layer
just to stream (2 tokens/s) and 8,064 chips to be resident. Simulating is slow because every multiply really
runs in numpy: well under a second per token for the tiny models the Hive trains, 8 seconds per token for a
1.3B, 46 minutes per token for a 500B. Training on the chip is for models the simulator can hold: the Hive's own
small models train in a fraction of a second per step; a big model's training plan is exact, its simulation is not
something to wait for.

## Event protocol

Browser → gateway (WebSocket `/ws`):

```json
{"type": "user_message", "chat": "c_…", "task": "t_…", "text": "…", "mode": "rush|peace", "bee": "<bee id, Peace only>"}
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
| `output` | A command's output on a Bee's computer | `bee`, `command`, `code` |
| `rush` | "Rush: N Bees on it" with their names | `bees` |
| `answer` / `answer_delta` | Final answer text (set / stream) | |
| `task_done` / `task_error` | Ends the task | |
| `bees_changed` `tools_changed` `bee_status` | Refreshes the Bees panel / tools | |
| `call` · `call_said` · `call_heard` · `call_event` | The call screen: state, a line the Bee spoke (`n`, `audio`), what you said, the request's steps | `call` |

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
2. Put its URL in Settings → Brain (or `MNX_BRAIN_URL`); bots, Peace and Rush start using it at once.
3. In `hive/queen.py`, replace the placeholder with real planning and Bee calls.
