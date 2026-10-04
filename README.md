# Mnx

Mnx is an AI assistant web app. Responses are generated live by Claude, not scripted. While Mnx works you see each step as it happens:

- **Thinking**: the model's reasoning summary streams into the thinking area. The step label follows the current phase, e.g. *Understanding the core problem* → *Planning the slides*.
- **Activity steps**: *Reading file*, *Searching the web*, *Finding your location*, *Checking the weather*, *Writing snake.py*, and so on. Each one has a small animation and can be expanded.
- **Custom loading animation**: an orbiting dot loader shows rotating phrases ("Connecting the dots…", "Reading the results…") whenever the model is between steps.
- **Answer**: markdown streams in token by token, with code highlighting and copy buttons.

## Features

| Feature | How |
|---|---|
| Live web search & page reading | Anthropic server tools `web_search` / `web_fetch` |
| Weather (animated card, 7-day forecast) | Open-Meteo (no key) |
| Places: restaurants, shops, attractions | OpenStreetMap Nominatim + map embed + Google Maps links |
| Directions (drive / walk / cycle) | OSRM routing + Google Maps directions link |
| Your location ("near me", "here") | Browser geolocation (opt in with the 📍 button) |
| Files in any language: Python, HTML, JSON, Ruby, C, C++, Lua… | `create_file` tool. File card with highlighting, copy, download and live HTML preview |
| Slides (PPTX), PDF, Word (DOCX) | `create_document` tool. In-chat slide/page preview; files are built in the browser |
| Attachments | Images, PDFs and text/code files (click 📎, paste or drag & drop) |
| Connectors (MCP) | Add any remote MCP server URL in the **MCP** panel. A single aggregator server can expose thousands of apps |
| Chat history | Saved in your browser (IndexedDB) |
| Models | Opus 5.5 (default), Sonnet 5.5, Fable 5.1, with an adjustable thinking effort |

Mnx helps with anything legal. The system prompt tells it to decline only requests that would facilitate crimes or serious harm.

## Run it

```bash
npm install
export ANTHROPIC_API_KEY=sk-ant-...   # your key from console.anthropic.com
npm start                              # → http://localhost:3000
```

Set `PORT` to change the port. Node 18+ is required.

## Use your own local model

Mnx can run your own GGUF model, such as **mnx-q4_k_m.gguf** (your Qwen2.5-3B fine-tune), on your own device with llama.cpp. It's private, it works offline, and it needs no API key. Mnx uses the first `.gguf` in `models/` (preferring `mnx*`), or the file you set in `MNX_LOCAL_MODEL`. It appears in the model picker as **Mnx · Local (your model)**, and it's the default when no `ANTHROPIC_API_KEY` is set.

Get the model into `models/` by one of these:
- **Keep it in this repo (Git LFS).** Run `bash scripts/add-model-to-repo.sh /path/to/mnx-q4_k_m.gguf` once, from any computer or phone that has the file. It commits the model with Git LFS and pushes it. After that, `git clone` followed by `git lfs pull` gets the model, and the Termux setup script does this for you. GitHub's free plan allows LFS files up to 2 GB, and LFS storage and download bandwidth count against your quota. Every clone downloads the full ~1.9 GB.
- `HF_TOKEN=hf_... npm run get-model`. This downloads `tharunmakes/mnx-qwen2.5-3b-gguf/mnx-q4_k_m.gguf`. The repo is private, so you need a read token from https://huggingface.co/settings/tokens.
- `npm run get-model -- <google-drive-link>`. The file must be shared as "Anyone with the link".
- Copy the file in by hand.

### On your Android phone (Termux)

1. Install **Termux** from F-Droid or GitHub. The Play Store version is outdated.
2. In Termux:
   ```bash
   pkg install -y git
   git clone https://github.com/Tharunmaks/Mnx && cd Mnx
   git checkout claude/mnx-ai-web-interface-f75vu6   # until it's merged
   bash scripts/termux-setup.sh
   ```
   The script installs Node.js and llama.cpp (`llama-server`), then installs Mnx. It finds `mnx*.gguf` in your phone's **Downloads** folder and links it (so the file isn't stored twice), or downloads it if `HF_TOKEN` is set.
3. `npm start`, then open **http://localhost:3000** in your phone's browser.

On Android, Mnx starts `llama-server` itself the first time you chat with the local model. Loading a 3B model takes a few seconds. `termux-wake-lock` stops Android from pausing it in the background. A 3B Q4 model needs roughly 2.5 GB of free RAM. If your phone is slow, set `MNX_THREADS` to its number of performance cores. For a shorter context, set `MNX_CTX=2048`. To use a llama-server you started yourself, set `MNX_LLAMA_SERVER=http://127.0.0.1:8080`.

The online models still work from the phone if you set `ANTHROPIC_API_KEY`.

### Tools for your local model

Your model can use tools on the phone. It writes a `<tool_call>` in Qwen2.5's native format, Mnx runs the tool, sends the result back as a `<tool_response>`, and the model writes the final answer. Every step shows up live in the chat:

| Tool | What it does |
|---|---|
| `search_web` / `read_webpage` | Free web search (DuckDuckGo, with Wikipedia as a fallback) and reading pages |
| `create_image` | Images from a text prompt through the free Pollinations service (no key) |
| `create_document` | Slides (PPTX), PDF and Word files, built in your browser |
| `create_file` | Code and text files |
| `get_weather`, `find_places`, `get_directions`, `get_user_location` | Weather, places and routes |

A 3B model sometimes misses a tool or formats a call badly. Mnx repairs small JSON mistakes and allows up to 5 tool steps per reply. The local model still can't see images or PDFs you attach.

## How it works

```
browser (public/)  ── POST /api/chat ──►  server.js  ── stream ──►  Claude API
      ▲                                      │  runs tools (tools.js)
      └──────── Server-Sent Events ◄─────────┘  thinking / text / tool steps / cards
```

- `server.js` runs the agent loop. It streams Claude's events (thinking deltas, text deltas, tool-input deltas) to the browser, runs the tools, and loops until the answer is complete. Adaptive thinking is on with summarized display, so the reasoning can be shown live. Server-side refusal fallbacks are enabled.
- `local.js` runs the local GGUF model, either in process (node-llama-cpp, desktop) or through `llama-server` (Termux), and streams it with the same events.
- `tools.js` defines the tools and calls the free public APIs.
- `public/app.js` renders the live timeline, cards and history. `public/docs.js` builds the PPTX, PDF and DOCX files.
