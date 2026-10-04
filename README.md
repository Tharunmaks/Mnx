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

On Android, Mnx starts `llama-server` itself the first time you chat with the local model. Loading a 3B model takes a few seconds. `termux-wake-lock` stops Android from pausing it in the background. A 3B Q4 model needs roughly 2.5 GB of free RAM. If your phone is slow, set `MNX_THREADS` to its number of performance cores. Keep the context at 6144 tokens or more (`MNX_CTX`, default 8192): Mnx's instructions with all 83 tools take about 2,900 tokens. To use a llama-server you started yourself, set `MNX_LLAMA_SERVER=http://127.0.0.1:8080`.

The online models still work from the phone if you set `ANTHROPIC_API_KEY`.

### Make it faster on your phone

A phone CPU runs a 3B model slowly, so every second counts. In order of impact:

1. **Use the Q4_0 copy of your model:** `npm run fast-model`, then restart Mnx. Phone CPUs have fast paths for Q4_0, so Mnx reads your message several times faster and writes the answer a bit faster. Mnx picks `models/mnx-q4_0.gguf` automatically on Android. It needs about 1.8 GB more storage.
2. **Battery: Unrestricted for Termux** (Settings → Apps → Termux → Battery). Otherwise Android slows Termux down while you're in the browser.
3. **A smaller model.** In `training/mnx_train.ipynb`, set `BASE` to Qwen2.5-1.5B (about 2× faster) or 0.5B (about 5× faster, but weaker), then retrain.
4. **Run the model on a computer and chat from your phone.** A laptop is many times faster than a phone, and a computer with a GPU (`MNX_GPU_LAYERS=99`) can be 20–50× faster. Start Mnx on the computer with `HOST=0.0.0.0 npm start` and open `http://<computer-ip>:3000` on your phone (only on a network you trust).

Mnx already uses only the phone's fast CPU cores, a 4096-token context, cached instructions and shorter tool results on Android. The speed of each reply appears in Termux, e.g. `[local] reply: read 300 new tokens in 9.0s (33/s), wrote 120 tokens at 8.5/s`.

### Tools for your local model

Your model can use tools on the phone. It writes a `<tool_call>` in Qwen2.5's native format, Mnx runs the tool, sends the result back as a `<tool_response>`, and the model writes the final answer. Every step shows up live in the chat:

| Tool | What it does |
|---|---|
| `search_web` / `read_webpage` | Free web search (DuckDuckGo, with Wikipedia as a fallback) and reading pages |
| `create_image` | Images from a text prompt through the free Pollinations service (no key) |
| `create_document` | Slides (PPTX), PDF and Word files, built in your browser |
| `create_file` | Code and text files |
| `get_weather`, `find_places`, `get_directions`, `get_user_location` | Weather, places and routes |
| `run_code` | Runs Python or JavaScript on your phone, **only after you tap Run**. Shows the output, errors and any charts the code saves |
| `calculate` | Exact math: percentages, powers, roots, factorials, trig (degrees), logs. Works offline |
| `convert_units` | Length, weight, temperature, volume, speed, area, data size, time, energy, pressure. Works offline |
| `convert_currency` | Today's exchange rates (open.er-api.com, no key) |
| `get_time` | The local time and date anywhere in the world |
| `wikipedia` | A short, reliable summary of a person, place, thing or event |
| `define_word` | Dictionary meanings, pronunciation, examples and synonyms (dictionaryapi.dev) |
| `translate` | Translations into ~40 languages (MyMemory, no key) |
| `create_qr_code` | QR codes for links, Wi-Fi details, contacts or any text |

**65 quick tools** (`tools-more.js`) are on top of these, for exact answers instead of guesses:

| Group | Tools |
|---|---|
| Money | loan EMI, savings growth (compound interest), tax/GST, discounts, bill split with tip, percentage change, trip fuel cost |
| Health & sport | BMI, daily calories (BMR/TDEE), running pace |
| Dates & time | exact age, days between dates, add to a date, day of the week, countdowns, Unix time, time-zone conversion, timers that ring |
| Numbers | statistics, GCD/LCM, prime factors, number in words (international or lakh/crore), Roman numerals, binary/hex |
| Text | word count, change case, slugs, sort/de-duplicate lines, extract emails/phones/links, text diff, lorem ipsum |
| Developer | JSON check & format, regex tester, Base64, URL encoding, hashes, JWT decode, cron explained, HTTP status codes, IP subnets, npm/PyPI and GitHub info |
| Fun & random | dice, coin flips, random numbers, random picks, jokes, quotes |
| Live info (free, no key) | public holidays, country facts, crypto prices, books (Open Library), synonyms & rhymes, air quality, sunrise/sunset, recent earthquakes |
| Memory & notes | remember / recall / forget facts about you, save and list notes, passwords and UUIDs |

The online models get every tool directly. The local model gets them as a compact one-line-each list, so its instructions stay small enough for a phone.

**Memory.** Tell Mnx "remember that I'm vegetarian" and it saves the fact in `data/memory.json` on your device and uses it in every chat, with the local and the online models alike. "What do you remember about me?" lists the facts, and "forget …" removes one.

**Learning from your chats.** Mnx gets better from the way you use it:
- Tap **👍** on a good answer. Mnx saves the conversation exactly as the model saw and wrote it.
- Tap **👎** and write what it should have said. Your answer replaces Mnx's in the saved example. A 👎 without a correction is kept for review but never trained on.
- With **Learn from my chats** on (Settings, on by default), chats with an online model (Fable, Opus, Sonnet) are saved as teacher examples in the local model's format, so your model learns from the bigger one. Chats that used tools the local model doesn't have, such as built-in web search or connectors, are skipped.

Everything is saved in `data/learned.jsonl` on your device. To include it in the next training run:
```bash
npm run learn                                   # checks it → training/data/extra.jsonl
HF_TOKEN=hf_... npm run learn -- --upload       # also uploads it (private) for the Colab notebook
```
The notebook downloads your learned examples automatically and weights them 3×. The model only actually changes when you retrain it: a 3B model can't fine-tune itself on a phone. Memory, on the other hand, works instantly.

**▶ Run on code in answers.** Python and JavaScript code blocks in answers have a **Run** button. Tapping it is the approval: the code runs on your device and the output appears under the block.

Other websites open in the same browser can't send requests to Mnx. Requests from another origin are refused, so a web page can't chat as you, change your memory or add training data.

**Code runner safety.** Every program appears in an approval card, and nothing runs until you tap **Run**. **Don't run** (or no answer within 10 minutes) skips it. Programs run in a temporary folder with a 30-second limit, and they don't get Mnx's API keys. They can still reach your files and the internet, so read the code before you approve it. To turn the runner off, set `MNX_CODE_RUNNER=off`. Charts need `pip install matplotlib`. Mnx only accepts connections from the device it runs on. To use it from another device, set `HOST=0.0.0.0`, but only on a network you trust.

**Answer quality.** The local model is told to start with the direct answer, use clear markdown (headings, lists, tables, code blocks), match the answer's length to the question, and cite its sources. It uses Qwen2.5's recommended sampling: temperature 0.7, top_p 0.8, top_k 20, repeat penalty 1.05. You can change these with `MNX_TEMPERATURE`, `MNX_TOP_P`, `MNX_TOP_K` and `MNX_REPEAT_PENALTY`.

Online models (Opus, Sonnet, Fable) get the same everyday tools. A 3B model sometimes misses a tool or formats a call badly. Mnx repairs common JSON mistakes (raw line breaks in code, trailing commas, code fences) and allows up to 5 tool steps per reply. If the model starts repeating itself, Mnx stops it and says so; if an answer hits the length limit, Mnx tells you to say “continue”. The local model still can't see images or PDFs you attach.

## Training your model

`training/` fine-tunes your local model to use Mnx's tools reliably, then measures how reliable it is.

1. **Data.** `npm run train:data` writes `training/data/train.jsonl` with **about 115,000 distinct conversations** (100,000 from `generate.py` plus 15,000 for the quick tools from `gen_quick.mjs`, whose results come from actually running the tools; ~700 MB, about a minute) in exactly the format Mnx uses at runtime (`<think>`, `<tool_call>`, `<tool_response>`):
   - Web search (prices, sports, launches, events, software versions, news, films; reading pages; retrying after errors)
   - Weather, location, places and directions in about 150 cities
   - Everyday tools: calculator, unit and currency conversion, world time, Wikipedia, dictionary, translation and QR codes (`training/content_tools.py`)
   - Images: combinations of subjects, settings, times of day and 25 styles
   - Slides, PDF and Word on 26 topics, including turning your notes into slides
   - **Code files: 772 checked programs.**
     - 8 classic programs in 10 languages (Python, JavaScript, TypeScript, C, C++, Java, Go, Rust, Ruby, PHP). Each one is compiled and run, and must print exactly the expected output.
     - Unit converters; data classes in Python, TypeScript and Java; SQL databases; web pages in different themes; Express and Flask REST APIs; JavaScript utilities; Bash scripts.
     - 30 hand-written programs.
     - `training/check_code.py` checks all of them.
   - Code follow-ups: "now in Rust" (the same program in another language) and "run it" (writes the file, then runs it with the code runner).
   - Running code: about 20 kinds of problems, with outputs from actually running them; includes the user saying no and fixing a failed run
   - Charts
   - Follow-ups ("make it portrait", "add a slide…", "what about Delhi?", "directions to the first one")
   - Clarifying questions, simple math without tools, direct answers, and requests Mnx can't do
   - Messy phone typing (lowercase, "pls", "wether", "tmrw")

   Every conversation is checked with Mnx's own parsers (`training/validate.mjs`). The 600 test cases use cities, topics and tasks that never appear in training. No other AI model wrote the data.
2. **Train.** Open `training/mnx_train.ipynb` in Google Colab with a T4 GPU (free tier works) and run the cells. It does a LoRA fine-tune of Qwen2.5-3B-Instruct with Unsloth, learning only the reply to your latest message (not the system prompt, earlier turns or tool results). It saves progress to Google Drive so it can resume after a disconnect. The full ~115,000 conversations take roughly 35–45 h on a free T4 (several sessions), 13–16 h on an L4, or 5–8 h on an A100; `USE = 25000` takes about 9 h on a T4. Your learned examples (see *Learning from your chats*) are added automatically. Afterwards it exports `mnx-q4_k_m.gguf` and the faster-on-phones `mnx-q4_0.gguf`, then uploads them to your Hugging Face repo.
3. **Measure.** Run `npm run eval -- --model models/mnx-q4_k_m.gguf --target 98`. It runs 800 held-out cases (cities, topics and tasks never seen in training) through Mnx's real tool loop, including the permission step for code, and prints the success rate per category and overall. A case only passes if the right tool was called with valid arguments and the final answer uses the result. On a phone, add `--limit 100` to keep it short.

For code, a test only passes if the file the model wrote actually compiles, using whichever compilers are installed.

**Smooth tool use.** Mnx maps common near-misses from small models to the right tool or argument: `web_search` becomes `search_web`, `code` becomes `content`, `city` becomes `location`, `from`/`to` become `origin`/`destination`, and `ppt` becomes `pptx`. It also strips stray markdown fences around file contents.

Permission for running code is enforced by Mnx itself, so it never depends on the model. Image quality comes from the image service. Speed depends on your phone; the Q4_0 file and shorter thinking help.

## Testing

```bash
npm test               # unit, integration and training-pipeline tests (mock Anthropic API, fake llama-server)
npm run test:runtime   # real llama.cpp with a tiny random-weight model (pip install gguf numpy)
MNX_LLAMA_SERVER_BIN=$(which llama-server) npm run test:runtime   # also the real llama-server (e.g. on Termux)
```

The tests cover the streaming parsers (`<think>`, `<tool_call>`), repair of malformed tool calls, every tool, the code runner's safety rules, full tool loops for both the online and local models, approvals, aborts, and the server's security checks.

## How it works

```
browser (public/)  ── POST /api/chat ──►  server.js  ── stream ──►  Claude API
      ▲                                      │  runs tools (tools.js)
      └──────── Server-Sent Events ◄─────────┘  thinking / text / tool steps / cards
```

- `server.js` runs the agent loop. It streams Claude's events (thinking deltas, text deltas, tool-input deltas) to the browser, runs the tools, and loops until the answer is complete. Adaptive thinking is on with summarized display, so the reasoning can be shown live. Server-side refusal fallbacks are enabled.
- `local.js` runs the local GGUF model, either in process (node-llama-cpp, desktop) or through `llama-server` (Termux), and streams it with the same events.
- `training/` makes the training data, runs the Colab fine-tune and scores the model.
- `tools.js` defines the tools and calls the free public APIs.
- `public/app.js` renders the live timeline, cards and history. `public/docs.js` builds the PPTX, PDF and DOCX files.
