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

## How it works

```
browser (public/)  ── POST /api/chat ──►  server.js  ── stream ──►  Claude API
      ▲                                      │  runs tools (tools.js)
      └──────── Server-Sent Events ◄─────────┘  thinking / text / tool steps / cards
```

- `server.js` runs the agent loop. It streams Claude's events (thinking deltas, text deltas, tool-input deltas) to the browser, runs the tools, and loops until the answer is complete. Adaptive thinking is on with summarized display, so the reasoning can be shown live. Server-side refusal fallbacks are enabled.
- `tools.js` defines the tools and calls the free public APIs.
- `public/app.js` renders the live timeline, cards and history. `public/docs.js` builds the PPTX, PDF and DOCX files.
