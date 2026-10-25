# Optional tools: the web, Chrome, Gmail and Drive

MNX1 is offline by default and needs no API key. These tool agents are
opt-in: nothing in this file loads unless you pass `--tools`. The offline
agents keep working exactly as before with or without them.

| Agent | Skill | Turn it on | What it can do |
|---|---|---|---|
| browser | `web` | `--tools web` | open a page, answer "what does this page say about X", follow links, search the web (DuckDuckGo, no key) |
| browser | `web` | `--tools web --chrome` | the same, through a real Chrome window (Playwright) |
| mail | `gmail` | `--tools gmail` | list, search and read your Gmail; save drafts and reply drafts (never sends) |
| files | `drive` | `--tools drive` | list, search and read your Google Drive (read-only): Docs, Sheets as CSV, Slides, text files |

`--tools google` means `gmail,drive` and `--tools all` turns everything on.

```bash
python -m mnx.agent --tools all          # interactive, with every tool
python -m mnx.agent --tools web "open https://en.wikipedia.org/wiki/Sea_otter"
python -m mnx.agent --tools web "what does https://docs.python.org/3/library/asyncio.html say about event loops"
python -m mnx.agent --tools web "search the web for python asyncio tutorial"
```

In the interactive chat:

```text
you> check my email
you> read email 2
you> draft a reply to email 2 saying thanks, see you Friday
you> emails from alice
you> search my email for invoice
you> draft an email to bob@example.com subject Lunch saying see you at 1
you> search my drive for budget
you> read file 1
you> what does file 1 say about the deadline
you> recent files in my drive
you> follow link 3
```

## How the brain uses them

MNX1 checks the tool agents first, right after feedback ("correct", "wrong,
the answer is ..."), because only they can answer "check my email" or
"open <url>". Anything that isn't an explicit tool request falls through to
the router and the offline specialists as usual, so `what is 15% of 240`
still goes to the mathematician.

Tool answers are live data, so they are marked unverified, never replayed
from experience and never used to self-train the router. The transcript in
`~/.mnx/transcript.jsonl` keeps only the first line of a tool answer, so the
text of your emails and files isn't copied to disk.

## Chrome

`--chrome` uses [Playwright](https://playwright.dev/python/) (`pip install
-e .[browser]`, or `pip install playwright`) to drive Chrome:

1. **Your own Chrome (recommended).** Start Chrome with remote debugging and
   MNX opens its pages in a new tab of that window, with your sign-ins:

   ```bash
   # macOS
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="$HOME/.mnx/chrome"
   # Windows
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=%USERPROFILE%\.mnx\chrome
   # Linux
   google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/.mnx/chrome"
   ```

   Recent Chrome versions only allow remote debugging with a separate
   `--user-data-dir`, so sign in to the sites you want once in that window.
   Use `--cdp http://host:port` if you picked another port.

2. **A fresh Chrome.** If nothing is listening on the port, MNX launches
   Chrome itself (or Playwright's Chromium after `playwright install
   chromium`) and closes it when MNX exits. MNX never closes a Chrome it
   attached to; it only closes its own tab.

The browser agent only reads. It doesn't type into pages, click buttons or
submit forms.

## Gmail and Drive: your own Google sign-in

MNX doesn't bundle a key or use a server of its own. You create a free OAuth
client in your own Google Cloud project, and the sign-in happens on your
machine (OAuth 2.0 for installed apps, with PKCE, standard library only).

1. Go to <https://console.cloud.google.com> and create a project.
2. **APIs & Services → Library**: enable **Gmail API** and **Google Drive API**.
3. **OAuth consent screen**: choose External, fill in the app name and your
   email, and add yourself under **Test users**.
4. **Credentials → Create credentials → OAuth client ID → Desktop app**,
   then **Download JSON**.
5. Save the file as `~/.mnx/google/credentials.json`.
6. Sign in:

   ```bash
   python -m mnx.agent.tools.google login           # Gmail + Drive
   python -m mnx.agent.tools.google login --gmail   # or just one
   python -m mnx.agent.tools.google status
   python -m mnx.agent.tools.google logout          # revoke and delete the token
   ```

   Your browser opens Google's consent page. Because the app is in testing
   mode, Google shows an "unverified app" warning; that's your own app, so
   choose Continue. The token is saved to `~/.mnx/google/token.json`,
   readable only by you, and refreshed automatically.

Scopes requested, the narrowest that work:

| Scope | Why |
|---|---|
| `gmail.readonly` | list, search and read messages |
| `gmail.compose` | save drafts. MNX never sends: "send an email ..." saves a draft too |
| `drive.readonly` | list, search and read files. MNX can't change, share or delete anything |

Set `MNX_GOOGLE_DIR` to keep the credentials somewhere else.

## Safety

- Pages, emails and files can contain text written to trick an assistant
  ("ignore previous instructions ..."). MNX has no language model to trick:
  it shows that text to you and never acts on it.
- Nothing is sent, posted, shared or deleted. The only write is a Gmail
  draft, which you review and send yourself.
- Errors (no internet, expired sign-in, a 403 from Google) are reported as
  "I couldn't ..." instead of a guess.

## On a phone

The web agent is standard library only, so it works in Termux. Gmail and
Drive work too: run `login --no-browser`, open the printed link in the
phone's browser, and the redirect comes back to Termux on `127.0.0.1`.
Chrome control (`--chrome`) needs a desktop.
