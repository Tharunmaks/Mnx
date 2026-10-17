"""Bees use any website, and (for the Hive's owner) any app on the paired phone, like a person does.

  "add KitKat in blinkit"            the Bee opens Blinkit in its own browser, finds KitKat and adds it to the cart
  "order 2 milk from zepto"          … up to checkout, then asks you before paying
  "search for running shoes on amazon"
  "open swiggy on my phone and …"    the owner's phone, through the Phone Drone (drone/drone.py)

How it works: the Bee looks at the page (the buttons, links and boxes it can see, and the text), the model picks
one step (click, type, scroll, go to a page …), the Bee does it, and looks again, until the goal is done.

Things only you know come from you, in the chat:
  - a login, an OTP, a password, an API key, a UPI ID, an address: a card with boxes to fill. Secret boxes are
    password fields; what you type goes straight into the website and is never saved, never shown to the model
    (it only sees {{placeholders}}) and never kept in the chat. The Bee's browser stays signed in afterwards, so
    you rarely log in twice.
  - paying, placing an order, booking or sending: a card with the total and a Pay button. The Bee never presses a
    pay / place order / book / send button without your OK in this chat (checked here, not left to the model).
    UPI collect requests are approved in your UPI app; OTPs come back to you as a box to fill.
  - captchas or anything stuck: "Watch live" opens the Bee's browser so you can tap it yourself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid
from collections import OrderedDict
from urllib.parse import quote_plus

from . import brain
from .core import Task

# Apps the Bees know by name: their website, a search page, and the Android package for the phone.
APPS = {
    "blinkit": {"name": "Blinkit", "web": "https://blinkit.com", "search": "https://blinkit.com/s/?q={q}",
                "package": "com.grofers.customerapp", "words": ["blinkit", "blink it", "grofers"]},
    "zepto": {"name": "Zepto", "web": "https://www.zeptonow.com", "search": "https://www.zeptonow.com/search?query={q}",
              "package": "com.zeptoconsumerapp", "words": ["zepto"]},
    "instamart": {"name": "Swiggy Instamart", "web": "https://www.swiggy.com/instamart", "search": "https://www.swiggy.com/instamart/search?query={q}",
                  "package": "in.swiggy.android", "words": ["instamart", "swiggy instamart"]},
    "swiggy": {"name": "Swiggy", "web": "https://www.swiggy.com", "search": "https://www.swiggy.com/search?query={q}",
               "package": "in.swiggy.android", "words": ["swiggy"]},
    "zomato": {"name": "Zomato", "web": "https://www.zomato.com", "search": None, "package": "com.application.zomato", "words": ["zomato"]},
    "bigbasket": {"name": "BigBasket", "web": "https://www.bigbasket.com", "search": "https://www.bigbasket.com/ps/?q={q}",
                  "package": "com.bigbasket.mobileapp", "words": ["bigbasket", "big basket"]},
    "amazon": {"name": "Amazon", "web": "https://www.amazon.in", "search": "https://www.amazon.in/s?k={q}",
               "package": "in.amazon.mShop.android.shopping", "words": ["amazon"]},
    "flipkart": {"name": "Flipkart", "web": "https://www.flipkart.com", "search": "https://www.flipkart.com/search?q={q}",
                 "package": "com.flipkart.android", "words": ["flipkart"]},
    "myntra": {"name": "Myntra", "web": "https://www.myntra.com", "search": None, "package": "com.myntra.android", "words": ["myntra"]},
    "jiomart": {"name": "JioMart", "web": "https://www.jiomart.com", "search": "https://www.jiomart.com/search/{q}",
                "package": "com.jpl.jiomart", "words": ["jiomart", "jio mart"]},
    "nykaa": {"name": "Nykaa", "web": "https://www.nykaa.com", "search": "https://www.nykaa.com/search/result/?q={q}",
              "package": "com.fsn.nykaa", "words": ["nykaa"]},
    "meesho": {"name": "Meesho", "web": "https://www.meesho.com", "search": "https://www.meesho.com/search?q={q}",
               "package": "com.meesho.supply", "words": ["meesho", "mesho"]},
    "bookmyshow": {"name": "BookMyShow", "web": "https://in.bookmyshow.com", "search": None, "package": "com.bt.bms",
                   "words": ["bookmyshow", "book my show", "bms"]},
    "makemytrip": {"name": "MakeMyTrip", "web": "https://www.makemytrip.com", "search": None, "package": "com.makemytrip",
                   "words": ["makemytrip", "make my trip", "mmt"]},
    "irctc": {"name": "IRCTC", "web": "https://www.irctc.co.in/nget/train-search", "search": None, "package": "cris.org.in.prs.ima",
              "words": ["irctc"]},
    "uber": {"name": "Uber", "web": "https://m.uber.com", "search": None, "package": "com.ubercab", "words": ["uber"]},
    "ola": {"name": "Ola", "web": "https://book.olacabs.com", "search": None, "package": "com.olacabs.customer", "words": ["ola", "ola cabs"]},
    "rapido": {"name": "Rapido", "web": "https://www.rapido.bike", "search": None, "package": "com.rapido.passenger", "words": ["rapido"]},
    "youtube": {"name": "YouTube", "web": "https://www.youtube.com", "search": "https://www.youtube.com/results?search_query={q}",
                "package": "com.google.android.youtube", "words": ["youtube"]},
    "gmail": {"name": "Gmail", "web": "https://mail.google.com", "search": None, "package": "com.google.android.gm", "words": ["gmail"]},
    "whatsapp": {"name": "WhatsApp", "web": "https://web.whatsapp.com", "search": None, "package": "com.whatsapp", "words": ["whatsapp"]},
    "dunzo": {"name": "Dunzo", "web": "https://www.dunzo.com", "search": None, "package": "com.dunzo.user", "words": ["dunzo"]},
    "dominos": {"name": "Domino's", "web": "https://pizzaonline.dominos.co.in", "search": None, "package": "com.Dominos", "words": ["dominos", "domino's", "dominoes"]},
    "pizzahut": {"name": "Pizza Hut", "web": "https://www.pizzahut.co.in", "search": None, "package": "com.pizzahutindia", "words": ["pizza hut", "pizzahut"]},
    "kfc": {"name": "KFC", "web": "https://online.kfc.co.in", "search": None, "package": "com.yum.kfc", "words": ["kfc"]},
    "mcdonalds": {"name": "McDonald's", "web": "https://www.mcdelivery.co.in", "search": None, "package": "com.mcdonalds.mobileapp", "words": ["mcdonalds", "mcdonald's", "mcdelivery"]},
    "ajio": {"name": "AJIO", "web": "https://www.ajio.com", "search": "https://www.ajio.com/search/?text={q}", "package": "com.ril.ajio", "words": ["ajio"]},
    "tatacliq": {"name": "Tata CLiQ", "web": "https://www.tatacliq.com", "search": "https://www.tatacliq.com/search/?searchCategory=all&text={q}", "package": "com.tul.tatacliq", "words": ["tata cliq", "tatacliq"]},
    "croma": {"name": "Croma", "web": "https://www.croma.com", "search": "https://www.croma.com/searchB?q={q}", "package": "com.croma.app", "words": ["croma"]},
    "reliancedigital": {"name": "Reliance Digital", "web": "https://www.reliancedigital.in", "search": "https://www.reliancedigital.in/search?q={q}", "package": "com.ril.digital", "words": ["reliance digital"]},
    "snapdeal": {"name": "Snapdeal", "web": "https://www.snapdeal.com", "search": "https://www.snapdeal.com/search?keyword={q}", "package": "com.snapdeal.main", "words": ["snapdeal"]},
    "pharmeasy": {"name": "PharmEasy", "web": "https://pharmeasy.in", "search": "https://pharmeasy.in/search/all?name={q}", "package": "com.phonegap.rxpal", "words": ["pharmeasy"]},
    "1mg": {"name": "Tata 1mg", "web": "https://www.1mg.com", "search": "https://www.1mg.com/search/all?name={q}", "package": "com.aranoah.healthkart.plus", "words": ["1mg", "tata 1mg"]},
    "netmeds": {"name": "Netmeds", "web": "https://www.netmeds.com", "search": None, "package": "com.NetmedsMarketplace.Netmeds", "words": ["netmeds"]},
    "lenskart": {"name": "Lenskart", "web": "https://www.lenskart.com", "search": "https://www.lenskart.com/search?q={q}", "package": "com.lenskart.app", "words": ["lenskart"]},
    "urbancompany": {"name": "Urban Company", "web": "https://www.urbancompany.com", "search": None, "package": "com.urbanclap.urbanclap", "words": ["urban company", "urbanclap"]},
    "goibibo": {"name": "Goibibo", "web": "https://www.goibibo.com", "search": None, "package": "com.goibibo", "words": ["goibibo"]},
    "cleartrip": {"name": "Cleartrip", "web": "https://www.cleartrip.com", "search": None, "package": "com.cleartrip.android", "words": ["cleartrip"]},
    "ixigo": {"name": "ixigo", "web": "https://www.ixigo.com", "search": None, "package": "com.ixigo", "words": ["ixigo"]},
    "redbus": {"name": "redBus", "web": "https://www.redbus.in", "search": None, "package": "in.redbus.android", "words": ["redbus", "red bus"]},
    "oyo": {"name": "OYO", "web": "https://www.oyorooms.com", "search": None, "package": "com.oyo.consumer", "words": ["oyo"]},
    "airbnb": {"name": "Airbnb", "web": "https://www.airbnb.co.in", "search": None, "package": "com.airbnb.android", "words": ["airbnb"]},
    "booking": {"name": "Booking.com", "web": "https://www.booking.com", "search": None, "package": "com.booking", "words": ["booking.com"]},
    "agoda": {"name": "Agoda", "web": "https://www.agoda.com", "search": None, "package": "com.agoda.mobile.consumer", "words": ["agoda"]},
    "indigo": {"name": "IndiGo", "web": "https://www.goindigo.in", "search": None, "package": "in.goindigo.android", "words": ["indigo"]},
    "airindia": {"name": "Air India", "web": "https://www.airindia.com", "search": None, "package": "com.airindia.mobile", "words": ["air india"]},
    "district": {"name": "District", "web": "https://www.district.in", "search": None, "package": "com.application.zomato.district", "words": ["district app"]},
    "paytm": {"name": "Paytm", "web": "https://paytm.com", "search": None, "package": "net.one97.paytm", "words": ["paytm"]},
    "phonepe": {"name": "PhonePe", "web": "https://www.phonepe.com", "search": None, "package": "com.phonepe.app", "words": ["phonepe", "phone pe"]},
    "gpay": {"name": "Google Pay", "web": "https://pay.google.com", "search": None, "package": "com.google.android.apps.nbu.paisa.user", "words": ["gpay", "google pay"]},
    "cred": {"name": "CRED", "web": "https://cred.club", "search": None, "package": "com.dreamplug.androidapp", "words": ["cred"]},
    "zerodha": {"name": "Zerodha Kite", "web": "https://kite.zerodha.com", "search": None, "package": "com.zerodha.kite3", "words": ["zerodha", "kite app"]},
    "groww": {"name": "Groww", "web": "https://groww.in", "search": None, "package": "com.nextbillion.groww", "words": ["groww"]},
    "dineout": {"name": "Swiggy Dineout", "web": "https://www.swiggy.com/dineout", "search": None, "package": "in.swiggy.android", "words": ["dineout", "swiggy dineout"]},
    "spotify": {"name": "Spotify", "web": "https://open.spotify.com", "search": "https://open.spotify.com/search/{q}", "package": "com.spotify.music", "words": ["spotify"]},
    "jiosaavn": {"name": "JioSaavn", "web": "https://www.jiosaavn.com", "search": "https://www.jiosaavn.com/search/{q}", "package": "com.jio.media.jiobeats", "words": ["jiosaavn", "saavn"]},
    "gaana": {"name": "Gaana", "web": "https://gaana.com", "search": "https://gaana.com/search/{q}", "package": "com.gaana", "words": ["gaana"]},
    "netflix": {"name": "Netflix", "web": "https://www.netflix.com", "search": "https://www.netflix.com/search?q={q}", "package": "com.netflix.mediaclient", "words": ["netflix"]},
    "primevideo": {"name": "Prime Video", "web": "https://www.primevideo.com", "search": None, "package": "com.amazon.avod.thirdpartyclient", "words": ["prime video", "amazon prime"]},
    "hotstar": {"name": "JioHotstar", "web": "https://www.hotstar.com", "search": None, "package": "in.startv.hotstar", "words": ["hotstar", "jiohotstar", "jio hotstar"]},
    "instagram": {"name": "Instagram", "web": "https://www.instagram.com", "search": None, "package": "com.instagram.android", "words": ["instagram"]},
    "facebook": {"name": "Facebook", "web": "https://www.facebook.com", "search": None, "package": "com.facebook.katana", "words": ["facebook"]},
    "x": {"name": "X (Twitter)", "web": "https://x.com", "search": "https://x.com/search?q={q}", "package": "com.twitter.android", "words": ["twitter"]},
    "linkedin": {"name": "LinkedIn", "web": "https://www.linkedin.com", "search": "https://www.linkedin.com/search/results/all/?keywords={q}", "package": "com.linkedin.android", "words": ["linkedin"]},
    "reddit": {"name": "Reddit", "web": "https://www.reddit.com", "search": "https://www.reddit.com/search/?q={q}", "package": "com.reddit.frontpage", "words": ["reddit"]},
    "telegram": {"name": "Telegram", "web": "https://web.telegram.org", "search": None, "package": "org.telegram.messenger", "words": ["telegram"]},
    "snapchat": {"name": "Snapchat", "web": "https://web.snapchat.com", "search": None, "package": "com.snapchat.android", "words": ["snapchat"]},
    "pinterest": {"name": "Pinterest", "web": "https://in.pinterest.com", "search": "https://in.pinterest.com/search/pins/?q={q}", "package": "com.pinterest", "words": ["pinterest"]},
    "outlook": {"name": "Outlook", "web": "https://outlook.live.com", "search": None, "package": "com.microsoft.office.outlook", "words": ["outlook mail", "outlook app", "hotmail"]},
    "googledrive": {"name": "Google Drive", "web": "https://drive.google.com", "search": None, "package": "com.google.android.apps.docs", "words": ["google drive", "gdrive"]},
    "googledocs": {"name": "Google Docs", "web": "https://docs.google.com", "search": None, "package": "com.google.android.apps.docs.editors.docs", "words": ["google docs"]},
    "googlemaps": {"name": "Google Maps", "web": "https://www.google.com/maps", "search": "https://www.google.com/maps/search/{q}", "package": "com.google.android.apps.maps", "words": ["google maps"]},
    "googlecalendar": {"name": "Google Calendar", "web": "https://calendar.google.com", "search": None, "package": "com.google.android.calendar", "words": ["google calendar"]},
    "chatgpt": {"name": "ChatGPT", "web": "https://chatgpt.com", "search": None, "package": "com.openai.chatgpt", "words": ["chatgpt"]},
    "github": {"name": "GitHub", "web": "https://github.com", "search": "https://github.com/search?q={q}", "package": "com.github.android", "words": ["github"]},
    "notion": {"name": "Notion", "web": "https://www.notion.so", "search": None, "package": "notion.id", "words": ["notion"]},
    "canva": {"name": "Canva", "web": "https://www.canva.com", "search": None, "package": "com.canva.editor", "words": ["canva"]},
    "wikipedia": {"name": "Wikipedia", "web": "https://en.wikipedia.org", "search": "https://en.wikipedia.org/w/index.php?search={q}", "package": "org.wikipedia", "words": ["wikipedia"]},
    "google": {"name": "Google", "web": "https://www.google.com", "search": "https://www.google.com/search?q={q}", "package": "com.google.android.googlequicksearchbox", "words": ["google"]},
    "ebay": {"name": "eBay", "web": "https://www.ebay.com", "search": "https://www.ebay.com/sch/i.html?_nkw={q}", "package": "com.ebay.mobile", "words": ["ebay"]},
    "aliexpress": {"name": "AliExpress", "web": "https://www.aliexpress.com", "search": "https://www.aliexpress.com/wholesale?SearchText={q}", "package": "com.alibaba.aliexpresshd", "words": ["aliexpress"]},
    "walmart": {"name": "Walmart", "web": "https://www.walmart.com", "search": "https://www.walmart.com/search?q={q}", "package": "com.walmart.android", "words": ["walmart"]},
    "doordash": {"name": "DoorDash", "web": "https://www.doordash.com", "search": None, "package": "com.dd.doordash", "words": ["doordash"]},
    "ubereats": {"name": "Uber Eats", "web": "https://www.ubereats.com", "search": None, "package": "com.ubercab.eats", "words": ["uber eats", "ubereats"]},
    "instacart": {"name": "Instacart", "web": "https://www.instacart.com", "search": None, "package": "com.instacart.client", "words": ["instacart"]},
    "dmart": {"name": "DMart Ready", "web": "https://www.dmart.in", "search": None, "package": "in.dmart", "words": ["dmart", "d mart"]},
    "licious": {"name": "Licious", "web": "https://www.licious.in", "search": None, "package": "com.licious", "words": ["licious"]},
    "countrydelight": {"name": "Country Delight", "web": "https://www.countrydelight.in", "search": None, "package": "app.mobile.countrydelight", "words": ["country delight"]},
    "practo": {"name": "Practo", "web": "https://www.practo.com", "search": None, "package": "com.practo.fabric", "words": ["practo"]},
    "naukri": {"name": "Naukri", "web": "https://www.naukri.com", "search": None, "package": "naukriApp.appModules.login", "words": ["naukri"]},
    "magicpin": {"name": "magicpin", "web": "https://magicpin.in", "search": None, "package": "com.magicpin.local", "words": ["magicpin"]},
    "olx": {"name": "OLX", "web": "https://www.olx.in", "search": None, "package": "com.olx.southasia", "words": ["olx"]},
    "99acres": {"name": "99acres", "web": "https://www.99acres.com", "search": None, "package": "com.nnacres.app", "words": ["99acres"]},
    "magicbricks": {"name": "MagicBricks", "web": "https://www.magicbricks.com", "search": None, "package": "com.timesgroup.magicbricks", "words": ["magicbricks"]},
    "myjio": {"name": "MyJio", "web": "https://www.jio.com", "search": None, "package": "com.jio.myjio", "words": ["myjio", "jio recharge"]},
    "airtel": {"name": "Airtel Thanks", "web": "https://www.airtel.in", "search": None, "package": "com.myairtelapp", "words": ["airtel"]},
    "digilocker": {"name": "DigiLocker", "web": "https://www.digilocker.gov.in", "search": None, "package": "com.digilocker.android", "words": ["digilocker"]},
}
VERB = (r"(?:add|put|order|buy|purchase|book|reserve|search(?:\s+for)?|find|look\s+up|get\s+me|get|check|track|cancel|reorder|"
        r"pay(?:\s+for)?|log\s*in|sign\s*in|open|play|send|message|post|subscribe|apply|remove|empty|show\s+me|compare)")
VERB_RE = re.compile(rf"\b{VERB}\b", re.I)
DOMAIN_RE = re.compile(r"\b((?:https?://)?(?:[a-z0-9-]+\.)+(?:com|in|co\.in|org|net|io|app|shop|store)(?:/[^\s]*)?)", re.I)
PHONE_RE = re.compile(r"\b(?:on|using|with|from)\s+my\s+phone\b|\bphone'?s?\b.*\bapp\b|\bin\s+the\s+[\w ]{2,20}\s+app\b", re.I)
# "order a cake on Bakingo app", "book a class on the cult.fit website": any app or site, by name
NAMED_APP = re.compile(r"\b(?:on|in|from|using|via|at|through)\s+(?:the\s+|my\s+)?(?P<name>[A-Za-z0-9][\w.&'+-]*(?:\s+[A-Za-z0-9][\w.&'+-]*){0,2})"
                       r"\s+(?:app|application|website|web\s*site|site)\b", re.I)
NOT_APPS = {"settings", "this", "that", "my", "your", "the", "same", "our", "chat", "hive", "mnx", "an", "a", "new", "other", "web",
            "mobile", "android", "ios", "phone", "test", "demo", "todo", "react", "flutter", "next", "next.js", "node", "django", "flask"}
SEARCH = "https://html.duckduckgo.com/html/?q={q}"  # an app or site we don't know yet: the Bee finds it first
ITEM_RE = re.compile(rf"\b{VERB}\s+(?:me\s+)?(?P<item>.+?)\s+(?:in|on|from|to|at|using|via)\s+(?:the\s+)?(?:my\s+)?(?P<where>[\w .-]+?)"
                     r"(?:\s+(?:app|cart|website|site|basket))?\s*[.!?]?$", re.I)

# Buttons the Bee never presses without your OK in this chat: they spend money or reach other people.
RISKY = re.compile(r"\b(?:pay|place\s+order|buy\s+now|confirm\s+(?:order|booking|payment)|proceed\s+to\s+pay|book\s+now|"
                   r"complete\s+(?:order|purchase|payment)|make\s+payment|send|submit\s+order|checkout\s+&\s+pay|order\s+now|"
                   r"slide\s+to\s+pay|swipe\s+to\s+pay)\b", re.I)
MONEY = re.compile(r"₹\s?([\d,]+(?:\.\d{1,2})?)")
STEPS = {"low": 15, "med": 25, "high": 40, "ultra": 60, "maxxxx": 60}
SHOTS: "OrderedDict[str, tuple[str, bytes, str]]" = OrderedDict()  # id → (user, image, mime); the newest 300
MAX_SHOTS = 300


# ---------------------------------------------------------------- what the user asked for
def _app_in(text: str) -> str | None:
    low = text.lower()
    for key, a in APPS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in a["words"]):
            return key
    return None


def parse(text: str) -> dict | None:
    """ "add KitKat in blinkit" → {"app": "blinkit", "item": "KitKat", "url": None, "phone": False, "goal": …}"""
    t = " ".join((text or "").split())
    if not t or not VERB_RE.search(t):
        return None
    from .cloud_chat import CODE_ASK
    if CODE_ASK.search(t):  # "write a script to scrape amazon", "how do I order on swiggy" are questions for the model
        return None
    app = _app_in(t)
    dm = DOMAIN_RE.search(t)
    named = NAMED_APP.search(t) if not app and not dm else None
    if named and named.group("name").lower() in NOT_APPS:
        named = None
    phone_app = None if app or dm else _phone_app_in(t)
    if not app and not dm and not named and not phone_app:
        return None
    if not app and re.search(r"@", t):  # an email address isn't a website
        return None
    item = None
    m = ITEM_RE.search(t)
    if m and (app is None or _app_in(m.group("where")) == app or (dm and dm.group(1) in m.group("where"))):
        item = re.sub(r"^(?:a|an|the|some)\s+", "", m.group("item").strip(" .,'\""), flags=re.I) or None
    url = None
    if dm and not app:
        url = dm.group(1) if dm.group(1).lower().startswith("http") else "https://" + dm.group(1)
    name = named.group("name").strip() if named else (phone_app["label"] if phone_app else None)
    if name and not url:
        url = SEARCH.format(q=quote_plus(f"{name} official website"))
    return {"app": app, "item": item, "url": url, "phone": bool(PHONE_RE.search(t)) or bool(phone_app and not named), "goal": t,
            "name": name, "package": phone_app["package"] if phone_app else None}


def _phone_app_in(text: str) -> dict | None:
    """An app installed on the owner's paired phone, named in the request ("open Cult.fit and book a class")."""
    from . import phone, users
    if not users.is_owner():
        return None
    low = text.lower()
    for a in phone.all_apps():
        label = (a.get("label") or "").strip().lower()
        if len(label) >= 4 and re.search(rf"\b{re.escape(label)}\b", low):
            return a
    return None


def start_url(intent: dict) -> str:
    a = APPS.get(intent.get("app") or "")
    if a:
        if intent.get("item") and a.get("search"):
            return a["search"].format(q=quote_plus(intent["item"]))
        return a["web"]
    return intent.get("url") or "https://www.google.com"


# ---------------------------------------------------------------- where the Bee works: a browser, or the phone
class WebSurface:
    """The Bee's own Chromium (it stays signed in between tasks)."""
    kind = "web"

    def __init__(self, session):
        self.s = session
        self.els: list[dict] = []

    async def open(self, url: str) -> None:
        await self.s.do("goto", url=url)

    async def observe(self) -> dict:
        st = await self.s.do("elements")
        self.els = st.get("elements") or []
        try:
            text = (await self.s.do("read")).get("text") or ""
        except Exception:
            text = ""
        return {"place": st.get("url", ""), "title": st.get("title", ""), "text": text[:2500],
                "elements": [{"label": e.get("label") or "", "kind": e.get("tag", "") + (f":{e['type']}" if e.get("type") else "")}
                             for e in self.els]}

    def _el(self, i) -> dict:
        try:
            return self.els[int(i)]
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"There's no element {i}") from exc

    async def click(self, i) -> None:
        e = self._el(i)
        await self.s.do("click", x=e["x"], y=e["y"])

    async def type(self, i, text: str, enter: bool = False) -> None:
        if i is not None:
            await self.click(i)
            await self.s.do("press", key="Control+A")
            await self.s.do("press", key="Backspace")
        await self.s.do("type", text=text)
        if enter:
            await self.s.do("press", key="Enter")

    async def press(self, key: str) -> None:
        await self.s.do("press", key={"enter": "Enter", "back": "Escape", "tab": "Tab"}.get(key.lower(), key))

    async def scroll(self, down: bool = True) -> None:
        await self.s.do("scroll", dy=600 if down else -600)

    async def back(self) -> None:
        await self.s.do("back")

    async def goto(self, url: str) -> None:
        await self.s.do("goto", url=url)

    async def shot(self) -> tuple[bytes, str]:
        st = await self.s.do("screenshot")
        data = st.get("image", "")
        return base64.b64decode(data.split(",", 1)[1]) if "," in data else b"", "image/jpeg"


class PhoneSurface:
    """The owner's paired phone (ADB through drone.py): taps, typing and keys on whatever app is open."""
    kind = "phone"

    def __init__(self, phone):
        self.p = phone
        self.els: list[dict] = []

    async def open(self, package: str) -> None:
        await self.p.call("open_app", {"package": package}, timeout=20)
        await asyncio.sleep(2.5)

    async def observe(self) -> dict:
        nodes = await self.p.call("ui_tree", {}, timeout=60)
        self.els = [n for n in nodes if n.get("clickable") or n.get("editable") or n.get("text") or n.get("desc")][:150]
        text = " · ".join(n.get("text") or n.get("desc") for n in self.els if (n.get("text") or n.get("desc")))[:2500]
        return {"place": "phone", "title": "", "text": text,
                "elements": [{"label": (n.get("text") or n.get("desc") or n.get("id") or "")[:80],
                              "kind": ("box" if n.get("editable") else "button" if n.get("clickable") else "text")} for n in self.els]}

    def _el(self, i) -> dict:
        try:
            return self.els[int(i)]
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"There's no element {i}") from exc

    async def click(self, i) -> None:
        e = self._el(i)
        await self.p.call("tap", {"x": e["x"], "y": e["y"]})
        await asyncio.sleep(1.2)

    async def type(self, i, text: str, enter: bool = False) -> None:
        if i is not None:
            await self.click(i)
        await self.p.call("text", {"text": text})
        if enter:
            await self.p.call("key", {"key": "enter"})
        await asyncio.sleep(0.8)

    async def press(self, key: str) -> None:
        await self.p.call("key", {"key": key.lower()})
        await asyncio.sleep(0.8)

    async def scroll(self, down: bool = True) -> None:
        y1, y2 = (1600, 600) if down else (600, 1600)
        await self.p.call("swipe", {"x1": 540, "y1": y1, "x2": 540, "y2": y2, "ms": 350})
        await asyncio.sleep(1)

    async def back(self) -> None:
        await self.press("back")

    async def goto(self, url: str) -> None:
        await self.p.call("open_url", {"url": url}, timeout=20)
        await asyncio.sleep(2.5)

    async def shot(self) -> tuple[bytes, str]:
        st = await self.p.call("screenshot", {"width": 540}, timeout=30)
        data = st.get("image", "")
        mime = data[5:data.index(";")] if data.startswith("data:") else "image/png"
        return base64.b64decode(data.split(",", 1)[1]) if "," in data else b"", mime


# ---------------------------------------------------------------- the loop
SYSTEM = """You operate {where} for the user, step by step, like a careful person. Goal: {goal}

Each turn you see the current screen: the place, its visible text, and numbered elements. Reply with ONE JSON
object and nothing else:
  {{"do": "click", "id": 3, "why": "…"}}
  {{"do": "type", "id": 5, "text": "KitKat", "enter": true}}        (type into box 5; use {{placeholders}} for the user's values)
  {{"do": "press", "key": "Enter"}}  {{"do": "scroll", "down": true}}  {{"do": "back"}}  {{"do": "goto", "url": "https://…"}}
  {{"do": "wait", "seconds": 10, "why": "…"}}                       (e.g. while the user approves a UPI request)
  {{"do": "ask", "question": "…", "fields": [{{"name": "phone", "label": "Mobile number", "secret": false}}]}}
      when you need something only the user knows: login, mobile number, OTP, password, API key, address, UPI ID.
      Mark passwords, OTPs, PINs, card numbers, CVVs and keys "secret": true. You will then see their values only
      as placeholders like {{{{phone}}}}; type the placeholder and the real value is typed for you.
  {{"do": "confirm", "summary": "2 × KitKat, delivery to Home", "amount": "₹45", "methods": ["UPI", "Cash on delivery"]}}
      BEFORE anything that spends money, places an order, books, or sends something to someone. Wait for the answer.
  {{"do": "done", "summary": "…"}}   {{"do": "fail", "reason": "…"}}
When you finish, the "summary" tells the user what you did and what you found: the facts, prices, headlines or
results on the page, what's in the cart, and anything they still need to do.
Rules: never guess a password or OTP; never pay or place an order without a confirm the user approved; if a
captcha appears, ask the user to solve it in Watch live (do "ask" with no fields). Stop with done as soon as the
goal is met (adding to the cart is done once it's in the cart). Values the user gave so far: {values}."""


def _shot_id(user: str, img: bytes, mime: str) -> str | None:
    if not img:
        return None
    sid = uuid.uuid4().hex[:16]
    SHOTS[sid] = (user, img, mime)
    while len(SHOTS) > MAX_SHOTS:
        SHOTS.popitem(last=False)
    return sid


def shot_for(sid: str, user: str) -> tuple[bytes, str] | None:
    hit = SHOTS.get(sid)
    if not hit or (hit[0] != user and user != "owner"):
        return None
    return hit[1], hit[2]


def _json(raw: str) -> dict | None:
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, TypeError):
        return None


class Vault:
    """What the user typed into the cards, for this task only. The model sees {{name}}; the page gets the value."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.secret: set[str] = set()

    def put(self, name: str, value: str, secret: bool) -> None:
        self.values[name] = value
        if secret:
            self.secret.add(name)

    def fill(self, text: str) -> str:
        return re.sub(r"\{\{\s*([\w-]+)\s*\}\}", lambda m: self.values.get(m.group(1), m.group(0)), text or "")

    def hide(self, text: str) -> str:
        """Every value the user gave, back to its placeholder: nothing they typed reaches the model or the chat."""
        for name, v in sorted(self.values.items(), key=lambda kv: -len(kv[1])):
            if len(v) >= 3:
                text = text.replace(v, "{{" + name + "}}")
        return text

    def describe(self) -> str:
        return ", ".join("{{" + n + "}}" + (" (secret)" if n in self.secret else "") for n in self.values) or "none yet"

    def clear(self) -> None:
        self.values.clear()


def _screen_text(obs: dict, vault: Vault) -> str:
    rows = [f"[{i}] {e['kind']} {json.dumps(e['label'][:80], ensure_ascii=False)}" for i, e in enumerate(obs["elements"])]
    return vault.hide(f"Place: {obs['place']}  {obs['title']}\nText: {obs['text']}\nElements:\n" + "\n".join(rows))


async def run(task: Task, surface, intent: dict, bee_name: str, live_url: str | None) -> None:
    """Work on the goal until done, failed, stopped or out of steps."""
    from .cloud_chat import ask
    vault = Vault()
    approved_amount: float | None = None  # the total you said yes to; a bigger one asks again
    history: list[str] = []
    where = APPS[intent["app"]]["name"] if intent.get("app") else (intent.get("name") or intent.get("url") or "the web")
    place = "your phone" if surface.kind == "phone" else f"{bee_name}'s browser"
    limit = STEPS.get(task.effort, 25)

    story: list[str] = []  # what the Bee did, for the "What happened" report at the end
    last: dict = {}

    async def screen(title: str, parent: str | None = None) -> None:
        try:
            img, mime = await surface.shot()
        except Exception:
            return
        sid = _shot_id(task.user, img, mime)
        if sid:
            await task.emit("screen", bee=bee_name, title=title, shot=sid, live=live_url, parent=parent)

    try:
        for n in range(limit):
            try:
                obs = await surface.observe()
            except Exception as exc:
                await task.answer(f"I couldn't see {where}: {str(exc)[:200]}")
                return
            if not last or obs["place"] != last.get("place"):
                story.append(f"Opened “{vault.hide(obs['title'] or obs['place'])[:70]}”" + (f" ({_short(obs['place'])})" if obs["place"] else ""))
            last.update(obs)
            msgs = [{"role": "system", "content": SYSTEM.format(where=f"{where} in {place}", goal=intent["goal"], values=vault.describe())},
                    {"role": "user", "content": ("Done so far:\n" + "\n".join(history[-14:]) + "\n\n" if history else "")
                     + "Screen now:\n" + _screen_text(obs, vault)}]
            try:
                raw = await brain.brain.chat(msgs, task.effort, max_tokens=300, temperature=0.1)
            except brain.BrainNotConnected as exc:
                await task.answer(f"The model didn't answer ({exc}), so I stopped.")
                return
            act = _json(raw)
            if not act or not act.get("do"):
                history.append("(your last reply wasn't one JSON object)")
                continue
            kind = str(act["do"]).lower()
            why = vault.hide(str(act.get("why") or ""))[:120]

            if kind == "done":
                await screen(f"Done on {where}")
                await task.answer(vault.hide(str(act.get("summary") or "Done.")))
                return
            if kind == "fail":
                await screen(f"Stuck on {where}")
                await task.answer(f"I couldn't finish on {where}: {vault.hide(str(act.get('reason') or 'no reason given'))}"
                                  + (" Open Watch live to do it yourself." if live_url else ""))
                return

            if kind == "ask":
                fields = [f for f in (act.get("fields") or []) if isinstance(f, dict) and re.fullmatch(r"[\w-]{1,30}", str(f.get("name", "")))][:6]
                await screen(f"{where} needs you")
                reply = await ask(task, "clarify", title=f"{where} needs you", service=where, ask_anyway=True,
                                  text=vault.hide(str(act.get("question") or "Fill these in to continue."))
                                  + ("\n\nSecret boxes go straight into the website; they aren't saved or shown to the model."
                                     if any(f.get("secret") for f in fields) else ""),
                                  fields=[{"name": f["name"], "label": str(f.get("label") or f["name"])[:60], "required": True,
                                           "input": "password" if f.get("secret") else "text", "secret": bool(f.get("secret"))}
                                          for f in fields],
                                  actions=[{"id": "go", "label": "Continue"}, {"id": "stop", "label": "Stop", "style": "danger"}])
                if reply["action"] != "go":
                    await task.answer(f"Okay, I stopped. {where} is still open in {place}.")
                    return
                for f in fields:
                    vault.put(f["name"], str((reply.get("values") or {}).get(f["name"]) or ""), bool(f.get("secret")))
                history.append(f"{n}. asked the user; they gave: {vault.describe()}")
                story.append("Asked you for " + ", ".join(str(f.get("label") or f["name"]) for f in fields))
                continue

            if kind == "confirm":
                amount = _money(str(act.get("amount") or "")) or _money(obs["text"])
                methods = [str(m)[:40] for m in (act.get("methods") or []) if str(m).strip()][:6]
                await screen(f"Ready to pay on {where}")
                fields = [{"name": "method", "label": "Pay with", "input": "select", "value": methods[0],
                           "options": methods}] if methods else []
                reply = await ask(task, "confirm" if not fields else "clarify", title=f"Pay on {where}?", service=where, ask_anyway=True,
                                  text=vault.hide(str(act.get("summary") or "")) + (f"\nTotal: ₹{amount:,.2f}" if amount else ""),
                                  details=[["Total", f"₹{amount:,.2f}" if amount else str(act.get("amount") or "see the screen")],
                                           ["Where", where]],
                                  fields=fields,
                                  actions=[{"id": "pay", "label": f"Pay ₹{amount:,.0f}" if amount else "Pay"},
                                           {"id": "stop", "label": "Don't pay", "style": "danger"}])
                if reply["action"] != "pay":
                    await task.answer(f"Okay, nothing was paid. Everything is still in the cart on {where}.")
                    return
                approved_amount = amount if amount else float("inf")
                chosen = (reply.get("values") or {}).get("method")
                history.append(f"{n}. the user approved paying" + (f" ₹{amount:,.2f}" if amount else "") + (f" with {chosen}" if chosen else ""))
                story.append("You approved paying" + (f" ₹{amount:,.2f}" if amount else "") + (f" with {chosen}" if chosen else ""))
                continue

            # actions on the screen
            label = ""
            if act.get("id") is not None:
                try:
                    label = obs["elements"][int(act["id"])]["label"]
                except (ValueError, TypeError, IndexError):
                    history.append(f"{n}. there's no element {act.get('id')}")
                    continue
            if kind == "click" and RISKY.search(label):
                total = _money(obs["text"])
                if approved_amount is None or (total and total > approved_amount + 0.5):
                    reply = await ask(task, "confirm", title=f"Press “{vault.hide(label)[:40]}” on {where}?", service=where, ask_anyway=True,
                                      text="This spends money, places an order or sends something. Nothing happens without your OK.",
                                      details=[["Total on the page", f"₹{total:,.2f}" if total else "not shown"]],
                                      actions=[{"id": "pay", "label": "Yes, go ahead"}, {"id": "stop", "label": "No", "style": "danger"}])
                    if reply["action"] != "pay":
                        await task.answer(f"Okay, I didn't press it. {where} is still open in {place}.")
                        return
                    approved_amount = total or float("inf")
            s = await task.step("running", _say(kind, act, vault.hide(label)), bee=bee_name, detail=why or None)
            try:
                if kind == "click":
                    await surface.click(act["id"])
                elif kind == "type":
                    text = str(act.get("text") or "")
                    await surface.type(act.get("id"), vault.fill(text), bool(act.get("enter")))
                elif kind == "press":
                    await surface.press(str(act.get("key") or "Enter"))
                elif kind == "scroll":
                    await surface.scroll(act.get("down", True) is not False)
                elif kind == "back":
                    await surface.back()
                elif kind == "goto":
                    url = str(act.get("url") or "")
                    if "{{" in url or not url.startswith(("http://", "https://")):
                        raise ValueError("only plain http(s) addresses")  # a secret never goes into an address
                    await surface.goto(url)
                elif kind == "wait":
                    await asyncio.sleep(max(1, min(float(act.get("seconds") or 5), 30)))
                else:
                    raise ValueError(f"unknown step {kind}")
            except Exception as exc:
                await task.finish_step(s, status="error", text=vault.hide(str(exc))[:160])
                history.append(f"{n}. {kind} failed: {vault.hide(str(exc))[:120]}")
                story.append(f"{_did(_say(kind, act, vault.hide(label)), tried=True)}: didn't work ({_why(exc)})")
                continue
            await task.finish_step(s)
            history.append(f"{n}. {_say(kind, act, vault.hide(label))}")
            if kind not in ("scroll", "wait"):
                story.append(_did(_say(kind, act, vault.hide(label))))
        await screen(f"Where I stopped on {where}")
        await task.answer(f"I used my {limit} steps and stopped. {where} is still open in {place}; ask again to carry on"
                          + (", or open Watch live." if live_url else "."))
    finally:
        if story:
            await report(task, bee_name, where, story, last, vault)
        vault.clear()


async def report(task: Task, bee: str, where: str, story: list[str], last: dict, vault: "Vault | None" = None) -> None:
    """The "What happened" card: each thing the Bee did, where it ended up, and what that page says."""
    hide = vault.hide if vault else (lambda t: t)
    details = [[f"{i}.", line] for i, line in enumerate(story[-20:], 1)]
    if last.get("place"):
        details.append(["Now on", f"{hide(last.get('title') or '')[:60]} · {_short(last['place'])}".strip(" ·")])
    said = " ".join(hide(last.get("text") or "").split())[:280]
    if said:
        details.append(["It says", said + ("…" if len(last.get("text") or "") > 280 else "")])
    await task.emit("result", bee=bee, service=where, title=f"What happened on {where}", badge="Report", details=details)


def _did(step: str, tried: bool = False) -> str:
    """ "Tapping “ADD”" → "Tapped “ADD”" (or "Tried to tap “ADD”"), for the report of what already happened."""
    past = {"Tapping": ("Tapped", "Tried to tap"), "Typing": ("Typed", "Tried to type"), "Pressing": ("Pressed", "Tried to press"),
            "Going back": ("Went back", "Tried to go back"), "Opening": ("Opened", "Tried to open"), "Scrolling": ("Scrolled", "Tried to scroll")}
    for now, (done, attempt) in past.items():
        if step.startswith(now):
            return (attempt if tried else done) + step[len(now):]
    return step


def _short(url: str) -> str:
    u = re.sub(r"^https?://(www\.)?", "", url or "")
    return u if len(u) <= 60 else u[:57] + "…"


def _why(exc: Exception) -> str:
    from .reader import explain_browser
    return explain_browser(str(exc))


def _say(kind: str, act: dict, label: str) -> str:
    lab = f"“{label[:50]}”" if label else ""
    text = str(act.get("text") or "")
    shown = text if text.startswith("{{") or len(text) < 40 else text[:37] + "…"
    return {"click": f"Tapping {lab}", "type": f"Typing {shown} {('into ' + lab) if lab else ''}".strip(),
            "press": f"Pressing {act.get('key') or 'Enter'}", "scroll": "Scrolling", "back": "Going back",
            "goto": f"Opening {str(act.get('url') or '')[:80]}", "wait": f"Waiting {act.get('seconds') or 5}s"}.get(kind, kind)


def _money(text: str) -> float | None:
    """The total on a page: the amount after "total" / "to pay" if there is one, else the largest ₹ amount."""
    m = re.search(r"(?:grand\s+total|to\s+pay|total\s+payable|order\s+total|bill\s+total|total)\D{0,20}₹\s?([\d,]+(?:\.\d{1,2})?)", text or "", re.I)
    if m:
        return float(m.group(1).replace(",", ""))
    vals = [float(v.replace(",", "")) for v in MONEY.findall(text or "") if v.replace(",", "").replace(".", "").isdigit()]
    return max(vals) if vals else None


# ---------------------------------------------------------------- in the chat
def _pick_bee(task: Task) -> dict | None:
    from .bees import registry
    if task.bee:
        return registry.mine(task.bee, task.user)
    mine = registry.of(task.user)
    words = re.compile(r"\b(?:shop\w*|order\w*|grocer\w*|buy\w*|cart|deliver\w*|food|app|web|brows\w*)\b", re.I)
    return next((b for b in mine if words.search(b["skill"]) or words.search(b["name"])), mine[0] if mine else None)


async def open_web(task: Task, b: dict, url: str, where: str, name: str, step=None):
    """The Bee's own Chromium on `url`; with no Chromium on this machine (a phone), its text-mode browser instead.
    Returns (surface, "where to watch it live", the current step)."""
    from .browser import BrowserError
    from .cells import cells
    from .reader import ReaderSurface
    cell = cells.get(b["id"], b["name"], b["computer"])
    surface, live = WebSurface(cell.browser), f"browser.html?bee={b['id']}"
    try:
        await surface.open(url)
    except BrowserError as exc:
        if "couldn't start" not in str(exc):
            raise
        if step is not None:
            await task.finish_step(step, status="done", text="No full browser on this machine: reading it in text mode")
        step = await task.step("opening", f"Opening {where} in {b['name']}'s text-mode browser", bee=name)
        surface, live = ReaderSurface(), None
        await surface.open(url)
    return surface, live, step


async def handle(task: Task, intent: dict) -> None:
    from .cells import cells
    from . import phone as phone_mod
    app = APPS.get(intent.get("app") or "")
    where = app["name"] if app else (intent.get("name") or intent.get("url") or "that website")
    package = (app or {}).get("package") or intent.get("package")
    use_phone = intent.get("phone") and task.user == "owner" and package and phone_mod.PHONES
    b = _pick_bee(task)
    if not use_phone and not b:
        await task.answer(f"A Bee does this in its own browser. Make one first (New Bee in Peace, or “make a bee called Shopper that "
                          f"orders my groceries”), then ask again: it will open {where} for you.")
        return
    name = "Phone" if use_phone else b["name"]
    s = await task.step("opening", f"Opening {where}" + (" on your phone" if use_phone else f" in {b['name']}'s browser"), bee=name)
    try:
        if use_phone:
            p = phone_mod.get()
            if "ui_tree" not in p.capabilities:
                raise RuntimeError("the phone needs ADB (Wireless debugging) for the Bee to see and tap apps")
            surface, live = PhoneSurface(p), "phone.html"
            await surface.open(package)
        else:
            surface, live, s = await open_web(task, b, start_url(intent), where, name, s)
    except Exception as exc:
        from .reader import explain_browser
        why = explain_browser(str(exc))
        await task.finish_step(s, status="error", text=why[:200])
        await task.answer(f"I couldn't open {where}: {why}.")
        return
    await task.finish_step(s)
    if not use_phone and intent.get("app"):                    # Blinkit and the like: a delivery place first (and a sign-in, if you want)
        from . import connections
        if intent["app"] in connections.SETUP:
            if not await connections.ensure_ready(task, surface, intent["app"], name, live, b["id"]):
                return
            try:
                await surface.open(start_url(intent))          # back to the search for what was asked
            except Exception:
                pass
    if not brain.is_connected():  # no model: say what's on the page and let the user take it from there
        img, mime = await surface.shot()
        sid = _shot_id(task.user, img, mime)
        if sid:
            await task.emit("screen", bee=name, title=f"{where}" + (f": {intent['item']}" if intent.get("item") else ""), shot=sid, live=live)
        try:
            obs = await surface.observe()
        except Exception:
            obs = {}
        await report(task, name, where, [f"Opened “{(obs.get('title') or where)[:70]}”"], obs)
        await task.answer(f"I opened {where}" + (f" with “{intent['item']}”" if intent.get("item") else "") + ". "
                          + ("Its page is summed up above. " if obs.get("text") else "")
                          + "No model is running on this Hive yet, so I can't go further by myself"
                          + (": open Watch live to finish it." if live else "."))
        return
    await run(task, surface, intent, name, live)
