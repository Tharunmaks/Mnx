"""Generate Mnx training data and the held-out evaluation set.

    python3 training/generate.py --train 4000 --test 400

Writes training/data/train.jsonl ({"category", "messages"}) and
training/data/test.jsonl ({"category", "messages", "mock", "expect"}).

Every conversation uses the exact format Mnx uses at runtime (see local.js):
the system prompt from training/system_prompt.json, <think>…</think>,
<tool_call>{json}</tool_call>, tool results as a user turn wrapped in
<tool_response>, and a final markdown answer. The test set is built from
held-out cities, topics, subjects and code tasks.
Standard library only, so it runs in Colab or Termux.
"""
import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content as C  # noqa: E402
import content_code as CC  # noqa: E402
import content_docs as D  # noqa: E402
import content_more as M  # noqa: E402
import content_qa as Q  # noqa: E402

PROMPT = json.load(open(os.path.join(HERE, "system_prompt.json"), encoding="utf-8"))
TOOLS = set(PROMPT["tools"])


class Gen:
    def __init__(self, seed, held_out):
        self.r = random.Random(seed)
        self.held_out = held_out
        if held_out:
            self.cities = C.HELD_OUT_CITIES + M.MORE_HELD_OUT_CITIES
            self.subjects = C.HELD_OUT_IMAGE_SUBJECTS
            self.animals, self.objects, self.settings = M.HELD_OUT_ANIMALS, [], M.HELD_OUT_SETTINGS
            self.topics = {**C.HELD_OUT_DOC_TOPICS, **D.MORE_HELD_OUT_DOC_TOPICS}
            self.code = C.HELD_OUT_CODE_TASKS + CC.MORE_HELD_OUT_CODE_TASKS
        else:
            self.cities = C.CITIES + M.MORE_CITIES
            self.subjects = C.IMAGE_SUBJECTS
            self.animals, self.objects, self.settings = M.ANIMALS, M.OBJECTS, M.SETTINGS
            self.topics = {**C.DOC_TOPICS, **D.MORE_DOC_TOPICS}
            self.code = C.CODE_TASKS + CC.MORE_CODE_TASKS
        self.styles = C.IMAGE_STYLES + M.MORE_IMAGE_STYLES
        self.place_kinds = C.PLACE_KINDS + M.MORE_PLACE_KINDS
        self.cuisines = C.CUISINES + M.MORE_CUISINES
        self.streets = C.STREETS + M.MORE_STREETS
        self.qa = C.QA + Q.MORE_QA
        self.unsupported_bank = C.UNSUPPORTED + Q.MORE_UNSUPPORTED

    # ───────── format helpers (must match local.js) ─────────
    def system(self):
        day = dt.date(2025, 1, 1) + dt.timedelta(days=self.r.randrange(900))
        name = self.r.choice(C.NAMES)
        text = f"{PROMPT['stable_prompt']}\n\nToday's date is {day.isoformat()}." + (f" The user's name is {name}." if name else "")
        return {"role": "system", "content": text}, day

    @staticmethod
    def think(title, text):
        return f"<think>**{title}**\n{text}</think>\n"

    @staticmethod
    def call(name, args):
        assert name in TOOLS, name
        return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"

    @staticmethod
    def response(obj):
        return f"<tool_response>\n{json.dumps(obj, ensure_ascii=False)}\n</tool_response>"

    def pick(self, items):
        return self.r.choice(items)

    # Real people type fast: lowercase, no punctuation, short forms, typos.
    NOISE = {"weather": ["wether", "weathr"], "please": ["pls", "plz"], "what's": ["whats", "wats"], "What's": ["whats", "Whats"],
             "tomorrow": ["tmrw", "tomorow"], "presentation": ["presentaion", "ppt"], "image": ["imgae", "pic"], "restaurants": ["restraunts", "restaurnts"],
             "directions": ["directons", "direction"], "you": ["u"], "Can you": ["can u", "Can u"], "picture": ["pic", "pictur"],
             "calculate": ["calc", "calculte"], "document": ["doc", "documnt"], "nearby": ["near by", "nearbyy"]}

    def noisy(self, text):
        if self.held_out:
            return text.lower() if self.r.random() < 0.15 else text
        if self.r.random() > 0.3:
            return text
        for word, subs in self.NOISE.items():
            if word in text and self.r.random() < 0.6:
                text = text.replace(word, self.pick(subs), 1)
        if self.r.random() < 0.6:
            text = text.lower()
        if self.r.random() < 0.5:
            text = text.rstrip("?.!")
        if self.r.random() < 0.15:
            text = self.pick(["pls ", "hey ", "mnx ", "yo "]) + text
        return text

    def conv(self, category, user, steps, final_think, answer, expect=None, history=None, noise=True):
        """steps: list of (think_title, think_text, tool, args, tool_result).
        history: earlier turns as plain [user, assistant, …] messages (as Mnx
        sends them: final answers only)."""
        sys_msg, _ = self.system()
        msgs = [sys_msg] + (history or []) + [{"role": "user", "content": self.noisy(user) if noise else user}]
        train_from = len(msgs) - 1
        mock = {}
        for title, text, tool, args, result in steps:
            msgs.append({"role": "assistant", "content": self.think(title, text) + self.call(tool, args)})
            msgs.append({"role": "user", "content": self.response(result)})
            mock.setdefault(tool, []).append(result)
        msgs.append({"role": "assistant", "content": self.think(*final_think) + answer})
        return {"category": category, "messages": msgs, "mock": mock, "expect": expect or {}, "train_from": train_from}

    # ───────── categories ─────────
    def weather(self, city=None, user=None, history=None, category="weather"):
        forced_user = user
        if city:
            country = next((c for n, c in self.cities if n == city), "")
        else:
            city, country = self.pick(self.cities)
        cond, icon = self.pick(C.CONDITIONS)
        temp = self.r.randint(-2, 40) if icon == "snow" else self.r.randint(8, 41)
        if icon == "snow":
            temp = self.r.randint(-6, 2)
        feels = temp + self.r.randint(-3, 4)
        hum = self.r.randint(30, 95)
        wind = self.r.randint(3, 35)
        no_loc = self.r.random() < 0.15 and not forced_user
        days = 7 if self.r.random() < 0.3 else 5
        place = f"{city}, {country}" if country else city
        asks = (
            ["What's the weather today?", "today's weather", "Is it going to rain today?", "How's the weather outside?"]
            if no_loc
            else [f"What's the weather in {city}?", f"weather {city}", f"Will it rain in {city} tomorrow?", f"How hot is it in {city} right now?",
                  f"Weather forecast for {city} this week", f"Do I need an umbrella in {city} today?"]
        )
        user = forced_user or self.pick(asks)
        if "week" in user:
            days = 7
        args = {} if no_loc else {"location": city}
        if days == 7:
            args["days"] = 7
        daily = []
        start = dt.date(2025, 6, 1) + dt.timedelta(days=self.r.randrange(300))
        for i in range(days):
            dcond, dicon = self.pick(C.CONDITIONS)
            hi = temp + self.r.randint(-3, 4)
            daily.append({"date": (start + dt.timedelta(days=i)).isoformat(), "condition": dcond, "icon": dicon, "max": hi, "min": hi - self.r.randint(4, 10),
                          "rain_chance": self.r.choice([0, 5, 10, 20, 40, 60, 80, 90]), "uv_max": round(self.r.uniform(1, 11), 1)})
        daily[0].update(condition=cond, icon=icon)
        result = {"location": place, "local_time": f"{start.isoformat()}T{self.r.randint(6, 21):02d}:00", "timezone": "auto",
                  "current": {"condition": cond, "temperature": f"{temp}°C", "feels_like": f"{feels}°C", "humidity": f"{hum}%", "wind": f"{wind} km/h", "precipitation_mm": 0.0},
                  "sunrise": f"{start.isoformat()}T06:{self.r.randint(0, 59):02d}", "sunset": f"{start.isoformat()}T18:{self.r.randint(0, 59):02d}", "daily": daily}
        tomorrow = daily[1]
        wet = any(w in user.lower() for w in ["rain", "umbrella"])
        lines = [f"It's **{temp}°C** and **{cond.lower()}** in {city} right now (feels like {feels}°C), with {hum}% humidity and {wind} km/h winds."]
        if wet:
            chance = daily[1 if "tomorrow" in user.lower() else 0]["rain_chance"]
            lines.insert(0, ("**Yes, take an umbrella.** " if chance >= 50 else "**Probably not.** ") + f"The chance of rain is **{chance}%**{' tomorrow' if 'tomorrow' in user.lower() else ' today'}.")
        rows = "\n".join(f"| {d['date']} | {d['condition']} | {d['max']}° / {d['min']}° | {d['rain_chance']}% |" for d in daily)
        lines.append(f"\n| Day | Conditions | High / Low | Rain |\n|---|---|---|---|\n{rows}")
        if temp >= 33:
            lines.append("\nIt's hot, so stay hydrated and avoid the midday sun.")
        elif temp <= 5:
            lines.append("\nIt's cold, so wear a warm jacket.")
        expect = {"tool": "get_weather", "args": {} if no_loc else {"location": city}, "answer": [str(temp)]}
        return self.conv(category, user, [("Checking the weather", f"The user wants the weather{'' if no_loc else ' in ' + city}. I'll call get_weather.",
                                           "get_weather", args, result)],
                         ("Reading the forecast", f"It's {temp}°C and {cond.lower()}. I'll give the current conditions first, then the forecast."), "\n".join(lines), expect, history=history)

    def search(self):
        kind = self.r.random()
        city, _ = self.pick(self.cities)
        n = self.r.randint(2, 9)
        if kind < 0.18:
            item = self.pick(M.PRICE_ITEMS)
            user = self.pick([f"What is the price of {item} today?", f"{item} price today", f"How much is {item} right now?"])
            query = f"{item} price today"
            price = round(self.r.uniform(80, 9000), 2)
            fact = f"{price:,}"
            results = [{"title": f"{item.title()} Price Today – Live Rates", "url": f"https://www.marketrates{n}.com/{item.split()[0].lower()}", "snippet": f"Today's {item} price is {fact}, up {round(self.r.uniform(0.1, 2.5), 1)}% from yesterday."},
                       {"title": f"{item.title()} rate history and trends", "url": f"https://finance.example{n}.org/{item.split()[0].lower()}-rates", "snippet": f"Track {item} prices over the last 30 days. Latest: {fact}."}]
            answer = f"The {item} price today is **{fact}**, according to [{results[0]['title']}]({results[0]['url']}). Prices change through the day, so check the latest rate before you buy."
        elif kind < 0.36:
            team1, team2 = self.r.sample(M.TEAMS, 2)
            event = self.pick(["final", "match yesterday", "last game", "semi-final"])
            user = self.pick([f"Who won the {team1} vs {team2} {event}?", f"{team1} vs {team2} result", f"Did {team1} win against {team2}?"])
            query = f"{team1} vs {team2} {event} result"
            winner = self.pick([team1, team2])
            margin = self.pick(["by 6 wickets", "by 23 runs", "2–1", "3–0", "on penalties"])
            results = [{"title": f"{team1} vs {team2}: {winner} win {margin}", "url": f"https://sports{n}.example.com/{team1.split()[0].lower()}-{team2.split()[0].lower()}", "snippet": f"{winner} won the {event} {margin} in front of a packed stadium."},
                       {"title": f"Highlights: {team1} vs {team2}", "url": f"https://www.sportsnews{n}.net/highlights", "snippet": f"Watch the highlights as {winner} sealed victory {margin}."}]
            answer = f"**{winner} won**, {margin}, according to [{results[0]['title']}]({results[0]['url']})."
            fact = winner
        elif kind < 0.54:
            company = self.pick(M.COMPANIES)
            product = self.pick(M.PRODUCTS)
            user = self.pick([f"When is the {company} {product} launching?", f"Latest news about {company}", f"What did {company} announce this week?"])
            query = f"{company} {product} launch" if "launch" in user else f"{company} latest news"
            month = self.pick(["March", "June", "September", "October", "November"])
            results = [{"title": f"{company} confirms {product} for {month}", "url": f"https://technews{n}.example.com/{company.lower().replace(' ', '-')}", "snippet": f"{company} said its {product} will arrive in {month}, with prices starting around ${self.r.randint(299, 1499)}."},
                       {"title": f"Everything we know about the {company} {product}", "url": f"https://www.gadgets{n}.org/{company.lower().replace(' ', '-')}-guide", "snippet": f"Expected features, price and release date for the {company} {product}."}]
            answer = f"{company} says its **{product} arrives in {month}** ([{results[0]['title']}]({results[0]['url']})). {results[0]['snippet'].split(', with ')[1].capitalize() if ', with ' in results[0]['snippet'] else ''}\n\nMore details: [{results[1]['title']}]({results[1]['url']})."
            fact = month
        elif kind < 0.66:
            event = self.pick(["concerts", "events", "festivals", "things to do"])
            user = self.pick([f"What {event} are happening in {city} this weekend?", f"{event} in {city} this weekend"])
            query = f"{event} in {city} this weekend"
            names = self.r.sample(["Jazz in the Park", "Night Food Market", "Indie Music Fest", "Art Walk", "Comedy Night", "Book Fair", "Craft Beer Festival", "Classical Evening"], 3)
            results = [{"title": f"{nm} – {city}", "url": f"https://events{n}.example.com/{city.lower().replace(' ', '-')}/{i}", "snippet": f"{nm} on Saturday from {self.r.randint(4, 8)} pm. Tickets from {self.r.randint(5, 40)}."} for i, nm in enumerate(names)]
            answer = f"Here's what's on in {city} this weekend:\n\n" + "\n".join(f"- **[{x['title'].split(' – ')[0]}]({x['url']})**: {x['snippet']}" for x in results) + "\n\nCheck the event pages for tickets and times."
            fact = names[0]
        elif kind < 0.78:
            sw = self.pick(M.SOFTWARE)
            major = self.r.randint(2, 30)
            ver = f"{major}.{self.r.randint(0, 12)}" + (f".{self.r.randint(0, 9)}" if self.r.random() < 0.5 else "")
            user = self.pick([f"What's the latest version of {sw}?", f"latest {sw} version", f"Is there a new {sw} release?"])
            query = f"{sw} latest version release"
            results = [{"title": f"{sw} {ver} released", "url": f"https://releases{n}.example.org/{sw.lower().replace(' ', '-').replace('.', '')}", "snippet": f"{sw} {ver} is now available with performance improvements and bug fixes."},
                       {"title": f"{sw} release notes", "url": f"https://docs{n}.example.com/{sw.lower().replace(' ', '-').replace('.', '')}/changelog", "snippet": f"Full changelog for {sw} {ver}."}]
            answer = f"The latest version of {sw} is **{ver}** ([{results[0]['title']}]({results[0]['url']})). It brings performance improvements and bug fixes; see the [release notes]({results[1]['url']}) for details."
            fact = ver
        elif kind < 0.9:
            topic = self.pick(M.NEWS_TOPICS)
            user = self.pick([f"What's the latest news on {topic}?", f"{topic} news today", f"Any updates about {topic} this week?"])
            query = f"{topic} latest news"
            heads = self.r.sample(["New record set", "Major announcement", "Experts weigh in", "Big milestone reached", "Plans revealed", "Funding boost announced"], 2)
            results = [{"title": f"{heads[i]} in {topic}", "url": f"https://news{n}.example.com/{topic.replace(' ', '-')}/{i}", "snippet": f"{heads[i]}: the latest development in {topic}, reported on {self.r.randint(1, 28)} {self.pick(['January', 'March', 'May', 'July', 'September', 'November'])}."} for i in range(2)]
            answer = f"Here's the latest on **{topic}**:\n\n" + "\n".join(f"- **{x['title']}**: {x['snippet']} ([source]({x['url']}))" for x in results)
            fact = heads[0]
        else:
            title = f"{self.pick(M.MOVIE_WORDS_A)} {self.pick(M.MOVIE_WORDS_B)}"
            user = self.pick([f"When does the movie {title} release?", f"{title} movie release date", f"Is {title} out in theatres yet?"])
            query = f"{title} movie release date"
            date = f"{self.r.randint(1, 28)} {self.pick(['January', 'April', 'June', 'August', 'October', 'December'])}"
            results = [{"title": f"{title}: release date, cast and trailer", "url": f"https://movies{n}.example.com/{title.lower().replace(' ', '-')}", "snippet": f"{title} hits theatres on {date}. The trailer has already crossed {self.r.randint(2, 60)} million views."}]
            answer = f"**{title}** releases in theatres on **{date}**, according to [{results[0]['title']}]({results[0]['url']})."
            fact = date
        steps = [("Searching the web", "This needs current information, so I'll search the web.", "search_web", {"query": query}, {"query": query, "results": results})]
        # Sometimes read the top page for details.
        if self.r.random() < 0.2:
            page = {"url": results[0]["url"], "title": results[0]["title"], "text": results[0]["snippet"] + " " + "Officials shared more details in a statement on the same day."}
            steps.append(("Reading the top result", "I'll open the first result for details.", "read_webpage", {"url": results[0]["url"]}, page))
        # Sometimes the first search fails and is retried.
        if self.r.random() < 0.1 and len(steps) == 1:
            steps.insert(0, ("Searching the web", "I need current information, so I'll search.", "search_web", {"query": query + " latest update today"}, {"error": "No search results"}))
            steps[1] = ("Trying a simpler search", "The first search found nothing; I'll try a shorter query.", "search_web", {"query": query}, {"query": query, "results": results})
        expect = {"tool": "search_web", "answer": [re.escape(str(fact))]}
        return self.conv("search", user, steps, ("Answering from the results", "The results answer the question. I'll give the answer first and cite the source."), answer, expect)

    def location(self):
        city, country = self.pick(self.cities)
        user = self.pick(["Where am I?", "What's my current location?", "Which city am I in right now?"])
        if self.r.random() < 0.25:
            result = {"available": False, "message": "The user hasn't shared their location. Ask them which city they're in."}
            answer = "I can't see your location yet. Tap the 📍 button next to the message box to share it, or tell me which city you're in."
            expect = {"tool": "get_user_location", "answer": ["📍|location|city"]}
        else:
            area = self.pick(["Anna Nagar", "Downtown", "Old Town", "Westside", "Central", "Harbour Front"])
            result = {"latitude": round(self.r.uniform(-40, 60), 4), "longitude": round(self.r.uniform(-120, 140), 4), "name": f"{area}, {city}, {country}"}
            answer = f"You're in **{area}, {city}**, {country}."
            expect = {"tool": "get_user_location", "answer": [re.escape(city)]}
        return self.conv("location", user, [("Finding the location", "The user asks where they are, so I'll check their location.", "get_user_location", {}, result)],
                         ("Answering", "I have the result."), answer, expect)

    def places(self):
        asks, query, osm, parts = self.pick(self.place_kinds)
        city, country = self.pick(self.cities)
        near_me = self.r.random() < 0.4
        what = self.pick(asks)
        if query == "restaurant" and self.r.random() < 0.4:
            cuisine = self.pick(self.cuisines)
            what, query = f"{cuisine} restaurants", f"{cuisine} restaurant"
        user = self.pick([f"Find {what} near me", f"Any {what} nearby?", f"Where can I find {what} around here?"] if near_me else
                         [f"Find {what} in {city}", f"Best {what} in {city}", f"Show me {what} near {city} station"])
        near = "my location" if near_me else (f"{city} station" if "station" in user else city)
        count = self.r.randint(3, 5)
        places = []
        for i in range(count):
            nm = f"{self.pick(parts)} {self.pick(['House', 'Corner', 'Point', 'Hub', 'Co.', 'Place', '& Co'])}" if osm not in ("atm", "fuel", "park") else f"{self.pick(parts)} {query.upper() if osm == 'atm' else query.title()}"
            addr = f"{self.r.randint(1, 220)} {self.pick(self.streets)}, {city}"
            p = {"name": nm, "type": osm.replace("_", " "), "address": addr, "latitude": round(self.r.uniform(-40, 60), 5), "longitude": round(self.r.uniform(-120, 140), 5),
                 "distance_km": round(0.2 + i * self.r.uniform(0.3, 1.2), 1), "maps_url": f"https://www.google.com/maps/search/?api=1&query={nm.replace(' ', '%20')}%20{city.replace(' ', '%20')}"}
            if self.r.random() < 0.5:
                p["opening_hours"] = self.pick(["Mo-Su 08:00-22:00", "Mo-Sa 09:00-21:00", "24/7", "Mo-Fr 07:00-19:00"])
            places.append(p)
        result = {"query": query, "near": f"{city}, {country}" if near_me else near, "count": count, "places": places}
        lines = [f"Here are the closest {what} I found{' near you' if near_me else ' in ' + city}:\n"]
        for p in places:
            extra = f" · open {p['opening_hours']}" if p.get("opening_hours") else ""
            lines.append(f"1. **{p['name']}**, {p['distance_km']} km away{extra}\n   {p['address']} · [Open in Maps]({p['maps_url']})")
        lines.append("\nWant directions to one of them?")
        expect = {"tool": "find_places", "args": {"near": "my location|me|here|current" if near_me else re.escape(city)}, "answer": [re.escape(places[0]["name"])]}
        return self.conv("places", user, [("Finding places", f"The user wants {what}{' near them' if near_me else ' in ' + city}. I'll use find_places.", "find_places", {"query": query, "near": near}, result)],
                         ("Listing the places", "I'll list them nearest first with map links."), "\n".join(lines), expect)

    def directions(self):
        city, _ = self.pick(self.cities)
        spots = ["Central Station", "City Museum", "the airport", "the Botanical Garden", "the Old Fort", "the University", "the Main Market", "the Beach"]
        a, b = self.r.sample(spots, 2)
        mode = self.pick(["driving", "walking", "cycling"])
        word = {"driving": "drive", "walking": "walk", "cycling": "cycle"}[mode]
        from_me = self.r.random() < 0.3
        origin = "my location" if from_me else f"{a}, {city}"
        dest = f"{b}, {city}"
        user = self.pick([f"How do I get to {b} in {city} from here?", f"Directions to {b}, {city}"] if from_me else
                         [f"How do I {word} from {a} to {b} in {city}?", f"{mode.title()} directions from {a} to {b}, {city}", f"Route from {a} to {b} in {city} by {'car' if mode == 'driving' else 'bike' if mode == 'cycling' else 'foot'}"])
        if from_me and mode != "driving" and word not in user:
            mode = "driving"
        km = round(self.r.uniform(0.8, 18), 1)
        speed = {"driving": 28, "walking": 4.8, "cycling": 14}[mode]
        mins = max(3, round(km / speed * 60))
        steps = [{"instruction": f"Head {self.pick(['north', 'south', 'east', 'west'])} on {self.pick(self.streets)}", "distance_m": self.r.randint(100, 900)},
                 {"instruction": f"Turn {self.pick(['left', 'right'])} onto {self.pick(self.streets)}", "distance_m": self.r.randint(300, 4000)},
                 {"instruction": f"Turn {self.pick(['left', 'right'])} onto {self.pick(self.streets)}", "distance_m": self.r.randint(200, 3000)},
                 {"instruction": f"Arrive at {b}", "distance_m": 0}]
        url = f"https://www.google.com/maps/dir/?api=1&origin=a&destination=b&travelmode={'bicycling' if mode == 'cycling' else mode}"
        result = {"origin": origin if not from_me else f"Current location, {city}", "destination": dest, "mode": mode, "distance_km": km, "duration_min": mins, "steps": steps, "maps_url": url}
        dur = f"{mins // 60} h {mins % 60} min" if mins >= 60 else f"{mins} min"
        answer = (f"It's about **{dur}** by {'car' if mode == 'driving' else 'bike' if mode == 'cycling' else 'foot'} (**{km} km**).\n\n" +
                  "\n".join(f"{i + 1}. {s['instruction']}" for i, s in enumerate(steps)) + f"\n\n[Open the route in Google Maps]({url})")
        expect = {"tool": "get_directions", "args": {"destination": re.escape(b.replace("the ", "")), "mode": mode}, "answer": [str(km)]}
        return self.conv("directions", user, [("Planning the route", f"The user wants {mode} directions to {b}. I'll call get_directions.", "get_directions", {"origin": origin, "destination": dest, "mode": mode}, result)],
                         ("Summarizing the route", "I'll give the time and distance first, then the steps."), answer, expect)

    def image(self):
        if self.r.random() < 0.15:
            brand = self.pick(C.LOGO_BRANDS)
            user = self.pick([f"Design a logo for {brand}", f"Make a logo for {brand}"])
            prompt = f"minimalist logo for {brand}, flat vector design, simple bold shapes, clean white background"
            aspect, key = "square", brand.split(" called ")[1].split()[0]
        else:
            subj = self.subject()
            style, desc = self.pick(self.styles)
            wall = self.r.random()
            if wall < 0.2:
                user, aspect = self.pick([f"Make a phone wallpaper of {subj}", f"Create a mobile wallpaper: {subj}"]), "portrait"
            elif wall < 0.35:
                user, aspect = self.pick([f"Make a desktop wallpaper of {subj}", f"Create a wide banner image of {subj}"]), "landscape"
            else:
                user = self.pick([f"Draw {subj}", f"Generate an image of {subj}", f"Create a picture of {subj} in {style} style", f"Can you make an image of {subj}?", f"Paint {subj} as a {style}"])
                aspect = "square"
            prompt = f"{subj}, {desc}"
            key = subj.split()[-1]
        args = {"prompt": prompt} if aspect == "square" and self.r.random() < 0.5 else {"prompt": prompt, "aspect": aspect}
        result = {"created": True, "shown_to_user": True, "prompt": prompt}
        answer = self.pick([f"Here's your image! 🎨 I went for {prompt.split(', ', 1)[1] if ', ' in prompt else 'a clean look'}. Want a different style or angle?",
                            "Done! Your image is above. I can try another style, colors or composition if you like.",
                            f"Here you go. 🎨 Tell me if you'd like it {self.pick(['brighter', 'more detailed', 'in another style', 'from a different angle'])}."])
        expect = {"tool": "create_image", "args": {"prompt": re.escape(key)}, "args_eq": {} if aspect == "square" else {"aspect": aspect}, "answer": [".+"]}
        return self.conv("image", user, [("Planning the image", "The user wants an image, so I'll write a detailed prompt for create_image.", "create_image", args, result)],
                         ("Image ready", "The image was created and shown, so I won't repeat the link."), answer, expect)

    def subject(self):
        if self.r.random() < 0.3 or not (self.animals or self.objects):
            return self.pick(self.subjects)
        base = self.pick(self.animals + self.objects)
        subj = f"{base} {self.pick(self.settings)}"
        if self.r.random() < 0.5:
            subj += " " + self.pick(M.TIMES)
        return subj

    def document(self, title=None, fmt=None, n=None, user=None, history=None, category="document"):
        if title is None:
            title, sections = self.pick(list(self.topics.items()))
        else:
            sections = self.topics[title]
        fmt = fmt or self.pick(["pptx", "pptx", "pdf", "docx"])
        n = n or self.r.randint(3, len(sections))
        word = {"pptx": self.pick(["presentation", "slide deck", "PPT", "slides"]), "pdf": self.pick(["PDF", "PDF report", "PDF document"]), "docx": self.pick(["Word document", "docx", "Word file"])}[fmt]
        user = user or self.pick([f"Make a {word} on {title.lower()}", f"Create a {n}-{'slide' if fmt == 'pptx' else 'section'} {word} about {title.lower()}", f"I need a {word} explaining {title.lower()}",
                                  f"Can you prepare a {word} on {title.lower()} for my class?", f"{word} about {title.lower()} please"])
        if fmt == "pptx":
            args = {"format": "pptx", "title": title, "subtitle": self.pick(["An overview", "Key ideas explained", "A quick guide"]), "slides": [{"title": t, "bullets": b} for t, b in sections[:n]]}
            unit = "slides"
        else:
            args = {"format": fmt, "title": title, "sections": [{"heading": t, "body": "\n".join(f"- {x}" for x in b)} for t, b in sections[:n]]}
            unit = "sections"
        fname = f"{title}.{fmt}"
        result = {"delivered": True, "filename": fname, "format": fmt, "pages_or_slides": n}
        answer = f"I've created **{fname}** with {n} {unit}" + (" plus a title slide" if fmt == "pptx" else "") + ":\n\n" + "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(sections[:n])) + "\n\nYou can preview it above and tap **Download**. Want me to add or change anything?"
        expect = {"tool": "create_document", "args_eq": {"format": fmt}, "min_items": 3, "answer": [re.escape(fmt)]}
        return self.conv(category, user, [("Planning the document", f"The user wants a {fmt} about {title.lower()}. I'll outline {n} {unit} and call create_document.", "create_document", args, result)],
                         ("Document ready", "The file was delivered; I'll summarize what's inside."), answer, expect, history=history)

    def code_file(self):
        task = self.pick(self.code)
        user = self.pick(task["asks"])
        args = {"filename": task["filename"], "content": task["content"]}
        result = {"delivered": True, "filename": task["filename"], "bytes": len(task["content"].encode())}
        answer = f"Here's **{task['filename']}**: {task['desc'][0].lower() + task['desc'][1:]}.\n\n{task['explain']}"
        ext = task["filename"].rsplit(".", 1)[1]
        expect = {"tool": "create_file", "args": {"filename": rf"\.{ext}$"}, "answer": [re.escape(task["filename"])]}
        return self.conv("code", user, [("Writing the code", f"The user wants {task['desc'][0].lower() + task['desc'][1:]}. I'll write the full file with create_file.", "create_file", args, result)],
                         ("File delivered", "I'll explain how to run it."), answer, expect)

    def _program(self):
        """A (user, code, explanation) program whose output we compute by running it."""
        k = self.r.randrange(8)
        if k == 0:
            p, rate, years = self.r.choice([1000, 5000, 10000, 25000, 50000]), self.r.choice([5, 6, 7, 8, 9, 12]), self.r.randint(2, 15)
            user = self.pick([f"Calculate compound interest on {p} at {rate}% for {years} years", f"If I invest {p} at {rate}% yearly for {years} years, how much will I have?"])
            code = f"principal = {p}\nrate = {rate} / 100\nyears = {years}\namount = principal * (1 + rate) ** years\nprint(f\"Final amount: {{amount:,.2f}}\")\nprint(f\"Interest earned: {{amount - principal:,.2f}}\")"
        elif k == 1:
            n = self.r.randint(10, 25)
            user = self.pick([f"What is {n} factorial?", f"Calculate {n}!", f"compute the factorial of {n}"])
            code = f"import math\nprint(math.factorial({n}))"
        elif k == 2:
            n = self.r.choice([50, 100, 150, 200, 500])
            user = self.pick([f"How many prime numbers are there below {n}?", f"List the primes under {n}"])
            code = f"primes = [n for n in range(2, {n}) if all(n % d for d in range(2, int(n ** 0.5) + 1))]\nprint(f\"Count: {{len(primes)}}\")\nprint(primes)"
        elif k == 3:
            a = dt.date(2024, 1, 1) + dt.timedelta(days=self.r.randrange(700))
            b = a + dt.timedelta(days=self.r.randint(10, 900))
            user = self.pick([f"How many days are there between {a.isoformat()} and {b.isoformat()}?", f"Days from {a.strftime('%B %d, %Y')} to {b.strftime('%B %d, %Y')}?"])
            code = f"from datetime import date\nd = date({b.year}, {b.month}, {b.day}) - date({a.year}, {a.month}, {a.day})\nprint(f\"{{d.days}} days\")"
        elif k == 4:
            nums = [self.r.randint(1, 100) for _ in range(self.r.randint(6, 12))]
            user = self.pick([f"Find the mean, median and standard deviation of {nums}", f"stats for these numbers: {', '.join(map(str, nums))}"])
            code = f"import statistics as s\nnums = {nums}\nprint(f\"Mean: {{s.mean(nums):.2f}}\")\nprint(f\"Median: {{s.median(nums)}}\")\nprint(f\"Std dev: {{s.stdev(nums):.2f}}\")"
        elif k == 5:
            p, rate, months = self.r.choice([200000, 500000, 1000000, 2500000]), self.r.choice([7.5, 8.5, 9, 10.5, 12]), self.r.choice([12, 24, 36, 60, 120])
            user = self.pick([f"What's the monthly EMI on a loan of {p} at {rate}% for {months} months?", f"Calculate EMI: {p} loan, {rate}% interest, {months} months"])
            code = f"p = {p}\nr = {rate} / 12 / 100\nn = {months}\nemi = p * r * (1 + r) ** n / ((1 + r) ** n - 1)\nprint(f\"EMI: {{emi:,.2f}} per month\")\nprint(f\"Total paid: {{emi * n:,.2f}}\")"
        elif k == 6:
            n = self.r.randint(15, 40)
            user = self.pick([f"Show the first {n} Fibonacci numbers", f"What's the {n}th Fibonacci number?"])
            code = f"a, b = 0, 1\nseq = []\nfor _ in range({n}):\n    seq.append(a)\n    a, b = b, a + b\nprint(seq)\nprint(f\"The {n}th number is {{seq[-1]}}\")"
        else:
            x, y = self.r.randint(2, 99), self.r.randint(10, 64)
            user = self.pick([f"What is {x} to the power of {y}?", f"Calculate {x}^{y} exactly"])
            code = f"print({x} ** {y})"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, {})  # our own fixed templates, run to get the exact output
        return user, code, out.getvalue()

    def _more_program(self):
        """Extra code-runner problems (deterministic, output computed by running)."""
        k = self.r.randrange(14)
        if k == 0:
            w, h = self.r.randint(45, 110), self.r.randint(150, 195)
            user = self.pick([f"What's my BMI if I weigh {w} kg and I'm {h} cm tall?", f"Calculate BMI for {w} kg and {h} cm"])
            code = f"weight = {w}\nheight = {h} / 100\nbmi = weight / height ** 2\ncategory = 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'\nprint(f\"BMI: {{bmi:.1f}} ({{category}})\")"
        elif k == 1:
            bill, tip, people = self.r.choice([850, 1240, 2360, 4999, 675]), self.r.choice([5, 10, 12, 15, 18, 20]), self.r.randint(2, 8)
            user = self.pick([f"Split a bill of {bill} with {tip}% tip between {people} people", f"{people} of us, bill is {bill}, {tip}% tip. how much each?"])
            code = f"bill = {bill}\ntip = {tip} / 100\npeople = {people}\ntotal = bill * (1 + tip)\nprint(f\"Total with tip: {{total:.2f}}\")\nprint(f\"Each person pays: {{total / people:.2f}}\")"
        elif k == 2:
            a, b = self.r.randint(12, 500), self.r.randint(12, 500)
            user = self.pick([f"Find the GCD and LCM of {a} and {b}", f"gcd and lcm of {a}, {b}"])
            code = f"import math\na, b = {a}, {b}\nprint(f\"GCD: {{math.gcd(a, b)}}\")\nprint(f\"LCM: {{a * b // math.gcd(a, b)}}\")"
        elif k == 3:
            d = dt.date(1950, 1, 1) + dt.timedelta(days=self.r.randrange(36000))
            user = self.pick([f"What day of the week was {d.strftime('%d %B %Y')}?", f"Which weekday is {d.isoformat()}?"])
            code = f"from datetime import date\nd = date({d.year}, {d.month}, {d.day})\nprint(f\"Weekday: {{d.strftime('%A')}}\")"
        elif k == 4:
            a, b, c = self.r.choice([1, 2, 3]), self.r.randint(-12, 12), self.r.randint(-20, 20)
            user = self.pick([f"Solve {a}x² + {b}x + {c} = 0", f"Find the roots of {a}x^2 + {b}x + {c}"])
            code = f"import cmath\na, b, c = {a}, {b}, {c}\nd = cmath.sqrt(b * b - 4 * a * c)\nr1, r2 = (-b + d) / (2 * a), (-b - d) / (2 * a)\nfmt = lambda z: f\"{{z.real:.3f}}\" if abs(z.imag) < 1e-12 else f\"{{z.real:.3f}} {{'+' if z.imag >= 0 else '-'}} {{abs(z.imag):.3f}}i\"\nprint(f\"Roots: {{fmt(r1)}} and {{fmt(r2)}}\")"
        elif k == 5:
            r = self.r.randint(2, 50)
            user = self.pick([f"Area and circumference of a circle with radius {r} cm", f"circle radius {r}: area?"])
            code = f"import math\nr = {r}\nprint(f\"Area: {{math.pi * r * r:.2f}} cm²\")\nprint(f\"Circumference: {{2 * math.pi * r:.2f}} cm\")"
        elif k == 6:
            n = self.r.randint(10 ** 5, 10 ** 12)
            user = self.pick([f"Convert {n} to binary and hexadecimal", f"{n} in binary and hex"])
            code = f"n = {n}\nprint(f\"Binary: {{bin(n)[2:]}}\")\nprint(f\"Hex: {{hex(n)[2:].upper()}}\")"
        elif k == 7:
            p, rate, years = self.r.choice([10000, 25000, 75000, 150000]), self.r.choice([4, 6.5, 7, 8.5, 10]), self.r.randint(1, 10)
            user = self.pick([f"Simple interest on {p} at {rate}% for {years} years?", f"calculate simple interest: principal {p}, rate {rate}%, time {years} years"])
            code = f"p, r, t = {p}, {rate}, {years}\ninterest = p * r * t / 100\nprint(f\"Interest: {{interest:,.2f}}\")\nprint(f\"Total: {{p + interest:,.2f}}\")"
        elif k == 8:
            old, new = self.r.randint(100, 5000), self.r.randint(100, 5000)
            user = self.pick([f"What's the percentage change from {old} to {new}?", f"percent change {old} -> {new}"])
            code = f"old, new = {old}, {new}\nchange = (new - old) / old * 100\nprint(f\"Change: {{change:+.2f}}%\")"
        elif k == 9:
            goal, monthly, rate = self.r.choice([50000, 100000, 250000, 500000]), self.r.choice([2000, 5000, 8000, 12000]), self.r.choice([0, 6, 8])
            user = self.pick([f"How many months to save {goal} if I save {monthly} a month at {rate}% interest?", f"Saving {monthly}/month, {rate}% yearly interest. When do I reach {goal}?"])
            code = f"goal, monthly, rate = {goal}, {monthly}, {rate} / 12 / 100\nbalance, months = 0.0, 0\nwhile balance < goal:\n    balance = balance * (1 + rate) + monthly\n    months += 1\nprint(f\"Months: {{months}} ({{months // 12}} years {{months % 12}} months)\")\nprint(f\"Final balance: {{balance:,.2f}}\")"
        elif k == 10:
            (c1, lat1, lon1), (c2, lat2, lon2) = self.r.sample([("Chennai", 13.0827, 80.2707), ("Mumbai", 19.076, 72.8777), ("Delhi", 28.6139, 77.209), ("London", 51.5074, -0.1278), ("New York", 40.7128, -74.006), ("Tokyo", 35.6762, 139.6503), ("Sydney", -33.8688, 151.2093), ("Paris", 48.8566, 2.3522), ("Dubai", 25.2048, 55.2708), ("Singapore", 1.3521, 103.8198)], 2)
            user = self.pick([f"How far is {c1} from {c2} in a straight line?", f"distance between {c1} and {c2} as the crow flies"])
            code = f"import math\nlat1, lon1, lat2, lon2 = map(math.radians, ({lat1}, {lon1}, {lat2}, {lon2}))\na = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2\nprint(f\"Distance: {{2 * 6371 * math.asin(math.sqrt(a)):,.0f}} km\")"
        elif k == 11:
            words = self.r.choices(["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "and", "a", "big", "red", "ball"], k=self.r.randint(12, 25))
            sentence = " ".join(words)
            user = f"Count the words in this sentence and show the most common ones: \"{sentence}\""
            code = f"from collections import Counter\ntext = \"{sentence}\"\nwords = text.split()\nprint(f\"Words: {{len(words)}}\")\nfor word, n in Counter(words).most_common(3):\n    print(f\"{{word}}: {{n}}\")"
        elif k == 12:
            a, b = sorted(self.r.sample(range(1900, 2101), 2))
            user = self.pick([f"How many leap years are there between {a} and {b}?", f"count leap years from {a} to {b}"])
            code = f"years = [y for y in range({a}, {b} + 1) if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)]\nprint(f\"Leap years: {{len(years)}}\")"
        else:
            n = self.r.randint(10 ** 6, 10 ** 15)
            user = self.pick([f"What is the sum of the digits of {n}?", f"sum of digits {n}"])
            code = f"n = {n}\nprint(f\"Sum of digits: {{sum(int(d) for d in str(n))}}\")"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, {})
        return user, code, out.getvalue()

    def run_code(self):
        user, code, stdout = self._program() if self.r.random() < 0.45 else self._more_program()
        variant = self.r.random()
        if variant < 0.15:
            result = {"ran": False, "reason": "The user chose not to run this code."}
            answer = f"No problem, I didn't run it. Here's what it would print:\n\n```\n{stdout.strip()}\n```\n\nYou can also run the code yourself any time."
            return self.conv("run_code_declined", user, [("Computing it", "This needs an exact calculation, so I'll run a short Python program.", "run_code", {"language": "python", "code": code}, result)],
                             ("Not run", "The user chose not to run it. I can still give the result."), answer,
                             {"tool": "run_code", "args_eq": {"language": "python"}, "decline": True, "answer": ["didn't run|did not run|not run|won't run"]})
        steps = []
        if variant < 0.25 and "import math" in code:
            bad = code.replace("import math\n", "")
            steps.append(("Computing it", "I'll run a short program for an exact answer.", "run_code", {"language": "python", "code": bad},
                          {"ran": True, "exit_code": 1, "timed_out": False, "stdout": "", "stderr": "Traceback (most recent call last):\n  File \"main.py\", line 1, in <module>\nNameError: name 'math' is not defined", "images_shown_to_user": []}))
            steps.append(("Fixing the error", "I forgot to import math. I'll fix it and run again.", "run_code", {"language": "python", "code": code},
                          {"ran": True, "exit_code": 0, "timed_out": False, "stdout": stdout, "stderr": "", "images_shown_to_user": []}))
        else:
            steps.append(("Computing it", "This needs an exact calculation, so I'll run a short Python program.", "run_code", {"language": "python", "code": code},
                          {"ran": True, "exit_code": 0, "timed_out": False, "stdout": stdout, "stderr": "", "images_shown_to_user": []}))
        first = stdout.strip().splitlines()[0]
        answer = f"{self._lead(first)}\n\n```\n{stdout.strip()}\n```"
        key = re.escape(re.sub(r"^[^:]*:\s*", "", first)[:40])
        return self.conv("run_code", user, steps, ("Reading the output", "The program ran successfully. I'll state the result first."), answer,
                         {"tool": "run_code", "args_eq": {"language": "python"}, "answer": [key]})

    @staticmethod
    def _lead(first):
        if ":" in first:
            label, value = first.split(":", 1)
            return f"**{label.strip()}: {value.strip()}**"
        return f"The answer is **{first.strip()}**."

    CHART_FUNCS = [("x ** 2", "y = x²"), ("2 ** x", "y = 2^x"), ("x ** 3 - 3 * x", "y = x³ − 3x"), ("10 * x + 5", "y = 10x + 5"),
                   ("math.sin(x)", "y = sin(x)"), ("math.cos(x)", "y = cos(x)"), ("math.sqrt(x)", "y = √x"), ("math.log(x + 1)", "y = ln(x + 1)"),
                   ("x ** 2 - 4 * x + 3", "y = x² − 4x + 3"), ("math.exp(-x) * 100", "y = 100·e^(−x)"), ("abs(x - 5)", "y = |x − 5|"), ("3 * x - 7", "y = 3x − 7")]

    def chart(self):
        if self.r.random() < 0.5:
            func, label = self.pick(self.CHART_FUNCS)
            lo, hi = self.pick([(0, 10), (0, 5), (0, 20), (1, 50), (0, 100), (0, 6)])
            user = self.pick([f"Plot {label} from {lo} to {hi}", f"Make a chart of {label} for x between {lo} and {hi}", f"Draw a graph of {label} from x={lo} to x={hi}", f"graph {label} {lo} to {hi}"])
            code = f"import math\nimport matplotlib.pyplot as plt\nxs = [{lo} + i * ({hi} - {lo}) / 200 for i in range(201)]\nys = [{func} for x in xs]\nplt.plot(xs, ys)\nplt.title(\"{label}\")\nplt.xlabel(\"x\")\nplt.ylabel(\"y\")\nplt.grid(True)\nplt.savefig(\"plot.png\")\nprint(\"Saved plot.png\")"
            answer = f"Here's the graph of **{label}** for x from {lo} to {hi}. It's shown above, and you can save it from the output card."
        else:
            kind = self.pick(["bar", "pie", "line"])
            labels = self.r.sample(["Jan", "Feb", "Mar", "Apr", "May", "Jun"] if kind == "line" else ["Rent", "Food", "Travel", "Bills", "Fun", "Savings", "Shopping"], self.r.randint(3, 6))
            if kind == "line":
                labels = sorted(labels, key=["Jan", "Feb", "Mar", "Apr", "May", "Jun"].index)
            vals = [self.r.randint(5, 95) * 100 for _ in labels]
            data = ", ".join(f"{l} {v}" for l, v in zip(labels, vals))
            user = self.pick([f"Make a {kind} chart of: {data}", f"{kind} chart for my expenses: {data}", f"Visualize this as a {kind} chart: {data}"])
            draw = {"bar": "plt.bar(labels, values)", "pie": "plt.pie(values, labels=labels, autopct=\"%1.0f%%\")", "line": "plt.plot(labels, values, marker=\"o\")"}[kind]
            code = f"import matplotlib.pyplot as plt\nlabels = {labels}\nvalues = {vals}\n{draw}\nplt.title(\"{kind.title()} chart\")\nplt.tight_layout()\nplt.savefig(\"chart.png\")\nprint(f\"Saved chart.png, total = {{sum(values)}}\")"
            answer = f"Here's your {kind} chart. It's shown above. The total is **{sum(vals):,}**, and the biggest item is **{labels[vals.index(max(vals))]}** ({max(vals):,})."
        out_name = "plot.png" if "plot.png" in code else "chart.png"
        stdout = "Saved plot.png\n" if out_name == "plot.png" else f"Saved chart.png, total = {sum(vals)}\n"
        result = {"ran": True, "exit_code": 0, "timed_out": False, "stdout": stdout, "stderr": "", "images_shown_to_user": [out_name]}
        return self.conv("chart", user, [("Making the chart", "I'll draw it with matplotlib and save the image.", "run_code", {"language": "python", "code": code}, result)],
                         ("Chart ready", "The chart was saved and shown."), answer, {"tool": "run_code", "args": {"code": "savefig"}, "answer": [".+"]})

    def multi(self):
        title, sections = self.pick(list(self.topics.items()))
        user = self.pick([f"Search for the latest on {title.lower()} and make a short presentation", f"Research {title.lower()} online and turn it into slides"])
        n = self.r.randint(2, 6)
        results = [{"title": f"{title}: what's new in {self.r.choice([2025, 2026])}", "url": f"https://news{n}.example.org/{title.lower().replace(' ', '-')}", "snippet": sections[0][1][0] + "."},
                   {"title": f"A beginner's guide to {title.lower()}", "url": f"https://learn{n}.example.com/{title.lower().replace(' ', '-')}", "snippet": sections[1][1][0] + "."}]
        slides = [{"title": t, "bullets": b} for t, b in sections[:4]]
        doc_args = {"format": "pptx", "title": title, "subtitle": "Based on the latest sources", "slides": slides}
        steps = [("Researching", "First I'll search for recent information.", "search_web", {"query": f"{title.lower()} latest"}, {"query": f"{title.lower()} latest", "results": results}),
                 ("Building the slides", "I have enough material; now I'll create the deck.", "create_document", doc_args, {"delivered": True, "filename": f"{title}.pptx", "format": "pptx", "pages_or_slides": len(slides)})]
        answer = f"I researched **{title.lower()}** and made **{title}.pptx** with {len(slides)} slides:\n\n" + "\n".join(f"{i + 1}. {s['title']}" for i, s in enumerate(slides)) + f"\n\nSources: [{results[0]['title']}]({results[0]['url']}), [{results[1]['title']}]({results[1]['url']})."
        return self.conv("multi_step", user, steps, ("Done", "Both steps worked; I'll summarize and cite sources."), answer,
                         {"tool": "search_web", "then": "create_document", "answer": ["pptx"]})

    # ───────── helpers for multi-turn conversations ─────────
    @staticmethod
    def final_text(conv):
        return conv["messages"][-1]["content"].split("</think>\n", 1)[1]

    def history_of(self, conv):
        """The earlier turn as Mnx sends it: the user's message and the final answer only."""
        m = conv["messages"]
        return m[1:conv["train_from"]] + [m[conv["train_from"]], {"role": "assistant", "content": self.final_text(conv)}]

    @staticmethod
    def first_call(conv):
        for msg in conv["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                return json.loads(msg["content"].split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0])
        return None

    @staticmethod
    def first_result(conv):
        for msg in conv["messages"]:
            if msg["role"] == "user" and msg["content"].startswith("<tool_response>"):
                return json.loads(msg["content"][len("<tool_response>\n"):-len("\n</tool_response>")])
        return None

    # ───────── new categories ─────────
    def simple_math(self):
        a, b = self.r.randint(2, 999), self.r.randint(2, 99)
        k = self.r.randrange(5)
        if k == 0:
            q, ans, work = self.pick([f"what is {a} + {b}", f"{a}+{b}", f"add {a} and {b}"]), a + b, f"{a} + {b} = {a + b}"
        elif k == 1:
            q, ans, work = self.pick([f"what is {a} - {b}", f"{a}-{b}"]), a - b, f"{a} − {b} = {a - b}"
        elif k == 2:
            q, ans, work = self.pick([f"what is {a} times {b}", f"{a} x {b}", f"{a}*{b}"]), a * b, f"{a} × {b} = {a * b}"
        elif k == 3:
            q, ans, work = self.pick([f"what is {b}% of {a * 10}", f"{b} percent of {a * 10}"]), round(a * 10 * b / 100, 2), f"{a * 10} × {b} / 100 = {round(a * 10 * b / 100, 2)}"
        else:
            q, ans, work = self.pick([f"what is {a} divided by {b}", f"{a}/{b}"]), round(a / b, 2), f"{a} ÷ {b} ≈ {round(a / b, 2)}"
        ans_s = f"{ans:g}" if isinstance(ans, float) else str(ans)
        sys_msg, _ = self.system()
        msgs = [sys_msg, {"role": "user", "content": self.noisy(q.capitalize() + "?" if self.r.random() < 0.5 else q)},
                {"role": "assistant", "content": self.think("Quick math", "This is simple arithmetic, so I'll answer directly without running code.") + f"**{ans_s}**\n\n{work}"}]
        return {"category": "no_tool", "messages": msgs, "mock": {}, "expect": {"tool": None, "answer": [re.escape(ans_s)]}, "train_from": 1}

    def clarify(self):
        asks, answer = self.pick(Q.CLARIFY)
        sys_msg, _ = self.system()
        msgs = [sys_msg, {"role": "user", "content": self.noisy(self.pick(asks))},
                {"role": "assistant", "content": self.think("Need more details", "The request is too vague to act on, so I'll ask one short question first.") + answer}]
        return {"category": "clarify", "messages": msgs, "mock": {}, "expect": {"tool": None, "answer": [r"\?"]}, "train_from": 1}

    def notes_slides(self):
        title, sections = self.pick(list(self.topics.items()))
        n = self.r.randint(3, len(sections))
        notes = "\n".join(f"{t.lower()}: " + "; ".join(b.lower() for b in bullets) for t, bullets in sections[:n])
        user = self.pick([f"Turn these notes into slides:\n{notes}", f"make a ppt from my notes on {title.lower()}\n\n{notes}", f"Here are my notes, make a presentation called \"{title}\":\n{notes}"])
        return self.document(title=title, fmt="pptx", n=n, user=user, category="notes_to_slides")

    def tool_error(self):
        k = self.r.randrange(4)
        if k == 0:
            city, country = self.pick(self.cities)
            typo = city[:2] + city[3:] if len(city) > 5 else city + city[-1]
            c = self.weather(city=city, user=self.pick([f"weather in {typo}", f"What's the weather in {typo}?"]), category="tool_error")
            first = ("Checking the weather", f"The user wants the weather in {typo}.", "get_weather", {"location": typo}, {"error": f"Couldn't find a place called \"{typo}\""})
            # Insert the failed attempt before the successful (corrected) one.
            m = c["messages"]
            m.insert(2, {"role": "assistant", "content": self.think(first[0], first[1]) + self.call(first[2], first[3])})
            m.insert(3, {"role": "user", "content": self.response(first[4])})
            m[4]["content"] = m[4]["content"].replace("**Checking the weather**\nThe user wants", f"**Fixing the place name**\n\"{typo}\" wasn't found; it's probably {city}. The user wants")
            c["mock"]["get_weather"].insert(0, first[4])
            c["expect"] = {"tool": "get_weather", "answer": c["expect"]["answer"]}
            return c
        if k == 1:
            asks, query, osm, parts = self.pick(self.place_kinds)
            city, _ = self.pick(self.cities)
            what = self.pick(asks)
            result = {"query": query, "near": city, "count": 0, "places": []}
            answer = f"I couldn't find any {what} in {city} on the map. Try a nearby area or a broader search (for example \"{query} near {city} station\"), or I can search the web instead."
            return self.conv("tool_error", f"Find {what} in {city}", [("Finding places", f"I'll search the map for {what} in {city}.", "find_places", {"query": query, "near": city}, result)],
                             ("No results", "Nothing was found, so I'll say so and suggest alternatives."), answer, {"tool": "find_places", "answer": ["couldn't|could not|no "]})
        if k == 2:
            # A temporary failure in the search service (not the model's fault): retry once.
            c = self.search()
            m = c["messages"]
            q = self.first_call(c)["arguments"]["query"]
            m.insert(2, {"role": "assistant", "content": self.think("Searching the web", "This needs current information, so I'll search the web.") + self.call("search_web", {"query": q})})
            m.insert(3, {"role": "user", "content": self.response({"error": "The search service timed out. Try again."})})
            m[4]["content"] = re.sub(r"^<think>\*\*[^*]+\*\*\n[^<]*</think>", "<think>**Trying again**\nThe search timed out, so I'll retry once.</think>", m[4]["content"])
            c["mock"]["search_web"].insert(0, {"error": "The search service timed out. Try again."})
            c["category"] = "tool_error"
            return c
        city, _ = self.pick(self.cities)
        a, b = self.r.sample(M.LANDMARKS, 2)
        answer = f"I couldn't find a route from {a} to {b} in {city}. Check that both places are spelled correctly, or try a nearby landmark or the full address."
        return self.conv("tool_error", f"Directions from {a} to {b} in {city}", [("Planning the route", "I'll ask for directions.", "get_directions", {"origin": f"{a}, {city}", "destination": f"{b}, {city}", "mode": "driving"}, {"error": "No route found between those places"})],
                         ("No route", "The routing service found no route; I'll explain and suggest what to try."), answer, {"tool": "get_directions", "answer": ["couldn't|could not"]})

    def followup(self):
        k = self.r.randrange(6)
        if k == 0:  # weather in another city
            c1 = self.weather()
            city2, _ = self.pick(self.cities)
            return self.weather(city=city2, user=self.pick([f"And in {city2}?", f"what about {city2}", f"How about {city2}?", f"Same for {city2}"]), history=self.history_of(c1), category="followup")
        if k == 1:  # restyle / resize the last image
            c1 = self.image()
            prompt1 = self.first_call(c1)["arguments"]["prompt"]
            subject = prompt1.split(", ")[0]
            if self.r.random() < 0.5:
                style, desc = self.pick(self.styles)
                user, args = self.pick([f"Make it {style} style", f"now do it as {style}", f"Can you try {style} instead?"]), {"prompt": f"{subject}, {desc}"}
            else:
                user, args = self.pick(["Make it a phone wallpaper", "make it portrait", "Now make it wide for my desktop"]), None
                args = {"prompt": prompt1, "aspect": "landscape" if "wide" in user or "desktop" in user else "portrait"}
            res = {"created": True, "shown_to_user": True, "prompt": args["prompt"]}
            return self.conv("followup", user, [("Updating the image", "The user wants a new version of the last image. I'll keep the subject and change what they asked.", "create_image", args, res)],
                             ("New version ready", "The new image is shown."), self.pick(["Here's the new version! 🎨 Want any other changes?", "Done. The updated image is above."]),
                             {"tool": "create_image", "args": {"prompt": re.escape(subject.split()[-1])}, "answer": [".+"]}, history=self.history_of(c1))
        if k == 2:  # add a slide to the deck
            title, sections = self.pick(list(self.topics.items()))
            c1 = self.document(title=title, fmt="pptx", n=3)
            extra_t, extra_b = sections[3]
            user = self.pick([f"Add a slide about {extra_t.lower()}", f"can you also add {extra_t.lower()}", f"Add one more slide: {extra_t}"])
            return self.document(title=title, fmt="pptx", n=4, user=user, history=self.history_of(c1), category="followup")
        if k == 3:  # directions to a place from the list
            c1 = self.places()
            places = self.first_result(c1)["places"]
            i = self.r.randrange(len(places))
            p = places[i]
            ordinal = ["first", "second", "third", "fourth", "fifth"][i]
            mode = "walking" if p["distance_km"] < 1.5 else "driving"
            user = self.pick([f"How do I get to the {ordinal} one?", f"directions to {p['name']}", f"Take me to {p['name']}"])
            mins = max(2, round(p["distance_km"] / (4.8 if mode == "walking" else 25) * 60))
            res = {"origin": "Current location", "destination": f"{p['name']}, {p['address']}", "mode": mode, "distance_km": p["distance_km"], "duration_min": mins,
                   "steps": [{"instruction": f"Head {self.pick(['north', 'east', 'south', 'west'])}", "distance_m": 120}, {"instruction": f"Arrive at {p['name']}", "distance_m": 0}],
                   "maps_url": "https://www.google.com/maps/dir/?api=1&destination=x"}
            answer = f"**{p['name']}** is about **{mins} min** away by {'foot' if mode == 'walking' else 'car'} ({p['distance_km']} km).\n\n1. {res['steps'][0]['instruction']}\n2. Arrive at {p['name']}\n\n[Open the route in Google Maps]({res['maps_url']})"
            return self.conv("followup", user, [("Getting directions", f"The user wants directions to {p['name']} from the list. I'll start from their location.", "get_directions",
                                                 {"origin": "my location", "destination": f"{p['name']}, {p['address']}", "mode": mode}, res)],
                             ("Route ready", "I'll give the time first."), answer, {"tool": "get_directions", "args": {"destination": re.escape(p["name"])}, "answer": [str(mins)]}, history=self.history_of(c1))
        if k == 4:  # rerun a calculation with a new number
            p, rate, y1, y2 = self.r.choice([1000, 5000, 20000, 100000]), self.r.choice([6, 7, 8, 10, 12]), self.r.randint(2, 10), self.r.randint(11, 30)
            def calc(years):
                code = f"principal = {p}\nrate = {rate} / 100\nyears = {years}\namount = principal * (1 + rate) ** years\nprint(f\"Final amount: {{amount:,.2f}}\")\nprint(f\"Interest earned: {{amount - principal:,.2f}}\")"
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    exec(code, {})
                return code, out.getvalue()
            code1, out1 = calc(y1)
            hist = [{"role": "user", "content": f"Calculate compound interest on {p} at {rate}% for {y1} years"}, {"role": "assistant", "content": f"{self._lead(out1.splitlines()[0])}\n\n```\n{out1.strip()}\n```"}]
            code2, out2 = calc(y2)
            ok = {"ran": True, "exit_code": 0, "timed_out": False, "stdout": out2, "stderr": "", "images_shown_to_user": []}
            return self.conv("followup", self.pick([f"What about {y2} years?", f"now for {y2} years", f"And if I keep it for {y2} years?"]),
                             [("Recalculating", f"Same calculation with {y2} years instead of {y1}.", "run_code", {"language": "python", "code": code2}, ok)],
                             ("Reading the output", "I'll state the new result first."), f"{self._lead(out2.splitlines()[0])}\n\n```\n{out2.strip()}\n```",
                             {"tool": "run_code", "answer": [re.escape(out2.splitlines()[0].split(': ')[1])]}, history=hist)
        # k == 5: turn a search answer into slides
        title, sections = self.pick(list(self.topics.items()))
        n = self.r.randint(2, 6)
        res1 = [{"title": f"{title}: what's new", "url": f"https://news{n}.example.org/{title.lower().replace(' ', '-')}", "snippet": sections[0][1][0] + "."}]
        hist = [{"role": "user", "content": f"What's new about {title.lower()}?"}, {"role": "assistant", "content": f"Here's a quick summary: {sections[0][1][0]} ([source]({res1[0]['url']}))."}]
        return self.document(title=title, fmt="pptx", n=self.r.randint(3, 5), user=self.pick(["Turn that into a presentation", "make slides about this", "Can you make a ppt on it?"]), history=hist, category="followup")

    def direct(self):
        asks, answer = self.pick(self.qa)
        user = self.pick(asks)
        sys_msg, _ = self.system()
        key = re.escape(re.findall(r"\*\*([^*]+)\*\*", answer)[0]) if "**" in answer else ".+"
        msgs = [sys_msg, {"role": "user", "content": self.noisy(user)}, {"role": "assistant", "content": self.think(self.pick(["Understanding the question", "Answering directly"]), "I can answer this from what I know, so no tool is needed.") + answer}]
        return {"category": "no_tool", "messages": msgs, "mock": {}, "expect": {"tool": None, "answer": [key]}, "train_from": 1}

    def unsupported(self):
        asks, answer = self.pick(self.unsupported_bank)
        sys_msg, _ = self.system()
        msgs = [sys_msg, {"role": "user", "content": self.noisy(self.pick(asks))}, {"role": "assistant", "content": self.think("Checking what I can do", "I don't have a tool for this, so I'll explain and offer what I can do.") + answer}]
        return {"category": "unsupported", "messages": msgs, "mock": {}, "expect": {"tool": None, "answer": ["can't|cannot"]}, "train_from": 1}


# Share of each category in the data.
MIX = [("weather", 0.09), ("search", 0.14), ("location", 0.02), ("places", 0.08), ("directions", 0.05), ("image", 0.10),
       ("document", 0.08), ("notes_slides", 0.02), ("code_file", 0.06), ("run_code", 0.10), ("chart", 0.03), ("multi", 0.03),
       ("direct", 0.06), ("simple_math", 0.03), ("clarify", 0.02), ("unsupported", 0.02), ("followup", 0.05), ("tool_error", 0.02)]


def generate(n, seed, held_out):
    """Yield n distinct conversations (no exact duplicates; unique prompts for the test set)."""
    g = Gen(seed, held_out)
    names, weights = zip(*MIX)
    seen = set()
    made = attempts = 0
    while made < n and attempts < n * 50:
        attempts += 1
        ex = getattr(g, g.r.choices(names, weights)[0])()
        m = ex["messages"]
        # The system prompt (date/name) doesn't make a conversation different.
        key = "\x1e".join(x["content"] for x in m[1:ex["train_from"] + 1] if x["role"] == "user") if held_out else hash(json.dumps(m[1:], ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        made += 1
        yield ex
    if made < n:
        print(f"  note: only {made} distinct conversations possible here (asked for {n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=100000)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    counts = {}
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for i, ex in enumerate(generate(a.train, a.seed, held_out=False), 1):
            counts[ex["category"]] = counts.get(ex["category"], 0) + 1
            f.write(json.dumps({"category": ex["category"], "messages": ex["messages"], "train_from": ex["train_from"]}, ensure_ascii=False) + "\n")
            if i % 10000 == 0:
                print(f"  {i:,} / {a.train:,}", flush=True)
    total = sum(counts.values())
    print(f"train: {total:,} conversations -> {a.out}/train.jsonl")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    tests = 0
    with open(os.path.join(a.out, "test.jsonl"), "w", encoding="utf-8") as f:
        for ex in generate(a.test, a.seed + 1000, held_out=True):
            t = ex["train_from"]
            f.write(json.dumps({"category": ex["category"], "messages": ex["messages"][:t + 1], "mock": ex["mock"], "expect": ex["expect"], "reference": ex["messages"][t + 1:]}, ensure_ascii=False) + "\n")
            tests += 1
    print(f"test:  {tests:,} held-out cases -> {a.out}/test.jsonl")


if __name__ == "__main__":
    main()
