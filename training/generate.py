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

PROMPT = json.load(open(os.path.join(HERE, "system_prompt.json"), encoding="utf-8"))
TOOLS = set(PROMPT["tools"])


class Gen:
    def __init__(self, seed, held_out):
        self.r = random.Random(seed)
        self.held_out = held_out
        self.cities = C.HELD_OUT_CITIES if held_out else C.CITIES
        self.subjects = C.HELD_OUT_IMAGE_SUBJECTS if held_out else C.IMAGE_SUBJECTS
        self.topics = C.HELD_OUT_DOC_TOPICS if held_out else C.DOC_TOPICS
        self.code = C.HELD_OUT_CODE_TASKS if held_out else C.CODE_TASKS

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

    def conv(self, category, user, steps, final_think, answer, expect=None):
        """steps: list of (think_title, think_text, tool, args, tool_result)."""
        sys_msg, _ = self.system()
        msgs = [sys_msg, {"role": "user", "content": user}]
        mock = {}
        for title, text, tool, args, result in steps:
            msgs.append({"role": "assistant", "content": self.think(title, text) + self.call(tool, args)})
            msgs.append({"role": "user", "content": self.response(result)})
            mock.setdefault(tool, []).append(result)
        msgs.append({"role": "assistant", "content": self.think(*final_think) + answer})
        return {"category": category, "messages": msgs, "mock": mock, "expect": expect or {}}

    # ───────── categories ─────────
    def weather(self):
        city, country = self.pick(self.cities)
        cond, icon = self.pick(C.CONDITIONS)
        temp = self.r.randint(-2, 40) if icon == "snow" else self.r.randint(8, 41)
        if icon == "snow":
            temp = self.r.randint(-6, 2)
        feels = temp + self.r.randint(-3, 4)
        hum = self.r.randint(30, 95)
        wind = self.r.randint(3, 35)
        no_loc = self.r.random() < 0.15
        days = 7 if self.r.random() < 0.3 else 5
        place = f"{city}, {country}"
        asks = (
            ["What's the weather today?", "today's weather", "Is it going to rain today?", "How's the weather outside?"]
            if no_loc
            else [f"What's the weather in {city}?", f"weather {city}", f"Will it rain in {city} tomorrow?", f"How hot is it in {city} right now?",
                  f"Weather forecast for {city} this week", f"Do I need an umbrella in {city} today?"]
        )
        user = self.pick(asks)
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
        return self.conv("weather", user, [("Checking the weather", f"The user wants the weather{'' if no_loc else ' in ' + city}. I'll call get_weather{'' if no_loc else ''}.",
                                            "get_weather", args, result)],
                         ("Reading the forecast", f"It's {temp}°C and {cond.lower()}. I'll give the current conditions first, then the forecast."), "\n".join(lines), expect)

    def search(self):
        kind = self.r.random()
        city, _ = self.pick(self.cities)
        n = self.r.randint(2, 9)
        if kind < 0.25:
            item = self.pick(["gold", "silver", "petrol", "diesel", "Bitcoin", "the US dollar in rupees"])
            user = self.pick([f"What is the price of {item} today?", f"{item} price today", f"How much is {item} right now?"])
            query = f"{item} price today"
            price = round(self.r.uniform(80, 9000), 2)
            fact = f"{price:,}"
            results = [{"title": f"{item.title()} Price Today – Live Rates", "url": f"https://www.marketrates{n}.com/{item.split()[0].lower()}", "snippet": f"Today's {item} price is {fact}, up {round(self.r.uniform(0.1, 2.5), 1)}% from yesterday."},
                       {"title": f"{item.title()} rate history and trends", "url": f"https://finance.example{n}.org/{item.split()[0].lower()}-rates", "snippet": f"Track {item} prices over the last 30 days. Latest: {fact}."}]
            answer = f"The {item} price today is **{fact}**, according to [{results[0]['title']}]({results[0]['url']}). Prices change through the day, so check the latest rate before you buy."
        elif kind < 0.5:
            team1, team2 = self.r.sample(["Chennai Super Kings", "Mumbai Indians", "Royal Challengers", "Kolkata Knight Riders", "Real Madrid", "Barcelona", "Manchester City", "Arsenal", "India", "Australia", "England"], 2)
            event = self.pick(["final", "match yesterday", "last game", "semi-final"])
            user = self.pick([f"Who won the {team1} vs {team2} {event}?", f"{team1} vs {team2} result", f"Did {team1} win against {team2}?"])
            query = f"{team1} vs {team2} {event} result"
            winner = self.pick([team1, team2])
            margin = self.pick(["by 6 wickets", "by 23 runs", "2–1", "3–0", "on penalties"])
            results = [{"title": f"{team1} vs {team2}: {winner} win {margin}", "url": f"https://sports{n}.example.com/{team1.split()[0].lower()}-{team2.split()[0].lower()}", "snippet": f"{winner} won the {event} {margin} in front of a packed stadium."},
                       {"title": f"Highlights: {team1} vs {team2}", "url": f"https://www.sportsnews{n}.net/highlights", "snippet": f"Watch the highlights as {winner} sealed victory {margin}."}]
            answer = f"**{winner} won**, {margin}, according to [{results[0]['title']}]({results[0]['url']})."
        elif kind < 0.75:
            company = self.pick(["Apple", "Samsung", "Google", "Tesla", "Tata Motors", "Infosys", "Microsoft", "OnePlus", "Nothing", "Sony"])
            product = self.pick(["new phone", "latest laptop", "next update", "new electric car", "latest smartwatch"])
            user = self.pick([f"When is the {company} {product} launching?", f"Latest news about {company}", f"What did {company} announce this week?"])
            query = f"{company} {product} launch" if "launch" in user else f"{company} latest news"
            month = self.pick(["March", "June", "September", "October", "November"])
            results = [{"title": f"{company} confirms {product} for {month}", "url": f"https://technews{n}.example.com/{company.lower().replace(' ', '-')}", "snippet": f"{company} said its {product} will arrive in {month}, with prices starting around ${self.r.randint(299, 1499)}."},
                       {"title": f"Everything we know about the {company} {product}", "url": f"https://www.gadgets{n}.org/{company.lower().replace(' ', '-')}-guide", "snippet": f"Expected features, price and release date for the {company} {product}."}]
            answer = f"{company} says its **{product} arrives in {month}** ([{results[0]['title']}]({results[0]['url']})). {results[0]['snippet'].split(', with ')[1].capitalize() if ', with ' in results[0]['snippet'] else ''}\n\nMore details: [{results[1]['title']}]({results[1]['url']})."
            fact = month
        else:
            event = self.pick(["concerts", "events", "festivals", "things to do"])
            user = self.pick([f"What {event} are happening in {city} this weekend?", f"{event} in {city} this weekend"])
            query = f"{event} in {city} this weekend"
            names = self.r.sample(["Jazz in the Park", "Night Food Market", "Indie Music Fest", "Art Walk", "Comedy Night", "Book Fair", "Craft Beer Festival", "Classical Evening"], 3)
            results = [{"title": f"{nm} – {city}", "url": f"https://events{n}.example.com/{city.lower().replace(' ', '-')}/{i}", "snippet": f"{nm} on Saturday from {self.r.randint(4, 8)} pm. Tickets from {self.r.randint(5, 40)}."} for i, nm in enumerate(names)]
            answer = f"Here's what's on in {city} this weekend:\n\n" + "\n".join(f"- **[{x['title'].split(' – ')[0]}]({x['url']})**: {x['snippet']}" for x in results) + "\n\nCheck the event pages for tickets and times."
            fact = names[0]
        if kind < 0.25:
            fact = fact
        elif kind < 0.5:
            fact = winner
        steps = [("Searching the web", "This needs current information, so I'll search the web.", "search_web", {"query": query}, {"query": query, "results": results})]
        # Sometimes read the top page for details.
        if self.r.random() < 0.2:
            page = {"url": results[0]["url"], "title": results[0]["title"], "text": results[0]["snippet"] + " " + "Officials shared more details in a statement on the same day."}
            steps.append(("Reading the top result", "I'll open the first result for details.", "read_webpage", {"url": results[0]["url"]}, page))
        # Sometimes the first search fails and is retried.
        if self.r.random() < 0.1:
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
        asks, query, osm, parts = self.pick(C.PLACE_KINDS)
        city, country = self.pick(self.cities)
        near_me = self.r.random() < 0.4
        what = self.pick(asks)
        if query == "restaurant" and self.r.random() < 0.4:
            cuisine = self.pick(C.CUISINES)
            what, query = f"{cuisine} restaurants", f"{cuisine} restaurant"
        user = self.pick([f"Find {what} near me", f"Any {what} nearby?", f"Where can I find {what} around here?"] if near_me else
                         [f"Find {what} in {city}", f"Best {what} in {city}", f"Show me {what} near {city} station"])
        near = "my location" if near_me else (f"{city} station" if "station" in user else city)
        count = self.r.randint(3, 5)
        places = []
        for i in range(count):
            nm = f"{self.pick(parts)} {self.pick(['House', 'Corner', 'Point', 'Hub', 'Co.', 'Place', '& Co'])}" if osm not in ("atm", "fuel", "park") else f"{self.pick(parts)} {query.upper() if osm == 'atm' else query.title()}"
            addr = f"{self.r.randint(1, 220)} {self.pick(C.STREETS)}, {city}"
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
        steps = [{"instruction": f"Head {self.pick(['north', 'south', 'east', 'west'])} on {self.pick(C.STREETS)}", "distance_m": self.r.randint(100, 900)},
                 {"instruction": f"Turn {self.pick(['left', 'right'])} onto {self.pick(C.STREETS)}", "distance_m": self.r.randint(300, 4000)},
                 {"instruction": f"Turn {self.pick(['left', 'right'])} onto {self.pick(C.STREETS)}", "distance_m": self.r.randint(200, 3000)},
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
            subj = self.pick(self.subjects)
            style, desc = self.pick(C.IMAGE_STYLES)
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

    def document(self):
        title, sections = self.pick(list(self.topics.items()))
        fmt = self.pick(["pptx", "pptx", "pdf", "docx"])
        n = self.r.randint(3, len(sections))
        word = {"pptx": self.pick(["presentation", "slide deck", "PPT", "slides"]), "pdf": self.pick(["PDF", "PDF report", "PDF document"]), "docx": self.pick(["Word document", "docx", "Word file"])}[fmt]
        user = self.pick([f"Make a {word} on {title.lower()}", f"Create a {n}-{'slide' if fmt == 'pptx' else 'section'} {word} about {title.lower()}", f"I need a {word} explaining {title.lower()}"])
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
        return self.conv("document", user, [("Planning the document", f"The user wants a {fmt} about {title.lower()}. I'll outline {n} {unit} and call create_document.", "create_document", args, result)],
                         ("Document ready", "The file was delivered; I'll summarize what's inside."), answer, expect)

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

    def run_code(self):
        user, code, stdout = self._program()
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

    def chart(self):
        func, label = self.pick([("x ** 2", "y = x²"), ("2 ** x", "y = 2^x"), ("x ** 3 - 3 * x", "y = x³ − 3x"), ("10 * x + 5", "y = 10x + 5")])
        user = self.pick([f"Plot {label} from 0 to 10", f"Make a chart of {label}", f"Draw a graph of {label}"])
        code = f"import matplotlib.pyplot as plt\nxs = [x / 10 for x in range(0, 101)]\nys = [{func.replace('x', 'x_')} for x_ in xs]\nplt.plot(xs, ys)\nplt.title(\"{label}\")\nplt.xlabel(\"x\")\nplt.ylabel(\"y\")\nplt.grid(True)\nplt.savefig(\"plot.png\")\nprint(\"Saved plot.png\")"
        result = {"ran": True, "exit_code": 0, "timed_out": False, "stdout": "Saved plot.png\n", "stderr": "", "images_shown_to_user": ["plot.png"]}
        answer = f"Here's the graph of **{label}** for x from 0 to 10. It's shown above, and you can save it from the output card."
        return self.conv("chart", user, [("Making the chart", "I'll plot it with matplotlib and save the image.", "run_code", {"language": "python", "code": code}, result)],
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

    def direct(self):
        asks, answer = self.pick(C.QA)
        user = self.pick(asks)
        sys_msg, _ = self.system()
        key = re.escape(re.findall(r"\*\*([^*]+)\*\*", answer)[0]) if "**" in answer else ".+"
        msgs = [sys_msg, {"role": "user", "content": user}, {"role": "assistant", "content": self.think(self.pick(["Understanding the question", "Answering directly"]), "I can answer this from what I know, so no tool is needed.") + answer}]
        return {"category": "no_tool", "messages": msgs, "mock": {}, "expect": {"tool": None, "answer": [key]}}

    def unsupported(self):
        asks, answer = self.pick(C.UNSUPPORTED)
        sys_msg, _ = self.system()
        msgs = [sys_msg, {"role": "user", "content": self.pick(asks)}, {"role": "assistant", "content": self.think("Checking what I can do", "I don't have a tool for this, so I'll explain and offer what I can do.") + answer}]
        return {"category": "unsupported", "messages": msgs, "mock": {}, "expect": {"tool": None, "answer": ["can't|cannot"]}}


# Share of each category in the data.
MIX = [("weather", 0.11), ("search", 0.16), ("location", 0.03), ("places", 0.09), ("directions", 0.06), ("image", 0.11),
       ("document", 0.10), ("code_file", 0.07), ("run_code", 0.11), ("chart", 0.03), ("multi", 0.04), ("direct", 0.07), ("unsupported", 0.02)]


def build(n, seed, held_out):
    g = Gen(seed, held_out)
    names, weights = zip(*MIX)
    out = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        ex = getattr(g, g.r.choices(names, weights)[0])()
        prompt = ex["messages"][1]["content"]
        # Test prompts are unique so each case is a distinct question.
        if held_out and prompt in seen:
            continue
        seen.add(prompt)
        out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train = build(a.train, a.seed, held_out=False)
    test = build(a.test, a.seed + 1000, held_out=True)
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps({"category": ex["category"], "messages": ex["messages"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "test.jsonl"), "w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps({"category": ex["category"], "messages": ex["messages"][:2], "mock": ex["mock"], "expect": ex["expect"], "reference": ex["messages"][2:]}, ensure_ascii=False) + "\n")
    counts = {}
    for ex in train:
        counts[ex["category"]] = counts.get(ex["category"], 0) + 1
    print(f"train: {len(train)} conversations -> {a.out}/train.jsonl")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"test:  {len(test)} held-out cases -> {a.out}/test.jsonl")


if __name__ == "__main__":
    main()
