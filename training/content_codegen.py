"""Code-task families: each yields many distinct, correct programs.

Every variant is checked by training/check_code.py (run, compile or parse).
A task is a dict: asks, filename, desc, content, explain (same shape as
content.CODE_TASKS). `variants(held_out)` lists every task in a split.
"""
import itertools
import json
import re


def art(w):
    """'a pub', 'an airline', 'an 18%', 'a user', 'an hour'."""
    s = str(w).lower()
    an = (s[:1] in "aeiou" and not s.startswith(("uni", "use", "usu", "eu", "one"))) or s.startswith(("hour", "honest", "8", "11", "18"))
    return f"{'an' if an else 'a'} {w}"


# ───────── unit converters (Python) ─────────
CONVERSIONS = [
    ("kilometres", "miles", "km", "mi", 0.621371, 0), ("miles", "kilometres", "mi", "km", 1.609344, 0),
    ("kilograms", "pounds", "kg", "lb", 2.20462, 0), ("pounds", "kilograms", "lb", "kg", 0.453592, 0),
    ("Celsius", "Fahrenheit", "°C", "°F", 1.8, 32), ("Fahrenheit", "Celsius", "°F", "°C", 5 / 9, -32 * 5 / 9),
    ("litres", "gallons", "L", "gal", 0.264172, 0), ("inches", "centimetres", "in", "cm", 2.54, 0),
    ("feet", "metres", "ft", "m", 0.3048, 0), ("metres", "feet", "m", "ft", 3.28084, 0),
    ("hours", "minutes", "h", "min", 60, 0), ("megabytes", "gigabytes", "MB", "GB", 1 / 1024, 0),
    ("square feet", "square metres", "sq ft", "m²", 0.092903, 0), ("knots", "km/h", "kn", "km/h", 1.852, 0),
]
HELD_OUT_CONVERSIONS = [("ounces", "grams", "oz", "g", 28.3495, 0), ("acres", "hectares", "ac", "ha", 0.404686, 0)]


def converter(c):
    a, b, ua, ub, factor, offset = c
    slug = re.sub(r"[^a-z]+", "_", f"{a}_to_{b}".lower()).strip("_")
    code = (f"import sys\n\n\ndef convert(value):\n    \"\"\"Convert {a} to {b}.\"\"\"\n    return value * {factor!r} + {offset!r}\n\n\n"
            f"if __name__ == \"__main__\":\n    value = float(sys.argv[1]) if len(sys.argv) > 1 else float(input(\"{a.capitalize()}: \"))\n"
            f"    print(f\"{{value:g}} {ua} = {{convert(value):.2f}} {ub}\")\n")
    return {"asks": [f"Write a Python script to convert {a} to {b}", f"{a} to {b} converter in python", f"Make {art(ua)} to {ub} converter",
                     f"python program that converts {a} into {b}"],
            "filename": f"{slug}.py", "desc": f"converts {a} to {b}", "content": code,
            "explain": f"Run `python {slug}.py 10` to convert 10 {a}, or run it without a number and type one in.", "check_args": ["10"]}


# ───────── data models (Python / TypeScript / Java) ─────────
ENTITIES = {
    "Student": [("name", "str", "Asha"), ("roll_number", "int", 12), ("grade", "str", "10A"), ("average", "float", 88.5)],
    "Book": [("title", "str", "Dune"), ("author", "str", "Frank Herbert"), ("year", "int", 1965), ("price", "float", 399.0)],
    "Product": [("name", "str", "Headphones"), ("sku", "str", "HP-100"), ("price", "float", 2499.0), ("stock", "int", 25)],
    "Employee": [("name", "str", "Ravi"), ("role", "str", "Engineer"), ("salary", "float", 85000.0), ("years", "int", 4)],
    "Car": [("make", "str", "Tata"), ("model", "str", "Nexon"), ("year", "int", 2023), ("mileage_km", "float", 12000.0)],
    "Movie": [("title", "str", "Interstellar"), ("director", "str", "Christopher Nolan"), ("year", "int", 2014), ("rating", "float", 8.7)],
    "Recipe": [("name", "str", "Masala Dosa"), ("servings", "int", 4), ("minutes", "int", 45), ("vegetarian", "bool", True)],
    "Task": [("title", "str", "Buy milk"), ("priority", "int", 2), ("done", "bool", False), ("due", "str", "2025-06-01")],
    "Song": [("title", "str", "Vaathi Coming"), ("artist", "str", "Anirudh"), ("seconds", "int", 228), ("liked", "bool", True)],
    "Order": [("order_id", "str", "ORD-1001"), ("customer", "str", "Meera"), ("total", "float", 1299.5), ("paid", "bool", True)],
    "Patient": [("name", "str", "Arjun"), ("age", "int", 34), ("blood_group", "str", "O+"), ("admitted", "bool", False)],
    "Flight": [("number", "str", "AI-202"), ("origin", "str", "MAA"), ("destination", "str", "DEL"), ("seats", "int", 180)],
}
HELD_OUT_ENTITIES = {
    "Hotel": [("name", "str", "Sea Breeze"), ("city", "str", "Goa"), ("rooms", "int", 40), ("rating", "float", 4.4)],
    "Course": [("code", "str", "CS101"), ("title", "str", "Intro to Programming"), ("credits", "int", 4), ("online", "bool", True)],
}


def camel(s):
    head, *rest = s.split("_")
    return head + "".join(w.capitalize() for w in rest)


def snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def model_task(entity, fields, lang):
    if lang == "python":
        lines = [f"    {f}: {t}" for f, t, _ in fields]
        args = ", ".join(f"{json.dumps(v) if t == 'str' else v}" for _, t, v in fields)
        code = (f"from dataclasses import dataclass, asdict\n\n\n@dataclass\nclass {entity}:\n" + "\n".join(lines) +
                f"\n\n    def to_dict(self):\n        return asdict(self)\n\n\nif __name__ == \"__main__\":\n    item = {entity}({args})\n    print(item)\n    print(item.to_dict())\n")
        fn, how = f"{snake(entity)}.py", f"Run `python {snake(entity)}.py` to see an example. `to_dict()` is handy for saving it as JSON."
    elif lang == "typescript":
        ts = {"str": "string", "int": "number", "float": "number", "bool": "boolean"}
        props = "\n".join(f"  {camel(f)}: {ts[t]};" for f, t, _ in fields)
        ctor = ", ".join(f"public {camel(f)}: {ts[t]}" for f, t, _ in fields)
        vals = ", ".join(json.dumps(v) for _, _, v in fields)
        code = (f"export interface {entity}Data {{\n{props}\n}}\n\nexport class {entity} implements {entity}Data {{\n  constructor({ctor}) {{}}\n\n"
                f"  toJSON(): {entity}Data {{\n    return {{ " + ", ".join(f"{camel(f)}: this.{camel(f)}" for f, _, _ in fields) + " };\n  }\n}\n\n"
                f"const example = new {entity}({vals});\nconsole.log(JSON.stringify(example));\n")
        fn, how = f"{snake(entity)}.ts", f"Compile with `tsc {snake(entity)}.ts` or run with `npx tsx {snake(entity)}.ts`."
    else:  # java
        jt = {"str": "String", "int": "int", "float": "double", "bool": "boolean"}
        decl = "\n".join(f"    private final {jt[t]} {camel(f)};" for f, t, _ in fields)
        params = ", ".join(f"{jt[t]} {camel(f)}" for f, t, _ in fields)
        assigns = "\n".join(f"        this.{camel(f)} = {camel(f)};" for f, _, _ in fields)
        getters = "\n\n".join(f"    public {jt[t]} {'is' if t == 'bool' else 'get'}{camel(f)[0].upper() + camel(f)[1:]}() {{\n        return {camel(f)};\n    }}" for f, t, _ in fields)
        tostr = " + \", \" + ".join(f"\"{camel(f)}=\" + {camel(f)}" for f, _, _ in fields)
        vals = ", ".join(json.dumps(v) if t == "str" else ("true" if v is True else "false" if v is False else str(v)) for _, t, v in fields)
        code = (f"public class {entity} {{\n{decl}\n\n    public {entity}({params}) {{\n{assigns}\n    }}\n\n{getters}\n\n"
                f"    @Override\n    public String toString() {{\n        return \"{entity}{{\" + {tostr} + \"}}\";\n    }}\n\n"
                f"    public static void main(String[] args) {{\n        System.out.println(new {entity}({vals}));\n    }}\n}}\n")
        fn, how = f"{entity}.java", f"Compile and run with `javac {entity}.java && java {entity}`."
    names = {"python": "Python", "typescript": "TypeScript", "java": "Java"}[lang]
    field_list = ", ".join(f.replace("_", " ") for f, _, _ in fields)
    return {"asks": [f"Create {art(entity)} class in {names} with {field_list}", f"{names} model for {art(entity.lower())}", f"Write {art(entity)} {('dataclass' if lang == 'python' else 'class')} in {names}",
                     f"I need {art(names)} class to store {entity.lower()} details"],
            "filename": fn, "desc": f"{art(entity)} {('dataclass' if lang == 'python' else 'class')} with {field_list}", "content": code, "explain": how, "lang": lang}


# ───────── SQL schemas (checked in SQLite) ─────────
SQL_DOMAINS = {
    "library": ("books", "members", "loans"), "school": ("students", "courses", "enrollments"), "shop": ("products", "customers", "orders"),
    "hospital": ("doctors", "patients", "appointments"), "blog": ("authors", "posts", "comments"), "gym": ("trainers", "members", "sessions"),
    "cinema": ("movies", "screens", "bookings"), "restaurant": ("dishes", "tables", "orders"),
}
HELD_OUT_SQL = {"airline": ("flights", "passengers", "tickets")}


def sql_task(domain, tables):
    a, b, link = tables
    code = (f"-- {domain.title()} database\nCREATE TABLE {a} (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL,\n  created_at TEXT DEFAULT CURRENT_TIMESTAMP\n);\n\n"
            f"CREATE TABLE {b} (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL,\n  email TEXT UNIQUE\n);\n\n"
            f"CREATE TABLE {link} (\n  id INTEGER PRIMARY KEY,\n  {a[:-1]}_id INTEGER NOT NULL REFERENCES {a}(id),\n  {b[:-1]}_id INTEGER NOT NULL REFERENCES {b}(id),\n  date TEXT NOT NULL\n);\n\n"
            f"INSERT INTO {a} (name) VALUES ('{a[:-1].title()} A'), ('{a[:-1].title()} B');\n"
            f"INSERT INTO {b} (name, email) VALUES ('Asha', 'asha@example.com'), ('Ben', 'ben@example.com');\n"
            f"INSERT INTO {link} ({a[:-1]}_id, {b[:-1]}_id, date) VALUES (1, 1, '2025-01-10'), (2, 1, '2025-01-12'), (1, 2, '2025-01-15');\n\n"
            f"-- How many {link} each {b[:-1]} has\nSELECT {b}.name, COUNT({link}.id) AS total\nFROM {b}\nLEFT JOIN {link} ON {link}.{b[:-1]}_id = {b}.id\nGROUP BY {b}.id\nORDER BY total DESC;\n")
    return {"asks": [f"Design a SQL database for {art(domain)}", f"sql schema for {art(domain)} management system", f"Create tables for {art(domain)} app in SQL",
                     f"Write SQL to create {art(domain)} database with sample data"],
            "filename": f"{domain}.sql", "desc": f"{art(domain)} database with {a}, {b} and {link} tables, sample data and a summary query",
            "content": code, "explain": f"Try it with `sqlite3 {domain}.db < {domain}.sql`. The last query counts {link} per {b[:-1]}."}


# ───────── web pages ─────────
PAGE_TYPES = ["landing page", "portfolio", "restaurant menu", "event invitation", "coming soon page"]
BUSINESSES = ["Bean There Café", "Nimbus Labs", "Sweet Crumbs Bakery", "Iron Pulse Gym", "Ink & Leaf Books", "Wanderly Travel", "PixelForge Studio",
              "Green Leaf Organics", "Spice Route Kitchen", "Blue Wave Surf School", "Lotus Yoga", "TechNest Repairs", "Aurora Photography",
              "Petals Florist", "Summit Coaching", "Orbit Robotics Club"]
HELD_OUT_BUSINESSES = ["Maple Street Diner", "Cosmo Dance Academy"]
THEMES = {"purple": ("#6d5dfc", "#f4f2ff"), "ocean": ("#0ea5e9", "#eef8fd"), "forest": ("#16a34a", "#effaf2"), "sunset": ("#f97316", "#fff4ec"),
          "rose": ("#e11d48", "#fff0f3"), "slate": ("#334155", "#f1f5f9")}


def page_task(kind, biz, theme):
    color, bg = THEMES[theme]
    slug = re.sub(r"[^a-z0-9]+", "-", biz.lower()).strip("-")
    body = {
        "landing page": f"<header><h1>{biz}</h1><p>Welcome! We're glad you're here.</p><a class=\"btn\" href=\"#contact\">Get in touch</a></header>\n<section class=\"grid\"><div class=\"card\"><h3>Quality</h3><p>Made with care.</p></div><div class=\"card\"><h3>Fast</h3><p>Quick and friendly service.</p></div><div class=\"card\"><h3>Local</h3><p>Proudly part of the community.</p></div></section>",
        "portfolio": f"<header><h1>{biz}</h1><p>Selected work</p></header>\n<section class=\"grid\">" + "".join(f"<div class=\"card\"><h3>Project {i}</h3><p>A short description of project {i}.</p></div>" for i in range(1, 5)) + "</section>",
        "restaurant menu": f"<header><h1>{biz}</h1><p>Our menu</p></header>\n<section class=\"grid\">" + "".join(f"<div class=\"card\"><h3>{d}</h3><p>₹{p}</p></div>" for d, p in [("Starter", 180), ("Main course", 320), ("Dessert", 150), ("Drinks", 90)]) + "</section>",
        "event invitation": f"<header><h1>You're invited!</h1><p>{biz} celebrates its anniversary</p><p id=\"countdown\"></p><a class=\"btn\" href=\"#rsvp\">RSVP</a></header>",
        "coming soon page": f"<header><h1>{biz}</h1><p>Something exciting is coming soon.</p><p id=\"countdown\"></p><a class=\"btn\" href=\"#notify\">Notify me</a></header>",
    }[kind]
    script = ("\n<script>\nconst target = new Date(Date.now() + 14 * 864e5);\nfunction tick() {\n  const s = Math.max(0, Math.floor((target - Date.now()) / 1000));\n"
              "  document.getElementById(\"countdown\").textContent = `${Math.floor(s / 86400)}d ${Math.floor(s % 86400 / 3600)}h ${Math.floor(s % 3600 / 60)}m`;\n}\ntick();\nsetInterval(tick, 1000);\n</script>"
              if "countdown" in body else "")
    code = (f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{biz}</title>\n<style>\n"
            f"*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui,sans-serif;background:{bg};color:#1f2330}}\n"
            f"header{{padding:72px 24px;text-align:center;background:{color};color:#fff}}h1{{margin:0 0 8px;font-size:clamp(28px,6vw,48px)}}\n"
            f".btn{{display:inline-block;margin-top:16px;padding:12px 26px;border-radius:999px;background:#fff;color:{color};font-weight:700;text-decoration:none}}\n"
            f".grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;max-width:960px;margin:40px auto;padding:0 20px}}\n"
            f".card{{background:#fff;padding:20px;border-radius:14px;box-shadow:0 4px 18px rgba(0,0,0,.06)}}\n</style>\n</head>\n<body>\n{body}{script}\n</body>\n</html>\n")
    return {"asks": [f"Make {art(kind)} for {biz}", f"Create {art(theme)} themed {kind} for {biz}", f"html {kind} for {biz}", f"Build a simple {kind} website for {biz} in {theme} colors"],
            "filename": f"{slug}.html", "desc": f"{art(theme)}-themed {kind} for {biz}", "content": code,
            "explain": "Open it in a browser. It works on phones too. Edit the text and the colors at the top of the `<style>` section."}


# ───────── REST APIs ─────────
RESOURCES = ["todos", "notes", "books", "products", "users", "contacts", "events", "recipes"]
HELD_OUT_RESOURCES = ["tickets"]


def api_task(resource, framework):
    one = resource[:-1]
    if framework == "express":
        code = (f"const express = require(\"express\");\n\nconst app = express();\napp.use(express.json());\n\nlet {resource} = [];\nlet nextId = 1;\n\n"
                f"app.get(\"/{resource}\", (req, res) => res.json({resource}));\n\n"
                f"app.get(\"/{resource}/:id\", (req, res) => {{\n  const item = {resource}.find((x) => x.id === Number(req.params.id));\n  item ? res.json(item) : res.status(404).json({{ error: \"Not found\" }});\n}});\n\n"
                f"app.post(\"/{resource}\", (req, res) => {{\n  const item = {{ id: nextId++, ...req.body }};\n  {resource}.push(item);\n  res.status(201).json(item);\n}});\n\n"
                f"app.put(\"/{resource}/:id\", (req, res) => {{\n  const i = {resource}.findIndex((x) => x.id === Number(req.params.id));\n  if (i < 0) return res.status(404).json({{ error: \"Not found\" }});\n  {resource}[i] = {{ ...{resource}[i], ...req.body }};\n  res.json({resource}[i]);\n}});\n\n"
                f"app.delete(\"/{resource}/:id\", (req, res) => {{\n  {resource} = {resource}.filter((x) => x.id !== Number(req.params.id));\n  res.status(204).end();\n}});\n\n"
                f"app.listen(3000, () => console.log(\"API running on http://localhost:3000/{resource}\"));\n")
        return {"asks": [f"Make a REST API for {resource} in Express", f"express crud api for {resource}", f"Node.js API with endpoints for {resource}"],
                "filename": f"{resource}_api.js", "desc": f"an Express REST API with create, read, update and delete for {resource}", "content": code,
                "explain": f"Install with `npm install express`, run `node {resource}_api.js`, then try `curl localhost:3000/{resource}`. Data is kept in memory."}
    code = (f"from flask import Flask, jsonify, request\n\napp = Flask(__name__)\n{resource} = {{}}\nnext_id = 1\n\n\n"
            f"@app.get(\"/{resource}\")\ndef list_{resource}():\n    return jsonify(list({resource}.values()))\n\n\n"
            f"@app.get(\"/{resource}/<int:item_id>\")\ndef get_{one}(item_id):\n    item = {resource}.get(item_id)\n    return (jsonify(item), 200) if item else (jsonify(error=\"Not found\"), 404)\n\n\n"
            f"@app.post(\"/{resource}\")\ndef create_{one}():\n    global next_id\n    item = {{\"id\": next_id, **(request.get_json() or {{}})}}\n    {resource}[next_id] = item\n    next_id += 1\n    return jsonify(item), 201\n\n\n"
            f"@app.put(\"/{resource}/<int:item_id>\")\ndef update_{one}(item_id):\n    if item_id not in {resource}:\n        return jsonify(error=\"Not found\"), 404\n    {resource}[item_id].update(request.get_json() or {{}})\n    return jsonify({resource}[item_id])\n\n\n"
            f"@app.delete(\"/{resource}/<int:item_id>\")\ndef delete_{one}(item_id):\n    {resource}.pop(item_id, None)\n    return \"\", 204\n\n\n"
            f"if __name__ == \"__main__\":\n    app.run(debug=True)\n")
    return {"asks": [f"Make a REST API for {resource} in Flask", f"python flask crud api for {resource}", f"Build a Flask backend for {resource}"],
            "filename": f"{resource}_api.py", "desc": f"a Flask REST API with create, read, update and delete for {resource}", "content": code,
            "explain": f"Install with `pip install flask`, run `python {resource}_api.py`, then open http://127.0.0.1:5000/{resource}. Data is kept in memory."}


# ───────── JavaScript utilities (each prints a demo) ─────────
JS_UTILS = [
    ("chunk", "split an array into chunks", "function chunk(items, size) {\n  const out = [];\n  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));\n  return out;\n}\n\nconsole.log(JSON.stringify(chunk([1, 2, 3, 4, 5], 2)));\n"),
    ("unique", "remove duplicates from an array", "const unique = (items) => [...new Set(items)];\n\nconsole.log(unique([1, 2, 2, 3, 3, 3]));\n"),
    ("group_by", "group objects by a key", "function groupBy(items, key) {\n  return items.reduce((acc, item) => {\n    (acc[item[key]] ||= []).push(item);\n    return acc;\n  }, {});\n}\n\nconsole.log(JSON.stringify(groupBy([{ type: \"fruit\", name: \"apple\" }, { type: \"veg\", name: \"carrot\" }, { type: \"fruit\", name: \"mango\" }], \"type\")));\n"),
    ("capitalize", "capitalize every word", "const capitalize = (text) => text.replace(/\\b\\w/g, (c) => c.toUpperCase());\n\nconsole.log(capitalize(\"hello mnx world\"));\n"),
    ("slugify", "turn a title into a URL slug", "function slugify(text) {\n  return text.toLowerCase().trim().replace(/[^a-z0-9]+/g, \"-\").replace(/^-+|-+$/g, \"\");\n}\n\nconsole.log(slugify(\"  Hello World! This is Mnx \"));\n"),
    ("throttle", "throttle a function", "function throttle(fn, wait = 200) {\n  let last = 0;\n  return function (...args) {\n    const now = Date.now();\n    if (now - last >= wait) {\n      last = now;\n      fn.apply(this, args);\n    }\n  };\n}\n\nconst log = throttle((x) => console.log(\"called with\", x), 1000);\nlog(1);\nlog(2); // ignored: too soon\n"),
    ("deep_clone", "deep-copy an object", "const deepClone = (value) => structuredClone(value);\n\nconst original = { a: 1, nested: { b: [1, 2] } };\nconst copy = deepClone(original);\ncopy.nested.b.push(3);\nconsole.log(original.nested.b, copy.nested.b);\n"),
    ("format_currency", "format numbers as money", "const formatINR = (n) => new Intl.NumberFormat(\"en-IN\", { style: \"currency\", currency: \"INR\" }).format(n);\n\nconsole.log(formatINR(1234567.5));\n"),
    ("random_int", "get a random integer in a range", "const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;\n\nconsole.log(randomInt(1, 6));\n"),
    ("days_between", "count days between two dates", "function daysBetween(a, b) {\n  return Math.round(Math.abs(new Date(b) - new Date(a)) / 864e5);\n}\n\nconsole.log(daysBetween(\"2025-01-01\", \"2025-03-01\"));\n"),
    ("flatten", "flatten nested arrays", "const flatten = (items) => items.flat(Infinity);\n\nconsole.log(flatten([1, [2, [3, [4]]]]));\n"),
    ("word_count", "count words in text", "const wordCount = (text) => (text.trim().match(/\\S+/g) || []).length;\n\nconsole.log(wordCount(\"Mnx counts the words in this sentence\"));\n"),
]


def js_util_task(name, what, code):
    return {"asks": [f"Write a JavaScript function to {what}", f"js function to {what}", f"How do I {what} in JavaScript?"],
            "filename": f"{name}.js", "desc": f"a JavaScript helper to {what}", "content": code, "explain": f"Run `node {name}.js` to see an example."}


# ───────── Bash scripts ─────────
EXTS = ["log", "tmp", "jpg", "png", "pdf", "txt"]


def bash_task(kind, ext, days):
    if kind == "cleanup":
        code = (f"#!/usr/bin/env bash\nset -euo pipefail\n\ndir=\"${{1:-.}}\"\n# Show, then delete, .{ext} files older than {days} days\n"
                f"find \"$dir\" -type f -name '*.{ext}' -mtime +{days} -print\nread -r -p \"Delete these files? [y/N] \" answer\n"
                f"if [[ \"$answer\" == [yY] ]]; then\n  find \"$dir\" -type f -name '*.{ext}' -mtime +{days} -delete\n  echo \"Deleted.\"\nfi\n")
        return {"asks": [f"Bash script to delete .{ext} files older than {days} days", f"clean up old {ext} files with a shell script", f"Write a script that removes {ext} files older than {days} days"],
                "filename": f"cleanup_{ext}.sh", "desc": f"removes .{ext} files older than {days} days after asking", "content": code,
                "explain": f"Run `bash cleanup_{ext}.sh /path/to/folder`. It lists the files first and only deletes them if you type y."}
    if kind == "count":
        code = (f"#!/usr/bin/env bash\nset -euo pipefail\n\ndir=\"${{1:-.}}\"\ncount=$(find \"$dir\" -type f -name '*.{ext}' | wc -l)\n"
                f"size=$(find \"$dir\" -type f -name '*.{ext}' -exec du -ch {{}} + 2>/dev/null | tail -n 1 | cut -f1)\necho \"$count .{ext} files, ${{size:-0}} in total\"\n")
        return {"asks": [f"Shell script to count .{ext} files in a folder", f"how many {ext} files do I have, bash script", f"bash: count and size of all .{ext} files"],
                "filename": f"count_{ext}.sh", "desc": f"counts .{ext} files and their total size", "content": code, "explain": f"Run `bash count_{ext}.sh ~/Documents`."}
    code = (f"#!/usr/bin/env bash\nset -euo pipefail\n\nsrc=\"${{1:?Usage: backup.sh <folder>}}\"\ndest=\"${{2:-$HOME/backups}}\"\nmkdir -p \"$dest\"\n"
            f"tar -czf \"$dest/$(basename \"$src\")-$(date +%F).tar.gz\" -C \"$(dirname \"$src\")\" \"$(basename \"$src\")\"\n"
            f"# Keep only the last {days} backups\nls -1t \"$dest\"/*.tar.gz | tail -n +{days + 1} | xargs -r rm --\necho \"Backup done; keeping the newest {days}.\"\n")
    return {"asks": [f"Backup script that keeps the last {days} backups", f"bash backup with rotation, keep {days}", f"Write a shell script to back up a folder and keep {days} copies"],
            "filename": f"backup_keep{days}.sh", "desc": f"backs up a folder and keeps the newest {days} backups", "content": code,
            "explain": f"Run `bash backup_keep{days}.sh ~/Documents`. Older backups beyond the newest {days} are removed."}


def variants(held_out=False):
    """Every code task in the requested split."""
    out = []
    out += [converter(c) for c in (HELD_OUT_CONVERSIONS if held_out else CONVERSIONS)]
    ents = HELD_OUT_ENTITIES if held_out else ENTITIES
    out += [model_task(e, f, lang) for e, f in ents.items() for lang in ("python", "typescript", "java")]
    out += [sql_task(d, t) for d, t in (HELD_OUT_SQL if held_out else SQL_DOMAINS).items()]
    bizs = HELD_OUT_BUSINESSES if held_out else BUSINESSES
    out += [page_task(k, b, t) for k, b, t in itertools.product(PAGE_TYPES, bizs, THEMES)]
    out += [api_task(r, fw) for r in (HELD_OUT_RESOURCES if held_out else RESOURCES) for fw in ("express", "flask")]
    if not held_out:
        out += [js_util_task(*u) for u in JS_UTILS]
        out += [bash_task(k, e, d) for k in ("cleanup", "count") for e in EXTS for d in (7, 30)]
        out += [bash_task("backup", "", d) for d in (3, 5, 7, 10)]
    else:
        out += [bash_task("cleanup", "bak", 14)]
    # Drop duplicates (e.g. "count" scripts don't depend on days).
    seen, unique = set(), []
    for t in out:
        if t["filename"] + t["content"] not in seen:
            seen.add(t["filename"] + t["content"])
            unique.append(t)
    return unique
