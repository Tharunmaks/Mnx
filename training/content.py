"""Content banks for the Mnx training-data generator.

Everything here is hand-written so the dataset never depends on another AI
model's output. The generator combines these with templates and Mnx's real
tool formats. Items marked HELD_OUT_* are reserved for the evaluation set, so
the test measures generalisation instead of memorisation.
"""

CITIES = [
    ("Chennai", "India"), ("Mumbai", "India"), ("Bengaluru", "India"), ("Delhi", "India"), ("Hyderabad", "India"),
    ("Kochi", "India"), ("Pune", "India"), ("Kolkata", "India"), ("Jaipur", "India"), ("Coimbatore", "India"),
    ("London", "United Kingdom"), ("Manchester", "United Kingdom"), ("Paris", "France"), ("Lyon", "France"),
    ("Berlin", "Germany"), ("Munich", "Germany"), ("Madrid", "Spain"), ("Barcelona", "Spain"), ("Rome", "Italy"),
    ("Milan", "Italy"), ("Amsterdam", "Netherlands"), ("Stockholm", "Sweden"), ("Oslo", "Norway"),
    ("New York", "United States"), ("San Francisco", "United States"), ("Chicago", "United States"),
    ("Austin", "United States"), ("Seattle", "United States"), ("Toronto", "Canada"), ("Vancouver", "Canada"),
    ("Mexico City", "Mexico"), ("São Paulo", "Brazil"), ("Buenos Aires", "Argentina"), ("Cape Town", "South Africa"),
    ("Nairobi", "Kenya"), ("Cairo", "Egypt"), ("Dubai", "United Arab Emirates"), ("Istanbul", "Turkey"),
    ("Tokyo", "Japan"), ("Osaka", "Japan"), ("Seoul", "South Korea"), ("Singapore", "Singapore"),
    ("Bangkok", "Thailand"), ("Kuala Lumpur", "Malaysia"), ("Jakarta", "Indonesia"), ("Sydney", "Australia"),
    ("Melbourne", "Australia"), ("Auckland", "New Zealand"),
]
HELD_OUT_CITIES = [
    ("Madurai", "India"), ("Lucknow", "India"), ("Edinburgh", "United Kingdom"), ("Lisbon", "Portugal"),
    ("Vienna", "Austria"), ("Prague", "Czech Republic"), ("Boston", "United States"), ("Denver", "United States"),
    ("Montreal", "Canada"), ("Lima", "Peru"), ("Hanoi", "Vietnam"), ("Manila", "Philippines"),
    ("Perth", "Australia"), ("Doha", "Qatar"), ("Athens", "Greece"),
]

NAMES = ["", "", "", "Tharun", "Priya", "Arjun", "Meera", "Sam", "Alex", "Fatima", "Kenji", "Lucas", "Ana", "Ravi", "Zara"]

CONDITIONS = [
    ("Clear sky", "sun"), ("Mainly clear", "sun"), ("Partly cloudy", "partly"), ("Overcast", "cloud"),
    ("Light rain", "rain"), ("Rain showers", "rain"), ("Thunderstorm", "storm"), ("Fog", "fog"),
    ("Light drizzle", "drizzle"), ("Light snow", "snow"),
]

PLACE_KINDS = [
    # (how users ask, query to send, osm type, name parts)
    (["cafes", "coffee shops", "a good café", "places for coffee"], "cafe", "cafe", ["Brew", "Bean", "Roast", "Cup", "Ember", "Grind", "Morning"]),
    (["restaurants", "places to eat", "good food", "dinner spots"], "restaurant", "restaurant", ["Spice", "Garden", "Table", "Kitchen", "Olive", "Saffron", "Harbor"]),
    (["pizza places", "pizza", "a pizzeria"], "pizza", "restaurant", ["Slice", "Napoli", "Crust", "Oven", "Basil"]),
    (["pharmacies", "a pharmacy", "a chemist", "medical shops"], "pharmacy", "pharmacy", ["Care", "Health", "Plus", "Wellness", "Apollo"]),
    (["hospitals", "a hospital", "an emergency room"], "hospital", "hospital", ["General", "City", "Mercy", "Lakeside", "St. Mary's"]),
    (["ATMs", "an ATM", "cash machines"], "atm", "atm", ["Central", "Union", "Metro", "City"]),
    (["gyms", "a gym", "fitness centers"], "gym", "fitness_centre", ["Iron", "Pulse", "Core", "Peak", "Flex"]),
    (["bookshops", "a bookstore", "book stores"], "bookshop", "books", ["Page", "Chapter", "Ink", "Story", "Leaf"]),
    (["supermarkets", "a grocery store", "groceries"], "supermarket", "supermarket", ["Fresh", "Daily", "Green", "Market", "Basket"]),
    (["petrol pumps", "gas stations", "a fuel station"], "fuel", "fuel", ["Speed", "Shell", "Highway", "Express"]),
    (["parks", "a park", "green spaces"], "park", "park", ["Riverside", "Lotus", "Central", "Hillview", "Sunset"]),
    (["hotels", "a place to stay", "hotels nearby"], "hotel", "hotel", ["Grand", "Residency", "Bay", "Comfort", "Palm"]),
]
CUISINES = ["South Indian", "North Indian", "Italian", "Chinese", "Japanese", "Mexican", "Thai", "Lebanese", "American", "Korean"]
STREETS = ["MG Road", "Main Street", "Park Avenue", "Station Road", "Lake View Road", "High Street", "Market Lane", "Church Street", "Beach Road", "Hill Road"]

IMAGE_SUBJECTS = [
    "a cozy cabin in snowy mountains", "a cyberpunk street market at night", "a golden retriever puppy in a field of sunflowers",
    "an astronaut floating above Earth", "a futuristic city with flying cars", "a peaceful Japanese zen garden",
    "a dragon flying over a medieval castle", "a cup of coffee on a rainy window sill", "a lighthouse on a stormy coast",
    "a tiger walking through a misty jungle", "a red sports car on a coastal highway", "a cute robot watering plants",
    "an underwater coral reef with tropical fish", "a hot air balloon festival at sunrise", "a wizard reading in a candle-lit library",
    "a minimalist mountain landscape", "a bowl of steaming ramen", "an owl perched on a branch under the full moon",
    "a sailboat on a calm lake at sunset", "a treehouse in an enchanted forest", "a samurai standing in falling cherry blossoms",
    "a cat wearing a tiny crown", "a waterfall in a tropical rainforest", "a vintage train crossing a stone bridge",
]
HELD_OUT_IMAGE_SUBJECTS = [
    "a panda eating bamboo in the rain", "a desert caravan under a starry sky", "a floating island with waterfalls",
    "a steampunk airship over London", "a fox curled up in autumn leaves", "a neon-lit arcade from the 1980s",
]
IMAGE_STYLES = [
    ("digital art", "highly detailed digital painting, vibrant colors, cinematic lighting"),
    ("watercolor", "soft watercolor painting, gentle washes of color, paper texture"),
    ("photo", "photorealistic photograph, 50mm lens, natural light, sharp focus"),
    ("anime", "anime style illustration, clean line art, bright cel shading"),
    ("oil painting", "classic oil painting, rich textures, dramatic chiaroscuro lighting"),
    ("pixel art", "16-bit pixel art, limited color palette, retro video game style"),
    ("3D render", "3D render, soft global illumination, octane render, high detail"),
    ("sketch", "pencil sketch, fine cross-hatching, black and white"),
    ("fantasy", "epic fantasy concept art, volumetric light, matte painting"),
    ("minimalist", "minimalist flat illustration, simple shapes, pastel colors"),
]
LOGO_BRANDS = ["a coffee shop called Bean There", "a tech startup called Nimbus", "a bakery called Sweet Crumbs", "a gym called Iron Pulse",
               "a bookstore called Ink & Leaf", "a travel agency called Wanderly", "a gaming channel called PixelForge"]

# Document topics: 5 sections, each with bullet points (all hand-written, factual).
DOC_TOPICS = {
    "How black holes form": [
        ("What a black hole is", ["A region where gravity is so strong that nothing, not even light, can escape", "Its boundary is called the event horizon", "Mass is concentrated in a tiny region called a singularity"]),
        ("Stellar collapse", ["Stars more than about 20 times the Sun's mass end in a core collapse", "The core runs out of nuclear fuel and can no longer resist gravity", "The outer layers explode as a supernova while the core collapses"]),
        ("Types of black holes", ["Stellar-mass: a few to tens of solar masses", "Supermassive: millions to billions of solar masses at galaxy centers", "Intermediate-mass black holes are rarer and still being studied"]),
        ("How we detect them", ["X-rays from hot gas in accretion disks", "Gravitational waves from merging black holes, first detected by LIGO in 2015", "The Event Horizon Telescope imaged the shadow of M87* in 2019"]),
        ("Key takeaways", ["Black holes are the end state of the most massive stars", "They shape the growth of galaxies", "They are natural laboratories for extreme physics"]),
    ],
    "Climate change basics": [
        ("What climate change is", ["A long-term shift in global temperatures and weather patterns", "Driven mainly by burning fossil fuels since the 1800s", "Earth has warmed about 1.1–1.2 °C above pre-industrial levels"]),
        ("The greenhouse effect", ["Gases like CO₂ and methane trap heat in the atmosphere", "CO₂ levels have risen above 420 ppm", "More trapped heat means a warmer planet"]),
        ("Impacts", ["More frequent heatwaves, droughts and heavy rain", "Rising sea levels from melting ice and warming oceans", "Stress on food, water and ecosystems"]),
        ("Solutions", ["Switching to renewable energy like solar and wind", "Improving energy efficiency in buildings and transport", "Protecting forests and restoring ecosystems"]),
        ("What individuals can do", ["Use public transport, cycle or walk when possible", "Reduce waste and energy use at home", "Support climate-friendly policies and businesses"]),
    ],
    "Introduction to Python": [
        ("Why Python", ["Readable syntax that is easy to learn", "Huge ecosystem of libraries", "Used in web development, data science, AI and automation"]),
        ("Core building blocks", ["Variables and data types: int, float, str, bool", "Collections: list, tuple, dict, set", "Control flow with if, for and while"]),
        ("Functions", ["Defined with def and called by name", "Parameters can have default values", "Return values with return"]),
        ("Working with libraries", ["Install packages with pip", "Import modules with import", "Popular libraries: requests, pandas, NumPy, Flask"]),
        ("Next steps", ["Build small projects like a to-do app", "Practice on coding challenge sites", "Read other people's code to learn patterns"]),
    ],
    "The solar system": [
        ("Overview", ["The Sun and everything bound to it by gravity", "Formed about 4.6 billion years ago", "Eight planets orbit the Sun"]),
        ("The inner planets", ["Mercury, Venus, Earth and Mars", "Small, rocky worlds", "Earth is the only known planet with life"]),
        ("The outer planets", ["Jupiter and Saturn are gas giants", "Uranus and Neptune are ice giants", "All four have ring systems and many moons"]),
        ("Smaller bodies", ["Dwarf planets like Pluto and Ceres", "The asteroid belt between Mars and Jupiter", "Comets from the distant Kuiper Belt and Oort Cloud"]),
        ("Exploration", ["Voyager 1 is the most distant human-made object", "Rovers like Perseverance explore Mars", "Missions keep revealing new moons and worlds"]),
    ],
    "Healthy eating habits": [
        ("Balanced plate", ["Half the plate vegetables and fruit", "A quarter whole grains", "A quarter protein such as beans, fish, eggs or chicken"]),
        ("Hydration", ["Drink water through the day", "Limit sugary drinks", "Thirst and urine color are good guides"]),
        ("Smart choices", ["Choose whole foods over ultra-processed ones", "Watch portion sizes", "Read nutrition labels for sugar and salt"]),
        ("Planning", ["Plan meals for the week", "Keep healthy snacks like nuts and fruit handy", "Cook at home more often"]),
        ("Sustainable habits", ["Make small changes one at a time", "Enjoy treats in moderation", "Consistency matters more than perfection"]),
    ],
    "Artificial intelligence explained": [
        ("What AI is", ["Computer systems that perform tasks needing human-like intelligence", "Includes learning, reasoning and understanding language", "Most modern AI is based on machine learning"]),
        ("Machine learning", ["Models learn patterns from data instead of fixed rules", "Supervised, unsupervised and reinforcement learning", "Neural networks power most recent progress"]),
        ("Language models", ["Trained on large amounts of text to predict the next word", "Can answer questions, write and summarize", "Can also make mistakes, so outputs should be checked"]),
        ("Everyday uses", ["Voice assistants and translation", "Recommendations on streaming and shopping apps", "Photo enhancement and spam filters"]),
        ("Using AI responsibly", ["Check important facts", "Protect personal data", "Be aware of bias in data and outputs"]),
    ],
    "Time management tips": [
        ("Plan your day", ["Write a short to-do list each morning", "Pick your top three priorities", "Estimate how long each task takes"]),
        ("Focus techniques", ["Work in focused blocks, like the Pomodoro technique (25 minutes on, 5 off)", "Silence notifications while working", "Do the hardest task first"]),
        ("Prioritize", ["Use the Eisenhower matrix: urgent vs important", "Say no to low-value commitments", "Delegate when you can"]),
        ("Avoid burnout", ["Schedule breaks and rest", "Sleep 7–9 hours", "Keep time for exercise and hobbies"]),
        ("Review", ["Reflect at the end of each week", "Adjust what isn't working", "Celebrate progress"]),
    ],
    "The water cycle": [
        ("Evaporation", ["The Sun heats water in oceans, lakes and rivers", "Water turns into water vapor and rises", "Plants also release water vapor through transpiration"]),
        ("Condensation", ["Water vapor cools as it rises", "It condenses into tiny droplets that form clouds", "Dust particles help droplets form"]),
        ("Precipitation", ["Droplets combine and fall as rain, snow, sleet or hail", "Type depends on temperature", "Precipitation returns water to the surface"]),
        ("Collection", ["Water gathers in oceans, lakes and rivers", "Some soaks into the ground as groundwater", "The cycle then starts again"]),
        ("Why it matters", ["Supplies fresh water for life", "Shapes weather and climate", "Moves heat around the planet"]),
    ],
    "Starting a small business": [
        ("Find your idea", ["Solve a real problem for a specific group of people", "Check if people will pay for it", "Study competitors"]),
        ("Plan", ["Write a simple business plan", "Estimate costs, pricing and break-even point", "Decide on a legal structure and register"]),
        ("Funding", ["Start with savings or small loans", "Consider grants and investors later", "Keep personal and business money separate"]),
        ("Marketing", ["Build a simple website and social media presence", "Ask happy customers for reviews", "Track what brings in customers"]),
        ("Grow", ["Listen to customer feedback", "Improve the product step by step", "Hire help when the workload demands it"]),
    ],
    "The human heart": [
        ("Structure", ["A muscular organ about the size of a fist", "Four chambers: two atria and two ventricles", "Valves keep blood flowing in one direction"]),
        ("How it works", ["The right side pumps blood to the lungs", "The left side pumps oxygen-rich blood to the body", "It beats about 100,000 times a day"]),
        ("The electrical system", ["The sinoatrial node acts as a natural pacemaker", "Electrical signals make the chambers contract in order", "An ECG records this activity"]),
        ("Keeping it healthy", ["Exercise regularly", "Eat less salt and saturated fat", "Don't smoke and manage stress"]),
        ("Warning signs", ["Chest pain or pressure", "Shortness of breath", "Seek emergency help for sudden symptoms"]),
    ],
}
HELD_OUT_DOC_TOPICS = {
    "Renewable energy": [
        ("What it is", ["Energy from sources that naturally replenish", "Includes solar, wind, hydro and geothermal", "Produces little or no greenhouse gas while running"]),
        ("Solar power", ["Panels turn sunlight into electricity", "Costs have fallen sharply over the last decade", "Works well on rooftops and in large farms"]),
        ("Wind power", ["Turbines turn moving air into electricity", "Onshore and offshore wind farms", "Output varies with the weather"]),
        ("Challenges", ["Storing energy for when the sun and wind are low", "Upgrading power grids", "Land use and materials"]),
        ("The future", ["Batteries and green hydrogen for storage", "Smarter grids", "A growing share of global electricity"]),
    ],
    "How the internet works": [
        ("Networks of networks", ["Millions of networks linked together", "Data travels in small packets", "Routers forward packets toward their destination"]),
        ("Addresses", ["Every device has an IP address", "DNS turns names like example.com into IP addresses", "Ports identify services on a device"]),
        ("Protocols", ["TCP makes delivery reliable", "HTTP and HTTPS carry web pages", "TLS encrypts the connection"]),
        ("Infrastructure", ["Undersea fiber cables connect continents", "Data centers host websites and apps", "Content delivery networks cache data near users"]),
        ("Staying safe", ["Use HTTPS websites", "Keep software updated", "Use strong, unique passwords"]),
    ],
}

# Code tasks: hand-written, working programs.
CODE_TASKS = [
    {"asks": ["Write a Python script that counts the words in a text file", "python program to count words in a file", "Make me a word counter in Python"],
     "filename": "word_count.py", "desc": "Counts words, lines and the most common words in a text file",
     "content": "import sys\nfrom collections import Counter\n\n\ndef main(path):\n    with open(path, encoding=\"utf-8\") as f:\n        text = f.read()\n    words = text.lower().split()\n    print(f\"Lines: {text.count(chr(10)) + 1}\")\n    print(f\"Words: {len(words)}\")\n    for word, count in Counter(words).most_common(5):\n        print(f\"{word}: {count}\")\n\n\nif __name__ == \"__main__\":\n    if len(sys.argv) != 2:\n        sys.exit(\"Usage: python word_count.py <file>\")\n    main(sys.argv[1])\n",
     "explain": "Run it with `python word_count.py notes.txt`. It prints the line count, the word count and the five most common words."},
    {"asks": ["Create a snake game in HTML", "make a playable snake game as a single html file", "Build me a snake game for the browser"],
     "filename": "snake.html", "desc": "A playable snake game in one HTML file",
     "content": "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>Snake</title>\n<style>body{margin:0;display:grid;place-items:center;height:100vh;background:#111;color:#eee;font-family:sans-serif}canvas{background:#222;border-radius:8px}</style>\n</head>\n<body>\n<div><canvas id=\"c\" width=\"400\" height=\"400\"></canvas><p id=\"s\">Score: 0</p></div>\n<script>\nconst c = document.getElementById(\"c\"), x = c.getContext(\"2d\"), size = 20;\nlet snake = [{x: 10, y: 10}], dir = {x: 1, y: 0}, food = {x: 15, y: 15}, score = 0;\ndocument.addEventListener(\"keydown\", e => {\n  const d = {ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0]}[e.key];\n  if (d && (d[0] !== -dir.x || d[1] !== -dir.y)) dir = {x: d[0], y: d[1]};\n});\nfunction step() {\n  const head = {x: (snake[0].x + dir.x + 20) % 20, y: (snake[0].y + dir.y + 20) % 20};\n  if (snake.some(p => p.x === head.x && p.y === head.y)) { snake = [{x: 10, y: 10}]; score = 0; }\n  snake.unshift(head);\n  if (head.x === food.x && head.y === food.y) { score++; food = {x: Math.floor(Math.random() * 20), y: Math.floor(Math.random() * 20)}; }\n  else snake.pop();\n  x.fillStyle = \"#222\"; x.fillRect(0, 0, 400, 400);\n  x.fillStyle = \"#e74c3c\"; x.fillRect(food.x * size, food.y * size, size - 2, size - 2);\n  x.fillStyle = \"#2ecc71\"; snake.forEach(p => x.fillRect(p.x * size, p.y * size, size - 2, size - 2));\n  document.getElementById(\"s\").textContent = \"Score: \" + score;\n}\nsetInterval(step, 120);\n</script>\n</body>\n</html>\n",
     "explain": "Open it in your browser and steer with the arrow keys. Eating the red food grows the snake; running into yourself restarts the game."},
    {"asks": ["Write a C program that prints a multiplication table", "c code for multiplication table", "multiplication table in C please"],
     "filename": "table.c", "desc": "Prints a multiplication table for a number",
     "content": "#include <stdio.h>\n\nint main(void) {\n    int n;\n    printf(\"Enter a number: \");\n    if (scanf(\"%d\", &n) != 1) {\n        return 1;\n    }\n    for (int i = 1; i <= 10; i++) {\n        printf(\"%d x %d = %d\\n\", n, i, n * i);\n    }\n    return 0;\n}\n",
     "explain": "Compile with `gcc table.c -o table` and run `./table`."},
    {"asks": ["Write a C++ program to reverse a string", "c++ code that reverses a string", "Reverse a string in C++"],
     "filename": "reverse.cpp", "desc": "Reverses a line of text",
     "content": "#include <algorithm>\n#include <iostream>\n#include <string>\n\nint main() {\n    std::string text;\n    std::getline(std::cin, text);\n    std::reverse(text.begin(), text.end());\n    std::cout << text << '\\n';\n    return 0;\n}\n",
     "explain": "Compile with `g++ reverse.cpp -o reverse`, run it and type a line of text."},
    {"asks": ["Write a Ruby script that checks if a word is a palindrome", "palindrome checker in ruby", "Ruby palindrome program"],
     "filename": "palindrome.rb", "desc": "Checks whether text is a palindrome",
     "content": "def palindrome?(text)\n  clean = text.downcase.gsub(/[^a-z0-9]/, \"\")\n  clean == clean.reverse\nend\n\nprint \"Enter text: \"\ninput = gets.to_s.chomp\nputs palindrome?(input) ? \"Yes, it's a palindrome!\" : \"No, it's not a palindrome.\"\n",
     "explain": "Run it with `ruby palindrome.rb`. It ignores case, spaces and punctuation."},
    {"asks": ["Write a Lua script for FizzBuzz", "fizzbuzz in lua", "Lua FizzBuzz please"],
     "filename": "fizzbuzz.lua", "desc": "Prints FizzBuzz from 1 to 100",
     "content": "for i = 1, 100 do\n  if i % 15 == 0 then\n    print(\"FizzBuzz\")\n  elseif i % 3 == 0 then\n    print(\"Fizz\")\n  elseif i % 5 == 0 then\n    print(\"Buzz\")\n  else\n    print(i)\n  end\nend\n",
     "explain": "Run it with `lua fizzbuzz.lua`."},
    {"asks": ["Make a JSON file of 5 sample users", "create a json with sample user data", "Give me a users.json with example data"],
     "filename": "users.json", "desc": "Five sample user records",
     "content": "[\n  {\"id\": 1, \"name\": \"Asha Rao\", \"email\": \"asha@example.com\", \"age\": 28, \"active\": true},\n  {\"id\": 2, \"name\": \"Ben Carter\", \"email\": \"ben@example.com\", \"age\": 34, \"active\": false},\n  {\"id\": 3, \"name\": \"Chen Wei\", \"email\": \"chen@example.com\", \"age\": 25, \"active\": true},\n  {\"id\": 4, \"name\": \"Diana Lopez\", \"email\": \"diana@example.com\", \"age\": 41, \"active\": true},\n  {\"id\": 5, \"name\": \"Elif Kaya\", \"email\": \"elif@example.com\", \"age\": 30, \"active\": false}\n]\n",
     "explain": "Each user has an id, name, email, age and an active flag."},
    {"asks": ["Write a Python to-do list app for the terminal", "simple todo app in python", "CLI to-do list in Python"],
     "filename": "todo.py", "desc": "A terminal to-do list that saves to todo.json",
     "content": "import json\nimport os\n\nFILE = \"todo.json\"\n\n\ndef load():\n    return json.load(open(FILE)) if os.path.exists(FILE) else []\n\n\ndef save(items):\n    json.dump(items, open(FILE, \"w\"), indent=2)\n\n\ndef main():\n    items = load()\n    while True:\n        for i, item in enumerate(items, 1):\n            print(f\"{i}. {'[x]' if item['done'] else '[ ]'} {item['text']}\")\n        cmd = input(\"\\n(a)dd, (d)one, (r)emove, (q)uit: \").strip().lower()\n        if cmd == \"a\":\n            items.append({\"text\": input(\"Task: \"), \"done\": False})\n        elif cmd in (\"d\", \"r\"):\n            n = int(input(\"Number: \")) - 1\n            if 0 <= n < len(items):\n                if cmd == \"d\":\n                    items[n][\"done\"] = True\n                else:\n                    items.pop(n)\n        elif cmd == \"q\":\n            break\n        save(items)\n\n\nif __name__ == \"__main__\":\n    main()\n",
     "explain": "Run `python todo.py`. Tasks are saved in todo.json, so they're still there next time."},
    {"asks": ["Write a JavaScript function to debounce", "debounce function in js", "JS debounce utility"],
     "filename": "debounce.js", "desc": "A reusable debounce helper",
     "content": "/**\n * Returns a function that waits `wait` ms after the last call before running `fn`.\n */\nexport function debounce(fn, wait = 300) {\n  let timer;\n  return function (...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), wait);\n  };\n}\n\n// Example: only search after the user stops typing for 300 ms.\n// input.addEventListener(\"input\", debounce((e) => search(e.target.value), 300));\n",
     "explain": "Wrap any function with `debounce(fn, 300)` and it only runs after calls stop for 300 ms."},
    {"asks": ["Write a bash script to back up a folder", "backup script in bash", "Shell script that zips a folder with the date"],
     "filename": "backup.sh", "desc": "Backs up a folder into a dated .tar.gz",
     "content": "#!/usr/bin/env bash\nset -euo pipefail\n\nsrc=\"${1:?Usage: backup.sh <folder> [destination]}\"\ndest=\"${2:-$HOME/backups}\"\nmkdir -p \"$dest\"\nname=\"$(basename \"$src\")-$(date +%Y-%m-%d_%H%M).tar.gz\"\ntar -czf \"$dest/$name\" -C \"$(dirname \"$src\")\" \"$(basename \"$src\")\"\necho \"Saved $dest/$name\"\n",
     "explain": "Run `bash backup.sh ~/Documents`. Backups go to ~/backups unless you pass another destination."},
    {"asks": ["Make a simple landing page in HTML and CSS", "html landing page for my app", "Create a landing page website"],
     "filename": "index.html", "desc": "A responsive landing page",
     "content": "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>Launch</title>\n<style>\n*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;color:#1f2330}\nheader{padding:80px 24px;text-align:center;background:linear-gradient(135deg,#6d5dfc,#3fb8ff);color:#fff}\nh1{font-size:clamp(32px,6vw,56px);margin:0 0 12px}.btn{display:inline-block;margin-top:20px;padding:12px 28px;border-radius:999px;background:#fff;color:#6d5dfc;font-weight:700;text-decoration:none}\n.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px;max-width:960px;margin:48px auto;padding:0 24px}\n.card{padding:24px;border-radius:16px;background:#f4f5fb}\n</style>\n</head>\n<body>\n<header><h1>Your app, but faster</h1><p>Everything you need in one simple place.</p><a class=\"btn\" href=\"#\">Get started</a></header>\n<section class=\"features\">\n<div class=\"card\"><h3>Fast</h3><p>Loads in a blink on any device.</p></div>\n<div class=\"card\"><h3>Secure</h3><p>Your data stays private.</p></div>\n<div class=\"card\"><h3>Simple</h3><p>No learning curve.</p></div>\n</section>\n</body>\n</html>\n",
     "explain": "Open it in a browser. Change the headline, button text and feature cards to match your app."},
    {"asks": ["Write a Python script that renames all photos in a folder by date", "rename photos with python", "Python batch file renamer"],
     "filename": "rename_photos.py", "desc": "Renames .jpg/.png files by their modified date",
     "content": "import os\nimport sys\nfrom datetime import datetime\n\nfolder = sys.argv[1] if len(sys.argv) > 1 else \".\"\nfor name in sorted(os.listdir(folder)):\n    if not name.lower().endswith((\".jpg\", \".jpeg\", \".png\")):\n        continue\n    path = os.path.join(folder, name)\n    stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime(\"%Y-%m-%d_%H-%M-%S\")\n    ext = os.path.splitext(name)[1].lower()\n    new = os.path.join(folder, f\"{stamp}{ext}\")\n    n = 1\n    while os.path.exists(new):\n        new = os.path.join(folder, f\"{stamp}_{n}{ext}\")\n        n += 1\n    os.rename(path, new)\n    print(f\"{name} -> {os.path.basename(new)}\")\n",
     "explain": "Run `python rename_photos.py path/to/photos`. Try it on a copy of the folder first."},
]
HELD_OUT_CODE_TASKS = [
    {"asks": ["Write a Python script that converts Celsius to Fahrenheit", "temperature converter in python"],
     "filename": "temperature.py", "desc": "Converts between Celsius and Fahrenheit",
     "content": "def c_to_f(c):\n    return c * 9 / 5 + 32\n\n\ndef f_to_c(f):\n    return (f - 32) * 5 / 9\n\n\nvalue = float(input(\"Temperature: \"))\nunit = input(\"Unit (C/F): \").strip().upper()\nif unit == \"C\":\n    print(f\"{value}°C = {c_to_f(value):.1f}°F\")\nelse:\n    print(f\"{value}°F = {f_to_c(value):.1f}°C\")\n",
     "explain": "Run `python temperature.py`, type a number and C or F."},
    {"asks": ["Write a JavaScript countdown timer for a web page", "countdown timer html js"],
     "filename": "countdown.html", "desc": "A countdown timer page",
     "content": "<!doctype html>\n<html>\n<body style=\"font-family:sans-serif;text-align:center;margin-top:20vh\">\n<h1 id=\"t\">05:00</h1>\n<button onclick=\"start()\">Start</button>\n<script>\nlet left = 300, timer;\nfunction show() { document.getElementById(\"t\").textContent = String(Math.floor(left / 60)).padStart(2, \"0\") + \":\" + String(left % 60).padStart(2, \"0\"); }\nfunction start() { clearInterval(timer); timer = setInterval(() => { if (left > 0) { left--; show(); } else clearInterval(timer); }, 1000); }\nshow();\n</script>\n</body>\n</html>\n",
     "explain": "Open it in a browser and press Start. Change `left = 300` to set the seconds."},
]

# Questions answered directly, without tools.
QA = [
    (["hi", "hello", "hey there", "hello Mnx"], "Hi! 👋 I'm **Mnx**. I can search the web, check the weather, find places, make images, slides and documents, write code and run it with your OK. What can I do for you?"),
    (["thanks", "thank you so much", "thanks, that helped"], "You're welcome! Let me know if there's anything else you need."),
    (["who are you", "what are you", "what can you do"], "I'm **Mnx**, an AI assistant running right on your device. I can:\n\n- **Search the web** and read pages\n- Check the **weather** and find **places** and **directions**\n- **Create images**, slides (PPTX), PDFs and Word documents\n- Write **code files** and **run code** after you approve it\n\nJust ask!"),
    (["what is photosynthesis", "explain photosynthesis simply"], "**Photosynthesis** is how plants make their own food using sunlight.\n\n1. **Light** is absorbed by chlorophyll in the leaves.\n2. Plants take in **carbon dioxide** from the air and **water** from the soil.\n3. Using light energy, they turn these into **glucose** (sugar) and release **oxygen**.\n\nThe overall reaction is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."),
    (["what is the difference between a list and a tuple in python", "list vs tuple python"], "The main difference is that **lists are mutable** and **tuples are immutable**.\n\n| | List | Tuple |\n|---|---|---|\n| Syntax | `[1, 2, 3]` | `(1, 2, 3)` |\n| Can change? | Yes | No |\n| Typical use | Collections that change | Fixed records, dict keys |\n| Speed | Slightly slower | Slightly faster |\n\nUse a tuple when the data shouldn't change, and a list when you'll add or remove items."),
    (["how do I make a cup of tea", "how to make tea"], "Here's a simple way to make a great cup of tea:\n\n1. **Boil** fresh water.\n2. **Warm** your cup or pot with a little hot water, then pour it out.\n3. Add **1 tea bag** (or 1 teaspoon of loose leaves) per cup.\n4. Pour in the hot water and **steep 3–5 minutes**.\n5. Remove the tea and add **milk, sugar or lemon** to taste.\n\nFor green tea, use water just below boiling (about 80 °C) and steep for 2–3 minutes."),
    (["what is 15% of 200", "15 percent of 200"], "**15% of 200 is 30.**\n\n200 × 0.15 = 30"),
    (["what is the capital of Australia", "capital city of australia"], "The capital of Australia is **Canberra**. Many people guess Sydney, but Canberra was chosen as a compromise between Sydney and Melbourne."),
    (["give me tips to sleep better", "how can I sleep better"], "Here are proven ways to sleep better:\n\n- **Keep a schedule**: go to bed and wake up at the same time every day.\n- **Limit screens** for 30–60 minutes before bed.\n- **Keep your room** cool, dark and quiet.\n- **Avoid caffeine** after early afternoon and heavy meals late at night.\n- **Get daylight** in the morning and move during the day.\n\nIf you often struggle for weeks, it's worth talking to a doctor."),
    (["explain recursion", "what is recursion in programming"], "**Recursion** is when a function calls itself to solve a smaller version of the same problem.\n\nEvery recursive function needs:\n1. A **base case** that stops the recursion.\n2. A **recursive case** that moves toward the base case.\n\n```python\ndef factorial(n):\n    if n <= 1:          # base case\n        return 1\n    return n * factorial(n - 1)  # recursive case\n```\n\n`factorial(4)` becomes 4 × 3 × 2 × 1 = **24**."),
    (["write a short poem about the sea", "poem about the ocean"], "**The Sea**\n\nThe waves roll in with silver light,\nThey whisper secrets to the night,\nThey carry stories, old and deep,\nOf sailors' dreams and sunken sleep.\n\nAnd when the morning paints the shore,\nThe sea begins its song once more."),
    (["how many days are in a leap year", "days in a leap year"], "A leap year has **366 days**, one more than a normal year, because February gets a 29th day. Leap years happen every 4 years, except years divisible by 100 but not by 400 (so 2000 was a leap year, but 1900 wasn't)."),
    (["what does HTTP stand for", "meaning of http"], "**HTTP** stands for **HyperText Transfer Protocol**. It's the set of rules browsers and web servers use to send web pages and other data. **HTTPS** is the secure version, which encrypts the connection with TLS."),
    (["give me a quick workout I can do at home", "home workout without equipment"], "Here's a **15-minute no-equipment workout**. Do each exercise for 40 seconds, rest 20 seconds, and repeat the circuit 3 times:\n\n1. **Jumping jacks**\n2. **Squats**\n3. **Push-ups** (on your knees if needed)\n4. **Lunges** (alternate legs)\n5. **Plank**\n\nWarm up for 2 minutes first and stretch afterwards."),
    (["translate good morning to spanish", "how do you say good morning in spanish"], "\"Good morning\" in Spanish is **\"Buenos días\"**."),
    (["what's the boiling point of water", "at what temperature does water boil"], "Water boils at **100 °C (212 °F)** at sea level. At higher altitudes it boils at a lower temperature because the air pressure is lower, about 90 °C at 3,000 m."),
    (["summarize the plot of Romeo and Juliet", "romeo and juliet summary"], "**Romeo and Juliet** is Shakespeare's tragedy about two young lovers from feuding families in Verona.\n\n- Romeo (a Montague) and Juliet (a Capulet) fall in love and **marry in secret**.\n- Romeo kills Juliet's cousin Tybalt in a fight and is **banished**.\n- Juliet fakes her death with a potion, but the message explaining the plan never reaches Romeo.\n- Believing she is dead, Romeo **poisons himself**; Juliet wakes, finds him and **stabs herself**.\n\nTheir deaths finally reconcile the two families."),
    (["what is an API", "explain what an api is"], "An **API** (Application Programming Interface) is a set of rules that lets programs talk to each other.\n\nThink of it like a restaurant menu: you (the app) choose from the menu (the API), the kitchen (the other service) prepares it, and the waiter brings back the result. For example, a weather app uses a weather service's API to get the forecast."),
]

# Requests Mnx has no tool for: explain honestly instead of pretending.
UNSUPPORTED = [
    (["send an email to my boss saying I'll be late", "email my friend the meeting notes"],
     "I can't send emails from here, because I don't have an email tool on this device. Here's a message you can copy and send:\n\n> Hi, I'm running a bit late today and will be in shortly. Sorry for the inconvenience!\n\nTip: if you add an email connector in the **MCP** panel and use an online Mnx model, I can send it for you."),
    (["call my mom", "make a phone call to the pizza place"],
     "I can't make phone calls. I can only search, create files, images and documents, check weather and places, and run code with your OK. If you'd like, I can find the phone number you need."),
    (["set an alarm for 7 am", "remind me tomorrow at 9"],
     "I can't set alarms or reminders on your phone. Your Clock app can do that: open it, tap **+**, and choose the time. I can help you plan your schedule if you like."),
]
