"""Training conversations for Mnx's everyday tools: calculate, convert_units,
convert_currency, get_time, wikipedia, define_word, translate, create_qr_code.

Results have exactly the shape the real tools return (tools-extra.js), and a
stable slice of every list is held out for the test set."""
import datetime as dt
import math
import re
import zlib
from zoneinfo import ZoneInfo


def held(key):
    """About 1 in 7 items is held out for testing (stable across runs)."""
    return zlib.crc32(str(key).encode()) % 7 == 0


def js_round(x, digits=12):
    """Like the app's round(): Number.parseFloat(x.toPrecision(digits)), and
    whole numbers as ints so JSON matches JavaScript's output."""
    v = float(f"{x:.{digits}g}")
    return int(v) if v.is_integer() and abs(v) < 1e15 else v


def pretty(x):
    v = js_round(x)
    if isinstance(v, int):
        return f"{v:,}"
    s = f"{v:,.10f}".rstrip("0").rstrip(".")
    return s


# ───────── calculate ─────────

def calc_tasks(r):
    """(question, expression, value, explanation) built from random numbers."""
    a, b, c = r.randint(12, 980), r.randint(3, 95), r.randint(2, 12)
    big = r.randint(1000, 99999)
    pct = r.choice([5, 7.5, 12, 15, 18, 22.5, 33, 45, 62])
    n = r.randint(5, 15)
    sq = r.randint(11, 99) ** 2
    deg = r.choice([30, 45, 60, 90, 120, 135, 150])
    p, rate, yrs = r.choice([1000, 5000, 25000, 100000]), r.choice([3, 4.5, 6, 7.25, 8, 9.5]), r.randint(3, 25)
    opts = [
        (f"What's {pct}% of {big:,}?", f"{pct}% of {big}", big * pct / 100, f"{pct}% of {big:,} = {big:,} × {pct}/100"),
        (f"calculate {a} * {b} + {c}^3", f"{a} * {b} + {c}^3", a * b + c ** 3, f"{a} × {b} = {a * b:,}, plus {c}³ = {c ** 3:,}"),
        (f"square root of {sq:,}", f"sqrt({sq})", math.sqrt(sq), f"{int(math.sqrt(sq))} × {int(math.sqrt(sq))} = {sq:,}"),
        (f"What is {n}! ?", f"{n}!", math.factorial(n), f"{n}! means 1 × 2 × … × {n}"),
        (f"what's 2 to the power of {n + 10}", f"2^{n + 10}", 2 ** (n + 10), f"2 multiplied by itself {n + 10} times"),
        (f"({a} + {b}) × {c} ÷ 4", f"({a} + {b}) * {c} / 4", (a + b) * c / 4, f"({a} + {b}) = {a + b}, × {c} = {(a + b) * c:,}, ÷ 4"),
        (f"sin of {deg} degrees", f"sin({deg})", math.sin(math.radians(deg)), f"sin({deg}°)"),
        (f"Compound interest: {p:,} at {rate}% for {yrs} years, what's the final amount?", f"{p} * (1 + {rate}/100)^{yrs}", p * (1 + rate / 100) ** yrs,
         f"Final amount = {p:,} × (1 + {rate}%)^{yrs}"),
        (f"{big:,} divided by {b}", f"{big} / {b}", big / b, f"{big:,} ÷ {b}"),
        (f"Increase {big:,} by {pct}%", f"{big} + {pct}%", big * (1 + pct / 100), f"{big:,} + {pct}% of {big:,}"),
        (f"What's {a} mod {c}?", f"{a} mod {c}", a % c, f"{a} = {c} × {a // c} + {a % c}"),
        (f"log base 10 of {10 ** c}", f"log({10 ** c})", c, f"10^{c} = {10 ** c:,}"),
        (f"{a}.{b} × {c}.5 - {b}", f"{a}.{b} * {c}.5 - {b}", float(f"{a}.{b}") * float(f"{c}.5") - b, f"{a}.{b} × {c}.5 = {js_round(float(f'{a}.{b}') * float(f'{c}.5')):,}, then − {b}"),
    ]
    return r.choice(opts)


# ───────── convert_units (same factors as tools-extra.js) ─────────

UNIT_TASKS = [
    ("km", "miles", 1000, 1609.344, [5, 10, 21.1, 42.195, 100, 3]),
    ("miles", "km", 1609.344, 1000, [1, 5, 26.2, 60, 13.1]),
    ("meters", "feet", 1, 0.3048, [1, 3, 10, 100, 1.8]),
    ("feet", "meters", 0.3048, 1, [6, 10, 1000, 5.5]),
    ("inches", "cm", 0.0254, 0.01, [1, 6, 12, 32, 55]),
    ("cm", "inches", 0.01, 0.0254, [10, 30, 175, 180]),
    ("kg", "lbs", 1, 0.45359237, [1, 5, 60, 72, 100]),
    ("lbs", "kg", 0.45359237, 1, [1, 10, 150, 200]),
    ("grams", "ounces", 0.001, 0.028349523125, [100, 250, 500]),
    ("liters", "gallons", 1, 3.785411784, [1, 4, 20, 50]),
    ("gallons", "liters", 3.785411784, 1, [1, 5, 15]),
    ("cups", "ml", 0.2365882365, 0.001, [1, 2, 3.5]),
    ("tbsp", "ml", 0.01478676478125, 0.001, [1, 2, 4]),
    ("mph", "km/h", 0.44704, 1 / 3.6, [30, 60, 70, 100]),
    ("km/h", "mph", 1 / 3.6, 0.44704, [50, 80, 100, 120]),
    ("GB", "MB", 1e9, 1e6, [1, 2.5, 16]),
    ("TB", "GB", 1e12, 1e9, [1, 2, 4]),
    ("hours", "minutes", 3600, 60, [2.5, 8, 24]),
    ("days", "hours", 86400, 3600, [3, 7, 30]),
    ("acres", "square meters", 4046.8564224, 1, [1, 5, 10]),
    ("square feet", "square meters", 0.09290304, 1, [500, 1200, 2000]),
    ("kcal", "kJ", 4184, 1000, [100, 250, 2000]),
]
TEMP_TASKS = [("°F", "°C", [32, 68, 98.6, 100, 212, 0, -40]), ("°C", "°F", [0, 25, 37, 100, -10, 40]), ("°C", "K", [0, 25, 100, -273.15])]


def to_celsius(v, u):
    return {"°C": v, "°F": (v - 32) * 5 / 9, "K": v - 273.15}[u]


def from_celsius(c, u):
    return {"°C": c, "°F": c * 9 / 5 + 32, "K": c + 273.15}[u]


def unit_task(r):
    if r.random() < 0.25:
        f, t, vals = r.choice(TEMP_TASKS)
        v = r.choice(vals)
        out = js_round(from_celsius(to_celsius(v, f), t), 10)
    else:
        f, t, a, b, vals = r.choice(UNIT_TASKS)
        v = r.choice(vals)
        out = js_round(v * a / b, 10)
    return f, t, (int(v) if float(v).is_integer() else v), out


# ───────── currency ─────────

CURRENCIES = {"USD": ("US dollars", "dollars"), "EUR": ("euros", "euros"), "GBP": ("British pounds", "pounds"), "INR": ("Indian rupees", "rupees"),
              "JPY": ("Japanese yen", "yen"), "AUD": ("Australian dollars", "AUD"), "CAD": ("Canadian dollars", "CAD"), "AED": ("UAE dirhams", "dirhams"),
              "SGD": ("Singapore dollars", "SGD"), "CHF": ("Swiss francs", "francs"), "CNY": ("Chinese yuan", "yuan"), "BRL": ("Brazilian reais", "reais"),
              "MXN": ("Mexican pesos", "pesos"), "ZAR": ("South African rand", "rand"), "KRW": ("South Korean won", "won"), "THB": ("Thai baht", "baht"),
              "LKR": ("Sri Lankan rupees", "LKR"), "MYR": ("Malaysian ringgit", "ringgit"), "SAR": ("Saudi riyals", "riyals"), "NZD": ("New Zealand dollars", "NZD")}
# Approximate units per 1 USD (realistic magnitudes; the tool supplies live rates).
PER_USD = {"USD": 1, "EUR": 0.92, "GBP": 0.79, "INR": 83.4, "JPY": 151.2, "AUD": 1.52, "CAD": 1.36, "AED": 3.6725, "SGD": 1.35, "CHF": 0.9,
           "CNY": 7.24, "BRL": 5.1, "MXN": 17.2, "ZAR": 18.6, "KRW": 1360.0, "THB": 36.4, "LKR": 300.5, "MYR": 4.7, "SAR": 3.75, "NZD": 1.66}


def currency_task(r, held_out):
    pairs = [(a, b) for a in CURRENCIES for b in CURRENCIES if a != b and held(a + b) == held_out]
    f, t = r.choice(pairs)
    amount = r.choice([1, 10, 20, 50, 100, 250, 500, 1000, 1500, 5000, 12000])
    rate = js_round(PER_USD[t] / PER_USD[f] * r.uniform(0.97, 1.03), 6)
    converted = round(amount * rate, 2)
    converted = int(converted) if float(converted).is_integer() else converted
    day = dt.date(2025, 1, 1) + dt.timedelta(days=r.randrange(600))
    updated = day.strftime("%a, %d %b %Y") + " 00:02:31 +0000"
    return f, t, amount, rate, converted, updated


# ───────── world time ─────────

ZONES = [("Tokyo", "Japan", "Asia/Tokyo"), ("London", "United Kingdom", "Europe/London"), ("New York", "United States", "America/New_York"),
         ("Los Angeles", "United States", "America/Los_Angeles"), ("Chicago", "United States", "America/Chicago"), ("Sydney", "Australia", "Australia/Sydney"),
         ("Dubai", "United Arab Emirates", "Asia/Dubai"), ("Singapore", "Singapore", "Asia/Singapore"), ("Mumbai", "India", "Asia/Kolkata"),
         ("Chennai", "India", "Asia/Kolkata"), ("Delhi", "India", "Asia/Kolkata"), ("Paris", "France", "Europe/Paris"), ("Berlin", "Germany", "Europe/Berlin"),
         ("Moscow", "Russia", "Europe/Moscow"), ("Beijing", "China", "Asia/Shanghai"), ("Hong Kong", "Hong Kong", "Asia/Hong_Kong"), ("Seoul", "South Korea", "Asia/Seoul"),
         ("Bangkok", "Thailand", "Asia/Bangkok"), ("Jakarta", "Indonesia", "Asia/Jakarta"), ("Cairo", "Egypt", "Africa/Cairo"), ("Lagos", "Nigeria", "Africa/Lagos"),
         ("Nairobi", "Kenya", "Africa/Nairobi"), ("Johannesburg", "South Africa", "Africa/Johannesburg"), ("São Paulo", "Brazil", "America/Sao_Paulo"),
         ("Mexico City", "Mexico", "America/Mexico_City"), ("Toronto", "Canada", "America/Toronto"), ("Vancouver", "Canada", "America/Vancouver"),
         ("Auckland", "New Zealand", "Pacific/Auckland"), ("Istanbul", "Turkey", "Europe/Istanbul"), ("Karachi", "Pakistan", "Asia/Karachi"),
         ("Dhaka", "Bangladesh", "Asia/Dhaka"), ("Kathmandu", "Nepal", "Asia/Kathmandu"), ("Colombo", "Sri Lanka", "Asia/Colombo"), ("Riyadh", "Saudi Arabia", "Asia/Riyadh"),
         ("Tehran", "Iran", "Asia/Tehran"), ("Honolulu", "United States", "Pacific/Honolulu"), ("Buenos Aires", "Argentina", "America/Argentina/Buenos_Aires"),
         ("Lima", "Peru", "America/Lima"), ("Madrid", "Spain", "Europe/Madrid"), ("Rome", "Italy", "Europe/Rome"), ("Amsterdam", "Netherlands", "Europe/Amsterdam"),
         ("Kuala Lumpur", "Malaysia", "Asia/Kuala_Lumpur"), ("Manila", "Philippines", "Asia/Manila"), ("Hanoi", "Vietnam", "Asia/Bangkok"), ("Reykjavik", "Iceland", "Atlantic/Reykjavik")]


def time_task(r, held_out):
    city, country, zone = r.choice([z for z in ZONES if held(z[0]) == held_out])
    utc = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=r.randrange(600 * 24 * 60))
    local = utc.astimezone(ZoneInfo(zone))
    off = local.utcoffset()
    mins = int(off.total_seconds() // 60)
    sign = "+" if mins >= 0 else "-"
    offset = f"UTC{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"
    result = {"location": f"{city}, {country}", "timezone": zone, "time": local.strftime("%H:%M"),
              "date": f"{local.strftime('%A')}, {local.day} {local.strftime('%B %Y')}", "utc_offset": offset}
    return city, result, local


# ───────── Wikipedia ─────────

WIKI = [
    ("Eiffel Tower", "Wrought-iron lattice tower in Paris, France", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel's company and built for the 1889 World's Fair. At about 330 metres tall, it was the tallest man-made structure in the world until 1930."),
    ("Albert Einstein", "German-born theoretical physicist (1879–1955)", "Albert Einstein was a German-born theoretical physicist best known for developing the theory of relativity. He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. His equation E = mc² expresses the equivalence of mass and energy."),
    ("Photosynthesis", "Process by which plants make food from light", "Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy. Using sunlight, they turn carbon dioxide and water into glucose and release oxygen. It takes place mainly in the chloroplasts of plant cells."),
    ("Mount Everest", "Earth's highest mountain above sea level", "Mount Everest is Earth's highest mountain above sea level, at 8,849 metres. It lies in the Himalayas on the border between Nepal and China. Edmund Hillary and Tenzing Norgay made the first confirmed ascent in 1953."),
    ("Taj Mahal", "Marble mausoleum in Agra, India", "The Taj Mahal is an ivory-white marble mausoleum on the bank of the Yamuna river in Agra, India. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his wife Mumtaz Mahal. It is a UNESCO World Heritage Site."),
    ("Great Wall of China", "Series of fortifications in northern China", "The Great Wall of China is a series of fortifications built across the historical northern borders of China. Walls were built from as early as the 7th century BC, and most of the existing wall dates from the Ming dynasty. Its many branches together stretch for thousands of kilometres."),
    ("Marie Curie", "Polish and naturalised-French physicist and chemist (1867–1934)", "Marie Curie was a physicist and chemist who did pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences, physics and chemistry. She discovered the elements polonium and radium."),
    ("Isaac Newton", "English mathematician and physicist (1643–1727)", "Isaac Newton was an English mathematician, physicist and astronomer. His book Principia Mathematica, published in 1687, set out the laws of motion and universal gravitation. He also made key contributions to optics and, with Leibniz, to the development of calculus."),
    ("Black hole", "Region of spacetime from which nothing escapes", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. Black holes form when very massive stars collapse at the end of their lives. The boundary beyond which escape is impossible is called the event horizon."),
    ("DNA", "Molecule that carries genetic instructions", "DNA (deoxyribonucleic acid) is the molecule that carries the genetic instructions of living organisms. It has a double-helix structure made of two strands of nucleotides. The sequence of its four bases, A, T, C and G, encodes genetic information."),
    ("Moon", "Earth's only natural satellite", "The Moon is Earth's only natural satellite, orbiting at an average distance of about 384,400 km. It is thought to have formed about 4.5 billion years ago. Its gravity causes the tides, and humans first landed on it in 1969 with Apollo 11."),
    ("Amazon rainforest", "Tropical rainforest in South America", "The Amazon rainforest is a moist broadleaf tropical rainforest covering much of the Amazon basin in South America. Most of it lies in Brazil, with parts in Peru, Colombia and other countries. It is the largest rainforest on Earth and is home to an enormous variety of species."),
    ("Leonardo da Vinci", "Italian Renaissance polymath (1452–1519)", "Leonardo da Vinci was an Italian polymath of the High Renaissance, active as a painter, engineer, scientist and inventor. His paintings include the Mona Lisa and The Last Supper. His notebooks contain designs for flying machines and detailed anatomical drawings."),
    ("Nelson Mandela", "South African anti-apartheid leader and president (1918–2013)", "Nelson Mandela was a South African anti-apartheid activist who served as President of South Africa from 1994 to 1999. He spent 27 years in prison before his release in 1990. He shared the 1993 Nobel Peace Prize with F. W. de Klerk."),
    ("Mahatma Gandhi", "Leader of India's independence movement (1869–1948)", "Mahatma Gandhi was an Indian lawyer and political leader who led India's movement for independence from British rule. He championed nonviolent resistance, including the 1930 Salt March. His birthday, 2 October, is observed as the International Day of Non-Violence."),
    ("A. P. J. Abdul Kalam", "Indian aerospace scientist and 11th President of India (1931–2015)", "A. P. J. Abdul Kalam was an Indian aerospace scientist who served as the 11th President of India from 2002 to 2007. He played a leading role in India's missile and space programmes, earning the nickname 'Missile Man of India'. He was awarded the Bharat Ratna in 1997."),
    ("Volcano", "Rupture in a planet's crust where magma escapes", "A volcano is a rupture in the crust of a planet through which molten rock, ash and gases escape from below the surface. On Earth, most volcanoes occur where tectonic plates diverge or converge. Eruptions can be explosive or produce slow-moving lava flows."),
    ("Bitcoin", "Decentralised digital currency launched in 2009", "Bitcoin is a decentralised digital currency created in 2009 by a person or group using the name Satoshi Nakamoto. Transactions are recorded on a public ledger called a blockchain. The total supply is capped at 21 million bitcoins."),
    ("Solar System", "The Sun and the objects that orbit it", "The Solar System is the Sun together with the objects that orbit it, including eight planets, their moons, dwarf planets, asteroids and comets. It formed about 4.6 billion years ago from a collapsing cloud of gas and dust. Jupiter is its largest planet."),
    ("Great Pyramid of Giza", "Oldest and largest pyramid in Giza, Egypt", "The Great Pyramid of Giza is the largest of the Egyptian pyramids and was built as a tomb for the pharaoh Khufu around 2560 BC. It is the oldest of the Seven Wonders of the Ancient World and the only one largely intact. It was the tallest man-made structure for over 3,800 years."),
    ("William Shakespeare", "English playwright and poet (1564–1616)", "William Shakespeare was an English playwright, poet and actor, widely regarded as the greatest writer in the English language. His works include Hamlet, Romeo and Juliet, Macbeth and A Midsummer Night's Dream. He wrote about 39 plays and 154 sonnets."),
    ("Olympic Games", "International multi-sport event", "The Olympic Games are the leading international sporting events, with thousands of athletes from around the world competing. The modern Games began in Athens in 1896, inspired by the ancient Olympics held in Olympia, Greece. Summer and Winter Games are each held every four years."),
    ("Cricket", "Bat-and-ball game played between two teams of eleven", "Cricket is a bat-and-ball game played between two teams of eleven players on a field with a 22-yard pitch at its centre. The batting side scores runs while the fielding side tries to dismiss batters. Popular formats include Test matches, One Day Internationals and Twenty20."),
    ("Chess", "Two-player strategy board game", "Chess is a board game for two players played on a checkered board of 64 squares. Each player starts with sixteen pieces, and the goal is to checkmate the opponent's king. The game in its modern form emerged in southern Europe in the 15th century."),
    ("Mars", "Fourth planet from the Sun", "Mars is the fourth planet from the Sun and is often called the Red Planet because of iron oxide on its surface. It has a thin atmosphere made mostly of carbon dioxide and two small moons, Phobos and Deimos. Several rovers, including Curiosity and Perseverance, have explored its surface."),
    ("Jupiter", "Largest planet in the Solar System", "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant made mostly of hydrogen and helium. Its Great Red Spot is a giant storm larger than Earth, and it has dozens of moons, including Ganymede, the largest moon in the Solar System."),
    ("Coffee", "Brewed drink made from roasted coffee beans", "Coffee is a drink brewed from the roasted seeds, called beans, of the Coffea plant. It is one of the most consumed beverages in the world and contains caffeine, a stimulant. The two most common species grown are arabica and robusta."),
    ("Sahara", "Largest hot desert in the world", "The Sahara is a desert spanning much of North Africa and is the largest hot desert in the world. It covers about 9 million square kilometres across more than ten countries. Its landscape includes sand dunes, gravel plains and rocky plateaus."),
    ("Nile", "Major river in northeastern Africa", "The Nile is a major north-flowing river in northeastern Africa that empties into the Mediterranean Sea. At about 6,650 km, it has long been considered the longest river in the world. Its two main tributaries are the White Nile and the Blue Nile, which meet at Khartoum."),
    ("Renaissance", "Period of European cultural rebirth (14th–17th century)", "The Renaissance was a period of European history marking the transition from the Middle Ages to modernity, roughly the 14th to 17th centuries. It began in Italy and saw a revival of interest in classical art, science and learning. Its figures include Leonardo da Vinci, Michelangelo and Raphael."),
    ("Electricity", "Phenomena associated with electric charge", "Electricity is the set of physical phenomena associated with the presence and motion of electric charge. It powers lighting, heating, electronics and motors. Electric current is measured in amperes and electric potential in volts."),
    ("Vaccine", "Biological preparation that provides immunity", "A vaccine is a biological preparation that helps the immune system learn to recognise and fight a specific disease. Vaccines often contain weakened or inactivated germs, or parts of them such as proteins or genetic instructions. Edward Jenner developed the first smallpox vaccine in 1796."),
    ("Himalayas", "Mountain range in Asia", "The Himalayas are a mountain range in Asia separating the Indian subcontinent from the Tibetan Plateau. They contain many of the world's highest peaks, including Mount Everest and Kangchenjunga. The range was formed by the collision of the Indian and Eurasian tectonic plates."),
    ("Roman Empire", "Ancient empire centred on Rome (27 BC – AD 476 in the west)", "The Roman Empire was the period of ancient Roman civilisation ruled by emperors, beginning with Augustus in 27 BC. At its height it controlled lands around the entire Mediterranean Sea. The Western Roman Empire fell in AD 476, while the Eastern Roman Empire lasted until 1453."),
    ("Artificial intelligence", "Intelligence shown by machines", "Artificial intelligence (AI) is the capability of computer systems to perform tasks normally associated with human intelligence, such as learning, reasoning and understanding language. Modern AI relies heavily on machine learning, especially neural networks. Applications include chatbots, image recognition and self-driving cars."),
    ("Python (programming language)", "General-purpose programming language", "Python is a high-level, general-purpose programming language known for its readable syntax. It was created by Guido van Rossum and first released in 1991. It is widely used for web development, data science, automation and artificial intelligence."),
    ("Internet", "Global system of interconnected computer networks", "The Internet is the global system of interconnected computer networks that uses the TCP/IP protocol suite to communicate. It grew out of the ARPANET research network of the late 1960s. It carries services such as the World Wide Web, email and video streaming."),
    ("Gravity", "Force of attraction between masses", "Gravity is the fundamental interaction that causes all things with mass or energy to be attracted to one another. On Earth it gives objects weight and makes them fall at about 9.8 m/s². Newton described it with his law of universal gravitation, and Einstein's general relativity explains it as the curvature of spacetime."),
    ("Ocean", "Body of salt water covering most of Earth", "The ocean is the body of salt water that covers about 71% of Earth's surface. It is commonly divided into five oceans: the Pacific, Atlantic, Indian, Southern and Arctic. The Pacific is the largest and deepest, containing the Mariana Trench."),
    ("Tea", "Drink made by steeping tea leaves", "Tea is an aromatic beverage made by pouring hot water over the cured leaves of the Camellia sinensis plant. It originated in China and is the most widely consumed drink in the world after water. Major types include green, black, oolong and white tea."),
]


def wiki_url(title):
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


# ───────── dictionary ─────────

WORDS = [
    ("serendipity", "/ˌsɛɹ.ənˈdɪp.ɪ.ti/", "noun", "The occurrence of happy or useful events by chance.", "Finding that old photo was pure serendipity.", ["chance", "luck", "fortune"]),
    ("ephemeral", "/ɪˈfɛm(ə)ɹəl/", "adjective", "Lasting for a very short time.", "Fame on social media is often ephemeral.", ["fleeting", "brief", "short-lived"]),
    ("ubiquitous", "/juːˈbɪkwɪtəs/", "adjective", "Present, appearing or found everywhere.", "Smartphones have become ubiquitous.", ["everywhere", "omnipresent", "pervasive"]),
    ("resilient", "/ɹɪˈzɪliənt/", "adjective", "Able to recover quickly from difficulties.", "Children are often surprisingly resilient.", ["tough", "hardy", "adaptable"]),
    ("meticulous", "/məˈtɪkjələs/", "adjective", "Showing great attention to detail; very careful and precise.", "She kept meticulous records of every expense.", ["careful", "precise", "thorough"]),
    ("pragmatic", "/pɹæɡˈmætɪk/", "adjective", "Dealing with things sensibly and realistically in a practical way.", "We need a pragmatic solution, not a perfect one.", ["practical", "realistic", "sensible"]),
    ("eloquent", "/ˈɛləkwənt/", "adjective", "Fluent or persuasive in speaking or writing.", "She gave an eloquent speech at the ceremony.", ["articulate", "expressive", "fluent"]),
    ("benevolent", "/bəˈnɛvələnt/", "adjective", "Well meaning and kindly.", "A benevolent stranger paid for our meal.", ["kind", "generous", "charitable"]),
    ("candid", "/ˈkændɪd/", "adjective", "Truthful and straightforward; frank.", "Thank you for your candid feedback.", ["frank", "honest", "open"]),
    ("diligent", "/ˈdɪlɪdʒənt/", "adjective", "Having or showing care and effort in one's work or duties.", "A diligent student reviews notes every day.", ["hard-working", "industrious", "conscientious"]),
    ("frugal", "/ˈfɹuːɡəl/", "adjective", "Sparing or economical with money or food.", "Living frugally helped them save for a house.", ["thrifty", "economical", "careful"]),
    ("gregarious", "/ɡɹɪˈɡɛəɹiəs/", "adjective", "Fond of company; sociable.", "He's gregarious and loves meeting new people.", ["sociable", "outgoing", "friendly"]),
    ("inevitable", "/ɪnˈɛvɪtəbəl/", "adjective", "Certain to happen; unavoidable.", "Some delays are inevitable on long trips.", ["unavoidable", "certain", "inescapable"]),
    ("lucid", "/ˈluːsɪd/", "adjective", "Expressed clearly; easy to understand.", "The teacher gave a lucid explanation of fractions.", ["clear", "understandable", "coherent"]),
    ("nostalgia", "/nɒˈstæl.dʒə/", "noun", "A sentimental longing for a period in the past.", "The old songs filled her with nostalgia.", ["wistfulness", "longing", "reminiscence"]),
    ("obsolete", "/ˌɒbsəˈliːt/", "adjective", "No longer produced or used; out of date.", "Floppy disks are now obsolete.", ["outdated", "outmoded", "old-fashioned"]),
    ("plausible", "/ˈplɔːzɪbəl/", "adjective", "Seeming reasonable or probable.", "That's a plausible explanation for the delay.", ["believable", "credible", "reasonable"]),
    ("tenacious", "/təˈneɪʃəs/", "adjective", "Holding firmly to something; persistent.", "She was tenacious in pursuing her goals.", ["persistent", "determined", "stubborn"]),
    ("verbose", "/vɜːˈbəʊs/", "adjective", "Using more words than needed.", "The report was verbose and hard to follow.", ["wordy", "long-winded", "rambling"]),
    ("whimsical", "/ˈwɪmzɪkəl/", "adjective", "Playfully quaint or fanciful.", "The café had whimsical decorations.", ["playful", "fanciful", "quirky"]),
    ("ambiguous", "/æmˈbɪɡjuəs/", "adjective", "Open to more than one interpretation.", "The instructions were ambiguous, so we asked again.", ["unclear", "vague", "equivocal"]),
    ("empathy", "/ˈɛmpəθi/", "noun", "The ability to understand and share the feelings of another.", "Good nurses show great empathy for patients.", ["compassion", "understanding", "sympathy"]),
    ("gratitude", "/ˈɡɹætɪtjuːd/", "noun", "The quality of being thankful.", "She expressed her gratitude with a handwritten note.", ["thankfulness", "appreciation", "thanks"]),
    ("hypothesis", "/haɪˈpɒθəsɪs/", "noun", "A proposed explanation made as a starting point for further investigation.", "The experiment was designed to test the hypothesis.", ["theory", "proposition", "assumption"]),
    ("integrity", "/ɪnˈtɛɡɹɪti/", "noun", "The quality of being honest and having strong moral principles.", "He is known for his integrity in business.", ["honesty", "principle", "uprightness"]),
    ("juxtapose", "/ˌdʒʌkstəˈpəʊz/", "verb", "To place close together for contrasting effect.", "The exhibit juxtaposes old and new photographs.", ["contrast", "compare", "set side by side"]),
    ("lethargic", "/ləˈθɑːdʒɪk/", "adjective", "Sluggish and lacking energy.", "The heat made everyone feel lethargic.", ["sluggish", "tired", "listless"]),
    ("mundane", "/mʌnˈdeɪn/", "adjective", "Lacking interest or excitement; dull and ordinary.", "He found the mundane tasks relaxing.", ["ordinary", "dull", "routine"]),
    ("paradox", "/ˈpæɹədɒks/", "noun", "A statement that seems self-contradictory but may express a truth.", "It's a paradox that the more you learn, the less you feel you know.", ["contradiction", "puzzle", "anomaly"]),
    ("procrastinate", "/pɹəˈkɹæstɪneɪt/", "verb", "To delay or postpone action.", "I tend to procrastinate when a task feels too big.", ["delay", "postpone", "put off"]),
    ("redundant", "/ɹɪˈdʌndənt/", "adjective", "No longer needed or useful; superfluous.", "The extra step in the process was redundant.", ["unnecessary", "superfluous", "surplus"]),
    ("versatile", "/ˈvɜːsətaɪl/", "adjective", "Able to adapt to many different functions or activities.", "A versatile actor can play comedy and drama.", ["adaptable", "flexible", "all-round"]),
    ("zealous", "/ˈzɛləs/", "adjective", "Having or showing great energy or enthusiasm for a cause.", "The zealous volunteers worked all weekend.", ["passionate", "fervent", "eager"]),
    ("altruism", "/ˈæltɹuɪzəm/", "noun", "Selfless concern for the well-being of others.", "Donating blood is an act of altruism.", ["selflessness", "generosity", "kindness"]),
    ("catalyst", "/ˈkætəlɪst/", "noun", "A person or thing that causes an important change.", "The new law was a catalyst for reform.", ["trigger", "spark", "stimulus"]),
    ("diligence", "/ˈdɪlɪdʒəns/", "noun", "Careful and persistent work or effort.", "Her diligence paid off in the exam.", ["effort", "industry", "persistence"]),
    ("innovate", "/ˈɪnəveɪt/", "verb", "To make changes by introducing new methods, ideas or products.", "Startups must innovate to survive.", ["invent", "create", "pioneer"]),
    ("mitigate", "/ˈmɪtɪɡeɪt/", "verb", "To make something less severe or painful.", "Planting trees can mitigate city heat.", ["reduce", "lessen", "ease"]),
    ("novice", "/ˈnɒvɪs/", "noun", "A person new to and inexperienced in a job or situation.", "This course is designed for novices.", ["beginner", "learner", "newcomer"]),
    ("optimistic", "/ˌɒptɪˈmɪstɪk/", "adjective", "Hopeful and confident about the future.", "She's optimistic about the job interview.", ["hopeful", "positive", "confident"]),
]

# ───────── translation (common phrases; checked translations) ─────────

PHRASES = {
    "Good morning": {"French": "Bonjour", "Spanish": "Buenos días", "German": "Guten Morgen", "Italian": "Buongiorno", "Portuguese": "Bom dia", "Hindi": "सुप्रभात", "Japanese": "おはようございます"},
    "Thank you very much": {"French": "Merci beaucoup", "Spanish": "Muchas gracias", "German": "Vielen Dank", "Italian": "Grazie mille", "Portuguese": "Muito obrigado", "Hindi": "बहुत धन्यवाद", "Japanese": "どうもありがとうございます"},
    "Where is the train station?": {"French": "Où est la gare ?", "Spanish": "¿Dónde está la estación de tren?", "German": "Wo ist der Bahnhof?", "Italian": "Dov'è la stazione?", "Portuguese": "Onde fica a estação de trem?", "Hindi": "रेलवे स्टेशन कहाँ है?", "Japanese": "駅はどこですか？"},
    "How much does this cost?": {"French": "Combien ça coûte ?", "Spanish": "¿Cuánto cuesta esto?", "German": "Wie viel kostet das?", "Italian": "Quanto costa questo?", "Portuguese": "Quanto custa isto?", "Hindi": "इसकी कीमत कितनी है?", "Japanese": "これはいくらですか？"},
    "Happy birthday": {"French": "Joyeux anniversaire", "Spanish": "Feliz cumpleaños", "German": "Alles Gute zum Geburtstag", "Italian": "Buon compleanno", "Portuguese": "Feliz aniversário", "Hindi": "जन्मदिन मुबारक हो", "Japanese": "お誕生日おめでとう"},
    "See you tomorrow": {"French": "À demain", "Spanish": "Hasta mañana", "German": "Bis morgen", "Italian": "A domani", "Portuguese": "Até amanhã", "Hindi": "कल मिलते हैं", "Japanese": "また明日"},
    "What is your name?": {"French": "Comment vous appelez-vous ?", "Spanish": "¿Cómo te llamas?", "German": "Wie heißen Sie?", "Italian": "Come ti chiami?", "Portuguese": "Qual é o seu nome?", "Hindi": "आपका नाम क्या है?", "Japanese": "お名前は何ですか？"},
    "I don't understand": {"French": "Je ne comprends pas", "Spanish": "No entiendo", "German": "Ich verstehe nicht", "Italian": "Non capisco", "Portuguese": "Eu não entendo", "Hindi": "मुझे समझ नहीं आया", "Japanese": "わかりません"},
    "Can you help me?": {"French": "Pouvez-vous m'aider ?", "Spanish": "¿Puedes ayudarme?", "German": "Können Sie mir helfen?", "Italian": "Puoi aiutarmi?", "Portuguese": "Você pode me ajudar?", "Hindi": "क्या आप मेरी मदद कर सकते हैं?", "Japanese": "手伝ってもらえますか？"},
    "Where is the bathroom?": {"French": "Où sont les toilettes ?", "Spanish": "¿Dónde está el baño?", "German": "Wo ist die Toilette?", "Italian": "Dov'è il bagno?", "Portuguese": "Onde fica o banheiro?", "Hindi": "शौचालय कहाँ है?", "Japanese": "トイレはどこですか？"},
    "I would like a coffee, please": {"French": "Je voudrais un café, s'il vous plaît", "Spanish": "Quisiera un café, por favor", "German": "Ich hätte gern einen Kaffee, bitte", "Italian": "Vorrei un caffè, per favore", "Portuguese": "Eu queria um café, por favor", "Hindi": "मुझे एक कॉफ़ी चाहिए, कृपया", "Japanese": "コーヒーをお願いします"},
    "Good night": {"French": "Bonne nuit", "Spanish": "Buenas noches", "German": "Gute Nacht", "Italian": "Buonanotte", "Portuguese": "Boa noite", "Hindi": "शुभ रात्रि", "Japanese": "おやすみなさい"},
    "Welcome": {"French": "Bienvenue", "Spanish": "Bienvenido", "German": "Willkommen", "Italian": "Benvenuto", "Portuguese": "Bem-vindo", "Hindi": "स्वागत है", "Japanese": "ようこそ"},
}
LANG_CODES = {"French": "fr", "Spanish": "es", "German": "de", "Italian": "it", "Portuguese": "pt", "Hindi": "hi", "Japanese": "ja"}

QR_TEXTS = [
    ("https://example.com/menu", "my restaurant menu link https://example.com/menu"), ("https://github.com/tharun/portfolio", "https://github.com/tharun/portfolio"),
    ("WIFI:T:WPA;S:HomeNet;P:sunflower42;;", "my wifi, network HomeNet password sunflower42"), ("https://wa.me/919876543210", "my whatsapp link https://wa.me/919876543210"),
    ("https://forms.gle/abc123", "this form https://forms.gle/abc123"), ("BEGIN:VCARD\nVERSION:3.0\nFN:Asha Rao\nTEL:+919800000000\nEND:VCARD", "my contact card: Asha Rao, +919800000000"),
    ("https://youtube.com/@mnxchannel", "my youtube channel https://youtube.com/@mnxchannel"), ("https://maps.google.com/?q=Marina+Beach", "a link to Marina Beach on Google Maps"),
    ("upi://pay?pa=shop@upi&pn=Corner%20Shop", "my shop's UPI payment link upi://pay?pa=shop@upi&pn=Corner%20Shop"), ("https://mnx.app/event", "the event page https://mnx.app/event"),
    ("WIFI:T:WPA;S:CafeGuest;P:latte2025;;", "cafe wifi CafeGuest with password latte2025"), ("https://linkedin.com/in/priya-s", "my LinkedIn https://linkedin.com/in/priya-s"),
]


class ToolsMixin:
    """Generator methods; mixed into training/generate.py's Gen."""

    def calc(self):
        q, expr, value, work = calc_tasks(self.r)
        result = {"expression": expr, "result": js_round(value)}
        shown = pretty(value)
        answer = self.pick([f"**{shown}**\n\n{work} = **{shown}**.", f"The answer is **{shown}**.\n\n{work}.", f"**{shown}**. ({work})"])
        return self.conv("calc", q, [("Calculating exactly", "This needs exact math, so I'll use the calculator.", "calculate", {"expression": expr}, result)],
                         ("Reading the result", "The calculator gave the exact value."), answer,
                         {"tool": "calculate", "answer": [re_num(shown)]})

    def units(self):
        f, t, v, out = unit_task(self.r)
        q = self.pick([f"convert {v} {f} to {t}", f"how many {t} is {v} {f}?", f"{v} {f} in {t}", f"What is {v} {f} in {t}?"])
        result = {"value": v, "from": f, "to": t, "result": out}
        shown = pretty(out)
        answer = self.pick([f"**{v} {f} = {shown} {t}**", f"{v} {f} is **{shown} {t}**."])
        return self.conv("units", q, [("Converting units", f"I'll convert {f} to {t} exactly.", "convert_units", {"value": v, "from": f, "to": t}, result)],
                         ("Reading the result", "Got the converted value."), answer,
                         {"tool": "convert_units", "answer": [re_num(shown)]})

    def currency(self):
        f, t, amount, rate, converted, updated = currency_task(self.r, self.held_out)
        fn, fs = CURRENCIES[f]
        tn, ts = CURRENCIES[t]
        q = self.pick([f"convert {amount} {f} to {t}", f"how much is {amount} {fs} in {tn}?", f"{amount} {f.lower()} to {t.lower()}",
                       f"What's {amount} {fn} in {t}?", f"{f} to {t} rate today"])
        if "rate" in q or amount == 1:
            amount, converted = 1, rate
        result = {"amount": amount, "from": f, "to": t, "rate": rate, "converted": converted, "updated": updated}
        shown = f"{converted:,}"
        answer = (f"**{amount:,} {f} = {shown} {t}** at today's rate (1 {f} = {rate} {t}).\n\nRates change constantly, so your bank or card may give slightly less."
                  if amount != 1 else f"**1 {f} = {rate} {t}** right now.\n\nRates move during the day, so treat this as a guide.")
        return self.conv("currency", q, [("Checking exchange rates", "Exchange rates change daily, so I'll get today's rate.", "convert_currency",
                                          {"amount": amount, "from": f, "to": t}, result)],
                         ("Reading the rate", "I have today's rate and the converted amount."), answer,
                         {"tool": "convert_currency", "args": {"from": f, "to": t}, "answer": [re_num(shown)]})

    def world_time(self):
        city, result, local = time_task(self.r, self.held_out)
        q = self.pick([f"what time is it in {city}?", f"time in {city}", f"What's the date and time in {city} right now?", f"current time {city}"])
        h = local.hour
        part = "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if 17 <= h < 21 else "night"
        answer = f"It's **{result['time']}** in {city} ({part}), on {result['date']}.\n\n{city} is on {result['timezone']} ({result['utc_offset']})."
        return self.conv("time", q, [("Checking the time", f"I'll get the current local time in {city}.", "get_time", {"location": city}, result)],
                         ("Reading the time", "Got the local time and date."), answer,
                         {"tool": "get_time", "args": {"location": re_escape(city)}, "answer": [re_escape(result["time"])]})

    def wiki(self):
        title, desc, summary = self.pick([w for w in WIKI if held(w[0]) == self.held_out])
        short = title.split(" (")[0]
        q = self.pick([f"Tell me about {short}", f"who was {short}?" if "–" in desc else f"what is {short}?", f"{short} wikipedia", f"Give me a quick summary of {short}"])
        result = {"title": title, "description": desc, "summary": summary, "url": wiki_url(title)}
        # Split into sentences, but not after initials like "A. P. J." or "F. W.".
        first, *rest = re.split(r"(?<=[a-zA-Z0-9)]{2})\. (?=[A-Z])", summary)
        first = first.rstrip(".")
        bolded = re.sub(re.escape(short), lambda m: f"**{m.group(0)}**", first, count=1, flags=re.I)
        first = bolded if bolded != first else f"**{short}**: {first}"
        answer = f"{first}.\n\n" + ("- " + "\n- ".join(x.rstrip(".") + "." for x in rest) if rest else "") + f"\n\nSource: [{title} on Wikipedia]({wiki_url(title)})"
        return self.conv("wiki", q, [("Looking it up", f"I'll get a reliable summary of {short} from Wikipedia.", "wikipedia", {"topic": short}, result)],
                         ("Summarising", "I'll explain the key facts and cite Wikipedia."), answer,
                         {"tool": "wikipedia", "args": {"topic": re_escape(short.split()[0])}, "answer": ["wikipedia"]})

    def dictionary(self):
        word, ph, pos, definition, example, syn = self.pick([w for w in WORDS if held(w[0]) == self.held_out])
        q = self.pick([f"what does {word} mean?", f"define {word}", f"meaning of {word}", f"{word} meaning and example"])
        result = {"word": word, "phonetic": ph, "meanings": [{"part_of_speech": pos, "definitions": [{"definition": definition, "example": example}], "synonyms": syn}]}
        answer = f"**{word}** ({pos}, {ph}): {definition}\n\n*Example:* “{example}”\n\n*Similar words:* {', '.join(syn)}."
        return self.conv("define", q, [("Looking up the word", f"I'll check the dictionary entry for “{word}”.", "define_word", {"word": word}, result)],
                         ("Explaining the word", "I'll give the meaning, an example and synonyms."), answer,
                         {"tool": "define_word", "args": {"word": word}, "answer": [re_escape(word)]})

    def translation(self):
        items = [(p, l) for p in PHRASES for l in PHRASES[p] if held(p + l) == self.held_out]
        phrase, lang = self.pick(items)
        out = PHRASES[phrase][lang]
        q = self.pick([f"translate \"{phrase}\" to {lang}", f"How do you say \"{phrase}\" in {lang}?", f"{phrase} in {lang}", f"what is '{phrase}' in {lang}"])
        result = {"text": phrase, "from": "en", "to": LANG_CODES[lang], "translation": out}
        answer = f"In {lang}, “{phrase}” is:\n\n**{out}**"
        return self.conv("translate", q, [("Translating", f"I'll translate this into {lang} with the translation tool.", "translate", {"text": phrase, "to": lang}, result)],
                         ("Giving the translation", "Got the translation."), answer,
                         {"tool": "translate", "args": {"to": lang}, "answer": [re_escape(out)]})

    def qr(self):
        text, said = self.pick([t for t in QR_TEXTS if held(t[0]) == self.held_out])
        q = self.pick([f"make a QR code for {said}", f"create a qr code: {said}", f"QR code for {said} please", f"generate qr for {said}"])
        result = {"created": True, "shown_to_user": True, "text": text}
        what = "your Wi-Fi network" if text.startswith("WIFI:") else "your contact card" if text.startswith("BEGIN:VCARD") else text
        answer = f"Here's your QR code. Scan it with any phone camera to open {what}.\n\nTap **Open / save** to download it for printing or sharing."
        return self.conv("qr", q, [("Making a QR code", "I'll turn this into a scannable QR code.", "create_qr_code", {"text": text}, result)],
                         ("QR code ready", "The QR code is shown above."), answer,
                         {"tool": "create_qr_code", "answer": ["QR|scan"]})


def re_escape(s):
    return re.escape(s)


def re_num(shown):
    """Accepts the number with or without thousands separators."""
    return re.escape(shown).replace(",", ",?")
