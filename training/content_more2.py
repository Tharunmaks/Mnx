"""More hand-written content (part 3): direct answers, vague requests,
unsupported requests and document topics."""

QA2 = [
    (["what is a computer virus", "explain computer viruses"], "A **computer virus** is a program that copies itself into other files or programs and spreads, often causing damage or stealing data.\n\nProtect yourself by:\n- Keeping your system and apps **updated**\n- Installing apps only from **trusted stores**\n- Not opening unexpected **attachments or links**\n- Using built-in protection like **Windows Security** or Google Play Protect"),
    (["what is the difference between http and https", "http vs https"], "**HTTPS** is HTTP with encryption (TLS).\n\n| | HTTP | HTTPS |\n|---|---|---|\n| Encrypted | No | Yes |\n| Padlock in browser | No | Yes |\n| Safe for passwords and payments | No | Yes |\n\nAlways look for **https://** before entering personal details."),
    (["what is a smart contract", "explain smart contracts"], "A **smart contract** is a program stored on a blockchain that runs automatically when its conditions are met. For example, it can release a payment once both sides confirm delivery, without a middleman. Ethereum is the best-known platform for them."),
    (["how does a refrigerator work", "how do fridges keep things cold"], "A fridge moves heat **out** of the inside and releases it **outside**.\n\n1. A **compressor** squeezes a refrigerant gas, making it hot.\n2. The hot gas cools down in the **coils at the back**, releasing heat into the room.\n3. The refrigerant passes through a **valve** and expands, becoming very cold.\n4. Inside the fridge, the cold refrigerant **absorbs heat** from your food.\n\nThe cycle repeats, keeping the inside cool."),
    (["what causes rainbows", "why do rainbows form"], "Rainbows form when **sunlight shines through raindrops**.\n\n- Each drop **bends (refracts)** the light as it enters.\n- The light **reflects** off the back of the drop.\n- It bends again as it leaves, **splitting** white light into colors.\n\nYou see a rainbow when the Sun is **behind you** and rain is in front of you. The colors are red, orange, yellow, green, blue, indigo and violet."),
    (["what is the difference between a virus and bacteria", "virus vs bacteria"], "| | Bacteria | Viruses |\n|---|---|---|\n| Living cells? | Yes, single cells | No, need a host cell to multiply |\n| Size | Larger | Much smaller |\n| Examples | Strep throat, tuberculosis | Flu, common cold, COVID-19 |\n| Treated with antibiotics? | Often, yes | **No** |\n\nThat's why antibiotics don't help with a cold."),
    (["what is compound interest", "explain compound interest"], "**Compound interest** is interest earned on both your original money **and** the interest it has already earned.\n\nExample: ₹10,000 at 10% a year:\n- Year 1: ₹11,000\n- Year 2: ₹12,100 (interest on ₹11,000)\n- Year 3: ₹13,310\n\nFormula: **A = P(1 + r)ⁿ**. The longer you leave it, the faster it grows."),
    (["how do airplanes fly", "how do planes stay in the air"], "Planes fly thanks to **lift** from their wings.\n\n- Wings are shaped and angled so air flowing over them is pushed **downward**, which pushes the wing **up**.\n- **Engines** provide thrust to keep air flowing fast over the wings.\n- When lift is greater than the plane's weight, it climbs.\n\nPilots control direction with movable surfaces like the ailerons, elevator and rudder."),
    (["what is a mutual fund", "explain mutual funds"], "A **mutual fund** pools money from many investors and a professional manager invests it in stocks, bonds or both.\n\n- **Pros:** diversification, professional management, you can start small (SIPs from ₹500 in India)\n- **Cons:** fees (the expense ratio), and returns aren't guaranteed\n\n**Index funds** are a low-cost type that simply track a market index."),
    (["what is a firewall", "explain firewalls"], "A **firewall** is a security system that controls network traffic based on rules, like a guard at a door. It blocks unwanted connections from the internet while allowing the ones you need. Your phone, computer and Wi-Fi router usually all have one."),
    (["what's the difference between affect and effect", "affect vs effect"], "- **Affect** is usually a **verb**: *The rain will affect the match.*\n- **Effect** is usually a **noun**: *The rain had a big effect on the match.*\n\nTip: **A**ffect = **A**ction, **E**ffect = **E**nd result."),
    (["what is the difference between its and it's", "its vs it's"], "- **It's** = *it is* or *it has*: *It's raining.*\n- **Its** = belonging to it: *The cat licked its paw.*\n\nTest: if you can replace it with \"it is\", use **it's**."),
    (["how do I write a good resume", "resume tips"], "Tips for a strong resume:\n\n1. **Keep it to 1 page** (2 if you have 10+ years of experience).\n2. Start with a short **summary** of who you are and what you want.\n3. List experience with **achievements and numbers**: \"Cut load time by 40%\" beats \"worked on performance\".\n4. Tailor **skills and keywords** to each job description.\n5. Use a **clean layout** and a simple font, and check spelling twice.\n6. Add links to your **portfolio, GitHub or LinkedIn** if relevant."),
    (["how do I prepare for an interview", "interview tips"], "How to prepare for an interview:\n\n1. **Research the company**: products, values and recent news.\n2. **Re-read the job description** and match your experience to it.\n3. Prepare **stories using STAR** (Situation, Task, Action, Result).\n4. Practise common questions: *Tell me about yourself*, *Why this role?*, *A challenge you solved*.\n5. Prepare **2–3 questions** to ask them.\n6. Plan your outfit and route (or test your camera for video calls) the day before."),
    (["what is a calorie", "explain calories"], "A **calorie** is a unit of energy. On food labels, \"calories\" are actually **kilocalories (kcal)**, the energy your body gets from food.\n\n- Carbohydrates: ~4 kcal per gram\n- Protein: ~4 kcal per gram\n- Fat: ~9 kcal per gram\n\nMost adults need roughly **1,800–2,500 kcal** a day, depending on size and activity."),
    (["what is an IP address", "explain ip address"], "An **IP address** is a number that identifies a device on a network so data knows where to go, like a postal address for the internet.\n\n- **IPv4** looks like `192.168.1.10`\n- **IPv6** looks like `2001:db8::1`\n\nYour home router gives your devices private addresses, and your internet provider gives your router a public one."),
    (["explain the water cycle", "what is the water cycle"], "The **water cycle** is how water moves around Earth:\n\n1. **Evaporation**: the Sun heats water into vapour.\n2. **Condensation**: vapour cools and forms clouds.\n3. **Precipitation**: water falls as rain, snow or hail.\n4. **Collection**: it gathers in oceans, rivers, lakes and underground, and the cycle repeats."),
    (["what's the difference between a meteor and a meteorite", "meteor vs meteorite"], "- A **meteoroid** is a small rock in space.\n- A **meteor** is the streak of light when it burns up in our atmosphere (a \"shooting star\").\n- A **meteorite** is a piece that survives and **lands on Earth**."),
    (["how can I improve my handwriting", "tips for better handwriting"], "To improve your handwriting:\n\n- **Slow down**; speed comes later.\n- Hold the pen **lightly** and write with your arm, not just your fingers.\n- Practise **basic strokes** (lines, circles) and then individual letters.\n- Keep letters a **consistent size and slant**, using lined paper.\n- Copy a sentence you like for **10 minutes a day**."),
    (["what is a leap second", "explain leap seconds"], "A **leap second** is an extra second occasionally added to clocks (UTC) to keep them in step with Earth's slightly irregular rotation. 27 have been added since 1972, and it has been agreed to stop adding them by 2035."),
    (["how do solar panels work", "explain solar panels"], "Solar panels turn sunlight into electricity using **photovoltaic cells**, usually made of silicon.\n\n1. Sunlight hits the cell and knocks **electrons** loose.\n2. The cell's layers push those electrons in one direction, creating a **current** (DC).\n3. An **inverter** converts it to AC for your home.\n\nOne panel typically produces 300–450 watts in full sun."),
    (["what is a semiconductor", "explain semiconductors"], "A **semiconductor** is a material, usually **silicon**, that conducts electricity better than an insulator but worse than a metal, and whose conductivity can be precisely controlled. That control lets engineers build **transistors**, the tiny switches inside every chip in phones and computers."),
    (["what does a VPN do", "explain vpn"], "A **VPN** (Virtual Private Network) creates an **encrypted tunnel** between your device and a VPN server.\n\n- It hides your traffic from others on the same Wi-Fi.\n- Websites see the VPN's IP address instead of yours.\n- It does **not** make you anonymous: the VPN provider can see your traffic, so choose a trustworthy one."),
    (["tell me a joke", "say something funny"], "Why do programmers prefer dark mode?\n\nBecause **light attracts bugs**. 🐛"),
    (["motivate me", "i need motivation"], "You don't have to do everything today, just the **next small step**. Pick one task, set a timer for 15 minutes, and start. Progress builds momentum, and momentum builds motivation. You've got this! 💪"),
    (["how are you", "how's it going"], "I'm doing great, thanks for asking! 😊 What can I help you with today?"),
    (["what is pi", "value of pi"], "**π (pi)** is the ratio of a circle's circumference to its diameter. It's about **3.14159** and its digits go on forever without repeating. It's used to calculate circles: area = πr², circumference = 2πr."),
    (["how many bones are in the human body", "number of bones in adults"], "An adult human has **206 bones**. Babies are born with around 270, and some fuse together as they grow. The smallest is the **stapes** in the ear; the longest is the **femur** (thigh bone)."),
    (["what is the largest planet", "biggest planet in the solar system"], "**Jupiter** is the largest planet in our solar system. It's more than **11 times wider than Earth** and more massive than all the other planets combined."),
    (["what language is spoken in Brazil", "brazil language"], "The official language of Brazil is **Portuguese**, which makes Brazil the largest Portuguese-speaking country in the world."),
]

CLARIFY2 = [
    (["what's the weather like there", "weather there?"], "Which city do you mean? Tell me the place, or tap 📍 to share your location and I'll check it."),
    (["translate this", "can you translate"], "Sure! What text would you like translated, and into which language?"),
    (["fix my code", "my code doesn't work"], "Happy to help! Please paste the code and the error message you see, and tell me which language it is."),
    (["calculate it", "do the math"], "What would you like me to calculate? Send me the numbers or the problem."),
    (["book it", "reserve a table"], "I can't make bookings, but I can find the place and its phone number or map link. Which place, and where?"),
    (["summarize", "give me a summary"], "Sure! Paste the text you'd like summarised, or tell me the topic and I'll search for it."),
    (["send it", "share this"], "I can't send or share things directly. I can create a file (like a PDF or document) that you can download and share yourself. Want me to do that?"),
    (["make it better", "improve this"], "Happy to! What should I improve? Paste the text or code, or tell me which result you mean and what you'd like changed."),
]

UNSUPPORTED2 = [
    (["play some music", "play a song on spotify"], "I can't play music or control other apps. I can recommend songs or playlists though. What mood are you in?"),
    (["take a photo", "open my camera"], "I can't use your camera. You can attach a photo with the 📎 button, and an online Mnx model can describe it for you."),
    (["pay my electricity bill", "transfer money to my friend"], "I can't make payments or access your bank. Your banking or UPI app is the safe way to do that. I can help you work out amounts or make a budget."),
    (["delete my files", "clean my phone storage"], "I can't change files on your phone myself. You can free up space in **Settings → Storage**, or I can explain what's usually safe to delete."),
    (["call an ambulance", "call emergency services"], "I can't make calls. **Please call your local emergency number right away.** In India it's **112**, in the US **911**, and in the UK **999**."),
]

DOC_TOPICS2 = {
    "Digital marketing basics": [
        ("What it is", ["Promoting products online", "Reaches people where they spend time", "Results can be measured precisely"]),
        ("Channels", ["Search engines (SEO and ads)", "Social media", "Email newsletters"]),
        ("Content", ["Useful blog posts and videos", "A consistent brand voice", "Clear calls to action"]),
        ("Measuring success", ["Track visits, sign-ups and sales", "Compare cost per customer", "Test and improve continuously"]),
        ("Getting started", ["Know your audience", "Pick one or two channels first", "Post consistently"]),
    ],
    "Volcanoes": [
        ("What a volcano is", ["An opening in Earth's crust", "Molten rock called magma rises through it", "Called lava once it reaches the surface"]),
        ("Types", ["Shield volcanoes with gentle slopes, like Mauna Loa", "Stratovolcanoes with steep cones, like Mount Fuji", "Cinder cones, small and steep"]),
        ("Eruptions", ["Pressure from gas in magma drives eruptions", "Can be explosive or effusive", "Release ash, gas and lava"]),
        ("Where they are", ["Most lie along tectonic plate boundaries", "The Pacific Ring of Fire has about 75% of active volcanoes", "Hotspots like Hawaii form volcanoes away from boundaries"]),
        ("Benefits and risks", ["Volcanic soil is very fertile", "Geothermal energy", "Eruptions can threaten nearby towns and air travel"]),
    ],
    "Mental health awareness": [
        ("Why it matters", ["Mental health affects how we think, feel and act", "Common conditions include anxiety and depression", "It's as important as physical health"]),
        ("Warning signs", ["Lasting sadness or worry", "Changes in sleep or appetite", "Withdrawing from friends and activities"]),
        ("Healthy habits", ["Regular sleep and exercise", "Staying connected with people", "Limiting alcohol and screen time"]),
        ("Getting help", ["Talk to someone you trust", "See a doctor or counsellor", "Use a helpline in a crisis"]),
        ("Supporting others", ["Listen without judging", "Encourage professional help", "Check in regularly"]),
    ],
    "The Internet of Things": [
        ("What it is", ["Everyday devices connected to the internet", "They collect and share data", "Examples: smart bulbs, watches, thermostats"]),
        ("How it works", ["Sensors collect data", "Devices connect via Wi-Fi, Bluetooth or mobile networks", "Apps and cloud services act on the data"]),
        ("Uses", ["Smart homes", "Health and fitness tracking", "Smart farming and factories"]),
        ("Risks", ["Security weaknesses in cheap devices", "Privacy concerns about data", "Dependence on internet connections"]),
        ("Staying safe", ["Change default passwords", "Update device firmware", "Put devices on a separate Wi-Fi network"]),
    ],
    "Basics of photography": [
        ("Exposure triangle", ["Aperture controls depth of field", "Shutter speed controls motion blur", "ISO controls sensitivity and noise"]),
        ("Composition", ["Rule of thirds", "Leading lines", "Frame your subject"]),
        ("Light", ["Golden hour gives warm, soft light", "Avoid harsh midday sun for portraits", "Use windows for natural indoor light"]),
        ("Phone photography", ["Tap to focus and adjust exposure", "Clean your lens", "Use portrait and night modes"]),
        ("Practice", ["Take photos every day", "Study photographers you admire", "Edit lightly"]),
    ],
    "Global warming and agriculture": [
        ("Overview", ["Farming both affects and is affected by climate change", "Rising temperatures change growing seasons", "Extreme weather damages crops"]),
        ("Impacts", ["Droughts and floods reduce yields", "New pests and diseases spread", "Heat stress affects livestock"]),
        ("Emissions from farming", ["Methane from livestock and rice paddies", "Nitrous oxide from fertilisers", "Deforestation for farmland"]),
        ("Solutions", ["Drought-resistant crops", "Efficient irrigation like drip systems", "Better soil management"]),
        ("The future", ["Precision farming with sensors", "Diverse crops for resilience", "Reducing food waste"]),
    ],
    "How to start coding": [
        ("Pick a goal", ["Websites, apps, games or data", "Your goal decides the language", "Python and JavaScript are great first choices"]),
        ("Learn the basics", ["Variables, conditions and loops", "Functions", "Lists and dictionaries"]),
        ("Practise", ["Code a little every day", "Build small projects", "Solve beginner challenges"]),
        ("Use tools", ["A code editor like VS Code", "Git for version control", "Online communities for help"]),
        ("Keep growing", ["Read other people's code", "Share your projects", "Learn from mistakes"]),
    ],
    "The importance of sleep": [
        ("Why we sleep", ["Restores the body and brain", "Strengthens memory", "Supports the immune system"]),
        ("How much", ["Adults need 7–9 hours", "Teenagers need 8–10 hours", "Children need even more"]),
        ("Sleep stages", ["Light sleep", "Deep sleep for physical recovery", "REM sleep for dreams and memory"]),
        ("Better sleep", ["Keep a regular schedule", "Avoid screens before bed", "Keep the room dark and cool"]),
        ("Effects of poor sleep", ["Lower focus and mood", "Higher risk of illness", "Slower reaction times"]),
    ],
}
HELD_OUT_DOC_TOPICS2 = {
    "Earthquakes": [
        ("What they are", ["Sudden shaking of the ground", "Caused by movement along faults", "Measured by magnitude scales"]),
        ("Causes", ["Tectonic plates grinding past each other", "Stress builds up and is suddenly released", "Some are triggered by volcanoes"]),
        ("Effects", ["Damage to buildings and roads", "Landslides", "Tsunamis from undersea quakes"]),
        ("Staying safe", ["Drop, cover and hold on", "Stay away from windows", "Have an emergency kit ready"]),
        ("Preparedness", ["Earthquake-resistant buildings", "Early warning systems", "Community drills"]),
    ],
}
