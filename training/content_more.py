"""More hand-written content for the Mnx training-data generator (part 2).

Adds breadth so 100,000 conversations stay diverse: more places, search
topics, image building blocks, document topics, code, maths problems and
direct answers. HELD_OUT_* items are only used for the evaluation set.
"""

MORE_CITIES = [
    ("Ahmedabad", "India"), ("Surat", "India"), ("Nagpur", "India"), ("Indore", "India"), ("Bhopal", "India"), ("Chandigarh", "India"),
    ("Visakhapatnam", "India"), ("Thiruvananthapuram", "India"), ("Mysuru", "India"), ("Mangaluru", "India"), ("Tiruchirappalli", "India"),
    ("Salem", "India"), ("Vellore", "India"), ("Puducherry", "India"), ("Goa", "India"), ("Patna", "India"), ("Ranchi", "India"),
    ("Guwahati", "India"), ("Bhubaneswar", "India"), ("Varanasi", "India"), ("Amritsar", "India"), ("Dehradun", "India"), ("Udaipur", "India"),
    ("Colombo", "Sri Lanka"), ("Kathmandu", "Nepal"), ("Dhaka", "Bangladesh"), ("Karachi", "Pakistan"), ("Lahore", "Pakistan"),
    ("Riyadh", "Saudi Arabia"), ("Abu Dhabi", "United Arab Emirates"), ("Muscat", "Oman"), ("Tehran", "Iran"), ("Tel Aviv", "Israel"),
    ("Beijing", "China"), ("Shanghai", "China"), ("Shenzhen", "China"), ("Hong Kong", "China"), ("Taipei", "Taiwan"), ("Busan", "South Korea"),
    ("Kyoto", "Japan"), ("Sapporo", "Japan"), ("Ho Chi Minh City", "Vietnam"), ("Phnom Penh", "Cambodia"), ("Yangon", "Myanmar"),
    ("Brisbane", "Australia"), ("Adelaide", "Australia"), ("Wellington", "New Zealand"), ("Dublin", "Ireland"), ("Glasgow", "United Kingdom"),
    ("Birmingham", "United Kingdom"), ("Liverpool", "United Kingdom"), ("Brussels", "Belgium"), ("Zurich", "Switzerland"), ("Geneva", "Switzerland"),
    ("Copenhagen", "Denmark"), ("Helsinki", "Finland"), ("Warsaw", "Poland"), ("Krakow", "Poland"), ("Budapest", "Hungary"), ("Bucharest", "Romania"),
    ("Sofia", "Bulgaria"), ("Belgrade", "Serbia"), ("Zagreb", "Croatia"), ("Porto", "Portugal"), ("Seville", "Spain"), ("Valencia", "Spain"),
    ("Naples", "Italy"), ("Florence", "Italy"), ("Venice", "Italy"), ("Hamburg", "Germany"), ("Frankfurt", "Germany"), ("Cologne", "Germany"),
    ("Marseille", "France"), ("Nice", "France"), ("Rotterdam", "Netherlands"), ("Los Angeles", "United States"), ("Miami", "United States"),
    ("Houston", "United States"), ("Dallas", "United States"), ("Atlanta", "United States"), ("Philadelphia", "United States"),
    ("Washington", "United States"), ("Las Vegas", "United States"), ("Phoenix", "United States"), ("San Diego", "United States"),
    ("Portland", "United States"), ("Minneapolis", "United States"), ("Calgary", "Canada"), ("Ottawa", "Canada"), ("Bogotá", "Colombia"),
    ("Santiago", "Chile"), ("Rio de Janeiro", "Brazil"), ("Quito", "Ecuador"), ("Havana", "Cuba"), ("Lagos", "Nigeria"), ("Accra", "Ghana"),
    ("Addis Ababa", "Ethiopia"), ("Casablanca", "Morocco"), ("Johannesburg", "South Africa"), ("Dar es Salaam", "Tanzania"), ("Kampala", "Uganda"),
]
MORE_HELD_OUT_CITIES = [
    ("Nashik", "India"), ("Rajkot", "India"), ("Thrissur", "India"), ("Tirunelveli", "India"), ("Bologna", "Italy"), ("Seville", "Spain"),
    ("Gothenburg", "Sweden"), ("Riga", "Latvia"), ("Tallinn", "Estonia"), ("Cebu", "Philippines"), ("Chiang Mai", "Thailand"),
    ("Penang", "Malaysia"), ("Hobart", "Australia"), ("Halifax", "Canada"), ("Nashville", "United States"), ("Salt Lake City", "United States"),
    ("Montevideo", "Uruguay"), ("Tunis", "Tunisia"), ("Kigali", "Rwanda"), ("Almaty", "Kazakhstan"),
]

COMPANIES = ["Apple", "Samsung", "Google", "Microsoft", "Amazon", "Meta", "Tesla", "Nvidia", "Intel", "AMD", "Sony", "Nintendo", "OnePlus",
             "Xiaomi", "Nothing", "Realme", "Vivo", "Oppo", "Motorola", "Nokia", "Tata Motors", "Mahindra", "Maruti Suzuki", "Ola Electric",
             "Ather", "Reliance Jio", "Airtel", "Infosys", "TCS", "Wipro", "Flipkart", "Zomato", "Swiggy", "Spotify", "Netflix", "Adobe",
             "Toyota", "Hyundai", "BMW", "Volkswagen"]
PRODUCTS = ["new phone", "latest laptop", "next software update", "new electric car", "latest smartwatch", "new earbuds", "new tablet",
            "next game console", "new electric scooter", "new subscription plan", "AI assistant feature", "foldable phone"]
TEAMS = ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru", "Kolkata Knight Riders", "Sunrisers Hyderabad",
         "Rajasthan Royals", "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants", "India", "Australia", "England",
         "South Africa", "New Zealand", "Pakistan", "Sri Lanka", "Real Madrid", "Barcelona", "Atlético Madrid", "Manchester City",
         "Manchester United", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Bayern Munich", "Borussia Dortmund", "PSG", "Juventus",
         "Inter Milan", "AC Milan", "Kerala Blasters", "Mohun Bagan", "Bengaluru FC", "Lakers", "Warriors", "Celtics", "Bulls", "Heat"]
PRICE_ITEMS = ["gold", "22 carat gold", "silver", "petrol", "diesel", "Bitcoin", "Ethereum", "the US dollar in rupees", "the euro in rupees",
               "onions", "tomatoes", "crude oil", "the Sensex", "the Nifty 50", "the iPhone 16", "a PS5", "CNG", "LPG cylinders"]
NEWS_TOPICS = ["space exploration", "electric vehicles", "renewable energy", "the stock market", "artificial intelligence", "climate summits",
               "cricket", "football transfers", "the Olympics", "smartphone launches", "cybersecurity", "the economy", "Mars missions",
               "chess tournaments", "the film industry", "Formula 1", "tennis", "startups", "medical research", "the monsoon"]
SOFTWARE = ["Python", "Node.js", "Android", "iOS", "Windows", "Ubuntu", "Chrome", "Firefox", "React", "Java", "Rust", "Go", "Flutter",
            "Blender", "VS Code", "WhatsApp", "Linux kernel", "PostgreSQL", "Kotlin", "TypeScript"]
MOVIE_WORDS_A = ["Silent", "Last", "Golden", "Broken", "Hidden", "Midnight", "Iron", "Crimson", "Lost", "Eternal", "Wild", "Neon"]
MOVIE_WORDS_B = ["Horizon", "Kingdom", "River", "Code", "Empire", "Storm", "Signal", "Garden", "Frontier", "Echo", "Legacy", "Tide"]

MORE_PLACE_KINDS = [
    (["bakeries", "a bakery", "cake shops"], "bakery", "bakery", ["Sweet", "Crumb", "Golden", "Oven", "Sugar"]),
    (["ice cream shops", "ice cream", "dessert places"], "ice cream", "ice_cream", ["Scoop", "Frost", "Cone", "Sundae"]),
    (["bars", "pubs", "a pub"], "pub", "pub", ["Anchor", "Fox", "Crown", "Barrel"]),
    (["cinemas", "movie theatres", "a cinema"], "cinema", "cinema", ["Grand", "Star", "Regal", "Silver Screen", "PVR"]),
    (["museums", "a museum", "art galleries"], "museum", "museum", ["City", "Heritage", "Modern Art", "History", "Science"]),
    (["libraries", "a library", "a quiet place to study"], "library", "library", ["Central", "Public", "Community", "City"]),
    (["banks", "a bank branch", "banks nearby"], "bank", "bank", ["State", "City", "Union", "National", "Federal"]),
    (["clinics", "a doctor", "a clinic"], "clinic", "clinic", ["Family", "City", "Care", "Health", "Wellness"]),
    (["dentists", "a dentist", "dental clinics"], "dentist", "dentist", ["Smile", "Bright", "Pearl", "Family"]),
    (["vets", "a veterinarian", "pet clinics"], "veterinary", "veterinary", ["Paws", "Pet Care", "Animal", "Furry Friends"]),
    (["bus stops", "a bus stop", "bus stations"], "bus station", "bus_station", ["Central", "City", "Main", "Market"]),
    (["train stations", "a railway station", "the nearest station"], "railway station", "station", ["Central", "Junction", "City", "Main"]),
    (["EV charging stations", "a charging point", "electric car chargers"], "charging station", "charging_station", ["Volt", "Charge", "Power", "Ion"]),
    (["electronics stores", "a phone shop", "mobile shops"], "electronics", "electronics", ["Gadget", "Tech", "Digital", "Smart"]),
    (["salons", "a barber", "hair salons"], "hairdresser", "hairdresser", ["Style", "Trim", "Glow", "Urban"]),
    (["temples", "a temple", "places of worship"], "place of worship", "place_of_worship", ["Sri", "Holy", "Divine", "Ancient"]),
    (["beaches", "a beach", "places to swim"], "beach", "beach", ["Golden", "Silver", "Sunset", "Palm"]),
    (["coworking spaces", "a coworking space", "places to work"], "coworking", "coworking_space", ["Hive", "Desk", "Hub", "Nest"]),
    (["laundromats", "a laundry", "dry cleaners"], "laundry", "laundry", ["Fresh", "Clean", "Spin", "Quick"]),
    (["hardware stores", "a hardware shop", "tool shops"], "hardware", "hardware", ["Bolt", "Builder", "Fix", "Handy"]),
]
MORE_CUISINES = ["Kerala", "Chettinad", "Hyderabadi biryani", "Punjabi", "Bengali", "Gujarati", "Vietnamese", "Greek", "Turkish", "Spanish",
                 "French", "Ethiopian", "Indonesian", "Sri Lankan", "vegan", "seafood", "street food", "Mediterranean", "burger", "sushi"]
MORE_STREETS = ["Gandhi Road", "Nehru Street", "Lakeshore Drive", "King Street", "Queen Street", "Ocean Drive", "Temple Street",
                "College Road", "Ring Road", "Mount Road", "Brigade Road", "Linking Road", "Elm Street", "Maple Avenue", "Harbour Road",
                "Sunset Boulevard", "Victoria Street", "Riverside Walk", "Garden Lane", "Airport Road"]
LANDMARKS = ["Central Station", "City Museum", "the airport", "the Botanical Garden", "the Old Fort", "the University", "the Main Market",
             "the Beach", "the Bus Terminal", "the City Library", "the Stadium", "the Zoo", "the Science Center", "the Art Gallery",
             "the Clock Tower", "the Central Park", "the Harbour", "the Old Town Square", "the Convention Center", "the General Hospital",
             "the Cathedral", "the Opera House", "the Mall", "the Lake"]

ANIMALS = ["a red fox", "a snow leopard", "a baby elephant", "a koala", "a hummingbird", "a humpback whale", "a husky puppy", "a peacock",
           "a sea turtle", "a lion cub", "a tabby cat", "a polar bear", "a barn owl", "a giraffe", "a dolphin", "a hedgehog", "a parrot",
           "a wolf pack", "a jellyfish", "a butterfly", "a panda", "a horse", "a kangaroo", "a chameleon", "a flamingo"]
OBJECTS = ["a vintage camera", "a steaming cup of chai", "an old typewriter", "a bonsai tree", "a stack of books", "a guitar", "a red bicycle",
           "a lantern", "a pocket watch", "a hot air balloon", "a sailing ship", "a classic motorcycle", "a treehouse", "a lighthouse",
           "a robot", "a spaceship", "a castle", "a windmill", "a train", "a teapot", "a chess board", "a crystal ball"]
SETTINGS = ["in a misty forest", "on a quiet beach", "in a neon city", "in snowy mountains", "in a field of wildflowers", "on the moon",
            "under the ocean", "in a cozy library", "on a rooftop", "in a desert", "in a rainforest", "in a Japanese garden",
            "in an old European street", "beside a waterfall", "in outer space", "in a sunflower field", "in a bustling market",
            "on a mountain peak", "in an autumn park", "in a cyberpunk alley", "in a fairy-tale village", "at a train station"]
TIMES = ["at sunrise", "at sunset", "at night", "during golden hour", "in the rain", "under a starry sky", "on a foggy morning",
         "in winter", "in spring", "during a thunderstorm"]
MORE_IMAGE_STYLES = [
    ("cinematic", "cinematic film still, anamorphic lens, moody color grading"),
    ("comic book", "comic book art, bold ink outlines, halftone shading"),
    ("low poly", "low poly 3D art, faceted geometry, soft lighting"),
    ("vaporwave", "vaporwave aesthetic, pink and teal neon, retro grid"),
    ("studio ghibli inspired", "whimsical hand-painted animation style, soft pastel skies"),
    ("isometric", "isometric illustration, clean geometry, bright colors"),
    ("charcoal", "charcoal drawing, expressive strokes, high contrast"),
    ("stained glass", "stained glass window art, rich jewel tones, black leading"),
    ("paper cut", "layered paper-cut art, soft shadows, depth"),
    ("ukiyo-e", "ukiyo-e woodblock print, flat colors, flowing lines"),
    ("macro photo", "macro photograph, extreme detail, shallow depth of field"),
    ("claymation", "claymation style, handmade clay textures, soft studio light"),
    ("synthwave", "synthwave poster, glowing sunset, chrome reflections"),
    ("children's book", "children's book illustration, warm colors, friendly shapes"),
    ("art nouveau", "art nouveau poster, ornate borders, flowing organic lines"),
]
HELD_OUT_ANIMALS = ["a red panda", "an arctic fox", "a sloth"]
HELD_OUT_SETTINGS = ["in a crystal cave", "on a floating island", "in a bamboo grove"]
