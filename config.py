import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Re-ranking ────────────────────────────────────────────────────────────────
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words
CHUNK_OVERLAP = 60    # words

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 12
TOP_K_RERANKED  = 4

# ── Paths ─────────────────────────────────────────────────────────────────────
BROCHURE_DIR      = "brochures"
VECTOR_STORE_PATH = "vectorstore"
LOG_DB_PATH       = "logs/drivewise.db"

# ── Section keywords for structured chunking ──────────────────────────────────
SECTION_KEYWORDS = {
    "Engine & Performance": ["engine", "performance", "powertrain", "horsepower",
                              "torque", "displacement", "rpm", "cylinder", "cc"],
    "Fuel & Mileage":       ["mileage", "fuel", "efficiency", "kmpl", "consumption",
                              "range", "arai", "petrol", "diesel", "electric", "hybrid"],
    "Safety":               ["safety", "airbag", "abs", "esp", "brake", "adas",
                              "collision", "alert", "sensor", "ebd", "isofix"],
    "Dimensions":           ["dimension", "length", "width", "height", "wheelbase",
                              "weight", "ground clearance", "boot", "luggage", "turning"],
    "Interior & Comfort":   ["interior", "comfort", "seating", "seat", "legroom",
                              "cabin", "ventilat", "sunroof", "panoramic", "upholstery"],
    "Infotainment":         ["infotainment", "connectivity", "audio", "bluetooth",
                              "touchscreen", "display", "apple carplay", "android auto",
                              "speaker", "navigation", "wireless"],
    "Exterior & Design":    ["exterior", "design", "style", "color", "colour",
                              "grille", "wheel", "alloy", "led", "headlamp", "fog"],
    "Transmission":         ["transmission", "gearbox", "automatic", "manual",
                              "amt", "cvt", "dct", "imt", "6-speed", "7-speed"],
    "Warranty & Service":   ["warranty", "service", "maintenance", "roadside",
                              "assistance", "year"],
    "Variants & Pricing":   ["variant", "trim", "version", "price", "ex-showroom",
                              "base", "mid", "top"],
}
