from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

BASELINE_PATH = MODELS_DIR / "baseline.joblib"
ARENA_MODELS_PATH = MODELS_DIR / "arena_models.joblib"
ARENA_EVAL_PATH = REPORTS_DIR / "arena_eval.json"
BERT_DIR = MODELS_DIR / "bert"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MOVIE_PROFILES = {
    "Neon Horizon": {
        "year": 2026,
        "genre": "Sci-Fi",
        "accent_primary": "#A855F7",
        "accent_secondary": "#22D3EE",
        "poster_motif": "🌆",
        "tagline": "In a city of synthetic memories, truth is the rarest glitch.",
    },
    "Midnight Symphony": {
        "year": 2026,
        "genre": "Mystery / Drama",
        "accent_primary": "#F43F5E",
        "accent_secondary": "#FACC15",
        "poster_motif": "🎻",
        "tagline": "One final movement before the curtain falls.",
    },
    "Shadow Protocol": {
        "year": 2026,
        "genre": "Action",
        "accent_primary": "#EF4444",
        "accent_secondary": "#F59E0B",
        "poster_motif": "🔥",
        "tagline": "Every operative has a cover. None survive exposure.",
    },
    "Whispering Hollow": {
        "year": 2026,
        "genre": "Horror",
        "accent_primary": "#8B5CF6",
        "accent_secondary": "#F43F5E",
        "poster_motif": "🕯️",
        "tagline": "Some houses remember every footstep in the dark.",
    },
    "Autumn in Lisbon": {
        "year": 2026,
        "genre": "Romance",
        "accent_primary": "#F59E0B",
        "accent_secondary": "#EC4899",
        "poster_motif": "🚋",
        "tagline": "Two strangers, seven hills, and a season that refused to end.",
    },
    "Cafe Chaos": {
        "year": 2026,
        "genre": "Comedy",
        "accent_primary": "#FACC15",
        "accent_secondary": "#38BDF8",
        "poster_motif": "🥐",
        "tagline": "One Michelin critic. Three burnt souffles. Zero chill.",
    },
}
