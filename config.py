"""
Hivedex Configuration
=====================
Central configuration for all Hivedex settings.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SIGNALS_DIR = DATA_DIR / "signals"

# Create directories if they don't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# Data files
EVENTS_CATALOG = DATA_DIR / "events_catalog.csv"
VALIDATION_RESULTS = DATA_DIR / "validation_results.csv"
MANUAL_OUTCOMES = DATA_DIR / "manual_outcomes.csv"

# =============================================================================
# API SETTINGS
# =============================================================================

# Arctic Shift (Reddit)
ARCTIC_SHIFT_BASE_URL = "https://arctic-shift.photon-reddit.com/api"
ARCTIC_SHIFT_RATE_LIMIT = 1.0  # seconds between requests

# GDELT
GDELT_LOOKBACK_DAYS = 90  # Max days GDELT API allows

# Request settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# =============================================================================
# SIGNAL CALCULATION
# =============================================================================

# Reddit signal weights (must sum to 1.0)
REDDIT_WEIGHTS = {
    "volume": 0.35,
    "sentiment": 0.30,
    "momentum": 0.20,
    "engagement": 0.15
}

# GDELT signal weights (must sum to 1.0)
GDELT_WEIGHTS = {
    "coverage": 0.40,
    "tone": 0.25,
    "velocity": 0.20,
    "diversity": 0.15
}

# Combined signal weights
HIVEMIND_WEIGHTS = {
    "reddit": 0.60,
    "gdelt": 0.40
}

# Signal thresholds
SIGNAL_BULLISH_THRESHOLD = 50  # Signal > this = bullish
SIGNAL_ALERT_THRESHOLD = 70   # Signal > this = alert

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Baseline calculation
BASELINE_DAYS = 30  # Days to use for volume baseline

# Lead time calculation
LEAD_TIME_WINDOW = 60  # Days around event to analyze

# Prediction window
PRE_EVENT_DAYS = 14  # Days before event to analyze for prediction

# =============================================================================
# CATEGORIES
# =============================================================================

VALID_CATEGORIES = {
    "stock",
    "movie",
    "tech",
    "gaming",
    "other",
    "politics",
    "entertainment",
    "health",
    "crypto"
}

CATEGORY_COLORS = {
    "stock": "#3B82F6",
    "movie": "#EC4899",
    "tech": "#8B5CF6",
    "gaming": "#10B981",
    "other": "#F59E0B",
    "politics": "#EF4444",
    "entertainment": "#F97316",
    "health": "#14B8A6",
    "crypto": "#FBBF24"
}

# =============================================================================
# VISUALIZATION
# =============================================================================

COLORS = {
    "reddit": "#FF4500",      # Reddit orange
    "gdelt": "#1E88E5",       # News blue
    "hivemind": "#7C3AED",    # Purple
    "positive": "#22C55E",    # Green
    "negative": "#EF4444",    # Red
    "neutral": "#6B7280",     # Gray
    "background": "#F8FAFC",
    "text": "#1E293B"
}

# Chart dimensions
CHART_WIDTH = 700
CHART_HEIGHT = 400

# =============================================================================
# CACHING
# =============================================================================

CACHE_ENABLED = True
CACHE_EXPIRY_HOURS = 24

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("HIVEDEX_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_path(filename: str) -> Path:
    """Get full path for a data file."""
    return DATA_DIR / filename

def get_cache_path(filename: str) -> Path:
    """Get full path for a cache file."""
    return CACHE_DIR / filename

def validate_weights(weights: dict) -> bool:
    """Validate that weights sum to 1.0."""
    return abs(sum(weights.values()) - 1.0) < 0.001


# Validate configuration on import
assert validate_weights(REDDIT_WEIGHTS), "Reddit weights must sum to 1.0"
assert validate_weights(GDELT_WEIGHTS), "GDELT weights must sum to 1.0"
assert validate_weights(HIVEMIND_WEIGHTS), "Hivemind weights must sum to 1.0"
