# Hivedex Scripts Package

# Core data fetching
from .data_fetcher import (
    fetch_reddit_posts,
    fetch_reddit_comments,
    fetch_gdelt_news,
    fetch_gdelt_timeline,
    fetch_stock_outcome,
    calculate_stock_outcome,
    fetch_movie_outcome,
    fetch_event_data
)

# Signal calculation
from .signal_calculator import (
    analyze_sentiment,
    add_sentiment_to_posts,
    calculate_reddit_signal,
    calculate_gdelt_signal,
    calculate_combined_signal,
    calculate_lead_time,
    validate_prediction,
    process_event_signals
)

# Visualizations (optional - requires altair)
try:
    from .visualizations import (
        create_accuracy_gauge,
        create_category_accuracy_chart,
        create_lead_time_chart,
        create_recent_predictions_table,
        create_signal_timeline,
        create_signal_components_chart,
        create_live_signal_gauge,
        create_trend_chart,
        format_summary_stats,
        COLORS,
        CATEGORY_COLORS
    )
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    COLORS = {
        "reddit": "#FF4500",
        "gdelt": "#1E88E5",
        "hivemind": "#7C3AED",
        "positive": "#22C55E",
        "negative": "#EF4444",
        "neutral": "#6B7280"
    }
    CATEGORY_COLORS = {
        "stock": "#3B82F6",
        "movie": "#EC4899",
        "tech": "#8B5CF6",
        "gaming": "#10B981",
        "other": "#F59E0B"
    }
