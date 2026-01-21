"""
Hivedex Signal Calculator
=========================
Calculates Reddit and GDELT signals for prediction validation.

Signal Components:
- Reddit Signal (0-100): Volume, Sentiment, Momentum, Engagement
- GDELT Signal (0-100): Coverage, Tone, Velocity, Diversity
- Combined Hivemind Signal: Weighted combination (60% Reddit, 40% GDELT)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
_analyzer = SentimentIntensityAnalyzer()


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

def analyze_sentiment(texts: list) -> pd.DataFrame:
    """
    Analyze sentiment of text using VADER.

    Args:
        texts: List of text strings

    Returns:
        DataFrame with neg, neu, pos, compound scores
    """
    results = []

    for text in texts:
        if pd.isna(text) or not str(text).strip():
            results.append({
                "neg": 0, "neu": 1, "pos": 0, "compound": 0
            })
        else:
            scores = _analyzer.polarity_scores(str(text))
            results.append(scores)

    return pd.DataFrame(results)


def add_sentiment_to_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment scores to Reddit posts DataFrame.

    Args:
        posts_df: DataFrame with 'title' and optionally 'selftext' columns

    Returns:
        DataFrame with added sentiment columns
    """
    if posts_df.empty:
        return posts_df

    # Combine title and selftext for full sentiment
    if "selftext" in posts_df.columns:
        full_text = posts_df["title"].fillna("") + " " + posts_df["selftext"].fillna("")
    else:
        full_text = posts_df["title"].fillna("")

    sentiment = analyze_sentiment(full_text.tolist())

    # Add sentiment columns
    for col in ["neg", "neu", "pos", "compound"]:
        posts_df[col] = sentiment[col].values

    return posts_df


# =============================================================================
# REDDIT SIGNAL CALCULATION
# =============================================================================

def calculate_reddit_signal(
    posts_df: pd.DataFrame,
    baseline_days: int = 30,
    date_col: str = "created_utc"
) -> pd.DataFrame:
    """
    Calculate daily Reddit signal from posts data.

    Signal Components:
    - Volume (35%): Post count relative to baseline
    - Sentiment (30%): Average compound sentiment
    - Momentum (20%): Rate of change in volume
    - Engagement (15%): Average post score

    Args:
        posts_df: DataFrame with Reddit posts (must have sentiment already)
        baseline_days: Days to use for baseline calculation
        date_col: Column name for timestamp

    Returns:
        DataFrame with daily signal scores
    """
    if posts_df.empty:
        return pd.DataFrame(columns=[
            "date", "volume", "sentiment", "momentum", "engagement",
            "reddit_signal", "signal_direction"
        ])

    # Ensure sentiment is calculated
    if "compound" not in posts_df.columns:
        posts_df = add_sentiment_to_posts(posts_df)

    # Convert timestamp to date
    if posts_df[date_col].dtype in ["int64", "float64"]:
        posts_df["date"] = pd.to_datetime(posts_df[date_col], unit="s").dt.date
    else:
        posts_df["date"] = pd.to_datetime(posts_df[date_col]).dt.date

    # Daily aggregations
    daily = posts_df.groupby("date").agg({
        "id": "count",               # Volume
        "compound": "mean",          # Sentiment
        "score": ["mean", "sum"]     # Engagement
    }).reset_index()

    daily.columns = ["date", "volume", "sentiment", "avg_score", "total_score"]

    # Fill missing dates
    all_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D"
    ).date
    daily = daily.set_index("date").reindex(all_dates).reset_index()
    daily.columns = ["date", "volume", "sentiment", "avg_score", "total_score"]
    daily = daily.fillna({"volume": 0, "sentiment": 0, "avg_score": 0, "total_score": 0})

    # Calculate baseline (first N days or rolling mean)
    if len(daily) > baseline_days:
        baseline_volume = daily["volume"].iloc[:baseline_days].mean()
    else:
        baseline_volume = daily["volume"].mean()

    baseline_volume = max(baseline_volume, 1)  # Avoid division by zero

    # Calculate momentum (7-day volume change)
    daily["momentum"] = daily["volume"].pct_change(periods=7).fillna(0)
    daily["momentum"] = daily["momentum"].clip(-3, 3)  # Cap extreme values

    # Normalize components to 0-100 scale
    daily["volume_norm"] = _normalize(
        daily["volume"] / baseline_volume,
        min_val=0, max_val=5,  # 0x to 5x baseline
        output_min=0, output_max=100
    )

    daily["sentiment_norm"] = _normalize(
        daily["sentiment"],
        min_val=-1, max_val=1,
        output_min=0, output_max=100
    )

    daily["momentum_norm"] = _normalize(
        daily["momentum"],
        min_val=-1, max_val=2,
        output_min=0, output_max=100
    )

    daily["engagement_norm"] = _normalize(
        daily["avg_score"],
        min_val=0, max_val=daily["avg_score"].quantile(0.95) if len(daily) > 10 else 100,
        output_min=0, output_max=100
    )

    # Combined Reddit Signal (weighted average)
    daily["reddit_signal"] = (
        daily["volume_norm"] * 0.35 +
        daily["sentiment_norm"] * 0.30 +
        daily["momentum_norm"] * 0.20 +
        daily["engagement_norm"] * 0.15
    )

    # Signal direction
    daily["signal_direction"] = daily["sentiment"].apply(
        lambda x: "bullish" if x > 0.2 else ("bearish" if x < -0.2 else "neutral")
    )

    return daily[[
        "date", "volume", "sentiment", "momentum", "avg_score",
        "volume_norm", "sentiment_norm", "momentum_norm", "engagement_norm",
        "reddit_signal", "signal_direction"
    ]]


# =============================================================================
# GDELT SIGNAL CALCULATION
# =============================================================================

def calculate_gdelt_signal(
    articles_df: pd.DataFrame,
    timeline_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate daily GDELT news signal.

    Signal Components:
    - Coverage (40%): Article count per day
    - Tone (25%): Average news sentiment
    - Velocity (20%): Rate of coverage increase
    - Diversity (15%): Number of unique sources

    Args:
        articles_df: DataFrame with news articles
        timeline_df: Optional timeline data with tone

    Returns:
        DataFrame with daily signal scores
    """
    if articles_df.empty:
        return pd.DataFrame(columns=[
            "date", "coverage", "tone", "velocity", "diversity",
            "gdelt_signal", "signal_direction"
        ])

    # Parse date from seendate or datetime column
    date_col = "seendate" if "seendate" in articles_df.columns else "datetime"
    if date_col in articles_df.columns:
        articles_df["date"] = pd.to_datetime(articles_df[date_col]).dt.date
    else:
        # Try to infer date
        for col in ["date", "publishDate", "url_pub_date"]:
            if col in articles_df.columns:
                articles_df["date"] = pd.to_datetime(articles_df[col]).dt.date
                break
        else:
            return pd.DataFrame()

    # Daily aggregations
    domain_col = "domain" if "domain" in articles_df.columns else "source"
    if domain_col not in articles_df.columns:
        articles_df[domain_col] = "unknown"

    daily = articles_df.groupby("date").agg({
        "url": "count",           # Coverage volume
        domain_col: "nunique"     # Source diversity
    }).reset_index()

    daily.columns = ["date", "coverage", "diversity"]

    # Add tone from timeline if available
    if timeline_df is not None and not timeline_df.empty:
        if "datetime" in timeline_df.columns:
            timeline_df["date"] = pd.to_datetime(timeline_df["datetime"]).dt.date
        elif "date" in timeline_df.columns:
            timeline_df["date"] = pd.to_datetime(timeline_df["date"]).dt.date

        if "value" in timeline_df.columns:
            tone_daily = timeline_df.groupby("date")["value"].mean().reset_index()
            tone_daily.columns = ["date", "tone"]
            daily = daily.merge(tone_daily, on="date", how="left")
        else:
            daily["tone"] = 0
    elif "tone" in articles_df.columns:
        tone_daily = articles_df.groupby("date")["tone"].mean().reset_index()
        daily = daily.merge(tone_daily, on="date", how="left")
    else:
        daily["tone"] = 0

    daily["tone"] = daily["tone"].fillna(0)

    # Calculate velocity (3-day coverage change)
    daily["velocity"] = daily["coverage"].pct_change(periods=3).fillna(0)
    daily["velocity"] = daily["velocity"].clip(-2, 5)  # Cap extreme values

    # Normalize components
    daily["coverage_norm"] = _normalize(
        np.log1p(daily["coverage"]),
        min_val=0, max_val=np.log1p(100),
        output_min=0, output_max=100
    )

    daily["tone_norm"] = _normalize(
        daily["tone"],
        min_val=-50, max_val=50,
        output_min=0, output_max=100
    )

    daily["velocity_norm"] = _normalize(
        daily["velocity"],
        min_val=-1, max_val=3,
        output_min=0, output_max=100
    )

    daily["diversity_norm"] = _normalize(
        daily["diversity"] / daily["coverage"].clip(1),
        min_val=0, max_val=1,
        output_min=0, output_max=100
    )

    # Combined GDELT Signal
    daily["gdelt_signal"] = (
        daily["coverage_norm"] * 0.40 +
        daily["tone_norm"] * 0.25 +
        daily["velocity_norm"] * 0.20 +
        daily["diversity_norm"] * 0.15
    )

    # Signal direction
    daily["signal_direction"] = daily["tone"].apply(
        lambda x: "bullish" if x > 10 else ("bearish" if x < -10 else "neutral")
    )

    return daily[[
        "date", "coverage", "tone", "velocity", "diversity",
        "coverage_norm", "tone_norm", "velocity_norm", "diversity_norm",
        "gdelt_signal", "signal_direction"
    ]]


# =============================================================================
# COMBINED SIGNAL
# =============================================================================

def calculate_combined_signal(
    reddit_signal: pd.DataFrame,
    gdelt_signal: pd.DataFrame,
    reddit_weight: float = 0.60,
    gdelt_weight: float = 0.40
) -> pd.DataFrame:
    """
    Combine Reddit and GDELT signals into unified hivemind signal.

    Args:
        reddit_signal: Reddit signal DataFrame
        gdelt_signal: GDELT signal DataFrame
        reddit_weight: Weight for Reddit signal (default 0.60)
        gdelt_weight: Weight for GDELT signal (default 0.40)

    Returns:
        DataFrame with combined signal
    """
    # Convert date columns to same type
    if not reddit_signal.empty:
        reddit_signal["date"] = pd.to_datetime(reddit_signal["date"]).dt.date
    if not gdelt_signal.empty:
        gdelt_signal["date"] = pd.to_datetime(gdelt_signal["date"]).dt.date

    # Merge on date
    if reddit_signal.empty and gdelt_signal.empty:
        return pd.DataFrame()
    elif reddit_signal.empty:
        combined = gdelt_signal.copy()
        combined["reddit_signal"] = 50  # Neutral
        combined["reddit_direction"] = "neutral"
    elif gdelt_signal.empty:
        combined = reddit_signal.copy()
        combined["gdelt_signal"] = 50  # Neutral
        combined["gdelt_direction"] = "neutral"
    else:
        combined = reddit_signal.merge(
            gdelt_signal[["date", "gdelt_signal", "signal_direction", "coverage", "tone"]],
            on="date",
            how="outer",
            suffixes=("_reddit", "_gdelt")
        )

    # Rename direction columns
    if "signal_direction_reddit" in combined.columns:
        combined = combined.rename(columns={
            "signal_direction_reddit": "reddit_direction",
            "signal_direction_gdelt": "gdelt_direction"
        })
    elif "signal_direction" in combined.columns and "gdelt_signal" in combined.columns:
        combined = combined.rename(columns={"signal_direction": "reddit_direction"})
        combined["gdelt_direction"] = combined.get("gdelt_direction", "neutral")

    # Fill missing values with neutral signal
    combined["reddit_signal"] = combined["reddit_signal"].fillna(50)
    combined["gdelt_signal"] = combined["gdelt_signal"].fillna(50)

    # Calculate combined hivemind signal
    combined["hivemind_signal"] = (
        combined["reddit_signal"] * reddit_weight +
        combined["gdelt_signal"] * gdelt_weight
    )

    # Alignment bonus/penalty
    combined["signals_aligned"] = (
        (combined.get("reddit_direction", "neutral") == combined.get("gdelt_direction", "neutral")) &
        (combined.get("reddit_direction", "neutral") != "neutral")
    )

    # Combined direction (prefer Reddit as leading indicator)
    combined["hivemind_direction"] = combined.get("reddit_direction", "neutral")

    return combined


# =============================================================================
# LEAD TIME CALCULATION
# =============================================================================

def calculate_lead_time(
    signal_df: pd.DataFrame,
    event_date: str,
    signal_threshold: float = 70
) -> dict:
    """
    Calculate how many days before the event signals peaked.

    Args:
        signal_df: Combined signal DataFrame
        event_date: Event date (YYYY-MM-DD)
        signal_threshold: Threshold to consider signal "activated"

    Returns:
        Dict with lead time metrics
    """
    event_dt = pd.to_datetime(event_date).date()
    signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date

    # Get pre-event signals
    pre_event = signal_df[signal_df["date"] < event_dt].copy()

    if pre_event.empty:
        return {
            "reddit_lead_days": None,
            "gdelt_lead_days": None,
            "reddit_beats_news_by": None,
            "reddit_peak_signal": None,
            "gdelt_peak_signal": None
        }

    # Find first day each signal exceeded threshold
    reddit_activated = pre_event[pre_event["reddit_signal"] >= signal_threshold]
    gdelt_activated = pre_event[pre_event["gdelt_signal"] >= signal_threshold]

    reddit_first_date = reddit_activated["date"].min() if not reddit_activated.empty else None
    gdelt_first_date = gdelt_activated["date"].min() if not gdelt_activated.empty else None

    # Calculate lead times
    reddit_lead = (event_dt - reddit_first_date).days if reddit_first_date else None
    gdelt_lead = (event_dt - gdelt_first_date).days if gdelt_first_date else None

    # How many days Reddit beat news
    if reddit_first_date and gdelt_first_date:
        reddit_beats_news = (gdelt_first_date - reddit_first_date).days
    else:
        reddit_beats_news = None

    # Peak signal values
    reddit_peak = pre_event["reddit_signal"].max()
    gdelt_peak = pre_event["gdelt_signal"].max()

    return {
        "reddit_lead_days": reddit_lead,
        "gdelt_lead_days": gdelt_lead,
        "reddit_beats_news_by": reddit_beats_news,
        "reddit_peak_signal": round(reddit_peak, 2),
        "gdelt_peak_signal": round(gdelt_peak, 2),
        "reddit_peak_date": str(pre_event.loc[pre_event["reddit_signal"].idxmax(), "date"]) if not pre_event.empty else None,
        "gdelt_peak_date": str(pre_event.loc[pre_event["gdelt_signal"].idxmax(), "date"]) if not pre_event.empty else None
    }


# =============================================================================
# PREDICTION VALIDATION
# =============================================================================

def validate_prediction(
    signal_df: pd.DataFrame,
    event_date: str,
    actual_outcome: str,
    pre_event_days: int = 14
) -> dict:
    """
    Validate if the signal correctly predicted the outcome.

    Args:
        signal_df: Combined signal DataFrame
        event_date: Event date
        actual_outcome: Actual outcome ('positive', 'negative', 'neutral')
        pre_event_days: Days before event to consider for prediction

    Returns:
        Dict with validation results
    """
    event_dt = pd.to_datetime(event_date).date()
    signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date

    # Get pre-event window
    start_window = event_dt - timedelta(days=pre_event_days)
    pre_event = signal_df[
        (signal_df["date"] >= start_window) &
        (signal_df["date"] < event_dt)
    ]

    if pre_event.empty:
        return {
            "prediction_correct": None,
            "predicted_direction": "unknown",
            "actual_outcome": actual_outcome,
            "signal_strength": None,
            "confidence": None
        }

    # Determine prediction from signal
    avg_reddit_signal = pre_event["reddit_signal"].mean()
    avg_sentiment = pre_event.get("sentiment", pd.Series([0])).mean()

    if avg_reddit_signal > 60 and avg_sentiment > 0.1:
        predicted_direction = "positive"
    elif avg_reddit_signal < 40 or avg_sentiment < -0.1:
        predicted_direction = "negative"
    else:
        predicted_direction = "neutral"

    # Check if prediction matches outcome
    if actual_outcome == "unknown":
        prediction_correct = None
    elif actual_outcome == "neutral":
        prediction_correct = predicted_direction == "neutral"
    else:
        prediction_correct = predicted_direction == actual_outcome

    # Calculate confidence
    signal_strength = pre_event["reddit_signal"].max()
    alignment = pre_event.get("signals_aligned", pd.Series([False])).mean()

    confidence = min(
        (signal_strength / 100 * 0.6 + alignment * 0.4) * 100,
        95
    )

    return {
        "prediction_correct": prediction_correct,
        "predicted_direction": predicted_direction,
        "actual_outcome": actual_outcome,
        "signal_strength": round(signal_strength, 2),
        "avg_signal": round(avg_reddit_signal, 2),
        "avg_sentiment": round(avg_sentiment, 3),
        "confidence": round(confidence, 1)
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize(
    values: pd.Series,
    min_val: float,
    max_val: float,
    output_min: float = 0,
    output_max: float = 100
) -> pd.Series:
    """
    Normalize values to a specified output range.
    """
    clipped = values.clip(min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    scaled = normalized * (output_max - output_min) + output_min
    return scaled.fillna((output_min + output_max) / 2)


def process_event_signals(event_data: dict) -> dict:
    """
    Process all signals for a single event.

    Args:
        event_data: Dict from data_fetcher.fetch_event_data()

    Returns:
        Dict with all signal analysis
    """
    event = event_data["event"]

    # Calculate Reddit signal
    reddit_signal = calculate_reddit_signal(
        add_sentiment_to_posts(event_data["reddit_posts"])
    )

    # Calculate GDELT signal
    gdelt_signal = calculate_gdelt_signal(
        event_data["news_articles"],
        event_data.get("news_timeline")
    )

    # Combine signals
    combined_signal = calculate_combined_signal(reddit_signal, gdelt_signal)

    # Calculate lead time
    lead_time = calculate_lead_time(
        combined_signal,
        event["event_date"]
    )

    # Validate prediction
    actual_outcome = event_data.get("outcome", {}).get("outcome", "unknown")
    validation = validate_prediction(
        combined_signal,
        event["event_date"],
        actual_outcome
    )

    return {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "category": event.get("category"),
        "event_date": event.get("event_date"),
        "reddit_signal": reddit_signal,
        "gdelt_signal": gdelt_signal,
        "combined_signal": combined_signal,
        "lead_time": lead_time,
        "validation": validation,
        "outcome": event_data.get("outcome", {})
    }


if __name__ == "__main__":
    # Test with sample data
    print("Testing signal calculator with sample data...")

    # Create sample Reddit posts
    sample_posts = pd.DataFrame({
        "id": [f"post_{i}" for i in range(100)],
        "title": [
            "NVDA to the moon! AI demand is insane" if i % 3 == 0
            else "Nvidia earnings will beat" if i % 3 == 1
            else "Just bought more NVDA calls"
            for i in range(100)
        ],
        "selftext": [""] * 100,
        "score": [100 + i * 10 for i in range(100)],
        "created_utc": [
            int((datetime(2024, 11, 1) + timedelta(days=i // 5)).timestamp())
            for i in range(100)
        ]
    })

    print("\nCalculating Reddit signal...")
    reddit_signal = calculate_reddit_signal(sample_posts)
    print(f"Days analyzed: {len(reddit_signal)}")
    print(f"Average signal: {reddit_signal['reddit_signal'].mean():.2f}")

    # Create sample news articles
    sample_news = pd.DataFrame({
        "url": [f"http://news.com/article_{i}" for i in range(50)],
        "seendate": [
            (datetime(2024, 11, 1) + timedelta(days=i // 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(50)
        ],
        "domain": [f"source{i % 5}.com" for i in range(50)],
        "tone": [10 + (i % 20) for i in range(50)]
    })

    print("\nCalculating GDELT signal...")
    gdelt_signal = calculate_gdelt_signal(sample_news)
    print(f"Days analyzed: {len(gdelt_signal)}")
    print(f"Average signal: {gdelt_signal['gdelt_signal'].mean():.2f}")

    print("\nCombining signals...")
    combined = calculate_combined_signal(reddit_signal, gdelt_signal)
    print(f"Combined days: {len(combined)}")
    print(f"Average hivemind signal: {combined['hivemind_signal'].mean():.2f}")

    print("\nCalculating lead time...")
    lead = calculate_lead_time(combined, "2024-11-20")
    print(f"Lead time metrics: {lead}")

    print("\nValidating prediction...")
    validation = validate_prediction(combined, "2024-11-20", "positive")
    print(f"Validation: {validation}")
