# Hivedex API Documentation

Complete reference for all Hivedex Python modules.

---

## scripts/data_fetcher.py

### fetch_reddit_posts()
Fetch Reddit posts from Arctic Shift API.

```python
fetch_reddit_posts(
    subreddits: list,      # List of subreddit names
    keywords: list,        # Search keywords
    start_date: str,       # "YYYY-MM-DD"
    end_date: str,         # "YYYY-MM-DD"
    use_cache: bool = True # Use cached results
) -> pd.DataFrame
```

**Returns:** DataFrame with columns: `id`, `title`, `selftext`, `score`, `num_comments`, `created_utc`, `subreddit`

**Example:**
```python
posts = fetch_reddit_posts(
    subreddits=["wallstreetbets", "stocks"],
    keywords=["NVDA", "nvidia"],
    start_date="2024-11-01",
    end_date="2024-11-20"
)
```

---

### fetch_gdelt_news()
Fetch news articles from GDELT DOC 2.0 API.

```python
fetch_gdelt_news(
    keywords: list,        # Search keywords
    start_date: str,       # "YYYY-MM-DD"
    end_date: str,         # "YYYY-MM-DD"
    use_cache: bool = True
) -> pd.DataFrame
```

**Returns:** DataFrame with columns: `url`, `title`, `seendate`, `domain`, `tone`

---

### fetch_stock_outcome()
Fetch stock price data using yfinance.

```python
fetch_stock_outcome(
    ticker: str,           # Stock symbol (e.g., "NVDA")
    event_date: str,       # "YYYY-MM-DD"
    window_days: int = 30  # Days around event
) -> pd.DataFrame
```

**Returns:** DataFrame with OHLCV data

---

## scripts/signal_calculator.py

### analyze_sentiment()
Analyze sentiment using VADER.

```python
analyze_sentiment(
    texts: list            # List of text strings
) -> list
```

**Returns:** List of compound sentiment scores (-1 to +1)

**Example:**
```python
scores = analyze_sentiment([
    "This is amazing!",
    "Terrible product"
])
# Returns: [0.718, -0.654]
```

---

### calculate_reddit_signal()
Calculate Reddit signal from posts.

```python
calculate_reddit_signal(
    posts_df: pd.DataFrame,
    baseline_days: int = 30,
    date_col: str = "created_utc"
) -> pd.DataFrame
```

**Returns:** DataFrame with daily signal values:
- `date`: Date
- `reddit_signal`: Combined signal (0-100)
- `volume_norm`: Volume component
- `sentiment_norm`: Sentiment component
- `momentum_norm`: Momentum component
- `engagement_norm`: Engagement component

---

### calculate_gdelt_signal()
Calculate GDELT news signal.

```python
calculate_gdelt_signal(
    articles_df: pd.DataFrame,
    timeline_df: pd.DataFrame = None
) -> pd.DataFrame
```

**Returns:** DataFrame with `date`, `gdelt_signal` columns

---

### calculate_combined_signal()
Combine Reddit and GDELT signals.

```python
calculate_combined_signal(
    reddit_signal: pd.DataFrame,
    gdelt_signal: pd.DataFrame,
    reddit_weight: float = 0.60,
    gdelt_weight: float = 0.40
) -> pd.DataFrame
```

**Returns:** DataFrame with `date`, `reddit_signal`, `gdelt_signal`, `hivemind_signal`

---

### calculate_lead_time()
Calculate how many days Reddit led GDELT.

```python
calculate_lead_time(
    signal_df: pd.DataFrame,
    event_date: str,
    signal_threshold: float = 70
) -> dict
```

**Returns:**
```python
{
    'reddit_peak_date': datetime,
    'gdelt_peak_date': datetime,
    'reddit_lead_days': int,
    'event_date': datetime
}
```

---

### validate_prediction()
Validate if prediction was correct.

```python
validate_prediction(
    signal_df: pd.DataFrame,
    event_date: str,
    actual_outcome: str,
    pre_event_days: int = 14
) -> dict
```

**Returns:**
```python
{
    'predicted_direction': 'positive' | 'negative',
    'actual_outcome': str,
    'prediction_correct': bool,
    'avg_signal': float,
    'confidence': float
}
```

---

## scripts/live_monitor.py

### LiveSignalMonitor
Real-time signal monitoring class.

```python
monitor = LiveSignalMonitor(
    subreddits: list,
    keywords: list,
    alert_threshold: float = 70.0,
    lookback_days: int = 30
)

# Fetch current signal
signal = monitor.fetch_current_signal()

# Check for alerts
alert = monitor.check_alert_status()
```

---

### quick_signal_check()
Quick one-off signal check.

```python
quick_signal_check(
    subreddits: list,
    keywords: list,
    days: int = 7
) -> dict
```

**Returns:**
```python
{
    'signal': float,        # 0-100
    'trend': float,         # Change from start
    'trend_direction': str, # 'up', 'down', 'stable'
    'posts_24h': int,
    'sentiment': float,
    'alert': bool,
    'timestamp': str,
    'status': str
}
```

---

## scripts/kalshi_fetcher.py

### calculate_market_signal()
Calculate prediction market signal.

```python
calculate_market_signal(
    market_price: float,      # 0-1 (yes price)
    price_change_24h: float,  # Price change
    volume_24h: float         # Trading volume
) -> dict
```

**Returns:**
```python
{
    'price_signal': float,
    'momentum_signal': float,
    'volume_signal': float,
    'market_signal': float,
    'market_direction': str
}
```

---

### compare_hivemind_vs_markets()
Compare Reddit vs prediction market signals.

```python
compare_hivemind_vs_markets(
    reddit_signal: float,
    reddit_direction: str,
    market_signal: float,
    market_direction: str
) -> dict
```

**Returns:**
```python
{
    'signals_aligned': bool,
    'divergence': float,
    'combined_confidence': float,
    'insight': str
}
```

---

## scripts/data_validator.py

### DataValidator
Validate data quality.

```python
validator = DataValidator(events_df, validations_df)
result = validator.validate_all()
```

**Returns:**
```python
{
    'valid': bool,
    'issues': list,
    'warnings': list,
    'summary': dict
}
```

---

### generate_quality_report()
Generate formatted quality report.

```python
report = generate_quality_report(events_df, validations_df)
print(report)
```

---

## scripts/visualizations.py

### Chart Functions

| Function | Purpose |
|----------|---------|
| `create_accuracy_gauge()` | Circular accuracy gauge |
| `create_category_accuracy_chart()` | Bar chart by category |
| `create_lead_time_chart()` | Box plot of lead times |
| `create_signal_timeline()` | Dual-line signal chart |
| `create_live_signal_gauge()` | Current signal gauge |
| `create_trend_chart()` | 7-day trend line |

All chart functions return `alt.Chart` objects.

---

## scripts/batch_process.py

### process_single_event()
Process one event through full pipeline.

```python
result = process_single_event(
    event_id: str,
    events_df: pd.DataFrame = None
) -> dict
```

---

### process_all_events()
Batch process all events.

```python
results_df = process_all_events(
    events_df: pd.DataFrame = None,
    limit: int = None,
    categories: list = None
) -> pd.DataFrame
```

---

## config.py

### Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `REDDIT_WEIGHTS` | dict | Signal component weights |
| `GDELT_WEIGHTS` | dict | News signal weights |
| `HIVEMIND_WEIGHTS` | dict | Combined signal weights |
| `SIGNAL_BULLISH_THRESHOLD` | 50 | Bullish prediction cutoff |
| `SIGNAL_ALERT_THRESHOLD` | 70 | Alert trigger level |
| `BASELINE_DAYS` | 30 | Volume baseline window |

---

## Error Handling

All functions include try/except blocks and return empty DataFrames or default values on error. Check return values:

```python
posts = fetch_reddit_posts(...)
if posts.empty:
    print("No data returned")
```

---

*API version 1.0 - Hivedex for Hex-a-thon 2025*
