# Hivedex

**Proving that Reddit can predict the future.**

[![Accuracy](https://img.shields.io/badge/accuracy-72.7%25-blue)]()
[![Events](https://img.shields.io/badge/events-55-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## The Result

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 72.7% |
| **Average Lead Time** | 7.4 days |
| **Best Category** | Movies (90.9%) |
| **Events Analyzed** | 55 |

Reddit communities correctly predicted 72.7% of historical events—with signals peaking **7-9 days before** mainstream news coverage.

---

## Signal Timeline

![Signal Timeline - NVIDIA Example](assets/signal_timeline_nvidia.png)

*Reddit signals (orange) peak before news coverage (blue), demonstrating predictive lead time.*

---

## Live Monitoring

![7-Day Signal Trend](assets/7-day%20signal%20trend.png)

*Real-time tracking of hivemind activity with configurable alert thresholds.*

---

## How It Works

### 1. Data Collection
- **Reddit**: Arctic Shift API (free, no auth required)
- **News**: GDELT DOC 2.0 API
- **Outcomes**: yfinance (stocks), TMDB (movies), manual curation

### 2. Signal Calculation

**Reddit Signal (0-100)**
```
Signal = Volume×0.35 + Sentiment×0.30 + Momentum×0.20 + Engagement×0.15
```

**GDELT Signal (0-100)**
```
Signal = Coverage×0.40 + Tone×0.25 + Velocity×0.20 + Diversity×0.15
```

**Combined Hivemind Signal**
```
Hivemind = Reddit×0.60 + GDELT×0.40
```

### 3. Validation
- Compare predicted direction to actual outcome
- Calculate lead time (days Reddit peaked before news)
- Track accuracy by category

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the APIs
python -c "from scripts.data_fetcher import fetch_reddit_posts; print('APIs working!')"

# Process all events
python scripts/batch_process.py

# Process specific categories
python scripts/batch_process.py --categories stock movie --limit 10
```

---

## Project Structure

```
Hivedex/
├── assets/                         # Images and media
│   ├── signal_timeline_nvidia.png
│   └── 7-day signal trend.png
│
├── data/
│   ├── events_catalog.csv          # 50+ historical events
│   ├── validation_results.csv      # Prediction accuracy results
│   └── manual_outcomes.csv         # Non-API outcome data
│
├── scripts/
│   ├── data_fetcher.py             # Arctic Shift + GDELT APIs
│   ├── signal_calculator.py        # Signal calculation + VADER
│   ├── batch_process.py            # Process all events
│   ├── visualizations.py           # Altair charts
│   ├── live_monitor.py             # Real-time monitoring
│   └── kalshi_fetcher.py           # Prediction market data
│
├── notebooks/
│   ├── test_apis.ipynb             # API testing
│   ├── hivedex_app.ipynb           # Main Hex app
│   └── hivedex_hex_app.ipynb       # Enhanced Hex app
│
├── hex_config/
│   ├── semantic_model.yaml         # Hex semantic layer
│   └── threads_prompts.md          # AI chatbot config
│
├── docs/
│   ├── METHODOLOGY.md              # Technical approach
│   ├── API.md                      # API documentation
│   └── TROUBLESHOOTING.md          # Common issues
│
├── config.py                       # Central configuration
├── requirements.txt
└── README.md
```

---

## Accuracy by Category

| Category | Events | Accuracy | Avg Lead Time |
|----------|--------|----------|---------------|
| **Movies** | 10 | 90% | 9 days |
| **Stocks** | 15 | 73% | 7 days |
| **Tech** | 10 | 70% | 8 days |
| **Gaming** | 10 | 70% | 5 days |
| **Other** | 5+ | 65% | 6 days |

---

## Case Studies

### SVB Collapse (March 2023)
- **Reddit signal**: Bearish, 8 days early
- **Key insight**: r/finance users identified bond portfolio risks before mainstream media
- **Result**: Correct prediction

### Barbie Box Office ($1.44B)
- **Reddit signal**: Bullish, 9 days early
- **Key insight**: Cross-community excitement (r/movies + r/femalefashionadvice + meme subs)
- **Result**: Correct prediction, outperformed analyst estimates

### The Marvels Underperformance
- **Reddit signal**: Bearish, 16 days early
- **Key insight**: Superhero fatigue sentiment in r/marvelstudios
- **Result**: Correct prediction

### Baldur's Gate 3 Success
- **Reddit signal**: Extremely bullish (92/100)
- **Key insight**: Early Access community validation over 3 years
- **Result**: Correct prediction, 10M+ copies sold

---

## Data Sources

| Source | What We Use | Rate Limit | Auth |
|--------|-------------|------------|------|
| **Arctic Shift** | Reddit posts/comments | ~1 req/sec | None |
| **GDELT** | News articles + tone | Generous | None |
| **yfinance** | Stock prices | Generous | None |
| **TMDB** | Movie box office | 40 req/10s | Free key |

---

## Key APIs

### Data Fetching
```python
from scripts.data_fetcher import fetch_reddit_posts, fetch_gdelt_news

# Fetch Reddit data
posts = fetch_reddit_posts(
    subreddits=["wallstreetbets", "stocks"],
    keywords=["NVDA", "earnings"],
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Fetch news data
articles = fetch_gdelt_news(
    keywords=["NVIDIA", "earnings"],
    start_date="2024-01-01",
    end_date="2024-01-31"
)
```

### Signal Calculation
```python
from scripts.signal_calculator import calculate_reddit_signal, validate_prediction

# Calculate signal
signal = calculate_reddit_signal(posts_df, baseline_days=30)

# Validate prediction
result = validate_prediction(signal_df, event_date, actual_outcome)
```

### Live Monitoring
```python
from scripts.live_monitor import quick_signal_check

signal = quick_signal_check(
    subreddits=["wallstreetbets", "stocks"],
    keywords=["NVDA", "earnings"],
    days=7
)
print(f"Current signal: {signal['signal']}")
```

---

## Technical Notes

### Sentiment Analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner):
- Compound score: -1 (negative) to +1 (positive)
- Optimized for social media text
- Handles slang, emojis, capitalization

### Lead Time Calculation
1. Find peak Reddit signal date
2. Find peak GDELT signal date
3. Lead time = GDELT peak - Reddit peak

### Prediction Logic
- Signal > 50 = Bullish prediction
- Signal < 50 = Bearish prediction

---

## Hex Deployment

1. Create new project at hex.tech
2. Upload data files from `data/`
3. Import `notebooks/hivedex_hex_app.ipynb`
4. Configure required packages (pandas, altair, vaderSentiment)

See `hex_config/` for semantic model and AI configuration.

---

## Documentation

- [Methodology](docs/METHODOLOGY.md) - Technical approach and validation
- [API Reference](docs/API.md) - Detailed API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

---

## License

MIT

---

## The Insight

> **The crowd knows. We just quantified how much.**

500 million Reddit users across 100K+ communities represent massive distributed intelligence. Domain experts share insights anonymously, upvotes surface quality analysis, and communities self-correct misinformation.

Hivedex proves this pattern exists—and makes it explorable.

---

*Built for Hex-a-thon 2025*
