"""
Hivedex Data Fetcher
====================
Fetches data from Arctic Shift (Reddit), GDELT (News), and outcome sources.
All APIs are free and require no authentication.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import hashlib
from typing import Optional

# Optional yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Stock data fetching disabled.")

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API endpoints
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"


def _get_cache_key(prefix: str, **kwargs) -> str:
    """Generate a unique cache key from parameters."""
    params_str = json.dumps(kwargs, sort_keys=True)
    hash_str = hashlib.md5(params_str.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_str}"


def _load_from_cache(cache_key: str) -> Optional[pd.DataFrame]:
    """Load data from cache if exists."""
    cache_file = CACHE_DIR / f"{cache_key}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)
    return None


def _save_to_cache(cache_key: str, df: pd.DataFrame) -> None:
    """Save data to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.csv"
    df.to_csv(cache_file, index=False)


# =============================================================================
# ARCTIC SHIFT (Reddit) API
# =============================================================================

def fetch_reddit_posts(
    subreddits: list,
    keywords: list,
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch Reddit posts from Arctic Shift API.

    Args:
        subreddits: List of subreddit names (without r/)
        keywords: List of keywords to search for
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        use_cache: Whether to use cached data

    Returns:
        DataFrame with posts
    """
    cache_key = _get_cache_key(
        "reddit_posts",
        subreddits=subreddits,
        keywords=keywords,
        start_date=start_date,
        end_date=end_date
    )

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            print(f"Loaded {len(cached)} posts from cache")
            return cached

    # Convert dates to Unix timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    all_posts = []

    for subreddit in subreddits:
        for keyword in keywords:
            try:
                # Arctic Shift API endpoint for posts
                url = f"{ARCTIC_SHIFT_BASE}/posts/search"
                params = {
                    "subreddit": subreddit,
                    "title": keyword,
                    "after": start_ts,
                    "before": end_ts,
                    "limit": 100,
                    "sort": "asc"
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    posts = data.get("data", [])
                    all_posts.extend(posts)
                    print(f"Fetched {len(posts)} posts from r/{subreddit} for '{keyword}'")
                else:
                    print(f"Error {response.status_code} for r/{subreddit}: {response.text[:100]}")

                # Rate limiting - be respectful
                time.sleep(0.5)

            except Exception as e:
                print(f"Error fetching r/{subreddit} for '{keyword}': {e}")

    if not all_posts:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_posts)

    # Deduplicate by post ID
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    # Save to cache
    if use_cache and not df.empty:
        _save_to_cache(cache_key, df)

    print(f"Total unique posts: {len(df)}")
    return df


def fetch_reddit_comments(
    post_ids: list,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch comments for specific Reddit posts from Arctic Shift.

    Args:
        post_ids: List of Reddit post IDs
        use_cache: Whether to use cached data

    Returns:
        DataFrame with comments
    """
    cache_key = _get_cache_key("reddit_comments", post_ids=sorted(post_ids[:20]))

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached

    all_comments = []

    for post_id in post_ids[:100]:  # Limit to avoid too many requests
        try:
            url = f"{ARCTIC_SHIFT_BASE}/comments/search"
            params = {
                "link_id": f"t3_{post_id}",
                "limit": 100
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                comments = data.get("data", [])
                all_comments.extend(comments)

            time.sleep(0.3)

        except Exception as e:
            print(f"Error fetching comments for {post_id}: {e}")

    df = pd.DataFrame(all_comments) if all_comments else pd.DataFrame()

    if use_cache and not df.empty:
        _save_to_cache(cache_key, df)

    return df


# =============================================================================
# GDELT (News) API
# =============================================================================

def fetch_gdelt_news(
    keywords: list,
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch news articles from GDELT DOC 2.0 API.

    Args:
        keywords: List of keywords to search for
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        use_cache: Whether to use cached data

    Returns:
        DataFrame with news articles
    """
    cache_key = _get_cache_key(
        "gdelt_news",
        keywords=keywords,
        start_date=start_date,
        end_date=end_date
    )

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            print(f"Loaded {len(cached)} articles from cache")
            return cached

    try:
        # Try using gdeltdoc library first
        from gdeltdoc import GdeltDoc, Filters

        gd = GdeltDoc()

        # Join keywords with OR for search
        keyword_query = " OR ".join(keywords)

        f = Filters(
            keyword=keyword_query,
            start_date=start_date,
            end_date=end_date,
            num_records=250
        )

        articles = gd.article_search(f)

        if articles is not None and not articles.empty:
            if use_cache:
                _save_to_cache(cache_key, articles)
            print(f"Fetched {len(articles)} articles from GDELT")
            return articles

    except ImportError:
        print("gdeltdoc not installed, falling back to direct API")
    except Exception as e:
        print(f"gdeltdoc error: {e}, falling back to direct API")

    # Fallback: Direct GDELT API call
    try:
        keyword_query = " ".join(keywords)  # Space-separated for GDELT

        # GDELT DOC API
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": keyword_query,
            "mode": "artlist",
            "maxrecords": 250,
            "format": "json",
            "startdatetime": start_date.replace("-", "") + "000000",
            "enddatetime": end_date.replace("-", "") + "235959"
        }

        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            # Check if response is JSON
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type.lower():
                print(f"GDELT returned non-JSON response: {content_type}")
                return pd.DataFrame()

            try:
                data = response.json()
            except Exception as json_err:
                print(f"GDELT JSON parse error: {json_err}")
                print(f"Response preview: {response.text[:200]}")
                return pd.DataFrame()

            articles = data.get("articles", [])
            df = pd.DataFrame(articles)

            if use_cache and not df.empty:
                _save_to_cache(cache_key, df)

            print(f"Fetched {len(df)} articles from GDELT direct API")
            return df
        else:
            print(f"GDELT API error: {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching GDELT news: {e}")
        return pd.DataFrame()


def fetch_gdelt_timeline(
    keywords: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch news timeline (daily tone/volume) from GDELT.

    Args:
        keywords: List of keywords
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with daily tone/volume
    """
    try:
        from gdeltdoc import GdeltDoc, Filters

        gd = GdeltDoc()
        keyword_query = " OR ".join(keywords)

        f = Filters(
            keyword=keyword_query,
            start_date=start_date,
            end_date=end_date
        )

        timeline = gd.timeline_search("timelinetone", f)
        return timeline if timeline is not None else pd.DataFrame()

    except Exception as e:
        print(f"Error fetching GDELT timeline: {e}")
        return pd.DataFrame()


# =============================================================================
# OUTCOME DATA (Stocks, Movies)
# =============================================================================

def fetch_stock_outcome(
    ticker: str,
    event_date: str,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Fetch stock price data around an event.

    Args:
        ticker: Stock ticker symbol
        event_date: Event date in YYYY-MM-DD format
        window_days: Days before and after event to fetch

    Returns:
        DataFrame with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        print(f"yfinance not available, cannot fetch stock data for {ticker}")
        return pd.DataFrame()

    try:
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        start = (event_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end = (event_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

        stock = yf.Ticker(ticker)
        history = stock.history(start=start, end=end)

        if history.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()

        # Calculate returns
        history["daily_return"] = history["Close"].pct_change()
        history["cumulative_return"] = (1 + history["daily_return"]).cumprod() - 1

        # Add event marker
        history["is_event_day"] = history.index.date == event_dt.date()

        print(f"Fetched {len(history)} days of data for {ticker}")
        return history.reset_index()

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_stock_outcome(
    stock_df: pd.DataFrame,
    event_date: str,
    days_after: int = 7
) -> dict:
    """
    Calculate stock outcome after an event.

    Args:
        stock_df: Stock price DataFrame
        event_date: Event date
        days_after: Days after event to measure

    Returns:
        Dict with outcome metrics
    """
    if stock_df.empty:
        return {"outcome": "unknown", "change_pct": None}

    event_dt = pd.to_datetime(event_date)

    # Get price on event day and after
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    event_price = stock_df[stock_df["Date"].dt.date == event_dt.date()]["Close"]
    if event_price.empty:
        # Use closest previous trading day
        before = stock_df[stock_df["Date"] < event_dt]["Close"]
        if before.empty:
            return {"outcome": "unknown", "change_pct": None}
        event_price = before.iloc[-1]
    else:
        event_price = event_price.iloc[0]

    # Get price after event
    after = stock_df[stock_df["Date"] > event_dt + timedelta(days=days_after)]["Close"]
    if after.empty:
        after_price = stock_df["Close"].iloc[-1]
    else:
        after_price = after.iloc[0]

    change_pct = ((after_price - event_price) / event_price) * 100

    # Determine outcome
    if change_pct > 5:
        outcome = "positive"
    elif change_pct < -5:
        outcome = "negative"
    else:
        outcome = "neutral"

    return {
        "outcome": outcome,
        "change_pct": round(change_pct, 2),
        "event_price": round(event_price, 2),
        "after_price": round(after_price, 2)
    }


def fetch_movie_outcome(
    movie_title: str,
    tmdb_api_key: Optional[str] = None
) -> dict:
    """
    Fetch movie box office data from TMDB (requires free API key).

    Args:
        movie_title: Movie title to search
        tmdb_api_key: TMDB API key (optional, returns mock data if not provided)

    Returns:
        Dict with movie outcome data
    """
    if not tmdb_api_key:
        print(f"No TMDB API key, returning placeholder for '{movie_title}'")
        return {
            "title": movie_title,
            "revenue": None,
            "budget": None,
            "outcome": "unknown"
        }

    try:
        # Search for movie
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": tmdb_api_key,
            "query": movie_title
        }

        response = requests.get(search_url, params=params)

        if response.status_code != 200:
            return {"title": movie_title, "outcome": "unknown"}

        results = response.json().get("results", [])
        if not results:
            return {"title": movie_title, "outcome": "unknown"}

        movie_id = results[0]["id"]

        # Get movie details
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": tmdb_api_key}

        response = requests.get(details_url, params=params)
        details = response.json()

        revenue = details.get("revenue", 0)
        budget = details.get("budget", 0)

        # Calculate outcome
        if budget > 0 and revenue > 0:
            multiplier = revenue / budget
            if multiplier > 2.5:
                outcome = "positive"
            elif multiplier < 1.5:
                outcome = "negative"
            else:
                outcome = "neutral"
        else:
            outcome = "unknown"

        return {
            "title": movie_title,
            "revenue": revenue,
            "budget": budget,
            "multiplier": revenue / budget if budget > 0 else None,
            "outcome": outcome
        }

    except Exception as e:
        print(f"Error fetching movie data: {e}")
        return {"title": movie_title, "outcome": "unknown"}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fetch_event_data(
    event: dict,
    days_before: int = 60,
    days_after: int = 30
) -> dict:
    """
    Fetch all data for a single event.

    Args:
        event: Event dict from events catalog
        days_before: Days before event to fetch
        days_after: Days after event to fetch

    Returns:
        Dict with reddit_posts, news_articles, outcome data
    """
    event_date = datetime.strptime(event["event_date"], "%Y-%m-%d")
    start_date = (event_date - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end_date = (event_date + timedelta(days=days_after)).strftime("%Y-%m-%d")

    # Parse subreddits and keywords
    subreddits = [s.strip() for s in event["subreddits"].split(",")]
    keywords = [k.strip() for k in event["keywords"].split(",")]

    result = {
        "event": event,
        "reddit_posts": fetch_reddit_posts(subreddits, keywords, start_date, end_date),
        "news_articles": fetch_gdelt_news(keywords, start_date, end_date),
        "news_timeline": fetch_gdelt_timeline(keywords, start_date, end_date),
        "outcome": None
    }

    # Fetch outcome based on category
    if event["category"] == "stock" and event.get("ticker"):
        stock_data = fetch_stock_outcome(event["ticker"], event["event_date"])
        result["stock_data"] = stock_data
        result["outcome"] = calculate_stock_outcome(stock_data, event["event_date"])
    elif event["category"] == "movie":
        result["outcome"] = fetch_movie_outcome(event["event_name"])
    else:
        result["outcome"] = {"outcome": event.get("expected_outcome", "unknown")}

    return result


if __name__ == "__main__":
    # Test the data fetchers
    print("Testing Arctic Shift API...")
    posts = fetch_reddit_posts(
        subreddits=["wallstreetbets"],
        keywords=["NVDA", "nvidia"],
        start_date="2024-11-01",
        end_date="2024-11-20"
    )
    print(f"Fetched {len(posts)} posts\n")

    print("Testing GDELT API...")
    news = fetch_gdelt_news(
        keywords=["NVIDIA", "earnings"],
        start_date="2024-11-01",
        end_date="2024-11-20"
    )
    print(f"Fetched {len(news)} articles\n")

    print("Testing yfinance...")
    stock = fetch_stock_outcome("NVDA", "2024-11-20")
    print(f"Fetched {len(stock)} days of stock data")
    outcome = calculate_stock_outcome(stock, "2024-11-20")
    print(f"Outcome: {outcome}")
