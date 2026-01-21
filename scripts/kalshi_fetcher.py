"""
Hivedex Kalshi Prediction Market Integration
=============================================
Fetches prediction market data from Kalshi to compare with Reddit hivemind signals.
This adds a third data source: Reddit + GDELT + Prediction Markets

Kalshi API is free and public for market data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time

# Kalshi API base URL (public endpoints)
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Alternative: Use the public market data endpoint
KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/v1/events"


def fetch_kalshi_markets(
    category: Optional[str] = None,
    status: str = "open",
    limit: int = 100
) -> pd.DataFrame:
    """
    Fetch active prediction markets from Kalshi.

    Args:
        category: Filter by category (politics, economics, tech, etc.)
        status: Market status (open, closed, settled)
        limit: Maximum number of markets to fetch

    Returns:
        DataFrame with market data
    """
    try:
        # Try the public events API first
        url = f"{KALSHI_BASE_URL}/markets"
        params = {
            "limit": limit,
            "status": status
        }

        if category:
            params["series_ticker"] = category

        headers = {
            "Accept": "application/json",
            "User-Agent": "Hivedex/1.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            markets = data.get("markets", [])
            df = pd.DataFrame(markets)
            print(f"Fetched {len(df)} markets from Kalshi")
            return df
        else:
            print(f"Kalshi API returned {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching Kalshi markets: {e}")
        return pd.DataFrame()


def fetch_market_history(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch price history for a specific market.

    Args:
        ticker: Market ticker (e.g., "PRES-2024-DEM")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with price history
    """
    try:
        url = f"{KALSHI_BASE_URL}/markets/{ticker}/history"

        params = {}
        if start_date:
            params["min_ts"] = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        if end_date:
            params["max_ts"] = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        headers = {
            "Accept": "application/json",
            "User-Agent": "Hivedex/1.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            history = data.get("history", [])
            df = pd.DataFrame(history)
            print(f"Fetched {len(df)} price points for {ticker}")
            return df
        else:
            print(f"Kalshi history API returned {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching market history: {e}")
        return pd.DataFrame()


def get_polymarket_data(market_slug: str) -> dict:
    """
    Fetch data from Polymarket (alternative prediction market).
    Uses their public GraphQL API.

    Args:
        market_slug: Market identifier

    Returns:
        Dict with market data
    """
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"slug": market_slug}

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data[0] if data else {}
        return {}

    except Exception as e:
        print(f"Error fetching Polymarket data: {e}")
        return {}


def calculate_market_signal(
    market_price: float,
    price_change_24h: float,
    volume_24h: float
) -> dict:
    """
    Calculate a prediction market signal similar to Reddit/GDELT signals.

    Args:
        market_price: Current yes price (0-1)
        price_change_24h: 24h price change
        volume_24h: 24h trading volume

    Returns:
        Dict with signal components
    """
    # Price component (0-100): Higher price = stronger signal
    price_signal = market_price * 100

    # Momentum component (0-100): Based on price change
    momentum_signal = min(max((price_change_24h + 0.1) / 0.2 * 100, 0), 100)

    # Volume component (0-100): Normalized volume
    volume_signal = min(volume_24h / 10000 * 100, 100)  # Normalize to 10k

    # Combined signal
    combined_signal = (
        price_signal * 0.50 +
        momentum_signal * 0.30 +
        volume_signal * 0.20
    )

    return {
        "price_signal": round(price_signal, 2),
        "momentum_signal": round(momentum_signal, 2),
        "volume_signal": round(volume_signal, 2),
        "market_signal": round(combined_signal, 2),
        "market_direction": "bullish" if market_price > 0.5 else "bearish"
    }


def compare_hivemind_vs_markets(
    reddit_signal: float,
    reddit_direction: str,
    market_signal: float,
    market_direction: str
) -> dict:
    """
    Compare Reddit hivemind predictions with prediction market odds.

    This is a KEY INSIGHT: When do crowds (Reddit) vs markets (Kalshi) diverge?

    Args:
        reddit_signal: Reddit signal strength (0-100)
        reddit_direction: Reddit prediction direction
        market_signal: Market signal strength (0-100)
        market_direction: Market prediction direction

    Returns:
        Dict with comparison analysis
    """
    # Check alignment
    signals_aligned = reddit_direction == market_direction

    # Calculate divergence
    divergence = abs(reddit_signal - market_signal)

    # Determine who's likely right based on historical patterns
    # (In a real implementation, this would use historical accuracy data)
    if signals_aligned:
        confidence_boost = min(divergence / 2, 15)  # Up to 15% boost when aligned
        combined_confidence = min((reddit_signal + market_signal) / 2 + confidence_boost, 95)
        insight = "Reddit and markets AGREE - high confidence signal"
    else:
        # When they diverge, who to trust?
        if reddit_signal > market_signal + 20:
            insight = "Reddit is MORE bullish than markets - potential early signal"
            combined_confidence = reddit_signal * 0.6 + market_signal * 0.4
        elif market_signal > reddit_signal + 20:
            insight = "Markets are MORE confident than Reddit - smart money signal"
            combined_confidence = market_signal * 0.6 + reddit_signal * 0.4
        else:
            insight = "Slight disagreement between crowd and markets"
            combined_confidence = (reddit_signal + market_signal) / 2

    return {
        "signals_aligned": signals_aligned,
        "divergence": round(divergence, 2),
        "combined_confidence": round(combined_confidence, 2),
        "insight": insight,
        "reddit_signal": reddit_signal,
        "market_signal": market_signal
    }


# Sample prediction market events to track (mapped to our events catalog)
MARKET_EVENT_MAPPING = {
    # Stock events that might have prediction markets
    "stock_001": {"market": "NVDA-EARNINGS-Q3-2024", "type": "earnings"},
    "stock_002": {"market": "TSLA-DELIVERY-Q4-2023", "type": "earnings"},

    # Election events
    "other_001": {"market": "PRES-2024", "type": "election"},

    # Tech events
    "tech_001": {"market": "CHATGPT-100M-USERS", "type": "milestone"},
    "tech_005": {"market": "CLAUDE-VS-GPT4", "type": "comparison"},

    # Movie box office
    "movie_001": {"market": "BARBIE-BOX-OFFICE-1B", "type": "entertainment"},
    "movie_002": {"market": "OPPENHEIMER-BOX-OFFICE", "type": "entertainment"},
}


def generate_sample_market_data() -> pd.DataFrame:
    """
    Generate sample prediction market data for demonstration.
    In production, this would fetch from Kalshi/Polymarket APIs.
    """
    import numpy as np
    np.random.seed(42)

    markets = []
    for event_id, market_info in MARKET_EVENT_MAPPING.items():
        # Generate realistic market data
        yes_price = np.random.uniform(0.3, 0.85)
        volume = np.random.randint(5000, 100000)

        markets.append({
            "event_id": event_id,
            "market_ticker": market_info["market"],
            "market_type": market_info["type"],
            "yes_price": round(yes_price, 3),
            "no_price": round(1 - yes_price, 3),
            "volume_24h": volume,
            "price_change_24h": round(np.random.uniform(-0.1, 0.1), 3),
            "open_interest": volume * np.random.randint(2, 10),
            "last_updated": datetime.now().isoformat()
        })

    return pd.DataFrame(markets)


if __name__ == "__main__":
    print("Testing Kalshi/Prediction Market Integration...")

    # Generate sample data
    market_data = generate_sample_market_data()
    print(f"\nGenerated {len(market_data)} sample market records")
    print(market_data.head())

    # Test signal calculation
    signal = calculate_market_signal(
        market_price=0.72,
        price_change_24h=0.05,
        volume_24h=25000
    )
    print(f"\nSample market signal: {signal}")

    # Test comparison
    comparison = compare_hivemind_vs_markets(
        reddit_signal=75,
        reddit_direction="bullish",
        market_signal=68,
        market_direction="bullish"
    )
    print(f"\nHivemind vs Markets comparison: {comparison}")
