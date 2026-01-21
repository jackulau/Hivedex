"""
Hivedex Live Signal Monitor
============================
Real-time monitoring of Reddit signals for emerging events.
Designed to work with Hex's refresh capabilities.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time

# Import our data fetchers
try:
    from .data_fetcher import fetch_reddit_posts, fetch_gdelt_news
    from .signal_calculator import calculate_reddit_signal, calculate_gdelt_signal, calculate_combined_signal
except ImportError:
    from data_fetcher import fetch_reddit_posts, fetch_gdelt_news
    from signal_calculator import calculate_reddit_signal, calculate_gdelt_signal, calculate_combined_signal


class LiveSignalMonitor:
    """
    Real-time signal monitoring for Reddit topics.
    Designed for use in Hex notebooks with periodic refresh.
    """

    def __init__(
        self,
        subreddits: List[str],
        keywords: List[str],
        alert_threshold: float = 70.0,
        lookback_days: int = 30
    ):
        """
        Initialize live monitor.

        Args:
            subreddits: List of subreddits to monitor
            keywords: Keywords to search for
            alert_threshold: Signal threshold for alerts (0-100)
            lookback_days: Days of history to analyze
        """
        self.subreddits = subreddits
        self.keywords = keywords
        self.alert_threshold = alert_threshold
        self.lookback_days = lookback_days
        self.last_update = None
        self.signal_history = []

    def fetch_current_signal(self) -> dict:
        """
        Fetch and calculate current signal strength.

        Returns:
            Dict with current signal data
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        try:
            # Fetch Reddit data
            posts_df = fetch_reddit_posts(
                self.subreddits,
                self.keywords,
                start_date,
                end_date,
                use_cache=False  # Always fetch fresh for live monitoring
            )

            # Calculate signal
            if not posts_df.empty:
                reddit_signal = calculate_reddit_signal(posts_df, baseline_days=14)

                # Get most recent signal value
                if not reddit_signal.empty:
                    latest = reddit_signal.iloc[-1]
                    current_signal = latest.get("reddit_signal", 0)

                    # Get 7-day trend
                    if len(reddit_signal) >= 7:
                        week_ago = reddit_signal.iloc[-7]["reddit_signal"]
                        trend = current_signal - week_ago
                        trend_direction = "up" if trend > 5 else ("down" if trend < -5 else "stable")
                    else:
                        trend = 0
                        trend_direction = "insufficient_data"

                    result = {
                        "signal": round(current_signal, 1),
                        "trend": round(trend, 1),
                        "trend_direction": trend_direction,
                        "posts_24h": len(posts_df[posts_df["created_utc"] >= datetime.now() - timedelta(days=1)]) if "created_utc" in posts_df.columns else 0,
                        "sentiment": round(latest.get("avg_sentiment", 0), 3),
                        "alert": current_signal >= self.alert_threshold,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }
                else:
                    result = self._empty_result("No signal data calculated")
            else:
                result = self._empty_result("No Reddit posts found")

        except Exception as e:
            result = self._empty_result(f"Error: {str(e)}")

        self.last_update = datetime.now()
        self.signal_history.append(result)

        # Keep only last 100 readings
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

        return result

    def _empty_result(self, message: str) -> dict:
        """Return empty result structure."""
        return {
            "signal": 0,
            "trend": 0,
            "trend_direction": "unknown",
            "posts_24h": 0,
            "sentiment": 0,
            "alert": False,
            "timestamp": datetime.now().isoformat(),
            "status": message
        }

    def get_signal_history_df(self) -> pd.DataFrame:
        """Get signal history as DataFrame."""
        if not self.signal_history:
            return pd.DataFrame()
        return pd.DataFrame(self.signal_history)

    def check_alert_status(self) -> dict:
        """
        Check if current conditions warrant an alert.

        Returns:
            Dict with alert information
        """
        if not self.signal_history:
            return {"alert": False, "reason": "No data"}

        latest = self.signal_history[-1]

        alerts = []

        # High signal alert
        if latest["signal"] >= self.alert_threshold:
            alerts.append(f"Signal above threshold: {latest['signal']:.1f} >= {self.alert_threshold}")

        # Rapid increase alert (if we have history)
        if len(self.signal_history) >= 3:
            recent_signals = [h["signal"] for h in self.signal_history[-3:]]
            if all(s > 0 for s in recent_signals):
                increase = recent_signals[-1] - recent_signals[0]
                if increase > 15:
                    alerts.append(f"Rapid signal increase: +{increase:.1f} in last 3 readings")

        return {
            "alert": len(alerts) > 0,
            "alerts": alerts,
            "current_signal": latest["signal"],
            "timestamp": latest["timestamp"]
        }


def create_watchlist_monitor(watchlist: List[dict]) -> dict:
    """
    Create monitors for multiple watchlist items.

    Args:
        watchlist: List of dicts with 'name', 'subreddits', 'keywords'

    Returns:
        Dict mapping names to current signals
    """
    results = {}

    for item in watchlist:
        name = item.get("name", "Unknown")
        subreddits = item.get("subreddits", [])
        keywords = item.get("keywords", [])

        if subreddits and keywords:
            monitor = LiveSignalMonitor(
                subreddits=subreddits,
                keywords=keywords,
                alert_threshold=item.get("threshold", 70),
                lookback_days=item.get("lookback", 30)
            )
            results[name] = monitor.fetch_current_signal()
        else:
            results[name] = {"status": "Invalid configuration", "signal": 0}

    return results


def quick_signal_check(
    subreddits: List[str],
    keywords: List[str],
    days: int = 7
) -> dict:
    """
    Quick signal check for immediate results.
    Useful for Hex cell quick checks.

    Args:
        subreddits: Subreddits to check
        keywords: Keywords to search
        days: Days to look back

    Returns:
        Dict with signal info
    """
    monitor = LiveSignalMonitor(
        subreddits=subreddits,
        keywords=keywords,
        lookback_days=days
    )
    return monitor.fetch_current_signal()


# Hex-compatible refresh helper
def get_refresh_timestamp() -> str:
    """Return current timestamp for Hex refresh cells."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


# Sample watchlist for demo
SAMPLE_WATCHLIST = [
    {
        "name": "NVIDIA",
        "subreddits": ["nvidia", "wallstreetbets", "stocks"],
        "keywords": ["nvidia", "NVDA", "GPU", "earnings"],
        "threshold": 75
    },
    {
        "name": "AI/ML Hype",
        "subreddits": ["artificial", "MachineLearning", "technology"],
        "keywords": ["GPT", "Claude", "AI", "LLM"],
        "threshold": 70
    },
    {
        "name": "Gaming Releases",
        "subreddits": ["games", "pcgaming", "gaming"],
        "keywords": ["release", "launch", "review"],
        "threshold": 65
    },
    {
        "name": "Box Office",
        "subreddits": ["movies", "boxoffice"],
        "keywords": ["opening", "weekend", "million"],
        "threshold": 70
    }
]


if __name__ == "__main__":
    print("Testing Live Signal Monitor...")

    # Quick test
    result = quick_signal_check(
        subreddits=["technology"],
        keywords=["AI"],
        days=7
    )
    print(f"\nQuick signal check result: {result}")

    # Watchlist test
    print("\nTesting sample watchlist...")
    for item in SAMPLE_WATCHLIST[:1]:  # Just test first one
        print(f"\nMonitoring: {item['name']}")
        monitor = LiveSignalMonitor(
            subreddits=item["subreddits"],
            keywords=item["keywords"],
            alert_threshold=item["threshold"]
        )
        signal = monitor.fetch_current_signal()
        print(f"  Signal: {signal['signal']}")
        print(f"  Status: {signal['status']}")
