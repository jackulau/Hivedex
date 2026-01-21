"""
Hivedex Batch Processor
=======================
Processes all events from the catalog and generates validation results.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import fetch_event_data
from signal_calculator import (
    add_sentiment_to_posts,
    calculate_reddit_signal,
    calculate_gdelt_signal,
    calculate_combined_signal,
    calculate_lead_time,
    validate_prediction
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVENTS_FILE = DATA_DIR / "events_catalog.csv"
RESULTS_FILE = DATA_DIR / "validation_results.csv"
SIGNALS_DIR = DATA_DIR / "signals"

# Create signals directory
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


def load_events_catalog() -> pd.DataFrame:
    """Load events catalog, skipping comment lines."""
    events = pd.read_csv(
        EVENTS_FILE,
        comment="#",
        skipinitialspace=True
    )
    return events


def process_single_event(event: pd.Series, save_signals: bool = True) -> dict:
    """
    Process a single event through the full pipeline.

    Args:
        event: Event row from catalog
        save_signals: Whether to save signal timelines

    Returns:
        Dict with validation results
    """
    event_dict = event.to_dict()
    event_id = event_dict.get("event_id", "unknown")

    print(f"\n{'='*60}")
    print(f"Processing: {event_dict.get('event_name', event_id)}")
    print(f"Category: {event_dict.get('category')}")
    print(f"Date: {event_dict.get('event_date')}")
    print(f"{'='*60}")

    try:
        # Step 1: Fetch all data for event
        print("\n1. Fetching data...")
        event_data = fetch_event_data(event_dict)

        reddit_count = len(event_data.get("reddit_posts", pd.DataFrame()))
        news_count = len(event_data.get("news_articles", pd.DataFrame()))
        print(f"   Reddit posts: {reddit_count}")
        print(f"   News articles: {news_count}")

        # Step 2: Calculate signals
        print("\n2. Calculating signals...")

        reddit_posts = event_data.get("reddit_posts", pd.DataFrame())
        if not reddit_posts.empty:
            reddit_posts = add_sentiment_to_posts(reddit_posts)

        reddit_signal = calculate_reddit_signal(reddit_posts)
        gdelt_signal = calculate_gdelt_signal(
            event_data.get("news_articles", pd.DataFrame()),
            event_data.get("news_timeline")
        )
        combined_signal = calculate_combined_signal(reddit_signal, gdelt_signal)

        print(f"   Reddit signal days: {len(reddit_signal)}")
        print(f"   GDELT signal days: {len(gdelt_signal)}")
        print(f"   Combined signal days: {len(combined_signal)}")

        if not combined_signal.empty:
            print(f"   Peak Reddit signal: {combined_signal['reddit_signal'].max():.2f}")
            print(f"   Peak GDELT signal: {combined_signal['gdelt_signal'].max():.2f}")

        # Step 3: Calculate lead time
        print("\n3. Calculating lead time...")
        lead_time = calculate_lead_time(
            combined_signal,
            event_dict["event_date"]
        )
        print(f"   Reddit lead: {lead_time.get('reddit_lead_days')} days")
        print(f"   GDELT lead: {lead_time.get('gdelt_lead_days')} days")
        print(f"   Reddit beats news by: {lead_time.get('reddit_beats_news_by')} days")

        # Step 4: Get outcome
        outcome = event_data.get("outcome", {})
        actual_outcome = outcome.get("outcome", event_dict.get("expected_outcome", "unknown"))
        print(f"\n4. Outcome: {actual_outcome}")
        if outcome.get("change_pct") is not None:
            print(f"   Price change: {outcome.get('change_pct')}%")

        # Step 5: Validate prediction
        print("\n5. Validating prediction...")
        validation = validate_prediction(
            combined_signal,
            event_dict["event_date"],
            actual_outcome
        )
        print(f"   Predicted: {validation.get('predicted_direction')}")
        print(f"   Correct: {validation.get('prediction_correct')}")
        print(f"   Confidence: {validation.get('confidence')}%")

        # Save signals if requested
        if save_signals and not combined_signal.empty:
            signal_file = SIGNALS_DIR / f"{event_id}_signals.csv"
            combined_signal.to_csv(signal_file, index=False)
            print(f"\n   Saved signals to: {signal_file.name}")

        # Compile results
        result = {
            "event_id": event_id,
            "event_name": event_dict.get("event_name"),
            "category": event_dict.get("category"),
            "event_date": event_dict.get("event_date"),
            "subreddits": event_dict.get("subreddits"),
            "reddit_posts_count": reddit_count,
            "news_articles_count": news_count,
            "reddit_peak_signal": lead_time.get("reddit_peak_signal"),
            "gdelt_peak_signal": lead_time.get("gdelt_peak_signal"),
            "reddit_lead_days": lead_time.get("reddit_lead_days"),
            "gdelt_lead_days": lead_time.get("gdelt_lead_days"),
            "reddit_beats_news_by": lead_time.get("reddit_beats_news_by"),
            "predicted_direction": validation.get("predicted_direction"),
            "actual_outcome": actual_outcome,
            "prediction_correct": validation.get("prediction_correct"),
            "signal_strength": validation.get("signal_strength"),
            "avg_signal": validation.get("avg_signal"),
            "avg_sentiment": validation.get("avg_sentiment"),
            "confidence": validation.get("confidence"),
            "outcome_details": json.dumps(outcome) if outcome else None,
            "processed_at": datetime.now().isoformat()
        }

        return result

    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()

        return {
            "event_id": event_id,
            "event_name": event_dict.get("event_name"),
            "category": event_dict.get("category"),
            "event_date": event_dict.get("event_date"),
            "error": str(e),
            "prediction_correct": None,
            "processed_at": datetime.now().isoformat()
        }


def process_all_events(
    categories: list = None,
    limit: int = None,
    skip_cached: bool = True
) -> pd.DataFrame:
    """
    Process all events from the catalog.

    Args:
        categories: List of categories to process (None = all)
        limit: Maximum number of events to process
        skip_cached: Skip events that already have results

    Returns:
        DataFrame with all validation results
    """
    print("Loading events catalog...")
    events = load_events_catalog()
    print(f"Found {len(events)} events")

    # Filter by category if specified
    if categories:
        events = events[events["category"].isin(categories)]
        print(f"Filtered to {len(events)} events in categories: {categories}")

    # Apply limit
    if limit:
        events = events.head(limit)
        print(f"Limited to first {limit} events")

    # Load existing results if skipping cached
    existing_results = pd.DataFrame()
    if skip_cached and RESULTS_FILE.exists():
        existing_results = pd.read_csv(RESULTS_FILE)
        processed_ids = set(existing_results["event_id"].dropna())
        events = events[~events["event_id"].isin(processed_ids)]
        print(f"Skipping {len(processed_ids)} already processed events")
        print(f"Processing {len(events)} remaining events")

    if events.empty:
        print("No events to process!")
        return existing_results

    # Process events
    results = []
    for _, event in tqdm(events.iterrows(), total=len(events), desc="Processing events"):
        result = process_single_event(event)
        results.append(result)

    # Create results DataFrame
    new_results = pd.DataFrame(results)

    # Merge with existing results
    if not existing_results.empty:
        all_results = pd.concat([existing_results, new_results], ignore_index=True)
    else:
        all_results = new_results

    # Save results
    all_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to: {RESULTS_FILE}")

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(results: pd.DataFrame):
    """Print summary statistics of validation results."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    # Overall accuracy
    valid_results = results[results["prediction_correct"].notna()]
    if len(valid_results) > 0:
        correct = valid_results["prediction_correct"].sum()
        total = len(valid_results)
        accuracy = correct / total * 100
        print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    else:
        print("\nNo validated predictions yet")

    # Accuracy by category
    print("\nAccuracy by Category:")
    for category in results["category"].unique():
        cat_results = valid_results[valid_results["category"] == category]
        if len(cat_results) > 0:
            cat_correct = cat_results["prediction_correct"].sum()
            cat_total = len(cat_results)
            cat_accuracy = cat_correct / cat_total * 100
            print(f"  {category}: {cat_accuracy:.1f}% ({cat_correct}/{cat_total})")

    # Lead time stats
    print("\nLead Time Statistics:")
    lead_times = results["reddit_lead_days"].dropna()
    if len(lead_times) > 0:
        print(f"  Average Reddit lead: {lead_times.mean():.1f} days")
        print(f"  Max Reddit lead: {lead_times.max():.0f} days")

    beats_news = results["reddit_beats_news_by"].dropna()
    if len(beats_news) > 0:
        print(f"  Average Reddit beats news by: {beats_news.mean():.1f} days")

    # Data quality
    print("\nData Quality:")
    avg_posts = results["reddit_posts_count"].mean()
    avg_articles = results["news_articles_count"].mean()
    print(f"  Avg Reddit posts per event: {avg_posts:.1f}")
    print(f"  Avg news articles per event: {avg_articles:.1f}")

    # Errors
    errors = results[results["prediction_correct"].isna() & results.get("error", pd.Series()).notna()]
    if len(errors) > 0:
        print(f"\n  Events with errors: {len(errors)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process Hivedex events")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Categories to process (stock, movie, tech, gaming, other)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of events to process"
    )
    parser.add_argument(
        "--no-skip-cached",
        action="store_true",
        help="Reprocess all events including cached ones"
    )
    parser.add_argument(
        "--event",
        type=str,
        help="Process a single event by ID"
    )

    args = parser.parse_args()

    if args.event:
        # Process single event
        events = load_events_catalog()
        event = events[events["event_id"] == args.event]
        if event.empty:
            print(f"Event not found: {args.event}")
            sys.exit(1)
        result = process_single_event(event.iloc[0])
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        # Process all events
        process_all_events(
            categories=args.categories,
            limit=args.limit,
            skip_cached=not args.no_skip_cached
        )


if __name__ == "__main__":
    main()
