"""
Hivedex Test Suite
==================
Comprehensive testing for all Hivedex components.
Run with: python -m pytest scripts/test_suite.py -v
Or directly: python scripts/test_suite.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  [FAIL] {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*50}")
        return self.failed == 0


results = TestResults()


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

def test_events_catalog_loads():
    """Test events catalog can be loaded."""
    try:
        df = pd.read_csv('data/events_catalog.csv', comment='#')
        passed = len(df) >= 50
        results.record("Events catalog loads", passed,
                      f"Only {len(df)} events" if not passed else None)
    except Exception as e:
        results.record("Events catalog loads", False, str(e))


def test_validation_results_loads():
    """Test validation results can be loaded."""
    try:
        df = pd.read_csv('data/validation_results.csv')
        passed = len(df) >= 50
        results.record("Validation results loads", passed,
                      f"Only {len(df)} results" if not passed else None)
    except Exception as e:
        results.record("Validation results loads", False, str(e))


def test_manual_outcomes_loads():
    """Test manual outcomes can be loaded."""
    try:
        df = pd.read_csv('data/manual_outcomes.csv', comment='#')
        passed = len(df) >= 10
        results.record("Manual outcomes loads", passed,
                      f"Only {len(df)} outcomes" if not passed else None)
    except Exception as e:
        results.record("Manual outcomes loads", False, str(e))


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================

def test_event_ids_unique():
    """Test event IDs are unique."""
    try:
        df = pd.read_csv('data/events_catalog.csv', comment='#')
        unique = df['event_id'].nunique() == len(df)
        results.record("Event IDs unique", unique,
                      "Duplicate event IDs found" if not unique else None)
    except Exception as e:
        results.record("Event IDs unique", False, str(e))


def test_categories_valid():
    """Test categories are from valid set."""
    try:
        df = pd.read_csv('data/events_catalog.csv', comment='#')
        valid = {'stock', 'movie', 'tech', 'gaming', 'other',
                'politics', 'entertainment', 'health', 'crypto'}
        categories = set(df['category'].str.lower().unique())
        all_valid = categories.issubset(valid)
        results.record("Categories valid", all_valid,
                      f"Invalid: {categories - valid}" if not all_valid else None)
    except Exception as e:
        results.record("Categories valid", False, str(e))


def test_accuracy_reasonable():
    """Test accuracy is in reasonable range."""
    try:
        df = pd.read_csv('data/validation_results.csv')
        accuracy = df['prediction_correct'].mean() * 100
        passed = 50 <= accuracy <= 95
        results.record("Accuracy reasonable", passed,
                      f"Accuracy {accuracy:.1f}% outside 50-95% range" if not passed else None)
    except Exception as e:
        results.record("Accuracy reasonable", False, str(e))


def test_lead_times_reasonable():
    """Test lead times are reasonable."""
    try:
        df = pd.read_csv('data/validation_results.csv')
        avg_lead = df['reddit_lead_days'].mean()
        passed = 0 <= avg_lead <= 30
        results.record("Lead times reasonable", passed,
                      f"Avg lead {avg_lead:.1f} days outside 0-30 range" if not passed else None)
    except Exception as e:
        results.record("Lead times reasonable", False, str(e))


def test_signals_in_range():
    """Test signal values are 0-100."""
    try:
        df = pd.read_csv('data/validation_results.csv')
        for col in ['reddit_peak_signal', 'gdelt_peak_signal', 'confidence']:
            if col in df.columns:
                values = df[col].dropna()
                in_range = (values >= 0).all() and (values <= 100).all()
                if not in_range:
                    results.record(f"Signals in range ({col})", False,
                                  f"{col} has values outside 0-100")
                    return
        results.record("Signals in range", True)
    except Exception as e:
        results.record("Signals in range", False, str(e))


# =============================================================================
# IMPORT TESTS
# =============================================================================

def test_data_fetcher_imports():
    """Test data_fetcher module imports."""
    try:
        from scripts.data_fetcher import fetch_reddit_posts, fetch_gdelt_news
        results.record("data_fetcher imports", True)
    except Exception as e:
        results.record("data_fetcher imports", False, str(e))


def test_signal_calculator_imports():
    """Test signal_calculator module imports."""
    try:
        from scripts.signal_calculator import (
            analyze_sentiment, calculate_reddit_signal
        )
        results.record("signal_calculator imports", True)
    except Exception as e:
        results.record("signal_calculator imports", False, str(e))


def test_live_monitor_imports():
    """Test live_monitor module imports."""
    try:
        from scripts.live_monitor import LiveSignalMonitor, quick_signal_check
        results.record("live_monitor imports", True)
    except Exception as e:
        results.record("live_monitor imports", False, str(e))


def test_kalshi_fetcher_imports():
    """Test kalshi_fetcher module imports."""
    try:
        from scripts.kalshi_fetcher import calculate_market_signal
        results.record("kalshi_fetcher imports", True)
    except Exception as e:
        results.record("kalshi_fetcher imports", False, str(e))


def test_data_validator_imports():
    """Test data_validator module imports."""
    try:
        from scripts.data_validator import DataValidator, generate_quality_report
        results.record("data_validator imports", True)
    except Exception as e:
        results.record("data_validator imports", False, str(e))


# =============================================================================
# FUNCTION TESTS
# =============================================================================

def test_sentiment_analysis():
    """Test sentiment analysis works."""
    try:
        from scripts.signal_calculator import analyze_sentiment
        result = analyze_sentiment(["This is great!", "This is terrible."])
        passed = len(result) == 2
        results.record("Sentiment analysis", passed,
                      "Wrong number of results" if not passed else None)
    except Exception as e:
        results.record("Sentiment analysis", False, str(e))


def test_market_signal_calculation():
    """Test market signal calculation."""
    try:
        from scripts.kalshi_fetcher import calculate_market_signal
        signal = calculate_market_signal(
            market_price=0.72,
            price_change_24h=0.05,
            volume_24h=25000
        )
        passed = 'market_signal' in signal and 0 <= signal['market_signal'] <= 100
        results.record("Market signal calculation", passed,
                      "Invalid signal output" if not passed else None)
    except Exception as e:
        results.record("Market signal calculation", False, str(e))


def test_data_validation():
    """Test data validation works."""
    try:
        from scripts.data_validator import DataValidator
        events = pd.read_csv('data/events_catalog.csv', comment='#')
        validations = pd.read_csv('data/validation_results.csv')
        validator = DataValidator(events, validations)
        result = validator.validate_all()
        passed = result['valid']
        results.record("Data validation", passed,
                      f"{len(result['issues'])} issues" if not passed else None)
    except Exception as e:
        results.record("Data validation", False, str(e))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_category_accuracy_calculation():
    """Test category accuracy can be calculated."""
    try:
        df = pd.read_csv('data/validation_results.csv')
        for cat in df['category'].unique():
            cat_data = df[df['category'] == cat]
            acc = cat_data['prediction_correct'].mean() * 100
            if not (0 <= acc <= 100):
                results.record("Category accuracy calculation", False,
                              f"{cat} accuracy {acc} invalid")
                return
        results.record("Category accuracy calculation", True)
    except Exception as e:
        results.record("Category accuracy calculation", False, str(e))


def test_summary_stats_generation():
    """Test summary statistics can be generated."""
    try:
        df = pd.read_csv('data/validation_results.csv')
        stats = {
            'accuracy': df['prediction_correct'].mean() * 100,
            'lead_time': df['reddit_lead_days'].mean(),
            'confidence': df['confidence'].mean(),
            'count': len(df)
        }
        passed = all(v is not None and not pd.isna(v) for v in stats.values())
        results.record("Summary stats generation", passed,
                      "NaN values in stats" if not passed else None)
    except Exception as e:
        results.record("Summary stats generation", False, str(e))


# =============================================================================
# FILE STRUCTURE TESTS
# =============================================================================

def test_required_files_exist():
    """Test all required files exist."""
    required = [
        'data/events_catalog.csv',
        'data/validation_results.csv',
        'data/manual_outcomes.csv',
        'scripts/data_fetcher.py',
        'scripts/signal_calculator.py',
        'scripts/batch_process.py',
        'scripts/visualizations.py',
        'scripts/live_monitor.py',
        'scripts/kalshi_fetcher.py',
        'scripts/data_validator.py',
        'requirements.txt',
        'README.md',
        'SUBMISSION.md',
        'DEMO_SCRIPT.md'
    ]

    missing = [f for f in required if not os.path.exists(f)]
    passed = len(missing) == 0
    results.record("Required files exist", passed,
                  f"Missing: {missing}" if not passed else None)


def test_notebooks_exist():
    """Test notebooks exist."""
    notebooks = [
        'notebooks/test_apis.ipynb',
        'notebooks/hivedex_app.ipynb'
    ]

    missing = [f for f in notebooks if not os.path.exists(f)]
    passed = len(missing) == 0
    results.record("Notebooks exist", passed,
                  f"Missing: {missing}" if not passed else None)


def test_hex_config_exists():
    """Test Hex configuration exists."""
    configs = [
        'hex_config/semantic_model.yaml',
        'hex_config/threads_prompts.md'
    ]

    missing = [f for f in configs if not os.path.exists(f)]
    passed = len(missing) == 0
    results.record("Hex config exists", passed,
                  f"Missing: {missing}" if not passed else None)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("HIVEDEX TEST SUITE")
    print("=" * 50)

    print("\n[Data Loading Tests]")
    test_events_catalog_loads()
    test_validation_results_loads()
    test_manual_outcomes_loads()

    print("\n[Data Quality Tests]")
    test_event_ids_unique()
    test_categories_valid()
    test_accuracy_reasonable()
    test_lead_times_reasonable()
    test_signals_in_range()

    print("\n[Import Tests]")
    test_data_fetcher_imports()
    test_signal_calculator_imports()
    test_live_monitor_imports()
    test_kalshi_fetcher_imports()
    test_data_validator_imports()

    print("\n[Function Tests]")
    test_sentiment_analysis()
    test_market_signal_calculation()
    test_data_validation()

    print("\n[Integration Tests]")
    test_category_accuracy_calculation()
    test_summary_stats_generation()

    print("\n[File Structure Tests]")
    test_required_files_exist()
    test_notebooks_exist()
    test_hex_config_exists()

    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
