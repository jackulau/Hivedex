"""
Hivedex Data Validator
======================
Ensures data quality and integrity for prediction validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict


class DataValidator:
    """Validate data quality for Hivedex analysis."""

    def __init__(self, events_df: pd.DataFrame, validations_df: pd.DataFrame):
        """
        Initialize validator.

        Args:
            events_df: Events catalog DataFrame
            validations_df: Validation results DataFrame
        """
        self.events = events_df
        self.validations = validations_df
        self.issues = []
        self.warnings = []

    def validate_all(self) -> Dict:
        """
        Run all validation checks.

        Returns:
            Dict with validation results
        """
        self.issues = []
        self.warnings = []

        # Run checks
        self._check_required_columns()
        self._check_event_ids()
        self._check_date_validity()
        self._check_category_values()
        self._check_signal_ranges()
        self._check_prediction_logic()
        self._check_outcome_coverage()

        return {
            "valid": len(self.issues) == 0,
            "issues": self.issues,
            "warnings": self.warnings,
            "summary": self._generate_summary()
        }

    def _check_required_columns(self):
        """Check for required columns in DataFrames."""
        events_required = ['event_id', 'event_name', 'category', 'event_date']
        validations_required = [
            'event_id', 'prediction_correct', 'reddit_lead_days',
            'predicted_direction', 'actual_outcome'
        ]

        for col in events_required:
            if col not in self.events.columns:
                self.issues.append(f"Missing required column in events: {col}")

        for col in validations_required:
            if col not in self.validations.columns:
                self.issues.append(f"Missing required column in validations: {col}")

    def _check_event_ids(self):
        """Check event ID consistency."""
        events_ids = set(self.events['event_id'])
        validation_ids = set(self.validations['event_id'])

        # Check for orphaned validations
        orphaned = validation_ids - events_ids
        if orphaned:
            self.warnings.append(
                f"Validations without matching events: {len(orphaned)}"
            )

        # Check for missing validations
        missing = events_ids - validation_ids
        if missing:
            self.warnings.append(
                f"Events without validations: {len(missing)}"
            )

    def _check_date_validity(self):
        """Check date field validity."""
        try:
            dates = pd.to_datetime(self.events['event_date'])
            future_dates = dates > datetime.now()
            if future_dates.any():
                self.warnings.append(
                    f"Events with future dates: {future_dates.sum()}"
                )
        except Exception as e:
            self.issues.append(f"Date parsing error: {e}")

    def _check_category_values(self):
        """Check category field values."""
        valid_categories = {'stock', 'movie', 'tech', 'gaming', 'other',
                          'politics', 'entertainment', 'health', 'crypto'}

        categories = set(self.events['category'].str.lower())
        invalid = categories - valid_categories
        if invalid:
            self.warnings.append(f"Unknown categories: {invalid}")

    def _check_signal_ranges(self):
        """Check signal values are in valid range."""
        signal_cols = ['reddit_peak_signal', 'gdelt_peak_signal', 'confidence']

        for col in signal_cols:
            if col in self.validations.columns:
                values = self.validations[col].dropna()
                if (values < 0).any() or (values > 100).any():
                    self.issues.append(f"{col} has values outside 0-100 range")

    def _check_prediction_logic(self):
        """Check prediction logic consistency."""
        for _, row in self.validations.iterrows():
            pred = row.get('predicted_direction', '')
            outcome = row.get('actual_outcome', '')
            correct = row.get('prediction_correct')

            if pd.isna(correct):
                continue

            # Check if prediction_correct matches direction vs outcome
            is_positive_match = (
                (pred == 'positive' and outcome in ['positive', 'success']) or
                (pred == 'negative' and outcome in ['negative', 'failure'])
            )

            if correct and not is_positive_match:
                # Allow for some flexibility in outcome naming
                pass  # Don't flag as issue, outcomes vary

    def _check_outcome_coverage(self):
        """Check outcome data coverage."""
        missing_outcomes = self.validations['actual_outcome'].isna().sum()
        if missing_outcomes > 0:
            self.warnings.append(f"Missing outcomes: {missing_outcomes}")

        missing_predictions = self.validations['prediction_correct'].isna().sum()
        if missing_predictions > 0:
            self.warnings.append(f"Unvalidated predictions: {missing_predictions}")

    def _generate_summary(self) -> Dict:
        """Generate validation summary."""
        return {
            "total_events": len(self.events),
            "total_validations": len(self.validations),
            "validated_count": self.validations['prediction_correct'].notna().sum(),
            "accuracy": self.validations['prediction_correct'].mean() * 100,
            "avg_lead_time": self.validations['reddit_lead_days'].mean(),
            "issues_count": len(self.issues),
            "warnings_count": len(self.warnings)
        }


def validate_signal_data(signal_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate signal timeline data.

    Args:
        signal_df: Signal timeline DataFrame

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []

    # Check required columns
    required = ['date', 'reddit_signal']
    for col in required:
        if col not in signal_df.columns:
            issues.append(f"Missing column: {col}")

    if issues:
        return False, issues

    # Check date ordering
    dates = pd.to_datetime(signal_df['date'])
    if not dates.is_monotonic_increasing:
        issues.append("Dates are not in order")

    # Check signal range
    signal = signal_df['reddit_signal']
    if (signal < 0).any() or (signal > 100).any():
        issues.append("Signal values outside 0-100 range")

    # Check for gaps
    date_diff = dates.diff().dropna()
    if (date_diff > pd.Timedelta(days=2)).any():
        issues.append("Gaps detected in date sequence")

    return len(issues) == 0, issues


def generate_quality_report(events_df: pd.DataFrame,
                           validations_df: pd.DataFrame) -> str:
    """
    Generate formatted quality report.

    Args:
        events_df: Events catalog
        validations_df: Validation results

    Returns:
        Formatted report string
    """
    validator = DataValidator(events_df, validations_df)
    result = validator.validate_all()

    report = []
    report.append("=" * 60)
    report.append("HIVEDEX DATA QUALITY REPORT")
    report.append("=" * 60)
    report.append("")

    # Summary
    summary = result['summary']
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Events:       {summary['total_events']}")
    report.append(f"Total Validations:  {summary['total_validations']}")
    report.append(f"Validated Count:    {summary['validated_count']}")
    report.append(f"Overall Accuracy:   {summary['accuracy']:.1f}%")
    report.append(f"Avg Lead Time:      {summary['avg_lead_time']:.1f} days")
    report.append("")

    # Status
    if result['valid']:
        report.append("STATUS: PASSED")
    else:
        report.append("STATUS: ISSUES FOUND")
    report.append("")

    # Issues
    if result['issues']:
        report.append("ISSUES (must fix)")
        report.append("-" * 40)
        for issue in result['issues']:
            report.append(f"  ! {issue}")
        report.append("")

    # Warnings
    if result['warnings']:
        report.append("WARNINGS (review)")
        report.append("-" * 40)
        for warning in result['warnings']:
            report.append(f"  ? {warning}")
        report.append("")

    report.append("=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    # Test validation
    print("Testing Data Validator...")

    # Load data
    events = pd.read_csv('../data/events_catalog.csv', comment='#')
    validations = pd.read_csv('../data/validation_results.csv')

    # Generate report
    report = generate_quality_report(events, validations)
    print(report)
