"""
Hivedex Visualization Helpers
=============================
Altair-based visualizations for the Hex dashboard.
"""

import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from typing import Optional

# Configure Altair for better rendering
alt.data_transformers.disable_max_rows()


# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLORS = {
    "reddit": "#FF4500",      # Reddit orange
    "gdelt": "#1E88E5",       # News blue
    "hivemind": "#7C3AED",    # Purple
    "positive": "#22C55E",    # Green
    "negative": "#EF4444",    # Red
    "neutral": "#6B7280",     # Gray
    "background": "#F8FAFC",
    "text": "#1E293B"
}

CATEGORY_COLORS = {
    "stock": "#3B82F6",
    "movie": "#EC4899",
    "tech": "#8B5CF6",
    "gaming": "#10B981",
    "other": "#F59E0B"
}


# =============================================================================
# DASHBOARD CHARTS
# =============================================================================

def create_accuracy_gauge(accuracy: float, total_predictions: int) -> alt.Chart:
    """
    Create a circular gauge showing overall accuracy.

    Args:
        accuracy: Accuracy percentage (0-100)
        total_predictions: Total number of predictions

    Returns:
        Altair chart
    """
    # Create data for gauge
    gauge_data = pd.DataFrame({
        "category": ["Correct", "Incorrect"],
        "value": [accuracy, 100 - accuracy],
        "color": [COLORS["positive"], COLORS["negative"]]
    })

    base = alt.Chart(gauge_data).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("category:N", scale=alt.Scale(
            domain=["Correct", "Incorrect"],
            range=[COLORS["positive"], COLORS["negative"]]
        ), legend=None)
    )

    pie = base.mark_arc(innerRadius=80, outerRadius=120)

    # Add center text
    text = alt.Chart(pd.DataFrame({
        "text": [f"{accuracy:.1f}%"],
        "subtext": [f"{total_predictions} predictions"]
    })).mark_text(
        fontSize=32,
        fontWeight="bold",
        color=COLORS["text"]
    ).encode(
        text="text:N"
    )

    return alt.layer(pie, text).properties(
        width=250,
        height=250,
        title="Hivemind Accuracy"
    )


def create_category_accuracy_chart(validations_df: pd.DataFrame) -> alt.Chart:
    """
    Create bar chart showing accuracy by category.

    Args:
        validations_df: DataFrame with validation results

    Returns:
        Altair chart
    """
    # Calculate accuracy by category
    category_stats = validations_df.groupby("category").agg({
        "prediction_correct": ["mean", "count"]
    }).reset_index()
    category_stats.columns = ["category", "accuracy", "count"]
    category_stats["accuracy"] = category_stats["accuracy"] * 100

    bars = alt.Chart(category_stats).mark_bar().encode(
        x=alt.X("category:N", title="Category", sort="-y"),
        y=alt.Y("accuracy:Q", title="Accuracy %", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("category:N", scale=alt.Scale(
            domain=list(CATEGORY_COLORS.keys()),
            range=list(CATEGORY_COLORS.values())
        ), legend=None),
        tooltip=[
            alt.Tooltip("category:N", title="Category"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".1f"),
            alt.Tooltip("count:Q", title="Events")
        ]
    )

    # Add count labels on bars
    text = alt.Chart(category_stats).mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        color=COLORS["text"]
    ).encode(
        x=alt.X("category:N", sort="-y"),
        y=alt.Y("accuracy:Q"),
        text=alt.Text("count:Q", format="d")
    )

    # Add 73% target line
    rule = alt.Chart(pd.DataFrame({"y": [73]})).mark_rule(
        color=COLORS["hivemind"],
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y="y:Q")

    return alt.layer(bars, text, rule).properties(
        width=400,
        height=300,
        title="Accuracy by Category (target: 73%)"
    )


def create_lead_time_chart(validations_df: pd.DataFrame) -> alt.Chart:
    """
    Create box plot showing lead time distribution by category.

    Args:
        validations_df: DataFrame with validation results

    Returns:
        Altair chart
    """
    # Filter to events with lead time data
    lead_data = validations_df[validations_df["reddit_lead_days"].notna()].copy()

    if lead_data.empty:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No lead time data available")
        )

    return alt.Chart(lead_data).mark_boxplot(
        extent="min-max",
        median={"color": COLORS["text"]}
    ).encode(
        x=alt.X("category:N", title="Category"),
        y=alt.Y("reddit_lead_days:Q", title="Days Reddit Led News"),
        color=alt.Color("category:N", scale=alt.Scale(
            domain=list(CATEGORY_COLORS.keys()),
            range=list(CATEGORY_COLORS.values())
        ), legend=None)
    ).properties(
        width=400,
        height=300,
        title="How Early Did Reddit Know?"
    )


def create_recent_predictions_table(validations_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Create formatted table of recent predictions.

    Args:
        validations_df: DataFrame with validation results
        limit: Number of recent predictions to show

    Returns:
        Formatted DataFrame
    """
    recent = validations_df.nlargest(limit, "event_date").copy()

    # Format columns
    recent["result"] = recent["prediction_correct"].map({
        True: "Correct",
        False: "Wrong",
        None: "Pending"
    })

    recent["lead"] = recent["reddit_lead_days"].apply(
        lambda x: f"{x:.0f} days" if pd.notna(x) else "N/A"
    )

    recent["confidence"] = recent["confidence"].apply(
        lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
    )

    return recent[[
        "event_name", "category", "predicted_direction",
        "actual_outcome", "result", "lead", "confidence"
    ]].rename(columns={
        "event_name": "Event",
        "category": "Category",
        "predicted_direction": "Prediction",
        "actual_outcome": "Outcome",
        "result": "Result",
        "lead": "Lead Time",
        "confidence": "Confidence"
    })


# =============================================================================
# EVENT DEEP DIVE CHARTS
# =============================================================================

def create_signal_timeline(
    signal_df: pd.DataFrame,
    event_date: str,
    event_name: str
) -> alt.Chart:
    """
    Create dual-line timeline showing Reddit vs GDELT signals.

    Args:
        signal_df: Combined signal DataFrame
        event_date: Event date for marker
        event_name: Event name for title

    Returns:
        Altair chart
    """
    if signal_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No signal data available")
        )

    # Prepare data
    signal_df = signal_df.copy()
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    # Melt for multi-line chart
    plot_data = signal_df[["date", "reddit_signal", "gdelt_signal"]].melt(
        id_vars=["date"],
        var_name="signal_type",
        value_name="signal_value"
    )

    plot_data["signal_type"] = plot_data["signal_type"].map({
        "reddit_signal": "Reddit",
        "gdelt_signal": "News (GDELT)"
    })

    # Base chart
    lines = alt.Chart(plot_data).mark_line(
        point=True,
        strokeWidth=2
    ).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("signal_value:Q", title="Signal Strength",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("signal_type:N", scale=alt.Scale(
            domain=["Reddit", "News (GDELT)"],
            range=[COLORS["reddit"], COLORS["gdelt"]]
        ), legend=alt.Legend(title="Signal Source")),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("signal_type:N", title="Source"),
            alt.Tooltip("signal_value:Q", title="Signal", format=".1f")
        ]
    )

    # Event date marker
    event_rule = alt.Chart(pd.DataFrame({
        "date": [pd.to_datetime(event_date)]
    })).mark_rule(
        color=COLORS["negative"],
        strokeWidth=2,
        strokeDash=[5, 5]
    ).encode(
        x="date:T"
    )

    # Event label
    event_text = alt.Chart(pd.DataFrame({
        "date": [pd.to_datetime(event_date)],
        "text": ["Event"]
    })).mark_text(
        align="left",
        dx=5,
        dy=-10,
        color=COLORS["negative"],
        fontWeight="bold"
    ).encode(
        x="date:T",
        y=alt.value(10),
        text="text:N"
    )

    return alt.layer(lines, event_rule, event_text).properties(
        width=700,
        height=400,
        title=f"Signal Timeline: {event_name}"
    ).interactive()


def create_signal_components_chart(signal_df: pd.DataFrame) -> alt.Chart:
    """
    Create stacked area chart showing Reddit signal components.

    Args:
        signal_df: Reddit signal DataFrame with component columns

    Returns:
        Altair chart
    """
    if signal_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No component data available")
        )

    # Check for component columns
    component_cols = ["volume_norm", "sentiment_norm", "momentum_norm", "engagement_norm"]
    available_cols = [c for c in component_cols if c in signal_df.columns]

    if not available_cols:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No component data available")
        )

    signal_df = signal_df.copy()
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    # Melt for stacked area
    plot_data = signal_df[["date"] + available_cols].melt(
        id_vars=["date"],
        var_name="component",
        value_name="value"
    )

    # Clean component names
    plot_data["component"] = plot_data["component"].str.replace("_norm", "").str.title()

    return alt.Chart(plot_data).mark_area(opacity=0.7).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Component Value", stack="zero"),
        color=alt.Color("component:N", scale=alt.Scale(scheme="category10"),
                       legend=alt.Legend(title="Component")),
        tooltip=[
            alt.Tooltip("date:T", format="%Y-%m-%d"),
            alt.Tooltip("component:N"),
            alt.Tooltip("value:Q", format=".1f")
        ]
    ).properties(
        width=700,
        height=250,
        title="Reddit Signal Components"
    )


# =============================================================================
# LIVE SIGNAL CHARTS
# =============================================================================

def create_live_signal_gauge(current_signal: float) -> alt.Chart:
    """
    Create gauge showing current signal strength.

    Args:
        current_signal: Current signal value (0-100)

    Returns:
        Altair chart
    """
    # Determine color based on signal strength
    if current_signal >= 70:
        color = COLORS["positive"]
        label = "Strong Signal"
    elif current_signal >= 50:
        color = COLORS["hivemind"]
        label = "Moderate Signal"
    else:
        color = COLORS["neutral"]
        label = "Weak Signal"

    gauge_data = pd.DataFrame({
        "category": ["Signal", "Remaining"],
        "value": [current_signal, 100 - current_signal],
        "color": [color, "#E5E7EB"]
    })

    pie = alt.Chart(gauge_data).mark_arc(innerRadius=60, outerRadius=100).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("category:N", scale=alt.Scale(
            domain=["Signal", "Remaining"],
            range=[color, "#E5E7EB"]
        ), legend=None)
    )

    text = alt.Chart(pd.DataFrame({
        "value": [f"{current_signal:.0f}"],
        "label": [label]
    })).mark_text(
        fontSize=28,
        fontWeight="bold",
        color=color
    ).encode(text="value:N")

    return alt.layer(pie, text).properties(
        width=200,
        height=200,
        title="Current Hivemind Signal"
    )


def create_trend_chart(signal_df: pd.DataFrame, days: int = 7) -> alt.Chart:
    """
    Create line chart showing signal trend over recent days.

    Args:
        signal_df: Signal DataFrame with daily data
        days: Number of days to show

    Returns:
        Altair chart
    """
    if signal_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No trend data available")
        )

    signal_df = signal_df.copy()
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    recent = signal_df.tail(days)

    # Determine trend
    if len(recent) >= 2:
        trend = recent["reddit_signal"].iloc[-1] - recent["reddit_signal"].iloc[0]
        trend_text = "Trending Up" if trend > 5 else ("Trending Down" if trend < -5 else "Stable")
    else:
        trend_text = "Insufficient data"

    line = alt.Chart(recent).mark_line(
        point=True,
        strokeWidth=3,
        color=COLORS["reddit"]
    ).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("reddit_signal:Q", title="Signal", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("date:T", format="%Y-%m-%d"),
            alt.Tooltip("reddit_signal:Q", title="Signal", format=".1f")
        ]
    )

    # Add threshold line at 70
    threshold = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule(
        color=COLORS["positive"],
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y="y:Q")

    return alt.layer(line, threshold).properties(
        width=500,
        height=250,
        title=f"7-Day Trend ({trend_text})"
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_summary_stats(validations_df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for dashboard display.

    Args:
        validations_df: DataFrame with validation results

    Returns:
        Dict with summary stats
    """
    valid = validations_df[validations_df["prediction_correct"].notna()]

    return {
        "total_events": len(validations_df),
        "validated_events": len(valid),
        "overall_accuracy": valid["prediction_correct"].mean() * 100 if len(valid) > 0 else 0,
        "avg_lead_time": validations_df["reddit_lead_days"].mean(),
        "avg_confidence": validations_df["confidence"].mean(),
        "best_category": valid.groupby("category")["prediction_correct"].mean().idxmax() if len(valid) > 0 else "N/A",
        "reddit_beats_news_avg": validations_df["reddit_beats_news_by"].mean()
    }


if __name__ == "__main__":
    # Test visualizations with sample data
    print("Testing visualization helpers...")

    # Create sample validation data
    sample_data = pd.DataFrame({
        "event_id": [f"event_{i}" for i in range(20)],
        "event_name": [f"Event {i}" for i in range(20)],
        "category": ["stock"] * 5 + ["movie"] * 5 + ["tech"] * 5 + ["gaming"] * 5,
        "event_date": pd.date_range("2024-01-01", periods=20, freq="W"),
        "prediction_correct": [True, True, False, True, True] * 4,
        "reddit_lead_days": [10, 5, 8, 12, 3] * 4,
        "confidence": [85, 70, 60, 90, 75] * 4,
        "predicted_direction": ["positive", "negative", "positive", "positive", "negative"] * 4,
        "actual_outcome": ["positive", "negative", "negative", "positive", "negative"] * 4,
        "reddit_beats_news_by": [5, 2, 4, 8, 1] * 4
    })

    # Test charts
    stats = format_summary_stats(sample_data)
    print(f"Summary stats: {stats}")

    accuracy_chart = create_category_accuracy_chart(sample_data)
    print("Category accuracy chart created")

    lead_chart = create_lead_time_chart(sample_data)
    print("Lead time chart created")

    table = create_recent_predictions_table(sample_data)
    print(f"Recent predictions table: {len(table)} rows")

    print("\nAll visualization tests passed!")
