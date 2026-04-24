"""
PancrAI — Visualization Utilities
Chart helpers for Plotly-based visualizations used in the frontend.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import numpy as np


def confidence_bar_chart(
    confidence_scores: List[float],
    class_names: List[str] = None,
    predicted_class: int = 0,
) -> go.Figure:
    """
    Create a horizontal bar chart of classifier confidence scores.

    Args:
        confidence_scores: List of [no_tumor, benign, malignant, cystic] probs.
        class_names: Display names for each class.
        predicted_class: Index of the predicted class (highlighted).

    Returns:
        Plotly Figure.
    """
    if class_names is None:
        class_names = ["No Tumor", "Benign", "Malignant (PDAC)", "Cystic (IPMN)"]

    colors = []
    for i in range(len(class_names)):
        if i == predicted_class:
            colors.append("#1565C0")
        elif i == 2:  # malignant — always red
            colors.append("#C62828")
        else:
            colors.append("#90A4AE")

    percentages = [round(s * 100, 1) for s in confidence_scores]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=class_names,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in percentages],
        textposition="auto",
    ))

    fig.update_layout(
        title="Classifier Confidence Scores",
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 100]),
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=12),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def tumor_type_pie(distribution: Dict[str, int]) -> go.Figure:
    """
    Pie chart of tumor type distribution across all patients.

    Args:
        distribution: Dict like {"Benign": 12, "Malignant (PDAC)": 5, ...}

    Returns:
        Plotly Figure.
    """
    colors = {
        "No Tumor": "#4CAF50",
        "Benign": "#FF9800",
        "Malignant (PDAC)": "#F44336",
        "Cystic (IPMN)": "#2196F3",
        "Unknown": "#9E9E9E",
    }

    labels = list(distribution.keys())
    values = list(distribution.values())
    color_seq = [colors.get(l, "#9E9E9E") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=color_seq),
        hole=0.4,
        textinfo="percent+label",
        textfont_size=12,
    ))

    fig.update_layout(
        title="Tumor Type Distribution",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(orientation="v", x=1.05),
    )
    return fig


def uncertainty_gauge(uncertainty_score: float) -> go.Figure:
    """
    Gauge chart for model uncertainty score (0–100).

    Color zones:
    - 0–25: Green (low uncertainty)
    - 25–50: Yellow (moderate)
    - 50–75: Orange (high)
    - 75–100: Red (very high)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=uncertainty_score,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": "Model Uncertainty"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1565C0"},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 25], "color": "#E8F5E9"},
                {"range": [25, 50], "color": "#FFF9C4"},
                {"range": [50, 75], "color": "#FFE0B2"},
                {"range": [75, 100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def tumor_growth_chart(
    scan_dates: List[str],
    tumor_areas: List[float],
    tumor_classes: List[str] = None,
) -> go.Figure:
    """
    Line chart showing tumor size over time for longitudinal analysis.

    Args:
        scan_dates: List of date strings (x-axis).
        tumor_areas: Corresponding tumor area in cm².
        tumor_classes: Optional class per scan for color coding.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scan_dates,
        y=tumor_areas,
        mode="lines+markers+text",
        line=dict(color="#1565C0", width=3),
        marker=dict(size=10, color="#1565C0"),
        text=[f"{a:.2f} cm²" for a in tumor_areas],
        textposition="top center",
        name="Tumor Area",
    ))

    fig.update_layout(
        title="Tumor Size Progression",
        xaxis_title="Scan Date",
        yaxis_title="Tumor Area (cm²)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
        yaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
    )
    return fig


def training_history_chart(history: Dict) -> go.Figure:
    """Multi-subplot chart for training history visualization."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Loss", "Dice Score", "IoU",
                        "Sensitivity", "Specificity", "Hausdorff"],
    )

    pairs = [
        ("train_loss", "val_loss", 1, 1),
        ("train_dice", "val_dice", 1, 2),
        (None, "val_iou", 1, 3),
        (None, "val_sensitivity", 2, 1),
        (None, "val_specificity", 2, 2),
        (None, "val_hausdorff", 2, 3),
    ]

    for train_key, val_key, row, col in pairs:
        epochs = list(range(1, len(history.get(val_key, [])) + 1))
        if train_key and train_key in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[train_key],
                name="Train", line=dict(color="#1565C0"),
                showlegend=(row == 1 and col == 1),
            ), row=row, col=col)
        if val_key in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[val_key],
                name="Val", line=dict(color="#E53935"),
                showlegend=(row == 1 and col == 1),
            ), row=row, col=col)

    fig.update_layout(
        height=500,
        title_text="Training History",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
