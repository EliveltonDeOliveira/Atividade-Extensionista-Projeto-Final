"""
Risk score engineering module for injury risk classification.

Responsibilities:
- Define heuristic rules for computing injury risk score from health indicators
- Classify risk scores into Low / Medium / High categories
- Attach risk labels to preprocessed dataset
"""

import pandas as pd
from typing import Tuple


# Risk scoring thresholds and weights
RISK_WEIGHTS = {
    "post_discomfort": 2,          # Pain after activity is strong indicator
    "rpe_high": 1,                  # RPE >= 8
    "fatigue_high": 1,              # Fatigue level >= 7
    "pre_discomfort": 1,            # Pre-existing discomfort
    "sleep_insufficient": 1,        # Sleep hours < 6
    "stress_high": 1,               # Stress level = 3 (Alto)
    "sleep_poor_quality": 1,        # Sleep quality = 1 (Ruim)
}

RISK_THRESHOLDS = {
    "rpe_threshold": 8,
    "fatigue_threshold": 7,
    "sleep_min_hours": 6,
}

RISK_BOUNDARIES = {
    "low": (0, 1),        # 0-1 points
    "medium": (2, 3),     # 2-3 points
    "high": (4, float("inf")),  # 4+ points
}


def compute_risk_score(row: pd.Series) -> int:
    """
    Compute injury risk score for a single player-match record.
    
    Scoring logic:
    - post_discomfort == 1 (Sim): +2
    - rpe >= 8: +1
    - fatigue_level >= 7: +1
    - pre_discomfort == 1 (Sim): +1
    - sleep_hours < 6: +1
    - stress_level == 3 (Alto): +1
    - sleep_quality == 1 (Ruim): +1
    
    Args:
        row: A pandas Series representing a single player-match record
        
    Returns:
        Integer risk score (0+)
    """
    score = 0
    
    # Post-activity discomfort (strongest indicator)
    if pd.notna(row.get("post_discomfort")) and row["post_discomfort"] == 1:
        score += RISK_WEIGHTS["post_discomfort"]
    
    # High exertion (RPE)
    if pd.notna(row.get("rpe")) and row["rpe"] >= RISK_THRESHOLDS["rpe_threshold"]:
        score += RISK_WEIGHTS["rpe_high"]
    
    # High fatigue
    if pd.notna(row.get("fatigue_level")) and row["fatigue_level"] >= RISK_THRESHOLDS["fatigue_threshold"]:
        score += RISK_WEIGHTS["fatigue_high"]
    
    # Pre-existing discomfort
    if pd.notna(row.get("pre_discomfort")) and row["pre_discomfort"] == 1:
        score += RISK_WEIGHTS["pre_discomfort"]
    
    # Insufficient sleep
    if pd.notna(row.get("sleep_hours")) and row["sleep_hours"] < RISK_THRESHOLDS["sleep_min_hours"]:
        score += RISK_WEIGHTS["sleep_insufficient"]
    
    # High stress
    if pd.notna(row.get("stress_level")) and row["stress_level"] == 3:
        score += RISK_WEIGHTS["stress_high"]
    
    # Poor sleep quality
    if pd.notna(row.get("sleep_quality")) and row["sleep_quality"] == 1:
        score += RISK_WEIGHTS["sleep_poor_quality"]
    
    return score


def classify_risk(score: int) -> str:
    """
    Classify numeric risk score into risk category.
    
    Args:
        score: Numeric risk score from compute_risk_score()
        
    Returns:
        One of "Low", "Medium", "High"
    """
    low_min, low_max = RISK_BOUNDARIES["low"]
    med_min, med_max = RISK_BOUNDARIES["medium"]
    high_min, high_max = RISK_BOUNDARIES["high"]
    
    if low_min <= score <= low_max:
        return "Low"
    elif med_min <= score <= med_max:
        return "Medium"
    elif high_min <= score <= high_max:
        return "High"
    else:
        # Fallback (shouldn't happen with proper scoring)
        return "Unknown"


def engineer_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk_score and risk_label columns for each player-match record.
    
    Args:
        df: Preprocessed DataFrame from preprocessing.load_and_preprocess()
        
    Returns:
        DataFrame with added columns: risk_score (int), risk_label (str)
    """
    df = df.copy()
    
    df["risk_score"] = df.apply(compute_risk_score, axis=1)
    df["risk_label"] = df["risk_score"].apply(classify_risk)
    
    return df


def get_risk_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics of risk labels in the dataset.
    
    Args:
        df: DataFrame with risk_label column (from engineer_risk_label())
        
    Returns:
        Dictionary with label counts and percentages
    """
    if "risk_label" not in df.columns:
        raise ValueError("DataFrame must contain 'risk_label' column. Call engineer_risk_label() first.")
    
    label_counts = df["risk_label"].value_counts()
    label_pct = (label_counts / len(df) * 100).round(2)
    
    stats = {
        "total_samples": len(df),
        "counts": label_counts.to_dict(),
        "percentages": label_pct.to_dict(),
    }
    
    return stats
