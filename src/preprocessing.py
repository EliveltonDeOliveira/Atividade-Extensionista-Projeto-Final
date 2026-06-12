"""
Data preprocessing module for football injury risk classification.

Responsibilities:
- Load CSV files from data/pre/, data/post/, data/match/
- Normalize column names to English snake_case
- Encode ordinal and categorical variables
- Merge datasets by player nickname and match ID
- Impute missing values using median (numeric) and mode (categorical)
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# Mapping of Portuguese ordinal/categorical values to numeric codes
ORDINAL_ENCODING = {
    "general_health": {"Excelente": 4, "Boa": 3, "Regular": 2, "Ruim": 1},
    "energy_level": {"Alto": 3, "Moderado": 2, "Baixo": 1},
    "sleep_quality": {"Boa": 3, "Regular": 2, "Ruim": 1},
    "wakes_rested": {"Sempre": 3, "Às vezes": 2, "Raramente": 1, "Nunca": 0},
    "stress_level": {"Baixo": 1, "Moderado": 2, "Alto": 3},
    "motivation": {"Alto": 3, "Sim": 2, "Moderado": 2, "Baixo": 1},
}

BINARY_ENCODING = {"Sim": 1, "Não": 0}


def load_pre(data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate pre-activity survey CSVs from data/pre/.
    
    Args:
        data_dir: Path to project root directory
        
    Returns:
        DataFrame with columns: timestamp, nickname, position, ... + match_id (1, 2, or 3)
    """
    pre_dir = data_dir / "data" / "pre"
    csv_files = sorted(pre_dir.glob("*.csv"))
    
    dfs = []
    for idx, csv_file in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_file)
        df["match_id"] = idx
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    return result


def load_post(data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate post-activity survey CSVs from data/post/.
    
    Args:
        data_dir: Path to project root directory
        
    Returns:
        DataFrame with columns: timestamp, nickname, position, ... + match_id (1, 2, or 3)
    """
    post_dir = data_dir / "data" / "post"
    csv_files = sorted(post_dir.glob("*.csv"))
    
    dfs = []
    for idx, csv_file in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_file)
        df["match_id"] = idx
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    return result


def load_match(data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate match performance CSVs from data/match/.
    
    Args:
        data_dir: Path to project root directory
        
    Returns:
        DataFrame with columns: nickname, ... (32 performance metrics) + match_id (1, 2, or 3)
    """
    match_dir = data_dir / "data" / "match"
    csv_files = sorted(match_dir.glob("*.csv"))
    
    dfs = []
    for idx, csv_file in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_file)
        df["match_id"] = idx
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    return result


def normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Rename columns from Portuguese to English snake_case and encode ordinals.
    
    Args:
        df: Input DataFrame with Portuguese column names
        source: One of 'pre', 'post', or 'match'
        
    Returns:
        DataFrame with normalized column names and encoded values
    """
    df = df.copy()
    
    if source == "pre":
        rename_map = {
            "Timestamp": "timestamp",
            "Apelido": "nickname",
            "Posição em campo": "position",
            "Como você avalia sua saúde geral nesta semana?": "general_health",
            "Nível médio de energia durante os treinos": "energy_level",
            "Você sentiu algum desconforto físico durante os treinos/jogos?": "pre_discomfort",
            "Se sim: descreva o desconforto": "discomfort_description",
            "Horas de sono média nos últimos 7 dias": "sleep_hours",
            "Qualidade do sono recentemente?": "sleep_quality",
            "Você acorda descansado(a)?": "wakes_rested",
            "Número de refeições completas por dia": "meals_per_day",
            "Hidratação adequada?": "adequate_hydration",
            "Você faz uso de suplementos alimentares ou isotônicos?": "uses_supplements",
            "Se sim: quais suplementos?": "supplement_details",
            "Nível de estresse nos últimos 7 dias": "stress_level",
            "Motivação para treinar/jogar": "motivation",
            "Algo afetando concentração ou desempenho?": "affecting_performance",
            "Se sim: detalhe": "performance_detail",
            "Comentários adicionais sobre bem-estar, saúde ou desempenho": "additional_comments",
        }
        df = df.rename(columns=rename_map)
        
        # Encode ordinals
        for col, encoding_dict in ORDINAL_ENCODING.items():
            if col in df.columns:
                df[col] = df[col].map(encoding_dict)
        
        # Encode binary fields
        for col in ["pre_discomfort", "adequate_hydration", "uses_supplements", "affecting_performance"]:
            if col in df.columns:
                df[col] = df[col].map(BINARY_ENCODING)
        
        # Convert numeric columns
        df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
        df["meals_per_day"] = pd.to_numeric(df["meals_per_day"], errors="coerce")
    
    elif source == "post":
        rename_map = {
            "Timestamp": "timestamp",
            "Apelido": "nickname",
            "Posição em campo": "position",
            "Esforço percebido (RPE)": "rpe",
            "Nível de fadiga atual": "fatigue_level",
            "Dor ou desconforto após a atividade?": "post_discomfort",
            "Se sim: local e intensidade": "pain_location_intensity",
            "Como você avalia seu desempenho técnico hoje?": "technical_performance",
            "Observações sobre seu desempenho": "performance_observations",
        }
        df = df.rename(columns=rename_map)
        
        # Encode binary field
        if "post_discomfort" in df.columns:
            df["post_discomfort"] = df["post_discomfort"].map(BINARY_ENCODING)
        
        # Convert numeric columns
        df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce")
        df["fatigue_level"] = pd.to_numeric(df["fatigue_level"], errors="coerce")
    
    elif source == "match":
        rename_map = {
            "Apelido": "nickname",
            "Passe Certo": "accurate_pass",
            "Passe Errado": "missed_pass",
            "Interceptação": "interception",
            "Chute Gol": "shot_on_goal",
            "Chute Fora": "shot_wide",
            "Chute Bloq": "shot_blocked",
            "Chute Trave": "shot_on_post",
            "Drible Certo": "successful_dribble",
            "Drible Sofrido": "dribble_suffered",
            "Drible Errado": "failed_dribble",
            "Drible Evitado": "dribble_avoided",
            "Dividida Ganha": "tackle_won",
            "Dividida Perdida": "tackle_lost",
            "Desarme": "clearance",
            "Fair Play": "fair_play",
            "Falta Feita": "foul_made",
            "Falta Sofrida": "foul_suffered",
            "Penalti Feito": "penalty_scored",
            "Penalti Sofrido": "penalty_against",
            "Gol Penalti": "penalty_goal",
            "Penalti Perdido": "penalty_missed",
            "Mão": "hand_ball",
            "Perda de Bola": "ball_loss",
            "Gol Perdido": "goal_missed",
            "Falha na Defesa": "defensive_error",
            "Assist": "assist",
            "Gol": "goal",
            "Pifada": "bad_pass",
            "Lança": "long_pass",
            "Lança Errado": "long_pass_missed",
        }
        df = df.rename(columns=rename_map)
        
        # Convert all match metrics to numeric (NaN for missing)
        match_cols = [col for col in df.columns if col not in ["nickname", "match_id"]]
        for col in match_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def merge_datasets(pre: pd.DataFrame, post: pd.DataFrame, match: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre, post, and match datasets by (nickname, match_id) using left join.
    Impute missing values: numeric columns with median, categorical with mode.
    
    Args:
        pre: Normalized pre-survey DataFrame
        post: Normalized post-survey DataFrame
        match: Normalized match performance DataFrame
        
    Returns:
        Merged DataFrame with imputed missing values
    """
    # Left join starting from pre
    merged = pre.merge(post, on=["nickname", "match_id"], how="left", suffixes=("_pre", "_post"))
    merged = merged.merge(match, on=["nickname", "match_id"], how="left")
    
    # Consolidate position column: prefer position_pre, then position_post, then position
    if "position_pre" in merged.columns and "position_post" in merged.columns:
        merged["position"] = merged["position_pre"].fillna(merged["position_post"])
        merged = merged.drop(columns=["position_pre", "position_post"])
    elif "position_pre" in merged.columns:
        merged = merged.rename(columns={"position_pre": "position"})
        if "position_post" in merged.columns:
            merged = merged.drop(columns=["position_post"])
    elif "position_post" in merged.columns:
        merged = merged.rename(columns={"position_post": "position"})
    
    # Consolidate timestamp column
    if "timestamp_pre" in merged.columns and "timestamp_post" in merged.columns:
        merged["timestamp"] = merged["timestamp_pre"].fillna(merged["timestamp_post"])
        merged = merged.drop(columns=["timestamp_pre", "timestamp_post"])
    elif "timestamp_pre" in merged.columns:
        merged = merged.rename(columns={"timestamp_pre": "timestamp"})
        if "timestamp_post" in merged.columns:
            merged = merged.drop(columns=["timestamp_post"])
    elif "timestamp_post" in merged.columns:
        merged = merged.rename(columns={"timestamp_post": "timestamp"})
    
    # Identify numeric and categorical columns for imputation
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    categorical_cols = merged.select_dtypes(include=["object"]).columns
    
    # Impute numeric columns with median by position (to preserve position-specific patterns)
    for col in numeric_cols:
        if merged[col].isna().any():
            if "position" in merged.columns:
                median_by_pos = merged.groupby("position")[col].transform("median")
                merged[col] = merged[col].fillna(median_by_pos)
            # If still NaN, use overall median
            merged[col] = merged[col].fillna(merged[col].median())
    
    # Impute categorical columns with mode by position
    for col in categorical_cols:
        if merged[col].isna().any() and col not in ["nickname", "position", "timestamp"]:
            if "position" in merged.columns:
                mode_by_pos = merged.groupby("position")[col].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
                merged[col] = merged[col].fillna(mode_by_pos)
            # If still NaN, use overall mode
            if merged[col].isna().any():
                mode_val = merged[col].mode()[0] if not merged[col].mode().empty else None
                if mode_val is not None:
                    merged[col] = merged[col].fillna(mode_val)
    
    # Remove duplicates by (nickname, match_id) — keep first occurrence
    merged = merged.drop_duplicates(subset=["nickname", "match_id"], keep="first")
    
    return merged.reset_index(drop=True)


def load_and_preprocess(data_dir: Path) -> pd.DataFrame:
    """
    Convenience function: load, normalize, and merge all datasets in one call.
    
    Args:
        data_dir: Path to project root directory
        
    Returns:
        Merged and preprocessed DataFrame
    """
    pre = load_pre(data_dir)
    post = load_post(data_dir)
    match = load_match(data_dir)
    
    pre = normalize_columns(pre, "pre")
    post = normalize_columns(post, "post")
    match = normalize_columns(match, "match")
    
    merged = merge_datasets(pre, post, match)
    
    return merged
