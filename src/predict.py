"""
Prediction module for injury risk inference on new data.

Responsibilities:
- Load trained model and preprocessing artifacts
- Make predictions on new player health data
- Return risk classification and confidence scores
"""

from pathlib import Path
from typing import Any, Dict, IO, Union
import pickle
import json
import re
import sys

import pandas as pd
import numpy as np

# Add src to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import merge_datasets, normalize_columns
from risk_score import compute_risk_score


RawInput = Union[pd.DataFrame, str, Path, IO[str], IO[bytes]]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_message(message: str) -> str:
    """Hide absolute project paths in CLI-facing error messages."""
    sanitized = str(message).replace(str(PROJECT_ROOT), ".")
    sanitized = sanitized.replace(f"file://{PROJECT_ROOT}", ".")
    sanitized = re.sub(r"/home/[^\s:'\")]+", "[absolute-path]", sanitized)
    return sanitized


class InjuryRiskPredictor:
    """Interface for making injury risk predictions on new data."""
    
    def __init__(self, models_dir: Path = Path("models")) -> None:
        """
        Initialize predictor by loading model and preprocessing artifacts.
        
        Args:
            models_dir: Path to models directory containing saved artifacts
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self._load_artifacts()

    def _format_project_path(self, path: Path) -> str:
        """Format artifact paths relative to the project when possible."""
        try:
            return str(path.relative_to(self.models_dir.parent))
        except ValueError:
            return str(path)

    def _require_metadata(self) -> Dict[str, Any]:
        """Return metadata or raise when preprocessing artifacts are unavailable."""
        if not self.metadata:
            raise ValueError(
                "Metadata not available. Run feature engineering to generate data/processed/metadata.json."
            )
        return self.metadata
    
    def _load_artifacts(self) -> None:
        """Load trained model, scaler, and metadata from disk."""
        # Load model
        model_path = self.models_dir / "final_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._format_project_path(model_path)}"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_path = self.models_dir / "../data/processed/metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"✓ Model loaded from {self._format_project_path(model_path)}")
        if self.metadata:
            print(f"✓ Metadata loaded: {self.metadata['n_features']} features")

    def _apply_saved_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the saved training statistics."""
        metadata = self._require_metadata()
        feature_names = metadata["feature_names"]
        X_aligned = X.reindex(columns=feature_names, fill_value=0.0).astype(float)

        mean = np.asarray(metadata["scaler_params"]["mean"], dtype=float)
        scale = np.asarray(metadata["scaler_params"]["scale"], dtype=float)
        scale = np.where(scale == 0, 1.0, scale)

        X_scaled = (X_aligned.to_numpy(dtype=float) - mean) / scale
        return pd.DataFrame(X_scaled, columns=feature_names, index=X_aligned.index)

    def _load_raw_input(self, raw_input: RawInput, source: str, match_id: int) -> pd.DataFrame:
        """Load a raw dataframe or CSV input and normalize it for batch inference."""
        if isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            df = pd.read_csv(raw_input)

        if "match_id" not in df.columns:
            df["match_id"] = match_id
        else:
            df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").fillna(match_id).astype(int)

        return normalize_columns(df, source)

    def _build_model_feature_matrix(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Rebuild the exact model input columns expected by the trained model."""
        metadata = self._require_metadata()
        feature_names = metadata["feature_names"]
        X = pd.DataFrame(0.0, index=merged.index, columns=feature_names)

        base_features = [name for name in feature_names if not name.startswith("pos_")]
        for feature_name in base_features:
            if feature_name in merged.columns:
                X[feature_name] = pd.to_numeric(merged[feature_name], errors="coerce").fillna(0.0)

        position_columns = [name for name in feature_names if name.startswith("pos_")]
        if position_columns and "position" in merged.columns:
            positions = merged["position"].fillna("").astype(str).str.strip()
            for column_name in position_columns:
                position_name = column_name.replace("pos_", "", 1)
                X[column_name] = (positions == position_name).astype(float)

        return X

    def build_model_matrix_from_raw_dataframes(
        self,
        pre: pd.DataFrame,
        post: pd.DataFrame,
        match: pd.DataFrame,
        match_id: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create a scaled model matrix from raw pre, post, and match dataframes."""
        pre_df = self._load_raw_input(pre, "pre", match_id)
        post_df = self._load_raw_input(post, "post", match_id)
        match_df = self._load_raw_input(match, "match", match_id)

        merged = merge_datasets(pre_df, post_df, match_df)
        X = self._build_model_feature_matrix(merged)
        X_scaled = self._apply_saved_scaler(X)

        return X_scaled, merged

    def build_model_matrix_from_raw_csvs(
        self,
        pre_file: RawInput,
        post_file: RawInput,
        match_file: RawInput,
        match_id: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create a scaled model matrix from raw pre, post, and match CSV inputs."""
        return self.build_model_matrix_from_raw_dataframes(
            pre=pre_file,
            post=post_file,
            match=match_file,
            match_id=match_id,
        )

    def _format_batch_results(self, X: pd.DataFrame, merged: pd.DataFrame | None = None) -> pd.DataFrame:
        """Run batch inference and optionally prepend identifying columns."""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        results = pd.DataFrame({
            'risk_code': predictions,
            'risk_label': [label_map[prediction] for prediction in predictions],
            'prob_low': probabilities[:, 0],
            'prob_medium': probabilities[:, 1],
            'prob_high': probabilities[:, 2],
        })
        results['confidence'] = probabilities.max(axis=1)

        if merged is not None:
            id_columns = [column for column in ['nickname', 'position', 'match_id'] if column in merged.columns]
            analysis_columns = [
                column
                for column in [
                    'general_health',
                    'energy_level',
                    'sleep_hours',
                    'sleep_quality',
                    'wakes_rested',
                    'stress_level',
                    'motivation',
                    'pre_discomfort',
                    'rpe',
                    'fatigue_level',
                    'post_discomfort',
                ]
                if column in merged.columns
            ]

            if 'risk_score' not in merged.columns:
                merged = merged.copy()
                merged['risk_score'] = merged.apply(compute_risk_score, axis=1)

            extra_columns = [column for column in analysis_columns + ['risk_score'] if column in merged.columns]
            if id_columns or extra_columns:
                results = pd.concat(
                    [merged[id_columns + extra_columns].reset_index(drop=True), results.reset_index(drop=True)],
                    axis=1,
                )

        return results
    
    def _preprocess_input(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess raw player input to match training data format.
        
        Args:
            features: Dictionary of player health indicators
                     (e.g., {'sleep_hours': 7, 'rpe': 8, ...})
        
        Returns:
            Preprocessed feature vector (1D numpy array)
        """
        metadata = self._require_metadata()
        X = pd.DataFrame([features], columns=metadata['feature_names']).fillna(0.0)
        return self._apply_saved_scaler(X).to_numpy(dtype=float)
    
    def predict_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict injury risk for a single player.
        
        Args:
            features: Dictionary of player indicators
            
        Returns:
            Dictionary with prediction results:
            {
                'risk_label': str,  # 'Low', 'Medium', 'High'
                'risk_code': int,   # 0, 1, 2
                'probabilities': dict,  # {'Low': 0.2, 'Medium': 0.5, 'High': 0.3}
                'confidence': float  # confidence of top prediction
            }
        """
        # Preprocess input
        X_scaled = self._preprocess_input(features)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Map to labels
        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_label = label_map[prediction]
        confidence = max(probabilities)
        
        # Build result
        result = {
            'risk_label': risk_label,
            'risk_code': int(prediction),
            'probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'confidence': float(confidence),
            'input_features': features
        }
        
        return result
    
    def batch_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on multiple samples.
        
        Args:
            X: Feature dataframe (n_samples, n_features)
            
        Returns:
            DataFrame with predictions and probabilities
        """
        return self._format_batch_results(X)

    def predict_batch_from_raw_dataframes(
        self,
        pre: pd.DataFrame,
        post: pd.DataFrame,
        match: pd.DataFrame,
        match_id: int = 1,
    ) -> pd.DataFrame:
        """Predict injury risk from raw pre, post, and match dataframes."""
        X_scaled, merged = self.build_model_matrix_from_raw_dataframes(
            pre=pre,
            post=post,
            match=match,
            match_id=match_id,
        )
        return self._format_batch_results(X_scaled, merged)

    def predict_batch_from_raw_csvs(
        self,
        pre_file: RawInput,
        post_file: RawInput,
        match_file: RawInput,
        match_id: int = 1,
    ) -> pd.DataFrame:
        """Predict injury risk from raw pre, post, and match CSV inputs."""
        X_scaled, merged = self.build_model_matrix_from_raw_csvs(
            pre_file=pre_file,
            post_file=post_file,
            match_file=match_file,
            match_id=match_id,
        )
        return self._format_batch_results(X_scaled, merged)


def load_model(models_dir: Path = Path("models")) -> InjuryRiskPredictor:
    """
    Convenience function to load and return a predictor.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        Initialized InjuryRiskPredictor instance
    """
    return InjuryRiskPredictor(models_dir)


def predict_single(
    features: Dict[str, Any],
    models_dir: Path = Path("models")
) -> Dict[str, Any]:
    """
    Make a single prediction (convenience wrapper).
    
    Args:
        features: Dictionary of player indicators
        models_dir: Path to models directory
        
    Returns:
        Prediction result dictionary
    """
    predictor = load_model(models_dir)
    return predictor.predict_risk(features)


if __name__ == "__main__":
    # Example usage
    example_features = {
        'general_health': 3,
        'energy_level': 2,
        'sleep_hours': 7,
        'sleep_quality': 3,
        'wakes_rested': 2,
        'stress_level': 2,
        'motivation': 2,
        'rpe': 7,
        'fatigue_level': 6,
        'post_discomfort': 0,
        'accurate_pass': 50,
        'missed_pass': 10,
        'interception': 5,
        'shot_on_goal': 3,
        'successful_dribble': 4,
        'tackle_won': 3,
        'foul_made': 2,
        'goal': 0,
    }
    
    try:
        result = predict_single(example_features)
        print("\nPrediction Result:")
        print(f"  Risk Level: {result['risk_label']}  ({result['confidence']:.2%} confidence)")
        print(f"  Probabilities: {result['probabilities']}")
    except Exception as e:
        print(f"Note: {_sanitize_message(str(e))}")
        print("Train the model first using train.py")
