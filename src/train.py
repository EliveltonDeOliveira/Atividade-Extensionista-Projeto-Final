"""
Model training module for football injury risk classification.

Responsibilities:
- Load and prepare training data
- Train multiple classification models
- Perform cross-validation with stratification
- Evaluate model performance
- Serialize best model to disk
"""

from pathlib import Path
from typing import Tuple, Dict, List
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    make_scorer,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import json


class InjuryRiskClassifier:
    """Unified interface for training and evaluating injury risk models."""
    
    def __init__(
        self,
        n_splits: int = 3,
        random_state: int = 42,
        models_dir: Path = Path("models")
    ) -> None:
        """
        Initialize classifier.
        
        Args:
            n_splits: Number of folds for cross-validation (default 3 for small datasets)
            random_state: Random seed for reproducibility
            models_dir: Directory to save trained models
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.models = {}
        self.results = {}

    def _format_saved_path(self, path: Path) -> str:
        """Format saved artifact paths relative to the project when possible."""
        try:
            return str(path.relative_to(self.models_dir.parent))
        except ValueError:
            return str(path)
    
    def build_models(self) -> Dict[str, object]:
        """
        Define and return candidate models.
        
        Returns:
            Dictionary of model_name: model_instance
        """
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'  # Handle class imbalance
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'SVM': CalibratedClassifierCV(
                estimator=SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=self.random_state,
                    class_weight='balanced'
                ),
                cv=self.n_splits,
                ensemble=False
            )
        }
        return models
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Train models using cross-validation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            verbose: Print training progress
            
        Returns:
            Dictionary with cross-validation results for each model
        """
        models = self.build_models()
        
        for name, model in models.items():
            if verbose:            
                print(f"Training: {name}")                
            
            # Define scoring metrics
            scoring = {
                'accuracy': 'accuracy',
                'f1_weighted': 'f1_weighted',
                'precision_weighted': make_scorer(
                    precision_score,
                    average='weighted',
                    zero_division=0
                ),
                'recall_weighted': 'recall_weighted'
            }
            
            # Cross-validate
            cv_results = cross_validate(
                model, X, y,
                cv=self.cv,
                scoring=scoring,
                return_train_score=True
            )
            
            # Store results
            self.results[name] = cv_results
            self.models[name] = model
            
            # Print results
            if verbose:
                self._print_cv_results(name, cv_results)
        
        return self.results
    
    def _print_cv_results(self, model_name: str, cv_results: Dict) -> None:
        """Print formatted cross-validation results."""
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        for metric in metrics:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            
            print(f"\n{metric.upper()}:")
            print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
            print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
    
    def get_best_model(self, metric: str = 'f1_weighted') -> Tuple[str, object, float]:
        """
        Get best model based on specified metric.
        
        Args:
            metric: Metric to use for selection (default 'f1_weighted')
            
        Returns:
            Tuple of (model_name, model_instance, best_score)
        """
        best_name = None
        best_score = -1
        
        for name, cv_results in self.results.items():
            test_metric = f'test_{metric}'
            mean_score = cv_results[test_metric].mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
        
        best_model = self.models[best_name]
        return best_name, best_model, best_score
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train final model on entire dataset (best model from CV).
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        model_name, model_template, _ = self.get_best_model()
        
        print(f"\nTraining final model: {model_name}")
        print(f"Dataset shape: {X.shape}")
        
        # Retrain on full data
        best_model = self.build_models()[model_name]
        best_model.fit(X, y)
        
        # Store as final model
        self.final_model = best_model
        self.final_model_name = model_name
    
    def save_model(self, filename: str = "final_model.pkl") -> Path:
        """
        Save final model to disk.
        
        Args:
            filename: Name of file to save
            
        Returns:
            Path to saved model
        """
        if not hasattr(self, 'final_model'):
            raise ValueError("No final model trained. Call train_final_model() first.")
        
        model_path = self.models_dir / filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.final_model, f)
        
        print(f"Model saved: {self._format_saved_path(model_path)}")
        return model_path
    
    def save_results(self, filename: str = "training_results.json") -> Path:
        """
        Save cross-validation results to JSON.
        
        Args:
            filename: Name of file to save
            
        Returns:
            Path to saved results
        """
        results_path = self.models_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        results_dict = {}
        for model_name, cv_results in self.results.items():
            results_dict[model_name] = {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in cv_results.items()
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved: {self._format_saved_path(results_path)}")
        return results_path


def load_processed_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load preprocessed feature matrix and target labels.
    
    Args:
        data_dir: Path to project root directory
        
    Returns:
        Tuple of (X, y)
    """
    processed_dir = data_dir / "data" / "processed"
    
    X = pd.read_csv(processed_dir / "X_scaled.csv")
    y = pd.read_csv(processed_dir / "y_encoded.csv", header=0).iloc[:, 0]
    
    return X, y


def main(data_dir: Path = Path.cwd(), verbose: bool = True) -> InjuryRiskClassifier:
    """
    Main training pipeline.
    
    Args:
        data_dir: Path to project root
        verbose: Print progress
        
    Returns:
        Trained classifier instance
    """
    if verbose:        
        print("INJURY RISK CLASSIFICATION - MODEL TRAINING")        
    
    # Load data
    if verbose:
        print("\nLoading preprocessed data...")
    X, y = load_processed_data(data_dir)
    
    if verbose:
        print(f"Loaded: X shape {X.shape}, y shape {y.shape}")
        print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
    
    # Initialize and train classifier
    clf = InjuryRiskClassifier(
        n_splits=3,
        random_state=42,
        models_dir=data_dir / "models"
    )
    
    # Cross-validation training
    clf.train_with_cv(X, y, verbose=verbose)
    
    # Get best model info
    best_name, _, best_score = clf.get_best_model()
    if verbose:
        print(f"\nBest model: {best_name} (F1: {best_score:.4f})")
    
    # Train final model on full dataset
    clf.train_final_model(X, y)
    
    # Save artifacts
    clf.save_model()
    clf.save_results()
    
    if verbose:        
        print("✓ Training complete!")        
    
    return clf


if __name__ == "__main__":
    main()
