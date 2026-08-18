"""
Baseline models for stress classification in SheSense.

Train and evaluate RandomForest and XGBoost models with comprehensive metrics.
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score,
                           precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Only RandomForest will be used.")

class BaselineModel:
    """Base class for baseline stress classification models."""
    
    def __init__(self, model_type: str = 'random_forest', **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance_ = None
        
    def _create_model(self):
        """Create the appropriate model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            feature_cols: Optional[List[str]] = None):
        """Train the model."""
        # Select feature columns
        if feature_cols is None:
            feature_cols = [col for col in X_train.columns 
                          if col not in ['window_id', 'subject', 'label', 'timestamp_start', 'timestamp_end']]
        
        self.feature_cols = feature_cols
        X_train_features = X_train[feature_cols]
        
        # Handle missing values
        X_train_features = X_train_features.fillna(X_train_features.median())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create and train model
        self._create_model()
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_test_features = X_test[self.feature_cols].fillna(X_test[self.feature_cols].median())
        X_test_scaled = self.scaler.transform(X_test_features)
        y_pred_encoded = self.model.predict(X_test_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X_test_features = X_test[self.feature_cols].fillna(X_test[self.feature_cols].median())
        X_test_scaled = self.scaler.transform(X_test_features)
        return self.model.predict_proba(X_test_scaled)

def evaluate_model(model: BaselineModel, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series,
                  output_dir: str) -> Dict:
    """Comprehensive model evaluation."""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Get positive class probability (assuming binary classification)
    if y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = np.max(y_pred_proba, axis=1)
    
    # Calculate metrics
    results = {}
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    results['classification_report'] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    # ROC AUC and PR AUC (for binary classification)
    if len(np.unique(y_test)) == 2:
        try:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba_pos)
            results['pr_auc'] = average_precision_score(y_test, y_pred_proba_pos)
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_pos, pos_label='stress')
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_pos, pos_label='stress')
            
            results['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            results['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        except ValueError as e:
            logging.warning(f"Could not compute ROC/PR curves: {e}")
    
    # Per-subject evaluation
    if 'subject' in X_test.columns:
        subject_results = []
        for subject in X_test['subject'].unique():
            mask = X_test['subject'] == subject
            if mask.sum() > 0:
                subj_y_true = y_test[mask]
                subj_y_pred = y_pred[mask]
                
                if len(np.unique(subj_y_true)) > 1:  # Only if both classes present
                    subj_report = classification_report(subj_y_true, subj_y_pred, output_dict=True)
                    subject_results.append({
                        'subject': subject,
                        'n_samples': len(subj_y_true),
                        'accuracy': subj_report['accuracy'],
                        'f1_weighted': subj_report['weighted avg']['f1-score']
                    })
        
        results['per_subject'] = subject_results
    
    # Feature importance
    if model.feature_importance_ is not None:
        results['feature_importance'] = model.feature_importance_.to_dict('records')
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualizations
    create_evaluation_plots(results, output_dir)
    
    return results

def create_evaluation_plots(results: Dict, output_dir: str):
    """Create evaluation visualizations."""
    
    # Confusion matrix heatmap
    if 'confusion_matrix' in results:
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # ROC curve
    if 'roc_curve' in results:
        plt.figure(figsize=(8, 6))
        fpr = results['roc_curve']['fpr']
        tpr = results['roc_curve']['tpr']
        auc = results.get('roc_auc', 0)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Precision-Recall curve
    if 'pr_curve' in results:
        plt.figure(figsize=(8, 6))
        precision = results['pr_curve']['precision']
        recall = results['pr_curve']['recall']
        auc = results.get('pr_auc', 0)
        
        plt.plot(recall, precision, label=f'PR Curve (AUC = {auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature importance
    if 'feature_importance' in results:
        feature_imp = pd.DataFrame(results['feature_importance'])
        if len(feature_imp) > 0:
            plt.figure(figsize=(10, max(6, len(feature_imp) * 0.3)))
            top_features = feature_imp.head(20)  # Top 20 features
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Per-subject performance
    if 'per_subject' in results and results['per_subject']:
        subj_results = pd.DataFrame(results['per_subject'])
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(subj_results)), subj_results['accuracy'])
        plt.xlabel('Subject')
        plt.ylabel('Accuracy')
        plt.title('Per-Subject Accuracy')
        plt.xticks(range(len(subj_results)), subj_results['subject'], rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(subj_results)), subj_results['f1_weighted'])
        plt.xlabel('Subject')
        plt.ylabel('F1 Score (Weighted)')
        plt.title('Per-Subject F1 Score')
        plt.xticks(range(len(subj_results)), subj_results['subject'], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_subject_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

def cross_validate_model(model_type: str, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        feature_cols: List[str],
                        cv_folds: int = 5,
                        **model_params) -> Dict:
    """Perform cross-validation."""
    
    # Prepare data
    X_features = X_train[feature_cols].fillna(X_train[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    # Create model
    if model_type == 'random_forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = {
        'accuracy': cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy'),
        'f1_weighted': cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='f1_weighted'),
        'roc_auc': cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='roc_auc') if len(np.unique(y_encoded)) == 2 else None
    }
    
    # Compute statistics
    cv_results = {}
    for metric, scores in cv_scores.items():
        if scores is not None:
            cv_results[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'scores': scores.tolist()
            }
    
    return cv_results

def main():
    parser = argparse.ArgumentParser(description="Train baseline models for stress classification")
    parser.add_argument('--data', required=True, help='Path to processed dataset directory or parquet file')
    parser.add_argument('--model', choices=['random_forest', 'xgboost'], default='random_forest', 
                       help='Model type to train')
    parser.add_argument('--output', required=True, help='Output directory for model and results')
    parser.add_argument('--cross-val', action='store_true', help='Perform cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    # Model-specific parameters
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum depth')
    parser.add_argument('--class-weight', choices=['balanced', 'balanced_subsample'], default='balanced',
                       help='Class weighting strategy')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load data
    if os.path.isdir(args.data):
        # Load from directory
        train_path = os.path.join(args.data, 'train.parquet')
        test_path = os.path.join(args.data, 'test.parquet')
        val_path = os.path.join(args.data, 'val.parquet')
        
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        val_df = pd.read_parquet(val_path) if os.path.exists(val_path) else pd.DataFrame()
        
        # Load metadata if available
        metadata_path = os.path.join(args.data, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            feature_cols = metadata.get('feature_columns', [])
        else:
            feature_cols = None
    else:
        # Single file - split it
        df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
        # Simple split by subject
        subjects = df['subject'].unique()
        np.random.seed(args.random_state)
        np.random.shuffle(subjects)
        
        n_test = max(1, len(subjects) // 5)
        test_subjects = subjects[:n_test]
        train_subjects = subjects[n_test:]
        
        train_df = df[df['subject'].isin(train_subjects)]
        test_df = df[df['subject'].isin(test_subjects)]
        val_df = pd.DataFrame()
        feature_cols = None
    
    # Determine feature columns
    if feature_cols is None:
        feature_cols = [col for col in train_df.columns 
                       if col not in ['window_id', 'subject', 'label', 'timestamp_start', 'timestamp_end']]
    
    logging.info(f"Training with {len(feature_cols)} features on {len(train_df)} samples")
    
    # Prepare model parameters
    model_params = {
        'n_estimators': args.n_estimators,
        'random_state': args.random_state,
        'class_weight': args.class_weight,
        'n_jobs': -1
    }
    
    if args.max_depth is not None:
        model_params['max_depth'] = args.max_depth
    
    # Cross-validation
    if args.cross_val:
        logging.info("Performing cross-validation...")
        cv_results = cross_validate_model(
            args.model, train_df, train_df['label'], feature_cols, 
            cv_folds=args.cv_folds, **model_params
        )
        logging.info(f"CV Results: {cv_results}")
    
    # Train final model
    logging.info(f"Training {args.model} model...")
    model = BaselineModel(args.model, **model_params)
    model.fit(train_df, train_df['label'], feature_cols)
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    os.makedirs(args.output, exist_ok=True)
    results = evaluate_model(model, test_df, test_df['label'], args.output)
    
    # Save model (simplified - in practice you'd use joblib or pickle)
    model_info = {
        'model_type': args.model,
        'model_params': model_params,
        'feature_cols': feature_cols,
        'label_classes': model.label_encoder.classes_.tolist(),
        'performance': {
            'test_accuracy': results['classification_report']['accuracy'],
            'test_f1_weighted': results['classification_report']['weighted avg']['f1-score']
        }
    }
    
    if args.cross_val:
        model_info['cv_results'] = cv_results
    
    with open(os.path.join(args.output, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    # Create predictions file for downstream analysis
    test_predictions = test_df[['window_id', 'subject', 'label']].copy()
    test_predictions['predicted_label'] = model.predict(test_df)
    test_pred_proba = model.predict_proba(test_df)
    
    # Add probability columns
    for i, class_name in enumerate(model.label_encoder.classes_):
        test_predictions[f'prob_{class_name}'] = test_pred_proba[:, i]
    
    test_predictions.to_parquet(os.path.join(args.output, 'predictions.parquet'), index=False)
    
    logging.info(f"Model training complete. Results saved to {args.output}")
    print(f"Test Accuracy: {results['classification_report']['accuracy']:.3f}")
    print(f"Test F1 (weighted): {results['classification_report']['weighted avg']['f1-score']:.3f}")

if __name__ == "__main__":
    main()
