#!/usr/bin/env python3
"""
Derive Best Features Using Representation Learning and Model-Based Importance

This script:
- Preprocesses data (imputation, scaling, classic feature engineering)
- Adds representation-learned features (UMAP/PCA/Autoencoder)
- Trains a strong baseline model (LightGBM)
- Ranks features by combined importance (tree gain + permutation)
- Outputs a CSV and prints top features split by type: raw, engineered, representation
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
from src.data_processing import DataProcessor
from src.model import ModelTrainer
from src.evaluation import ModelEvaluator
import config as cfg

# Optional RAG guidance
try:
    from rag_system.research_engine import research_engine
except Exception:
    research_engine = None

from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("derive_best_features")


def make_output_dirs():
    out_dir = cfg.EVALUATION_CONFIG['results_dir'] / 'feature_engineering'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def infer_feature_type(name: str) -> str:
    if name.startswith("rep_"):
        return "representation"
    if any(s in name for s in ["_squared", "_sqrt", "_x_", "_log"]):
        return "engineered"
    return "raw"


def main():
    parser = argparse.ArgumentParser(description="Derive best features via representation learning")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV with raw data")
    parser.add_argument("--target-column", type=str, default=cfg.DATA_CONFIG['target_column'])
    parser.add_argument("--rep-method", type=str, choices=["umap", "pca", "autoencoder"],
                        default=cfg.FEATURE_CONFIG['representation_learning']['method'])
    parser.add_argument("--rep-components", type=int, default=cfg.FEATURE_CONFIG['representation_learning']['n_components'])
    parser.add_argument("--no-engineer", action="store_true", help="Disable classic feature engineering")
    parser.add_argument("--no-select", action="store_true", help="Disable SelectKBest feature selection (we do model-based ranking instead)")

    args = parser.parse_args()

    out_dir = make_output_dirs()

    # Load data
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded data: {df.shape}")

    # Build representation config from cfg + CLI overrides
    rep_cfg = dict(cfg.FEATURE_CONFIG['representation_learning'])
    rep_cfg['enabled'] = True
    rep_cfg['method'] = args.rep_method
    rep_cfg['n_components'] = int(args.rep_components)

    # Initialize processor using project config
    processor = DataProcessor(
        scaler_type=cfg.FEATURE_CONFIG['scaler_type'],
        imputer_strategy=cfg.FEATURE_CONFIG['imputer_strategy'],
        test_size=cfg.DATA_CONFIG['test_size'],
        random_state=cfg.DATA_CONFIG['random_state'],
        representation_config=rep_cfg,
        categorical_config=cfg.FEATURE_CONFIG.get('categorical_encoding', {'enabled': True})
    )

    # Run preprocessing with representation learning and without SelectKBest
    processed = processor.preprocess_pipeline(
        df,
        target_column=args.target_column,
        engineer_features=not args.no_engineer,
        select_features=not args.no_select,
        k_features=cfg.FEATURE_CONFIG['n_features_to_select'],
        use_representation=True,
    )

    X_train = processed['X_train']
    X_test = processed['X_test']
    y_train = processed['y_train']
    y_test = processed['y_test']

    # Encode labels if they are strings ('ASD_DD'/'TD')
    if y_train.dtype == object:
        mapping = {"TD": 0, "ASD_DD": 1}
        try:
            y_train = y_train.map(mapping).astype(int)
            y_test = y_test.map(mapping).astype(int)
        except Exception:
            # Fallback: factorize
            y_train, uniques = pd.factorize(y_train)
            y_test = pd.Series(pd.Categorical(y_test, categories=list(uniques))).cat.codes

    # Train a strong baseline model: LightGBM (fast + good importance)
    trainer = ModelTrainer(random_state=cfg.DATA_CONFIG['random_state'])
    model, params, cv_score = trainer.train_model(
        model_name='lightgbm',
        X_train=X_train,
        y_train=y_train,
        tune_hyperparameters=False,
        cv_folds=5,
        sensitivity_weight=cfg.MODEL_CONFIG['sensitivity_weight']
    )

    # Evaluate quickly to ensure model works
    evaluator = ModelEvaluator()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
    logger.info(f"Quick eval - Sensitivity: {metrics['sensitivity']:.3f}, Specificity: {metrics['specificity']:.3f}, AUC: {metrics.get('auc_roc', np.nan):.3f}")

    feature_names = X_train.columns.tolist()

    # 1) Built-in gain/importance (if available)
    gain_importance = None
    if hasattr(model, 'feature_importances_'):
        gain_importance = np.array(model.feature_importances_, dtype=float)
        if gain_importance.sum() > 0:
            gain_importance = gain_importance / gain_importance.sum()
    else:
        gain_importance = np.zeros(len(feature_names), dtype=float)

    # 2) Permutation importance on test set (robust)
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=cfg.DATA_CONFIG['random_state'], n_jobs=-1)
        perm_importance = perm.importances_mean
        perm_importance = np.clip(perm_importance, a_min=0, a_max=None)
        if perm_importance.max() > 0:
            perm_importance = perm_importance / perm_importance.max()
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        perm_importance = np.zeros(len(feature_names), dtype=float)

    # Combine: weighted average (can tweak weights)
    combined = 0.6 * gain_importance + 0.4 * perm_importance

    # Optional: RAG-guided boost for recommended features
    rag_boost = np.ones(len(feature_names), dtype=float)
    rag_notes = {}
    try:
        if research_engine is not None and getattr(research_engine, 'behavioral_database', None) is not None:
            recs = research_engine.get_feature_recommendations()
            recommended = (recs or {}).get('recommended_features', {})
            if recommended:
                # Map by substring match of recommended keys into our feature names
                for rec_key, meta in recommended.items():
                    boost = 1.15 if meta.get('recommendation') == 'Strong' else 1.08 if meta.get('recommendation') == 'Moderate' else 1.03
                    for idx, fname in enumerate(feature_names):
                        if rec_key in fname:
                            rag_boost[idx] *= boost
                            rag_notes[fname] = f"RAG-boosted via {rec_key} ({meta.get('recommendation')})"
    except Exception as e:
        print(f"RAG boost skipped: {e}")

    combined = combined * rag_boost

    # Build DataFrame
    types = [infer_feature_type(f) for f in feature_names]
    rankings = pd.DataFrame({
        'feature': feature_names,
        'gain_importance': gain_importance,
        'permutation_importance': perm_importance,
        'combined_importance': combined,
        'type': types,
        'rag_note': [rag_notes.get(f, '') for f in feature_names]
    }).sort_values('combined_importance', ascending=False)

    # Save CSV
    csv_path = out_dir / 'feature_rankings.csv'
    rankings.to_csv(csv_path, index=False)

    # Print top features by type
    def print_top(df, typ: str, k: int = 15):
        subset = df[df['type'] == typ].head(k)
        if subset.empty:
            return
        print(f"\nTop {k} {typ} features:")
        for i, row in subset.iterrows():
            print(f"  - {row['feature']}\t({row['combined_importance']:.4f})")

    print("\n===== Top Features (All) =====")
    for i, row in rankings.head(25).iterrows():
        print(f"{i+1:2d}. {row['feature']}\t({row['type']})\t{row['combined_importance']:.4f}")

    print_top(rankings, 'raw', 15)
    print_top(rankings, 'engineered', 15)
    print_top(rankings, 'representation', 15)

    # Save a text summary
    summary_path = out_dir / 'top_features_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Top 25 features (feature, type, combined_importance)\n")
        for i, row in rankings.head(25).iterrows():
            f.write(f"{row['feature']}, {row['type']}, {row['combined_importance']:.6f}\n")

    print(f"\nSaved rankings to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

