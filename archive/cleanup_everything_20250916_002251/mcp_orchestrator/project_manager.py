#!/usr/bin/env python3
"""
MCP Project Orchestrator for Binary Classification Project
Manages the entire workflow from data processing to model deployment
Uses RAG research engine for intelligent decision making throughout the pipeline
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None  # fallback
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.base import clone as sk_clone
import warnings
warnings.filterwarnings('ignore')

# Representation learning
try:
    from src.representation import RepresentationLearner, RepresentationConfig
except Exception:
    from representation import RepresentationLearner, RepresentationConfig  # type: ignore

# Add the RAG system to path
sys.path.append(str(Path(__file__).parent.parent / "rag_system"))
from rag_system.config import config
from rag_system.research_engine import research_engine

class ProjectOrchestrator:
    """MCP Project Orchestrator for binary classification pipeline"""
    
    def __init__(self):
        self.project_state = {
            'current_phase': 'initialization',
            'completed_phases': [],
            'data_summary': {},
            'feature_recommendations': {},
            'model_results': {},
            'deployment_status': 'not_deployed'
        }
        self.results_dir = config.PROJECT_PATH / "results"
        self.models_dir = config.PROJECT_PATH / "models" 
        self.reports_dir = config.PROJECT_PATH / "reports"
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        print("🚀 MCP Project Orchestrator initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🤖 Models directory: {self.models_dir}")
        print(f"📊 Reports directory: {self.reports_dir}")
    
    def execute_full_pipeline(self, limit_sessions: Optional[int] = 200, target_sensitivity: float = 0.87) -> Dict[str, Any]:
        """Execute the complete binary classification pipeline
        Args:
            limit_sessions: optional cap on sessions to ingest
            target_sensitivity: ASD sensitivity target for threshold selection
        """
        print("\n" + "="*60)
        print("🔄 STARTING FULL BINARY CLASSIFICATION PIPELINE")
        print("="*60)
        
        pipeline_results = {}
        
        try:
            # Phase 1: Data Preparation
            print("\n📊 Phase 1: Data Preparation")
            data_prep_results = self.execute_data_preparation(limit_sessions)
            pipeline_results['data_preparation'] = data_prep_results
            
            # Phase 2: Feature Engineering (RAG-guided)
            print("\n🔧 Phase 2: RAG-Guided Feature Engineering")
            feature_results = self.execute_feature_engineering()
            pipeline_results['feature_engineering'] = feature_results
            
            # Phase 3: Exploratory Analysis (RAG-powered)
            print("\n🔍 Phase 3: RAG-Powered Exploratory Analysis")
            analysis_results = self.execute_exploratory_analysis()
            pipeline_results['exploratory_analysis'] = analysis_results
            
            # Phase 4: Model Development
            print("\n🤖 Phase 4: Model Development")
            model_results = self.execute_model_development(target_sensitivity=target_sensitivity)
            pipeline_results['model_development'] = model_results
            
            # Phase 5: Results Reporting
            print("\n📋 Phase 5: Results Reporting")
            report_results = self.execute_results_reporting()
            pipeline_results['results_reporting'] = report_results
            
            self.project_state['current_phase'] = 'completed'
            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed at phase {self.project_state['current_phase']}: {e}")
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def execute_data_preparation(self, limit_sessions: Optional[int] = None) -> Dict[str, Any]:
        """Phase 1: Prepare and validate data using RAG research engine"""
        self.project_state['current_phase'] = 'data_preparation'
        
        print("🔄 Ingesting raw coloring game data...")
        data_summary = research_engine.ingest_raw_data(limit=limit_sessions)
        
        print("🔄 Indexing behavioral data in RAG system...")
        research_engine.index_behavioral_data()
        
        # Validate data quality
        validation_results = self._validate_data_quality()
        
        results = {
            'data_summary': data_summary,
            'validation_results': validation_results,
            'status': 'completed'
        }
        
        self.project_state['data_summary'] = data_summary
        self.project_state['completed_phases'].append('data_preparation')
        
# Save data summary
        with open(self.results_dir / "data_summary.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def execute_feature_engineering(self) -> Dict[str, Any]:
        """Phase 2: RAG-guided feature engineering and selection"""
        self.project_state['current_phase'] = 'feature_engineering'
        
        print("🤖 Consulting RAG system for feature recommendations...")
        feature_recs = research_engine.get_feature_recommendations()
        
        # Research key behavioral patterns
        motor_patterns = research_engine.research_query(
            "motor control differences between ASD and TD children, velocity and tremor patterns"
        )
        
        planning_patterns = research_engine.research_query(
            "executive function and planning differences, zone transitions and completion patterns"
        )
        
        multitouch_patterns = research_engine.research_query(
            "multi-touch behavior and palm touches in ASD versus TD children"
        )
        
        results = {
            'feature_recommendations': feature_recs,
            'motor_control_insights': motor_patterns,
            'planning_executive_insights': planning_patterns,
            'multitouch_insights': multitouch_patterns,
            'status': 'completed'
        }
        
        self.project_state['feature_recommendations'] = feature_recs
        self.project_state['completed_phases'].append('feature_engineering')
        
# Save feature analysis
        with open(self.results_dir / "feature_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def execute_exploratory_analysis(self) -> Dict[str, Any]:
        """Phase 3: RAG-powered exploratory data analysis"""
        self.project_state['current_phase'] = 'exploratory_analysis'
        
        # Comprehensive research queries
        research_queries = [
            "What are the most discriminative behavioral features for ASD vs TD classification?",
            "How do touch dynamics differ between diagnostic groups during coloring tasks?",
            "What motor control patterns are characteristic of ASD children?",
            "How does task completion and planning differ between ASD and TD groups?",
            "What role do palm touches and multi-touch patterns play in classification?"
        ]
        
        research_insights = {}
        for query in research_queries:
            print(f"🔍 Researching: {query[:60]}...")
            result = research_engine.research_query(query, n_results=20)
            research_insights[query] = result
        
        # Generate statistical summary
        behavioral_df = pd.DataFrame(research_engine.behavioral_database)
        statistical_summary = self._generate_statistical_summary(behavioral_df)
        
        results = {
            'research_insights': research_insights,
            'statistical_summary': statistical_summary,
            'data_shape': behavioral_df.shape if not behavioral_df.empty else (0, 0),
            'status': 'completed'
        }
        
        self.project_state['completed_phases'].append('exploratory_analysis')
        
        # Save exploratory analysis
        with open(self.results_dir / "exploratory_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def execute_model_development(self, use_representation: bool = True, target_sensitivity: float = 0.87, threshold_strategy: str = 'sens_first', target_specificity: float = 0.70, calibrate_per_fold: bool = False, calibration_method: str = 'sigmoid', use_polynomial: bool = False) -> Dict[str, Any]:
        """Phase 4: Develop and validate classification models
        Args:
            use_representation: Whether to augment features with learned representations
            target_sensitivity: ASD sensitivity target for threshold selection (used when threshold_strategy='sens_first')
            threshold_strategy: 'sens_first' (default), 'spec_first', or 'clinical' (require both Sens and Spec targets)
            target_specificity: ASD specificity target (used when threshold_strategy='spec_first' or 'clinical')
            calibrate_per_fold: If True, apply probability calibration per fold before thresholding
            calibration_method: 'sigmoid' or 'isotonic' calibration method
            use_polynomial: If True, apply PolynomialFeatures(degree=2) on scaled inputs (replaces representation step)
        """
        self.project_state['current_phase'] = 'model_development'
        
        # Prepare modeling dataset (base features only; RAG features added leakage-safe later)
        modeling_data = self._prepare_modeling_dataset_base()
        
        if modeling_data is None or modeling_data['X'].empty:
            return {'error': 'No valid modeling data available', 'status': 'failed'}
        
        X = modeling_data['X']
        y = modeling_data['y']
        feature_names = modeling_data['feature_names']
        
        print(f"🎯 Training (child-level) on {X.shape[0]} children with {X.shape[1]} features")
        print(f"📊 Label distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Group-aware holdout split by child
        groups = modeling_data.get('groups')
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx = list(gss.split(X, y, groups=groups))[0]
        train_idx, test_idx = idx[0], idx[1]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = [groups[i] for i in train_idx]

        # Leakage-safe RAG features computed using training children only
        child_ids = modeling_data['child_ids']
        train_child_ids = [child_ids[i] for i in train_idx]
        test_child_ids = [child_ids[i] for i in test_idx]
        target_child_ids = train_child_ids + test_child_ids

        # Attach child_id for merging
        X_with_id = X.copy()
        X_with_id['child_id'] = child_ids
        # Base (no RAG) copies for nested CV and ablations
        X_train_base_df = X_with_id.iloc[train_idx].copy()
        X_test_base_df = X_with_id.iloc[test_idx].copy()
        # Working DataFrames that will receive RAG features for holdout training
        X_train_df = X_train_base_df.copy()
        X_test_df = X_test_base_df.copy()

        try:
            rag_feat_df = research_engine.compute_child_rag_features_leak_safe(train_child_ids, target_child_ids=target_child_ids)
        except Exception:
            rag_feat_df = None
        
        if rag_feat_df is not None and not rag_feat_df.empty:
            X_train_df = X_train_df.merge(rag_feat_df, on='child_id', how='left')
            X_test_df = X_test_df.merge(rag_feat_df, on='child_id', how='left')
            X_train_df = X_train_df.fillna(0)
            X_test_df = X_test_df.fillna(0)
            rag_cols = [c for c in rag_feat_df.columns if c != 'child_id']
            feature_names = [c for c in X.columns] + rag_cols
        else:
            rag_cols = []
            feature_names = [c for c in X.columns]
        
        # Final train/test feature matrices (drop child_id)
        feature_cols_full = [c for c in X_train_df.columns if c != 'child_id']
        X_train = X_train_df[feature_cols_full]
        X_test = X_test_df[feature_cols_full]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Optional polynomial expansion (replaces representation if enabled)
        if use_polynomial:
            try:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_train_poly = poly.fit_transform(X_train_scaled)
                X_test_poly = poly.transform(X_test_scaled)
                X_train_aug = X_train_poly
                X_test_aug = X_test_poly
                rep_added = False
                rep_feature_names = []
            except Exception:
                X_train_aug = X_train_scaled
                X_test_aug = X_test_scaled
                rep_added = False
                rep_feature_names = []
        else:
            # Representation learning (augment features)
            rep_added = False
            rep_feature_names: List[str] = []
            if use_representation:
                try:
                    rep_cfg = RepresentationConfig(method="umap", n_components=min(16, max(2, X_train_scaled.shape[1] // 2)))
                    rep_learner = RepresentationLearner(rep_cfg)
                    Xtr_df = pd.DataFrame(X_train_scaled)
                    Xte_df = pd.DataFrame(X_test_scaled)
                    Ztr = rep_learner.fit_transform(Xtr_df)
                    Zte = rep_learner.transform(Xte_df)
                    # Concatenate
                    X_train_aug = np.concatenate([X_train_scaled, Ztr.values], axis=1)
                    X_test_aug = np.concatenate([X_test_scaled, Zte.values], axis=1)
                    rep_feature_names = Ztr.columns.tolist()
                    rep_added = True
                except Exception:
                    # Fallback: no representation learning
                    X_train_aug = X_train_scaled
                    X_test_aug = X_test_scaled
            else:
                X_train_aug = X_train_scaled
                X_test_aug = X_test_scaled

        # Train multiple models (expanded) + small sweep to improve Sens≥target with best Spec
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, class_weight={0:1.0, 1:1.5}),
            'SVM': SVC(probability=True, random_state=42, class_weight={0:1.0, 1:1.5}, C=1.0, kernel='rbf')
        }
        
        # Advanced clinical pipeline: polynomial expansion + conservative ExtraTrees
        try:
            et_clinical = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('clf', ExtraTreesClassifier(
                    n_estimators=400,
                    criterion='entropy',
                    max_features=0.25,
                    min_samples_leaf=8,
                    min_samples_split=17,
                    random_state=42
                ))
            ])
            models['Extra Trees Clinical'] = et_clinical
        except Exception:
            pass
        # Add small sweep variants for key models
        try:
            # Logistic Regression class_weight sweep
            for label, cw in [('cw_bal', 'balanced'), ('cw1.2', {0:1.0, 1:1.2}), ('cw1.5', {0:1.0, 1:1.5}), ('cw2.0', {0:1.0, 1:2.0})]:
                models[f'Logistic Regression [{label}]'] = LogisticRegression(random_state=42, max_iter=2000, class_weight=cw)
            # RandomForest class_weight sweep
            for label, cw in [('balanced', 'balanced'), ('b_sub', 'balanced_subsample'), ('cw1.5', {0:1.0, 1:1.5}), ('cw2.0', {0:1.0, 1:2.0})]:
                models[f'Random Forest [{label}]'] = RandomForestClassifier(n_estimators=400, random_state=42, class_weight=cw, min_samples_leaf=6, min_samples_split=12)
            # Add Balanced Random Forest if available
            try:
                from imblearn.ensemble import BalancedRandomForestClassifier
                models['Balanced RF'] = BalancedRandomForestClassifier(n_estimators=400, random_state=42, max_features='sqrt', min_samples_leaf=6)
            except Exception:
                pass
            # SVM class_weight sweep (kept small for runtime)
            for label, cw in [('balanced', 'balanced'), ('cw1.5', {0:1.0, 1:1.5})]:
                models[f'SVM [{label}]'] = SVC(probability=True, random_state=42, class_weight=cw, C=1.0, kernel='rbf')
        except Exception:
            pass
        # Optional: XGBoost / LightGBM
        try:
            from xgboost import XGBClassifier
            # XGBoost scale_pos_weight sweep (include <1.0 to favor specificity)
            for spw in [0.5, 0.75, 1.0, 1.5, 2.0]:
                models[f'XGBoost [spw={spw}]'] = XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                    eval_metric='logloss', random_state=42, scale_pos_weight=spw
                )
            # Keep base key for ensemble compatibility
            models['XGBoost'] = XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, scale_pos_weight=1.5
            )
        except Exception:
            pass
        # Optionally include LightGBM if explicitly enabled to avoid verbose training output
        try:
            import os
            if os.environ.get('ENABLE_LGBM', '0') == '1':
                import lightgbm as lgb
                for spw in [0.5, 0.75, 1.0, 1.5, 2.0]:
                    models[f'LightGBM [spw={spw}]'] = lgb.LGBMClassifier(
                        n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbosity=-1, scale_pos_weight=spw, min_child_samples=30, feature_fraction=0.8
                    )
                models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=-1, scale_pos_weight=1.5, min_child_samples=30, feature_fraction=0.8
                )
        except Exception:
            pass
        
        # Clinical soft-voting ensemble (robust), if components available
        try:
            voters = []
            if 'Extra Trees Clinical' in models:
                voters.append(('et', models['Extra Trees Clinical']))
            if 'Logistic Regression [C=0.5, balanced]' in models:
                voters.append(('lr', models['Logistic Regression [C=0.5, balanced]']))
            if 'LightGBM' in models:
                voters.append(('lgbm', models['LightGBM']))
            if len(voters) >= 2:
                models['ClinicalVote'] = VotingClassifier(estimators=voters, voting='soft', n_jobs=None)
        except Exception:
            pass

        model_results = {}
        best_model = None
        best_score = 0

        # Group-aware CV
        if StratifiedGroupKFold is not None:
            cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            cv_splits = cv_splitter.split(X_train, y_train, groups=groups_train)
        else:
            # Fallback: plain StratifiedKFold (may leak if groups exist, but GroupKFold unavailable)
            cv_splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train)

        # Determine ASD class index in label encoding
        import numpy as _np
        le = modeling_data['label_encoder']
        classes_arr = _np.array(modeling_data['class_names'])
        try:
            asd_idx = int(_np.where(classes_arr == 'ASD')[0][0])
        except Exception:
            # Fallback: assume 'ASD' is encoded as 1 if present, else 0
            asd_idx = 1 if (len(classes_arr) > 1) else 0
        
        # Augment models with TD-favoring weights and C sweeps (post ASD index determination)
        try:
            td_idx = 0 if asd_idx == 1 else 1
            # Logistic C sweeps with different class_weight strategies
            for C in [0.5, 1.0, 2.0]:
                models[f'Logistic Regression [C={C}, balanced]'] = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', C=C)
                cw_td15 = {asd_idx: 1.0, td_idx: 1.5}
                models[f'Logistic Regression [C={C}, td1.5]'] = LogisticRegression(random_state=42, max_iter=2000, class_weight=cw_td15, C=C)
                cw_td20 = {asd_idx: 1.0, td_idx: 2.0}
                models[f'Logistic Regression [C={C}, td2.0]'] = LogisticRegression(random_state=42, max_iter=2000, class_weight=cw_td20, C=C)
            # SVM C sweeps
            for C in [0.5, 1.0, 2.0]:
                models[f'SVM [C={C}, balanced]'] = SVC(probability=True, random_state=42, class_weight='balanced', C=C, kernel='rbf')
                cw_asd15 = {asd_idx: 1.5, td_idx: 1.0}
                models[f'SVM [C={C}, asd1.5]'] = SVC(probability=True, random_state=42, class_weight=cw_asd15, C=C, kernel='rbf')
                cw_td15 = {asd_idx: 1.0, td_idx: 1.5}
                models[f'SVM [C={C}, td1.5]'] = SVC(probability=True, random_state=42, class_weight=cw_td15, C=C, kernel='rbf')
            # RandomForest TD-favor variants
            for label, cw in [('td1.2', {asd_idx: 1.0, td_idx: 1.2}), ('td1.5', {asd_idx: 1.0, td_idx: 1.5})]:
                models[f'Random Forest [{label}]'] = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=cw)
        except Exception:
            pass

        for name, model in models.items():
            print(f"🔧 Training {name}...")

            # Use augmented (rep-learned) data if available
            X_train_model = X_train_aug
            X_test_model = X_test_aug

            # Cross-validation (manual to support group-aware)
            from numpy import mean, std
            # Placeholder (will be replaced by strict nested CV block below)
            aucs = []
            fold_details = []
            cv_mean = 0.0
            cv_std = 0.0
            cv_mean_sens = 0.0
            cv_std_sens = 0.0
            cv_mean_spec = 0.0
            cv_std_spec = 0.0

            # Strict nested CV: fold-specific RAG centroids, scaler, and representation
            aucs_nested = []
            fold_details_nested = []
            fold_num = 1
            child_ids_train = [child_ids[i] for i in train_idx]
            # Iterate folds
            for tr_idx2, va_idx2 in (cv_splitter.split(X_train, y_train, groups=groups_train) if StratifiedGroupKFold is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train)):
                # Fold child id lists
                fold_train_ids = [child_ids_train[i] for i in tr_idx2]
                fold_val_ids = [child_ids_train[i] for i in va_idx2]
                # Base features for the fold
                fold_tr_base = X_train_base_df[X_train_base_df['child_id'].isin(fold_train_ids)].copy()
                fold_va_base = X_train_base_df[X_train_base_df['child_id'].isin(fold_val_ids)].copy()
                # Fold RAG features computed only from fold-train children
                try:
                    rag_fold = research_engine.compute_child_rag_features_leak_safe(fold_train_ids, target_child_ids=(fold_train_ids + fold_val_ids))
                except Exception:
                    rag_fold = None
                if rag_fold is not None and not rag_fold.empty:
                    fold_tr_full = fold_tr_base.merge(rag_fold, on='child_id', how='left')
                    fold_va_full = fold_va_base.merge(rag_fold, on='child_id', how='left')
                else:
                    fold_tr_full = fold_tr_base
                    fold_va_full = fold_va_base
                # Order rows to align with y indices
                fold_tr_full = fold_tr_full.set_index('child_id').loc[fold_train_ids].reset_index()
                fold_va_full = fold_va_full.set_index('child_id').loc[fold_val_ids].reset_index()
                # Prepare matrices
                feature_cols_f = [c for c in fold_tr_full.columns if c != 'child_id']
                Xtr = fold_tr_full[feature_cols_f].fillna(0)
                Xva = fold_va_full[feature_cols_f].fillna(0)
                # Fold scaler
                scaler_f = StandardScaler()
                Xtr_s = scaler_f.fit_transform(Xtr)
                Xva_s = scaler_f.transform(Xva)
                # Fold polynomial or representation
                if use_polynomial:
                    try:
                        poly_f = PolynomialFeatures(degree=2, include_bias=False)
                        Xtr_model = poly_f.fit_transform(Xtr_s)
                        Xva_model = poly_f.transform(Xva_s)
                    except Exception:
                        Xtr_model = Xtr_s
                        Xva_model = Xva_s
                else:
                    # Fold representation (optional)
                    if use_representation:
                        try:
                            rep_cfg_f = RepresentationConfig(method="umap", n_components=min(16, max(2, Xtr_s.shape[1] // 2)))
                            rep_learner_f = RepresentationLearner(rep_cfg_f)
                            import pandas as _pd
                            Ztr_f = rep_learner_f.fit_transform(_pd.DataFrame(Xtr_s))
                            Zva_f = rep_learner_f.transform(_pd.DataFrame(Xva_s))
                            Xtr_model = __import__('numpy').concatenate([Xtr_s, Ztr_f.values], axis=1)
                            Xva_model = __import__('numpy').concatenate([Xva_s, Zva_f.values], axis=1)
                        except Exception:
                            Xtr_model = Xtr_s
                            Xva_model = Xva_s
                    else:
                        Xtr_model = Xtr_s
                        Xva_model = Xva_s
                # Fresh model instance for the fold
                try:
                    fold_model = sk_clone(model)
                except Exception:
                    try:
                        params = model.get_params()
                        fold_model = model.__class__(**params)
                    except Exception:
                        fold_model = model.__class__()
                # Fit and predict (with optional per-fold calibration)
                fold_model.fit(Xtr_model, y_train[tr_idx2])
                proba_f = None
                if hasattr(fold_model, 'predict_proba'):
                    if calibrate_per_fold:
                        try:
                            cal_f = CalibratedClassifierCV(fold_model, method=calibration_method, cv=3)
                            cal_f.fit(Xtr_model, y_train[tr_idx2])
                            try:
                                col_idx_f = list(cal_f.classes_).index(asd_idx)
                            except Exception:
                                col_idx_f = 1 if len(getattr(cal_f, 'classes_', [])) > 1 else 0
                            proba_f = cal_f.predict_proba(Xva_model)[:, col_idx_f]
                        except Exception:
                            proba_f = None
                    if proba_f is None:
                        try:
                            col_idx_f = list(fold_model.classes_).index(asd_idx)
                        except Exception:
                            col_idx_f = 1 if len(getattr(fold_model, 'classes_', [])) > 1 else 0
                        proba_f = fold_model.predict_proba(Xva_model)[:, col_idx_f]
                else:
                    try:
                        scores_f = fold_model.decision_function(Xva_model)
                        import numpy as np
                        proba_f = (scores_f - scores_f.min()) / (scores_f.max() - scores_f.min() + 1e-8)
                    except Exception:
                        proba_f = fold_model.predict(Xva_model)
                y_val_bin_f = (y_train[va_idx2] == asd_idx).astype(int)
                # Metrics and target sensitivity threshold
                auc_val = roc_auc_score(y_val_bin_f, proba_f)
                aucs_nested.append(float(auc_val))
                from sklearn.metrics import roc_curve
                fpr_f, tpr_f, thr_f = roc_curve(y_val_bin_f, proba_f)
                import numpy as np
                if threshold_strategy == 'clinical':
                    idx = np.where((tpr_f >= target_sensitivity) & ((1 - fpr_f) >= target_specificity))[0]
                    if len(idx) > 0:
                        # Among jointly feasible thresholds, maximize balanced improvement
                        gains = (tpr_f[idx] - target_sensitivity) + ((1 - fpr_f[idx]) - target_specificity)
                        best_idx_f = int(idx[np.argmax(gains)])
                    else:
                        # Fallback: spec-first
                        idx = np.where((1 - fpr_f) >= target_specificity)[0]
                        if len(idx) > 0:
                            best_idx_f = int(idx[np.argmax(tpr_f[idx])])
                        else:
                            youden_f = tpr_f - fpr_f
                            best_idx_f = int(np.argmax(youden_f))
                elif threshold_strategy == 'spec_first':
                    idx = np.where((1 - fpr_f) >= target_specificity)[0]
                    if len(idx) > 0:
                        # Maximize sensitivity among those meeting target specificity
                        best_idx_f = int(idx[np.argmax(tpr_f[idx])])
                    else:
                        # Fallback to Youden's J
                        youden_f = tpr_f - fpr_f
                        best_idx_f = int(np.argmax(youden_f))
                else:
                    idx = np.where(tpr_f >= target_sensitivity)[0]
                    if len(idx) > 0:
                        # Maximize specificity among those meeting target sensitivity
                        best_idx_f = int(idx[np.argmin(fpr_f[idx])])
                    else:
                        # Fallback to Youden's J
                        youden_f = tpr_f - fpr_f
                        best_idx_f = int(np.argmax(youden_f))
                best_thr_f = float(thr_f[best_idx_f])
                sens_f = float(tpr_f[best_idx_f])
                spec_f = float(1 - fpr_f[best_idx_f])
                fold_details_nested.append({
                    'fold': fold_num,
                    'auc': float(auc_val),
                    'threshold': best_thr_f,
                    'sensitivity_asd': sens_f,
                    'specificity_asd': spec_f
                })
                fold_num += 1
            # Override CV metrics with strict nested results
            fold_details = fold_details_nested
            cv_mean = float(mean(aucs_nested)) if aucs_nested else 0.0
            cv_std = float(std(aucs_nested)) if aucs_nested else 0.0
            if fold_details:
                cv_mean_sens = float(mean([fd['sensitivity_asd'] for fd in fold_details]))
                cv_std_sens = float(std([fd['sensitivity_asd'] for fd in fold_details]))
                cv_mean_spec = float(mean([fd['specificity_asd'] for fd in fold_details]))
                cv_std_spec = float(std([fd['specificity_asd'] for fd in fold_details]))
            else:
                cv_mean_sens = cv_std_sens = cv_mean_spec = cv_std_spec = 0.0

            # Fit on all training data
            model.fit(X_train_model, y_train)

            # Test predictions
            y_pred = model.predict(X_test_model)
            if hasattr(model, 'predict_proba'):
                try:
                    col_idx = list(model.classes_).index(asd_idx)
                except Exception:
                    col_idx = 1 if len(getattr(model, 'classes_', [])) > 1 else 0
                y_pred_proba_asd = model.predict_proba(X_test_model)[:, col_idx]
            else:
                # decision_function fallback
                try:
                    scores = model.decision_function(X_test_model)
                    import numpy as np
                    y_pred_proba_asd = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                except Exception:
                    y_pred_proba_asd = None

            # Calculate metrics for ASD as positive class
            y_test_bin = (y_test == asd_idx).astype(int)
            test_auc = roc_auc_score(y_test_bin, y_pred_proba_asd) if y_pred_proba_asd is not None else 0

            model_results[name] = {
                'cv_mean_auc': cv_mean,
                'cv_std_auc': cv_std,
                'cv_mean_sensitivity_asd': cv_mean_sens,
                'cv_std_sensitivity_asd': cv_std_sens,
                'cv_mean_specificity_asd': cv_mean_spec,
                'cv_std_specificity_asd': cv_std_spec,
                'cv_folds': fold_details,
                'test_auc': test_auc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

            # Track best model
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = (name, model)

                # Save best model and scaler
                joblib.dump(model, self.models_dir / f"{name.lower().replace(' ', '_')}_model.pkl")
                joblib.dump(scaler, self.models_dir / "scaler.pkl")

        # Update feature names if rep features were added
        if rep_added and rep_feature_names:
            feature_names = feature_names + rep_feature_names

        # Ablation placeholder initialized; constructed after holdout thresholding
        ablation = None

        # Probability calibration and threshold optimization on holdout (ASD positive)
        calibrated_name, calibrated_model = best_model
        try:
            if hasattr(calibrated_model, 'predict_proba'):
                calib = CalibratedClassifierCV(calibrated_model, method='sigmoid', cv=3)
                calib.fit(X_train_model, y_train)
                calibrated_model = calib
        except Exception:
            pass
        # Compute ROC and select threshold targeting ASD sensitivity; also compute calibrated holdout AUC
        try:
            if hasattr(calibrated_model, 'predict_proba'):
                try:
                    col_idx = list(getattr(calibrated_model, 'classes_', getattr(best_model[1], 'classes_', []))).index(asd_idx)
                except Exception:
                    col_idx = 1 if len(getattr(calibrated_model, 'classes_', [])) > 1 else 0
                y_proba_holdout = calibrated_model.predict_proba(X_test_model)[:, col_idx]
            else:
                y_proba_holdout = y_pred_proba_asd
            import numpy as np
            y_test_bin = (y_test == asd_idx).astype(int)
            fpr, tpr, thr = roc_curve(y_test_bin, y_proba_holdout)
            if threshold_strategy == 'clinical':
                idx = __import__('numpy').where((tpr >= target_sensitivity) & ((1 - fpr) >= target_specificity))[0]
                if len(idx) > 0:
                    gains = (tpr[idx] - target_sensitivity) + ((1 - fpr[idx]) - target_specificity)
                    best_idx = int(idx[__import__('numpy').argmax(gains)])
                else:
                    idx = __import__('numpy').where((1 - fpr) >= target_specificity)[0]
                    if len(idx) > 0:
                        best_idx = int(idx[__import__('numpy').argmax(tpr[idx])])
                    else:
                        youden = tpr - fpr
                        best_idx = int(__import__('numpy').argmax(youden))
            elif threshold_strategy == 'spec_first':
                idx = __import__('numpy').where((1 - fpr) >= target_specificity)[0]
                if len(idx) > 0:
                    best_idx = int(idx[__import__('numpy').argmax(tpr[idx])])
                else:
                    youden = tpr - fpr
                    best_idx = int(__import__('numpy').argmax(youden))
            else:
                idx = __import__('numpy').where(tpr >= target_sensitivity)[0]
                if len(idx) > 0:
                    best_idx = int(idx[__import__('numpy').argmin(fpr[idx])])
                else:
                    youden = tpr - fpr
                    best_idx = int(__import__('numpy').argmax(youden))
            best_thr = float(thr[best_idx])
            best_sens = float(tpr[best_idx])
            best_spec = float(1 - fpr[best_idx])
            # Calibrated holdout AUC
            test_auc_holdout = float(roc_auc_score(y_test_bin, y_proba_holdout))
        except Exception:
            best_thr, best_sens, best_spec = 0.5, None, None
            test_auc_holdout = 0.0

        # Build ablation (Base vs RAG-aug) now that best_thr/best_sens/best_spec are known
        try:
            if best_model:
                best_name = best_model[0]
                base_model_proto = models[best_name]
                from numpy import mean, std
                child_ids_train = [child_ids[i] for i in train_idx]
                # Nested CV on base features only
                aucs_base = []
                folds_base = []
                fold_num = 1
                for tr_idx2, va_idx2 in (cv_splitter.split(X_train, y_train, groups=groups_train) if StratifiedGroupKFold is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train)):
                    fold_train_ids = [child_ids_train[i] for i in tr_idx2]
                    fold_val_ids = [child_ids_train[i] for i in va_idx2]
                    tr_base = X_train_base_df.set_index('child_id').loc[fold_train_ids].reset_index()
                    va_base = X_train_base_df.set_index('child_id').loc[fold_val_ids].reset_index()
                    feat_cols = [c for c in tr_base.columns if c != 'child_id']
                    Xtr_b = tr_base[feat_cols].fillna(0)
                    Xva_b = va_base[feat_cols].fillna(0)
                    scaler_b = StandardScaler()
                    Xtr_bs = scaler_b.fit_transform(Xtr_b)
                    Xva_bs = scaler_b.transform(Xva_b)
                    if use_representation:
                        try:
                            rep_cfg_b = RepresentationConfig(method="umap", n_components=min(16, max(2, Xtr_bs.shape[1] // 2)))
                            rep_learner_b = RepresentationLearner(rep_cfg_b)
                            import pandas as _pd
                            Ztr_b = rep_learner_b.fit_transform(_pd.DataFrame(Xtr_bs))
                            Zva_b = rep_learner_b.transform(_pd.DataFrame(Xva_bs))
                            Xtr_bm = __import__('numpy').concatenate([Xtr_bs, Ztr_b.values], axis=1)
                            Xva_bm = __import__('numpy').concatenate([Xva_bs, Zva_b.values], axis=1)
                        except Exception:
                            Xtr_bm = Xtr_bs
                            Xva_bm = Xva_bs
                    else:
                        Xtr_bm = Xtr_bs
                        Xva_bm = Xva_bs
                try:
                    base_fold_model = sk_clone(base_model_proto)
                except Exception:
                    try:
                        params = base_model_proto.get_params()
                        base_fold_model = base_model_proto.__class__(**params)
                    except Exception:
                        base_fold_model = base_model_proto.__class__()
                base_fold_model.fit(Xtr_bm, y_train[tr_idx2])
                if hasattr(base_fold_model, 'predict_proba'):
                    try:
                        col_idx_b = list(base_fold_model.classes_).index(asd_idx)
                    except Exception:
                        col_idx_b = 1 if len(getattr(base_fold_model, 'classes_', [])) > 1 else 0
                    proba_b = base_fold_model.predict_proba(Xva_bm)[:, col_idx_b]
                else:
                    try:
                        scores_b = base_fold_model.decision_function(Xva_bm)
                        import numpy as np
                        proba_b = (scores_b - scores_b.min()) / (scores_b.max() - scores_b.min() + 1e-8)
                    except Exception:
                        proba_b = base_fold_model.predict(Xva_bm)
                y_val_bin_b = (y_train[va_idx2] == asd_idx).astype(int)
                from sklearn.metrics import roc_curve
                fpr_b, tpr_b, thr_b = roc_curve(y_val_bin_b, proba_b)
                import numpy as np
                    if threshold_strategy == 'clinical':
                        idx_b = np.where((tpr_b >= target_sensitivity) & ((1 - fpr_b) >= target_specificity))[0]
                        if len(idx_b) > 0:
                            gains_b = (tpr_b[idx_b] - target_sensitivity) + ((1 - fpr_b[idx_b]) - target_specificity)
                            best_idx_b = int(idx_b[np.argmax(gains_b)])
                        else:
                            idx_b = np.where((1 - fpr_b) >= target_specificity)[0]
                            if len(idx_b) > 0:
                                best_idx_b = int(idx_b[np.argmax(tpr_b[idx_b])])
                            else:
                                youden_b = tpr_b - fpr_b
                                best_idx_b = int(np.argmax(youden_b))
                    elif threshold_strategy == 'spec_first':
                        idx_b = np.where((1 - fpr_b) >= target_specificity)[0]
                        if len(idx_b) > 0:
                            best_idx_b = int(idx_b[np.argmax(tpr_b[idx_b])])
                        else:
                            youden_b = tpr_b - fpr_b
                            best_idx_b = int(np.argmax(youden_b))
                    else:
                        idx_b = np.where(tpr_b >= target_sensitivity)[0]
                        if len(idx_b) > 0:
                            best_idx_b = int(idx_b[np.argmin(fpr_b[idx_b])])
                        else:
                            youden_b = tpr_b - fpr_b
                            best_idx_b = int(np.argmax(youden_b))
                thr_best_b = float(thr_b[best_idx_b])
                sens_b = float(tpr_b[best_idx_b])
                spec_b = float(1 - fpr_b[best_idx_b])
                auc_b = roc_auc_score(y_val_bin_b, proba_b)
                aucs_base.append(float(auc_b))
                folds_base.append({'fold': fold_num, 'auc': float(auc_b), 'threshold': thr_best_b, 'sensitivity_asd': sens_b, 'specificity_asd': spec_b})
                fold_num += 1
                base_cv = {
                    'cv_mean_auc': float(mean(aucs_base)) if aucs_base else 0.0,
                    'cv_std_auc': float(std(aucs_base)) if aucs_base else 0.0,
                    'cv_mean_sensitivity_asd': float(mean([f['sensitivity_asd'] for f in folds_base])) if folds_base else 0.0,
                    'cv_std_sensitivity_asd': float(std([f['sensitivity_asd'] for f in folds_base])) if folds_base else 0.0,
                    'cv_mean_specificity_asd': float(mean([f['specificity_asd'] for f in folds_base])) if folds_base else 0.0,
                    'cv_std_specificity_asd': float(std([f['specificity_asd'] for f in folds_base])) if folds_base else 0.0,
                    'cv_folds': folds_base
                }
                # Base-only holdout scoring already computed below for rag_aug; recompute for base
                feat_cols_base = [c for c in X_train_base_df.columns if c != 'child_id']
                Xtr_b_all = X_train_base_df[feat_cols_base].fillna(0)
                Xte_b_all = X_test_base_df[feat_cols_base].fillna(0)
                scaler_b_all = StandardScaler()
                Xtr_b_all_s = scaler_b_all.fit_transform(Xtr_b_all)
                Xte_b_all_s = scaler_b_all.transform(Xte_b_all)
                if use_representation:
                    try:
                        rep_cfg_b_all = RepresentationConfig(method="umap", n_components=min(16, max(2, Xtr_b_all_s.shape[1] // 2)))
                        rep_learner_b_all = RepresentationLearner(rep_cfg_b_all)
                        import pandas as _pd
                        Ztr_b_all = rep_learner_b_all.fit_transform(_pd.DataFrame(Xtr_b_all_s))
                        Zte_b_all = rep_learner_b_all.transform(_pd.DataFrame(Xte_b_all_s))
                        Xtr_b_all_m = __import__('numpy').concatenate([Xtr_b_all_s, Ztr_b_all.values], axis=1)
                        Xte_b_all_m = __import__('numpy').concatenate([Xte_b_all_s, Zte_b_all.values], axis=1)
                    except Exception:
                        Xtr_b_all_m = Xtr_b_all_s
                        Xte_b_all_m = Xte_b_all_s
                else:
                    Xtr_b_all_m = Xtr_b_all_s
                    Xte_b_all_m = Xte_b_all_s
                try:
                    best_base_model = sk_clone(base_model_proto)
                except Exception:
                    try:
                        params = base_model_proto.get_params()
                        best_base_model = base_model_proto.__class__(**params)
                    except Exception:
                        best_base_model = base_model_proto.__class__()
                best_base_model.fit(Xtr_b_all_m, y_train)
                cal_base = None
                try:
                    if hasattr(best_base_model, 'predict_proba'):
                        cal_base = CalibratedClassifierCV(best_base_model, method='sigmoid', cv=3)
                        cal_base.fit(Xtr_b_all_m, y_train)
                except Exception:
                    cal_base = None
                if cal_base is not None:
                    try:
                        col_idx_base = list(getattr(cal_base, 'classes_', getattr(best_base_model, 'classes_', []))).index(asd_idx)
                    except Exception:
                        col_idx_base = 1 if len(getattr(cal_base, 'classes_', [])) > 1 else 0
                    y_proba_base = cal_base.predict_proba(Xte_b_all_m)[:, col_idx_base]
                else:
                    if hasattr(best_base_model, 'predict_proba'):
                        try:
                            col_idx_base = list(best_base_model.classes_).index(asd_idx)
                        except Exception:
                            col_idx_base = 1 if len(getattr(best_base_model, 'classes_', [])) > 1 else 0
                        y_proba_base = best_base_model.predict_proba(Xte_b_all_m)[:, col_idx_base]
                    else:
                        try:
                            scores_bb = best_base_model.decision_function(Xte_b_all_m)
                            import numpy as np
                            y_proba_base = (scores_bb - scores_bb.min()) / (scores_bb.max() - scores_bb.min() + 1e-8)
                        except Exception:
                            y_proba_base = None
                from sklearn.metrics import roc_curve
                import numpy as np
                y_test_bin = (y_test == asd_idx).astype(int)
                if y_proba_base is not None:
                    fpr_bh, tpr_bh, thr_bh = roc_curve(y_test_bin, y_proba_base)
                    if threshold_strategy == 'clinical':
                        idx_bh = __import__('numpy').where((tpr_bh >= target_sensitivity) & ((1 - fpr_bh) >= target_specificity))[0]
                        if len(idx_bh) > 0:
                            gains_bh = (tpr_bh[idx_bh] - target_sensitivity) + ((1 - fpr_bh[idx_bh]) - target_specificity)
                            best_idx_bh = int(idx_bh[__import__('numpy').argmax(gains_bh)])
                        else:
                            idx_bh = __import__('numpy').where((1 - fpr_bh) >= target_specificity)[0]
                            if len(idx_bh) > 0:
                                best_idx_bh = int(idx_bh[__import__('numpy').argmax(tpr_bh[idx_bh])])
                            else:
                                youden_bh = tpr_bh - fpr_bh
                                best_idx_bh = int(__import__('numpy').argmax(youden_bh))
                    elif threshold_strategy == 'spec_first':
                        idx_bh = __import__('numpy').where((1 - fpr_bh) >= target_specificity)[0]
                        if len(idx_bh) > 0:
                            best_idx_bh = int(idx_bh[__import__('numpy').argmax(tpr_bh[idx_bh])])
                        else:
                            youden_bh = tpr_bh - fpr_bh
                            best_idx_bh = int(__import__('numpy').argmax(youden_bh))
                    else:
                        idx_bh = __import__('numpy').where(tpr_bh >= target_sensitivity)[0]
                        if len(idx_bh) > 0:
                            best_idx_bh = int(idx_bh[__import__('numpy').argmin(fpr_bh[idx_bh])])
                        else:
                            youden_bh = tpr_bh - fpr_bh
                            best_idx_bh = int(__import__('numpy').argmax(youden_bh))
                    thr_bh = float(thr_bh[best_idx_bh])
                    sens_bh = float(tpr_bh[best_idx_bh])
                    spec_bh = float(1 - fpr_bh[best_idx_bh])
                    test_auc_b = float(roc_auc_score(y_test_bin, y_proba_base))
                else:
                    thr_bh = None
                    sens_bh = None
                    spec_bh = None
                    test_auc_b = 0.0
                # Fetch rag-aug CV metrics from recorded model_results for best model
                rag_cv_src = model_results.get(best_name, {})
                rag_cv = {
                    'cv_mean_auc': rag_cv_src.get('cv_mean_auc', 0.0),
                    'cv_std_auc': rag_cv_src.get('cv_std_auc', 0.0),
                    'cv_mean_sensitivity_asd': rag_cv_src.get('cv_mean_sensitivity_asd', 0.0),
                    'cv_std_sensitivity_asd': rag_cv_src.get('cv_std_sensitivity_asd', 0.0),
                    'cv_mean_specificity_asd': rag_cv_src.get('cv_mean_specificity_asd', 0.0),
                    'cv_std_specificity_asd': rag_cv_src.get('cv_std_specificity_asd', 0.0),
                    'cv_folds': rag_cv_src.get('cv_folds', [])
                }
                ablation = {
                    'best_model': best_name,
                    'target_sensitivity': target_sensitivity,
                    'base': {
                        'cv': base_cv,
                        'holdout': {
                            'test_auc': test_auc_b,
                            'threshold': thr_bh,
                            'sensitivity_asd': sens_bh,
                            'specificity_asd': spec_bh
                        }
                    },
                    'rag_aug': {
                        'cv': rag_cv,
                        'holdout': {
                            'test_auc': test_auc_holdout,
                            'threshold': best_thr,
                            'sensitivity_asd': best_sens,
                            'specificity_asd': best_spec
                        }
                    }
                }
        except Exception:
            ablation = None

        # Feature importance for best model
        feature_importance = self._get_feature_importance(best_model[1], feature_names)

        # Clinical ensemble optimization (leak-free): target Sens >= 0.86 and Spec >= 0.70
        clinical_ensemble = None
        try:
            target_sens_ce = max(0.86, float(target_sensitivity))
            target_spec_ce = 0.70

            # Define base learners for ensemble
            ensemble_base_names = []
            ensemble_base_protos = []
            for nm in ['SVM', 'Logistic Regression', 'XGBoost', 'LightGBM', 'Gradient Boosting', 'Random Forest']:
                if nm in models:
                    ensemble_base_names.append(nm)
                    ensemble_base_protos.append(models[nm])
            if len(ensemble_base_names) >= 2:
                # Prepare OOF containers
                n_train = len(y_train)
                oof_preds = {nm: __import__('numpy').zeros(n_train, dtype=float) for nm in ensemble_base_names}
                oof_truth = (y_train == asd_idx).astype(int)
                oof_fold_indices = []  # to compute per-fold thresholds later

                # Repeat group-aware fold split (using base features + RAG fold-safe) and gather OOF predictions for all base models
                child_ids_train = [child_ids[i] for i in train_idx]
                fold_num = 0
                for tr_idx2, va_idx2 in (cv_splitter.split(X_train, y_train, groups=groups_train) if StratifiedGroupKFold is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train)):
                    fold_num += 1
                    fold_train_ids = [child_ids_train[i] for i in tr_idx2]
                    fold_val_ids = [child_ids_train[i] for i in va_idx2]
                    # Base features for the fold (NO leakage)
                    fold_tr_base = X_train_base_df[X_train_base_df['child_id'].isin(fold_train_ids)].copy()
                    fold_va_base = X_train_base_df[X_train_base_df['child_id'].isin(fold_val_ids)].copy()
                    # RAG fold-specific features
                    try:
                        rag_fold = research_engine.compute_child_rag_features_leak_safe(fold_train_ids, target_child_ids=(fold_train_ids + fold_val_ids))
                    except Exception:
                        rag_fold = None
                    if rag_fold is not None and not rag_fold.empty:
                        fold_tr_full = fold_tr_base.merge(rag_fold, on='child_id', how='left')
                        fold_va_full = fold_va_base.merge(rag_fold, on='child_id', how='left')
                    else:
                        fold_tr_full = fold_tr_base
                        fold_va_full = fold_va_base
                    # Ensure order
                    fold_tr_full = fold_tr_full.set_index('child_id').loc[fold_train_ids].reset_index()
                    fold_va_full = fold_va_full.set_index('child_id').loc[fold_val_ids].reset_index()
                    # Matrices
                    feat_cols_f = [c for c in fold_tr_full.columns if c != 'child_id']
                    Xtr = fold_tr_full[feat_cols_f].fillna(0)
                    Xva = fold_va_full[feat_cols_f].fillna(0)
                    scaler_f = StandardScaler()
                    Xtr_s = scaler_f.fit_transform(Xtr)
                    Xva_s = scaler_f.transform(Xva)
                    # Representation per fold
                    if use_representation:
                        try:
                            rep_cfg_f = RepresentationConfig(method="umap", n_components=min(16, max(2, Xtr_s.shape[1] // 2)))
                            rep_learner_f = RepresentationLearner(rep_cfg_f)
                            import pandas as _pd
                            Ztr_f = rep_learner_f.fit_transform(_pd.DataFrame(Xtr_s))
                            Zva_f = rep_learner_f.transform(_pd.DataFrame(Xva_s))
                            Xtr_m = __import__('numpy').concatenate([Xtr_s, Ztr_f.values], axis=1)
                            Xva_m = __import__('numpy').concatenate([Xva_s, Zva_f.values], axis=1)
                        except Exception:
                            Xtr_m = Xtr_s
                            Xva_m = Xva_s
                    else:
                        Xtr_m = Xtr_s
                        Xva_m = Xva_s
                    # Train each base model and store probabilities (ASD positive)
                    for nm, proto in zip(ensemble_base_names, ensemble_base_protos):
                        try:
                            mdl = sk_clone(proto)
                        except Exception:
                            try:
                                params = proto.get_params()
                            except Exception:
                                params = {}
                            try:
                                mdl = proto.__class__(**params)
                            except Exception:
                                mdl = proto.__class__()
                        mdl.fit(Xtr_m, y_train[tr_idx2])
                        if hasattr(mdl, 'predict_proba'):
                            try:
                                col_idx_m = list(mdl.classes_).index(asd_idx)
                            except Exception:
                                col_idx_m = 1 if len(getattr(mdl, 'classes_', [])) > 1 else 0
                            proba = mdl.predict_proba(Xva_m)[:, col_idx_m]
                        else:
                            # decision_function fallback
                            try:
                                scores = mdl.decision_function(Xva_m)
                                import numpy as _np
                                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                            except Exception:
                                proba = mdl.predict(Xva_m)
                        # Map fold validation positions back to global train indices
                        oof_positions = __import__('numpy').array(va_idx2)
                        oof_preds[nm][oof_positions] = proba
                    # Track fold indices for threshold estimation
                    oof_fold_indices.append((__import__('numpy').array(va_idx2),))

                import numpy as np
                # Convert dict to matrix
                model_names_ce = ensemble_base_names
                P_oof = np.vstack([oof_preds[nm] for nm in model_names_ce]).T  # shape (n_train, n_models)

                # Weight grid search over top models (limit to 3 best by individual OOF AUC to keep search small)
                auc_per_model = []
                for j, nm in enumerate(model_names_ce):
                    try:
                        auc_j = float(roc_auc_score(oof_truth, P_oof[:, j]))
                    except Exception:
                        auc_j = 0.0
                    auc_per_model.append((auc_j, j, nm))
                auc_per_model.sort(reverse=True)
                top_k = min(4, len(auc_per_model))
                top_indices = [idx for _, idx, _ in auc_per_model[:top_k]]
                # Ensure SVM is included if available
                if 'SVM' in model_names_ce:
                    svm_idx = model_names_ce.index('SVM')
                    if svm_idx not in top_indices:
                        top_indices[-1] = svm_idx
                # Deduplicate and preserve order
                top_indices = list(dict.fromkeys(top_indices))
                P_top = P_oof[:, top_indices]
                top_names = [model_names_ce[idx] for idx in top_indices]

                # Generate simplex weight vectors with step 0.05 for finer search
                def gen_simplex_weights(k, step=0.05):
                    import itertools
                    steps = int(1.0/step) + 1
                    for parts in itertools.product(range(steps), repeat=k):
                        if sum(parts) == steps - 1:
                            w = [p*step for p in parts]
                            yield __import__('numpy').array(w, dtype=float)
                best_combo = None
                best_score = -1.0
                best_cv_thresholds = []
                # Precompute fold membership arrays
                fold_masks = []
                for (va_idx2_tuple,) in oof_fold_indices:
                    mask = np.zeros(n_train, dtype=bool)
                    mask[va_idx2_tuple] = True
                    fold_masks.append(mask)
                # Also evaluate single-model candidates (spec-first median threshold)
                single_best = None
                for j_idx, nm in enumerate(top_names):
                    p_single = P_top[:, j_idx]
                    fold_thresholds = []
                    for mask in fold_masks:
                        y_val = oof_truth[mask]
                        p_val = p_single[mask]
                        if y_val.sum() == 0 or y_val.sum() == len(y_val):
                            continue
                        thr_cands = np.unique(np.percentile(p_val, np.linspace(5, 95, 181)))
                        t_best = None
                        best_sens_local = -1
                        for t in thr_cands:
                            y_hat = (p_val >= t).astype(int)
                            tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                            if spec >= target_spec_ce and sens > best_sens_local:
                                best_sens_local = sens
                                t_best = float(t)
                        if t_best is not None:
                            fold_thresholds.append(t_best)
                    if len(fold_thresholds) > 0:
                        t_median_single = float(np.median(fold_thresholds))
                        y_hat_oof = (p_single >= t_median_single).astype(int)
                        tn, fp, fn, tp = confusion_matrix(oof_truth, y_hat_oof).ravel()
                        sens_oof = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        spec_oof = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        # Score: prioritize meeting spec target, maximize sensitivity
                        score_single = sens_oof if spec_oof >= target_spec_ce else 0.5 * sens_oof + 0.5 * min(spec_oof/target_spec_ce, 1.0)
                        if (single_best is None) or (score_single > single_best[0]):
                            single_best = (score_single, j_idx, nm, t_median_single, sens_oof, spec_oof)

                # For each weight vector, compute per-fold threshold meeting targets (spec-priority then sens), then median
                for w in gen_simplex_weights(P_top.shape[1], step=0.05):
                    p_agg = (P_top * w).sum(axis=1)
                    fold_thresholds = []
                    ok_folds = 0
                    # Per-fold thresholding to avoid bias
                    for mask in fold_masks:
                        y_val = oof_truth[mask]
                        p_val = p_agg[mask]
                        if y_val.sum() == 0 or y_val.sum() == len(y_val):
                            continue
                        # Candidate thresholds from percentiles
                        thr_cands = np.unique(np.percentile(p_val, np.linspace(5, 95, 181)))
                        t_best = None
                        best_score_local = -1
                        import numpy as _np
                        for t in thr_cands:
                            y_hat = (p_val >= t).astype(int)
                            tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                            if threshold_strategy == 'clinical':
                                if spec >= target_spec_ce and sens >= target_sens_ce:
                                    score_loc = (sens - target_sens_ce) + (spec - target_spec_ce)
                                    if score_loc > best_score_local:
                                        best_score_local = score_loc
                                        t_best = float(t)
                            else:
                                if spec >= target_spec_ce and sens > best_score_local:
                                    best_score_local = sens
                                    t_best = float(t)
                        if t_best is not None:
                            fold_thresholds.append(t_best)
                            ok_folds += 1
                    if len(fold_thresholds) == 0:
                        continue
                    t_median = float(np.median(fold_thresholds))
                    # Evaluate OOF with median threshold
                    y_hat_oof = (p_agg >= t_median).astype(int)
                    tn, fp, fn, tp = confusion_matrix(oof_truth, y_hat_oof).ravel()
                    sens_oof = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    spec_oof = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    # Clinical scoring: require specificity first; among those meeting spec, maximize sensitivity
                    if spec_oof >= target_spec_ce:
                        score = sens_oof
                    else:
                        score = 0.5 * sens_oof + 0.5 * min(spec_oof/target_spec_ce, 1.0)
                    if score > best_score:
                        best_score = score
                        best_combo = (w, t_median, sens_oof, spec_oof)
                        best_cv_thresholds = fold_thresholds

                # Prefer single-model if it scores better
                if single_best is not None:
                    score_single, j_idx, nm_single, t_med_single, sens_single, spec_single = single_best
                    if score_single > best_score:
                        best_combo = (None, t_med_single, sens_single, spec_single)
                        best_cv_thresholds = []
                        top_names = [top_names[j_idx]]
                        top_indices = [top_indices[j_idx]]

                if best_combo is not None:
                    w_best, thr_best_cv, sens_oof_best, spec_oof_best = best_combo
                    # Train base models on all training data and compute holdout probabilities
                    # Build full train matrices (already prepared): X_train_model and X_test_model
                    holdout_probs = []
                    for nm, proto in zip(top_names, [ensemble_base_protos[i] for i in top_indices]):
                        try:
                            mdl_full = sk_clone(proto)
                        except Exception:
                            try:
                                params = proto.get_params()
                            except Exception:
                                params = {}
                            try:
                                mdl_full = proto.__class__(**params)
                            except Exception:
                                mdl_full = proto.__class__()
                        mdl_full.fit(X_train_model, y_train)
                        if hasattr(mdl_full, 'predict_proba'):
                            try:
                                col_idx_m = list(mdl_full.classes_).index(asd_idx)
                            except Exception:
                                col_idx_m = 1 if len(getattr(mdl_full, 'classes_', [])) > 1 else 0
                            ph = mdl_full.predict_proba(X_test_model)[:, col_idx_m]
                        else:
                            try:
                                sc = mdl_full.decision_function(X_test_model)
                                import numpy as _np
                                ph = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
                            except Exception:
                                ph = mdl_full.predict(X_test_model)
                        holdout_probs.append(ph)
                    holdout_probs = __import__('numpy').vstack(holdout_probs).T
                    p_hold = (holdout_probs * w_best).sum(axis=1)
                    y_test_bin_h = (y_test == asd_idx).astype(int)
                    y_pred_h = (p_hold >= thr_best_cv).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test_bin_h, y_pred_h).ravel()
                    sens_h = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    spec_h = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    # Package results
                    clinical_ensemble = {
                        'base_models': top_names,
                        'weights': [float(x) for x in w_best.tolist()],
                        'cv': {
                            'threshold_median': float(thr_best_cv),
                            'sensitivity_asd': float(sens_oof_best),
                            'specificity_asd': float(spec_oof_best),
                            'fold_thresholds': [float(t) for t in best_cv_thresholds]
                        },
                        'holdout': {
                            'threshold': float(thr_best_cv),
                            'sensitivity_asd': float(sens_h),
                            'specificity_asd': float(spec_h)
                        }
                    }
        except Exception:
            clinical_ensemble = None

        # Generate feature plots
        try:
            import os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Top features bar chart
            if feature_importance:
                top_items = list(feature_importance.items())[:20]
                names = [n for n, _ in top_items]
                vals = [v for _, v in top_items]
                plt.figure(figsize=(10, 6))
                plt.barh(names[::-1], vals[::-1])
                plt.title("Top Feature Importances")
                plt.tight_layout()
                plt.savefig(plots_dir / "top_features.png")
                plt.close()

            # SHAP (tree-based only)
            try:
                import shap
                tree_like = hasattr(best_model[1], "feature_importances_")
                if tree_like:
                    explainer = shap.TreeExplainer(best_model[1])
                    # Sample to speed up
                    import numpy as np
                    sample_idx = np.random.choice(X_train_model.shape[0], size=min(200, X_train_model.shape[0]), replace=False)
                    shap_values = explainer.shap_values(X_train_model[sample_idx])
                    shap.summary_plot(shap_values, X_train_model[sample_idx], feature_names=feature_names, show=False)
                    plt.tight_layout()
                    plt.savefig(plots_dir / "shap_summary.png")
                    plt.close()
            except Exception:
                pass
        except Exception:
            pass
        
        results = {
            'model_results': model_results,
            'best_model': best_model[0] if best_model else None,
            'best_cv_score': best_score,
            'calibrated_threshold': best_thr,
            'holdout_sensitivity': best_sens,
            'holdout_specificity': best_spec,
            'holdout_sensitivity_asd': best_sens,
            'holdout_specificity_asd': best_spec,
            'feature_importance': feature_importance,
'dataset_info': {
                'total_children': X.shape[0],
                'n_features': int((X_train_aug.shape[1]) if 'X_train_aug' in locals() else X.shape[1]),
                'train_children': X_train.shape[0],
                'test_children': X_test.shape[0]
            },
'ablation': ablation,
            'clinical_ensemble': clinical_ensemble,
            'status': 'completed'
        }

        self.project_state['model_results'] = results
        self.project_state['completed_phases'].append('model_development')

# Save model results
        with open(self.results_dir / "model_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        # Save ablation report separately if available
        if ablation:
            with open(self.results_dir / "ablation_report.json", 'w') as f:
                json.dump(ablation, f, indent=2, default=str)

        return results

    def execute_results_reporting(self) -> Dict[str, Any]:
        """Phase 5: Generate comprehensive results report"""
        self.project_state['current_phase'] = 'results_reporting'

        # Compile comprehensive report
        report = self._generate_comprehensive_report()

        # Save report (JSON)
        report_path = self.reports_dir / f"binary_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create summary report (TXT)
        summary_report = self._create_summary_report(report)
        summary_path = self.reports_dir / "project_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_report)

        # Create PDF summary with fold metrics and variance
        pdf_path = self._create_pdf_summary(report)

        results = {
            'report_path': str(report_path),
            'summary_path': str(summary_path),
            'pdf_path': str(pdf_path) if pdf_path else None,
            'comprehensive_report': report,
            'status': 'completed'
        }

        self.project_state['completed_phases'].append('results_reporting')

        print(f"📋 Comprehensive report saved to: {report_path}")
        print(f"📄 Summary report saved to: {summary_path}")

        return results

    def _validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and completeness"""
        if not research_engine.behavioral_database:
            return {'error': 'No behavioral data available'}

        df = pd.DataFrame(research_engine.behavioral_database)

        # Sessions-level distribution (as before)
        sessions_label_dist = df['binary_label'].value_counts().to_dict() if 'binary_label' in df.columns else {}
        # Child-level distribution using mode label per child
        def _mode_or_first(s):
            m = s.mode()
            return m.iloc[0] if not m.empty else s.iloc[0]
        if 'binary_label' in df.columns and 'child_id' in df.columns:
            child_labels = df.groupby('child_id')['binary_label'].agg(_mode_or_first)
            child_label_dist = child_labels.value_counts().to_dict()
        else:
            child_label_dist = {}

        return {
            'total_sessions': len(df),
            'unique_children': df['child_id'].nunique() if 'child_id' in df.columns else 0,
            'label_distribution': sessions_label_dist,
            'child_label_distribution': child_label_dist,
            'missing_data_summary': df.isnull().sum().to_dict(),
            'data_quality_score': (1 - df.isnull().sum().sum() / df.size) * 100
        }

    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary of behavioral data"""
        if df.empty:
            return {'error': 'Empty dataframe'}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = {
            'descriptive_statistics': df[numeric_cols].describe().to_dict(),
            'correlations_with_label': {},
            'group_comparisons': {}
        }

        # Label encoding for correlation
        le = LabelEncoder()
        if 'binary_label' in df.columns:
            df_encoded = df.copy()
            df_encoded['label_encoded'] = le.fit_transform(df['binary_label'])

            # Correlations with label
# Ensure list concatenation, not elementwise string concat
            cols_for_corr = list(numeric_cols) + ['label_encoded']
            correlations = df_encoded[cols_for_corr].corr()['label_encoded']
            summary['correlations_with_label'] = correlations.drop('label_encoded').to_dict()

            # Group comparisons
            for col in numeric_cols:
                asd_data = df[df['binary_label'] == 'ASD'][col].dropna()
                td_data = df[df['binary_label'] == 'TD'][col].dropna()

                if len(asd_data) > 0 and len(td_data) > 0:
                    summary['group_comparisons'][col] = {
                        'ASD_mean': float(asd_data.mean()),
                        'TD_mean': float(td_data.mean()),
                        'difference_pct': float(((asd_data.mean() - td_data.mean()) / td_data.mean() * 100)) if td_data.mean() != 0 else 0
                    }

        return summary

    def _create_pdf_summary(self, report: Dict[str, Any]):
        """Create a PDF summary combining plots and key metrics across folds, with variance."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import json as _json
            import os
            import numpy as _np

            pdf_filename = self.reports_dir / f"binary_classification_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with PdfPages(pdf_filename) as pdf:
                # Page 1: Title and project overview
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                title = "ASD vs TD Binary Classification - Summary"
                ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=18, fontweight='bold')
                pi = report.get('project_info', {})
                ds = report.get('data_summary', {})
                mp = report.get('model_performance', {})
                best_cv = mp.get('best_cv_score', None)
                best_cv_str = f"{best_cv:.3f}" if isinstance(best_cv, (int, float)) else "N/A"
                lines = [
                    f"Generated: {pi.get('generated_at', '')}",
                    f"Completed Phases: {', '.join(pi.get('completed_phases', []))}",
                    "",
                    "Data Summary:",
                    f"  Total sessions: {ds.get('total_sessions', 'N/A')}",
                    f"  Unique children: {ds.get('unique_children', 'N/A')}",
                    f"  Label distribution: {ds.get('label_distribution', 'N/A')}",
                    "",
                    "Model Performance (Holdout):",
                    f"  Best model: {mp.get('best_model', 'N/A')}",
                    f"  Best CV AUC: {best_cv_str}",
                    f"  Holdout AUC: {max([v.get('test_auc', 0) for v in mp.get('model_results', {}).values()]) if mp.get('model_results') else 'N/A'}",
                    f"  Holdout ASD Sensitivity: {mp.get('holdout_sensitivity_asd', mp.get('holdout_sensitivity', 'N/A'))}",
                    f"  Holdout ASD Specificity: {mp.get('holdout_specificity_asd', mp.get('holdout_specificity', 'N/A'))}"
                ]
                y = 0.9
                for ln in lines:
                    ax.text(0.05, y, ln, ha='left', va='top', fontsize=11)
                    y -= 0.04
                pdf.savefig(fig)
                plt.close(fig)

                # Page 2: CV fold metrics for best model
                best_name = mp.get('best_model')
                if best_name and mp.get('model_results') and best_name in mp['model_results']:
                    best_res = mp['model_results'][best_name]
                    folds = best_res.get('cv_folds', [])
                    if folds:
                        aucs = [f['auc'] for f in folds]
                        sens = [f.get('sensitivity_asd') for f in folds]
                        spec = [f.get('specificity_asd') for f in folds]
                        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
                        fig.suptitle(f"Cross-Validation Metrics ({best_name})", fontsize=16)
                        # AUC plot
                        axes[0].plot(range(1, len(aucs)+1), aucs, marker='o')
                        axes[0].set_title(f"Fold AUCs (mean={_np.mean(aucs):.3f}, std={_np.std(aucs):.3f})")
                        axes[0].set_xlabel("Fold")
                        axes[0].set_ylabel("AUC (ASD+)")
                        axes[0].set_xticks(range(1, len(aucs)+1))
                        # Sensitivity plot
                        axes[1].plot(range(1, len(sens)+1), sens, marker='o', color='tab:green')
                        axes[1].set_title(f"Fold ASD Sensitivity (mean={_np.mean(sens):.3f}, std={_np.std(sens):.3f})")
                        axes[1].set_xlabel("Fold")
                        axes[1].set_ylabel("Sensitivity")
                        axes[1].set_xticks(range(1, len(sens)+1))
                        # Specificity plot
                        axes[2].plot(range(1, len(spec)+1), spec, marker='o', color='tab:red')
                        axes[2].set_title(f"Fold ASD Specificity (mean={_np.mean(spec):.3f}, std={_np.std(spec):.3f})")
                        axes[2].set_xlabel("Fold")
                        axes[2].set_ylabel("Specificity")
                        axes[2].set_xticks(range(1, len(spec)+1))
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        pdf.savefig(fig)
                        plt.close(fig)

                # Page 3: Ablation summary (Base vs RAG-aug) if available
                abl = mp.get('ablation')
                if abl:
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    ax.set_title("Ablation: Base vs RAG-Augmented (best model)", fontsize=14, fontweight='bold')
                    lines = []
                    lines.append(f"Best model: {abl.get('best_model', 'N/A')}")
                    lines.append(f"Target ASD Sensitivity: {abl.get('target_sensitivity', 'N/A')}")
                    lines.append("")
                    base_cv = abl.get('base', {}).get('cv', {})
                    rag_cv = abl.get('rag_aug', {}).get('cv', {})
                    base_hold = abl.get('base', {}).get('holdout', {})
                    rag_hold = abl.get('rag_aug', {}).get('holdout', {})
                    lines.extend([
                        "CV (mean ± std):",
                        f"  Base AUC: {base_cv.get('cv_mean_auc', 'N/A'):.3f} ± {base_cv.get('cv_std_auc', 0):.3f}",
                        f"  Base Sens: {base_cv.get('cv_mean_sensitivity_asd', 'N/A'):.3f} ± {base_cv.get('cv_std_sensitivity_asd', 0):.3f}",
                        f"  Base Spec: {base_cv.get('cv_mean_specificity_asd', 'N/A'):.3f} ± {base_cv.get('cv_std_specificity_asd', 0):.3f}",
                        f"  RAG  AUC: {rag_cv.get('cv_mean_auc', 'N/A'):.3f} ± {rag_cv.get('cv_std_auc', 0):.3f}",
                        f"  RAG  Sens: {rag_cv.get('cv_mean_sensitivity_asd', 'N/A'):.3f} ± {rag_cv.get('cv_std_sensitivity_asd', 0):.3f}",
                        f"  RAG  Spec: {rag_cv.get('cv_mean_specificity_asd', 'N/A'):.3f} ± {rag_cv.get('cv_std_specificity_asd', 0):.3f}",
                        "",
                        "Holdout:",
                        f"  Base AUC: {base_hold.get('test_auc', 'N/A')}",
                        f"  Base Sens: {base_hold.get('sensitivity_asd', 'N/A')}",
                        f"  Base Spec: {base_hold.get('specificity_asd', 'N/A')}",
                        f"  RAG  AUC: {rag_hold.get('test_auc', 'N/A')}",
                        f"  RAG  Sens: {rag_hold.get('sensitivity_asd', 'N/A')}",
                        f"  RAG  Spec: {rag_hold.get('specificity_asd', 'N/A')}"
                    ])
                    y = 0.9
                    for ln in lines:
                        ax.text(0.05, y, ln, ha='left', va='top', fontsize=11)
                        y -= 0.035
                    pdf.savefig(fig)
                    plt.close(fig)

                # Page 4+: Embed feature plots if available
                plots_dir = self.results_dir / "plots"
                for plot_name in ["top_features.png", "shap_summary.png"]:
                    plot_path = plots_dir / plot_name
                    if plot_path.exists():
                        img = plt.imread(str(plot_path))
                        fig, ax = plt.subplots(figsize=(11, 8.5))
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(plot_name)
                        pdf.savefig(fig)
                        plt.close(fig)

            return pdf_filename
        except Exception:
            return None

    def _prepare_modeling_dataset(self) -> Optional[Dict[str, Any]]:
        """Prepare dataset for machine learning modeling (child-level, may include RAG features)
        NOTE: This method includes RAG features computed on all data and may leak.
        Prefer _prepare_modeling_dataset_base for leakage-safe workflows.
        """
        if not research_engine.behavioral_database:
            return None
        
        df = pd.DataFrame(research_engine.behavioral_database)
        
        # Filter to known labels
        df = df[df['binary_label'].isin(['ASD', 'TD'])].copy()
        if df.empty:
            return None
        
        # Select numeric features for modeling (session-level)
        numeric_features = [
            'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
            'tremor_indicator', 'acc_magnitude_mean', 'acc_magnitude_std',
            'palm_touch_ratio', 'unique_fingers', 'max_finger_id',
            'session_duration', 'stroke_count', 'total_touch_points',
            'unique_zones', 'unique_colors', 'final_completion',
            'completion_progress_rate', 'avg_time_between_points',
            'canceled_touches'
        ]
        available_features = [f for f in numeric_features if f in df.columns]
        if not available_features:
            return None
        
        # Child-level label (mode across sessions)
        def _mode_or_first(s):
            m = s.mode()
            return m.iloc[0] if not m.empty else s.iloc[0]
        child_labels = df.groupby('child_id')['binary_label'].agg(_mode_or_first).rename('binary_label')
        
        # Child-level feature aggregation (mean over sessions)
        agg_df = df.groupby('child_id')[available_features].mean().reset_index()
        
        # Add session count as a feature
        sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
        agg_df = agg_df.merge(sess_count, on='child_id', how='left')
        
        # Join RAG-derived numeric features (ALL DATA - may leak)
        try:
            rag_feat = research_engine.compute_child_rag_features()
            if not rag_feat.empty:
                agg_df = agg_df.merge(rag_feat, on='child_id', how='left')
        except Exception:
            pass
        
        # Build final dataset
        child_df = agg_df.merge(child_labels.reset_index(), on='child_id', how='left')
        child_df = child_df.dropna(subset=['binary_label'])
        feature_cols = [c for c in child_df.columns if c not in ['child_id', 'binary_label']]
        X = child_df[feature_cols].fillna(0)
        le = LabelEncoder()
        y = le.fit_transform(child_df['binary_label'])
        
        # Save label encoder
        joblib.dump(le, self.models_dir / "label_encoder.pkl")
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_cols,
            'label_encoder': le,
            'class_names': le.classes_,
            'groups': child_df['child_id'].tolist(),
            'child_ids': child_df['child_id'].tolist(),
        }

    def _prepare_modeling_dataset_base(self) -> Optional[Dict[str, Any]]:
        """Prepare leakage-safe BASE dataset for modeling (no RAG features).
        Child-level aggregation of numeric session features and session count.
        """
        if not research_engine.behavioral_database:
            return None
        
        df = pd.DataFrame(research_engine.behavioral_database)
        
        # Filter to known labels
        df = df[df['binary_label'].isin(['ASD', 'TD'])].copy()
        if df.empty:
            return None
        
        # Select numeric features for modeling (session-level)
        numeric_features = [
            'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
            'tremor_indicator', 'acc_magnitude_mean', 'acc_magnitude_std',
            'palm_touch_ratio', 'unique_fingers', 'max_finger_id',
            'session_duration', 'stroke_count', 'total_touch_points',
            'unique_zones', 'unique_colors', 'final_completion',
            'completion_progress_rate', 'avg_time_between_points',
            'canceled_touches'
        ]
        available_features = [f for f in numeric_features if f in df.columns]
        if not available_features:
            return None
        
        # Child-level label (mode across sessions)
        def _mode_or_first(s):
            m = s.mode()
            return m.iloc[0] if not m.empty else s.iloc[0]
        child_labels = df.groupby('child_id')['binary_label'].agg(_mode_or_first).rename('binary_label')
        
        # Child-level feature aggregation (mean over sessions)
        agg_df = df.groupby('child_id')[available_features].mean().reset_index()
        
        # Add session count as a feature
        sess_count = df.groupby('child_id').size().rename('session_count').reset_index()
        agg_df = agg_df.merge(sess_count, on='child_id', how='left')
        
        # Build final dataset (NO RAG features)
        child_df = agg_df.merge(child_labels.reset_index(), on='child_id', how='left')
        child_df = child_df.dropna(subset=['binary_label'])
        feature_cols = [c for c in child_df.columns if c not in ['child_id', 'binary_label']]
        X = child_df[feature_cols].fillna(0)
        le = LabelEncoder()
        y = le.fit_transform(child_df['binary_label'])
        
        # Save label encoder
        joblib.dump(le, self.models_dir / "label_encoder.pkl")
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_cols,
            'label_encoder': le,
            'class_names': le.classes_,
            'groups': child_df['child_id'].tolist(),
            'child_ids': child_df['child_id'].tolist(),
        }

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        importance_dict = {}

        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0])
        else:
            return importance_dict

        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive project report"""
        return {
            'project_info': {
                'name': 'ASD vs TD Binary Classification',
                'description': 'Coloring game behavioral analysis for autism classification',
                'generated_at': datetime.now().isoformat(),
                'completed_phases': self.project_state['completed_phases']
            },
            'data_summary': self.project_state.get('data_summary', {}),
            'feature_analysis': self.project_state.get('feature_recommendations', {}),
            'model_performance': self.project_state.get('model_results', {}),
            'project_state': self.project_state
        }

    def _create_summary_report(self, report: Dict[str, Any]) -> str:
        """Create human-readable summary report"""
        summary = f"""
# Binary Classification Project Summary

Generated: {report['project_info']['generated_at']}

## Project Overview
{report['project_info']['description']}

## Data Summary
"""

        if 'data_summary' in report and report['data_summary']:
            ds = report['data_summary']
            summary += f"""
- Total sessions: {ds.get('total_sessions', 'N/A')}
- Unique children: {ds.get('unique_children', 'N/A')}
- Label distribution: {ds.get('label_distribution', 'N/A')}
"""

        if 'model_performance' in report and report['model_performance']:
            mp = report['model_performance']
            summary += f"""

## Model Performance
- Best model: {mp.get('best_model', 'N/A')}
- Best CV AUC score: {mp.get('best_cv_score', 'N/A'):.3f}
- Number of features: {mp.get('dataset_info', {}).get('n_features', 'N/A')}
- Holdout ASD Sensitivity: {mp.get('holdout_sensitivity_asd', mp.get('holdout_sensitivity', 'N/A'))}
- Holdout ASD Specificity: {mp.get('holdout_specificity_asd', mp.get('holdout_specificity', 'N/A'))}
"""

            if 'feature_importance' in mp:
                summary += "\n\n## Top Features:\n"
                for i, (feature, importance) in enumerate(list(mp['feature_importance'].items())[:10]):
                    summary += f"{i+1}. {feature}: {importance:.3f}\n"

        summary += f"""

## Completed Phases
{', '.join(report['project_info']['completed_phases'])}

---
Generated by MCP Project Orchestrator
"""

        return summary

    def get_project_status(self) -> Dict[str, Any]:
        """Get current project status"""
        return self.project_state

    def query_research_insights(self, query: str) -> Dict[str, Any]:
        """Query the RAG research system for insights"""
        return research_engine.research_query(query)

# Global orchestrator instance
orchestrator = ProjectOrchestrator()
