#!/usr/bin/env python3
"""
Master Analysis Script for Clinical Performance Improvement
Runs comprehensive analysis to achieve clinical targets and reduce variance

Current Performance Gap:
- Target: 86% sensitivity, 75% specificity
- Current: ~82% sensitivity, ~87% specificity  
- Issue: 60-90% sensitivity range across CV folds (30% variance)

Usage: python analysis/run_analysis.py
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

def print_header():
    """Print analysis header"""
    print("\n🔬 Clinical ML Performance Analysis & Improvement")
    print("=" * 55)
    print("🎯 Current Targets: 86% sensitivity, 75% specificity")
    print("⚠️  Current Issue: High CV variance (60-90% sensitivity range)")
    print("🚀 Goal: Systematic performance improvement\n")

def load_labels_from_processed() -> Optional[pd.DataFrame]:
    """Load labels from data/processed/labels.csv (PRIORITY method)"""
    try:
        labels_file = Path("data/processed/labels.csv")
        if not labels_file.exists():
            print("   ⚠️ No processed labels file found")
            return None
            
        df_labels = pd.read_csv(labels_file)
        
        # Check for required columns
        if 'Unity_id' not in df_labels.columns or 'Group' not in df_labels.columns:
            print(f"   ⚠️ Missing required columns in {labels_file}")
            return None
        
        # Convert Group to target (ASD=1, TD=0)
        df_labels['target'] = (df_labels['Group'] == 'ASD').astype(int)
        
        # Rename Unity_id to child_id for consistency
        df_labels = df_labels.rename(columns={'Unity_id': 'child_id'})
        
        child_labels = df_labels[['child_id', 'target']].copy()
        
        print(f"   ✅ Loaded {len(child_labels)} REAL labels from processed/labels.csv")
        print(f"   📊 Label distribution: ASD={child_labels['target'].sum()}, TD={len(child_labels)-child_labels['target'].sum()}")
        
        return child_labels
        
    except Exception as e:
        print(f"   ⚠️ Error loading processed labels: {e}")
        return None

def load_labels_from_rag() -> Optional[pd.DataFrame]:
    """Try to load labels using the RAG system (FALLBACK method)"""
    try:
        print("🔄 Attempting to load labels via RAG system...")
        
        # Import the research engine
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from rag_system.research_engine import research_engine
        
        # Check if behavioral database exists
        if not research_engine.behavioral_database:
            print("   📥 RAG database not loaded, ingesting data...")
            research_engine.ingest_raw_data(limit=None)
            research_engine.index_behavioral_data()
        
        if not research_engine.behavioral_database:
            return None
            
        # Build dataframe from behavioral database
        df = pd.DataFrame(research_engine.behavioral_database)
        if df.empty:
            return None
            
        # Filter to labeled data and get labels per child
        df = df[df['binary_label'].isin(['ASD', 'TD'])].copy()
        if df.empty:
            return None
            
        # Get most common label per child (mode)
        def mode_or_first(s: pd.Series):
            m = s.mode()
            return m.iloc[0] if not m.empty else s.iloc[0]
        
        child_labels = df.groupby('child_id')['binary_label'].agg(mode_or_first).reset_index()
        child_labels['target'] = (child_labels['binary_label'] == 'ASD').astype(int)
        child_labels = child_labels[['child_id', 'target']]
        
        print(f"   ✅ Loaded labels for {len(child_labels)} children from RAG system")
        return child_labels
        
    except Exception as e:
        print(f"   ⚠️ RAG system failed: {e}")
        return None

def load_labels_from_sample() -> Optional[pd.DataFrame]:
    """Load labels from sample_clinical_data.csv as fallback"""
    try:
        sample_file = Path("sample_clinical_data.csv")
        if not sample_file.exists():
            return None
            
        df = pd.read_csv(sample_file)
        if 'target' not in df.columns or 'child_id' not in df.columns:
            return None
            
        # Get unique child-label pairs
        child_labels = df[['child_id', 'target']].drop_duplicates()
        print(f"   ✅ Loaded labels for {len(child_labels)} children from sample_clinical_data.csv")
        return child_labels
        
    except Exception as e:
        print(f"   ⚠️ Sample data loading failed: {e}")
        return None

def create_synthetic_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic labels for analysis when real labels aren't available"""
    print("   ⚠️ Creating synthetic labels for analysis purposes...")
    
    # Use a feature-based heuristic to create realistic synthetic labels
    # This is ONLY for analysis - not for training!
    child_labels = features_df[['child_id']].copy()
    
    # Create labels based on velocity patterns (ASD children often have different motor patterns)
    # This is a simplified heuristic for analysis purposes only
    velocity_feature = features_df.get('velocity_cv', features_df.get('vel_std_over_mean', 0))
    tremor_feature = features_df.get('tremor_indicator', 0)
    
    # Simple heuristic: higher velocity variation + tremor indicates ASD
    # This is SYNTHETIC and only for testing the analysis pipeline!
    if hasattr(velocity_feature, 'quantile'):
        high_var_threshold = velocity_feature.quantile(0.6)
        synthetic_labels = ((velocity_feature > high_var_threshold) | (tremor_feature > tremor_feature.quantile(0.7))).astype(int)
    else:
        # Fallback: alternate labels
        synthetic_labels = (features_df.index % 2).astype(int)
    
    child_labels['target'] = synthetic_labels
    
    print(f"   🧪 Created synthetic labels: {synthetic_labels.sum()} ASD, {len(synthetic_labels) - synthetic_labels.sum()} TD")
    print(f"   ⚠️ WARNING: These are SYNTHETIC labels for analysis only!")
    
    return child_labels

def load_features_with_labels() -> Tuple[Optional[pd.DataFrame], bool]:
    """Load features and merge with labels from available sources"""
    
    # PRIORITY: Look for fileKeys_features_aligned.csv first (correct Unity_id format)
    feature_files = []
    
    # Check for the correct file with Unity_id format first
    correct_file = Path("results/fileKeys_features_aligned.csv")
    if correct_file.exists():
        feature_files.append(correct_file)
        print(f"🎯 Found CORRECT features file: {correct_file.name} (Unity_id format)")
    
    # Then check for any other features files as fallback
    other_files = list(Path("results").glob("*features_aligned.csv"))
    for file in other_files:
        if file not in feature_files:
            feature_files.append(file)
    
    if not feature_files:
        print("❌ No processed feature files found in results/")
        return None, False
    
    # Use the first (priority) file
    feature_file = feature_files[0]
    print(f"📊 Loading features from: {feature_file.name}")
    
    try:
        df_features = pd.read_csv(feature_file)
        print(f"   Shape: {df_features.shape[0]} samples, {df_features.shape[1]} features")
        
        # Check if labels already exist
        if 'target' in df_features.columns or 'label' in df_features.columns:
            target_col = 'target' if 'target' in df_features.columns else 'label'
            print(f"   ✅ Found existing target column: {target_col}")
            if target_col != 'target':
                df_features = df_features.rename(columns={target_col: 'target'})
            return df_features, True
        
        # Need to merge with labels
        print("   ⚠️ No target column found in features, attempting to merge with labels...")
        
        if 'child_id' not in df_features.columns:
            print("   ❌ No child_id column found for merging")
            return None, False
        
        # PRIORITY: Try processed labels first
        print("🎯 Trying REAL labels from processed/labels.csv first...")
        child_labels = load_labels_from_processed()
        
        # FALLBACK: Try RAG system
        if child_labels is None:
            print("   🔄 Trying RAG system as fallback...")
            child_labels = load_labels_from_rag()
        
        # FALLBACK: Try sample data
        if child_labels is None:
            print("   🔄 Trying sample data as fallback...")
            child_labels = load_labels_from_sample()
        
        # LAST RESORT: Synthetic labels
        if child_labels is None:
            print("   ⚠️ No real labels available, creating synthetic labels for analysis...")
            child_labels = create_synthetic_labels(df_features)
        
        # Merge features with labels
        print(f"   🔄 Merging {len(df_features)} feature rows with {len(child_labels)} labeled children...")
        
        # Check for child ID format compatibility
        features_ids = set(df_features['child_id'].astype(str))
        label_ids = set(child_labels['child_id'].astype(str))
        overlap = features_ids.intersection(label_ids)
        
        print(f"   🔍 ID format check: {len(overlap)} overlapping IDs")
        
        if len(overlap) == 0:
            print(f"   ⚠️ No matching child IDs - ID format mismatch detected")
            print(f"   Features ID example: {list(features_ids)[0] if features_ids else 'None'}")
            print(f"   Labels ID example: {list(label_ids)[0] if label_ids else 'None'}")
            print(f"   ⚠️ Using synthetic labels for analysis (FEATURE PROCESSING may use different raw data)")
            
            # Use synthetic labels based on features
            child_labels = create_synthetic_labels(df_features)
        
        df_merged = df_features.merge(child_labels, on='child_id', how='inner')
        
        if df_merged.empty:
            print("   ❌ Merge failed - using features with synthetic labels")
            df_merged = df_features.copy()
            synthetic_labels = create_synthetic_labels(df_features)
            df_merged = df_merged.merge(synthetic_labels, on='child_id', how='left')
        
        # Show what type of labels we're using
        if len(overlap) > 0:
            print(f"   ✅ SUCCESS: Using REAL labels for {len(df_merged)} samples!")
        else:
            print(f"   ⚠️ Using SYNTHETIC labels for {len(df_merged)} samples (analysis only)")
        
        return df_merged, True
        
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return None, False

def check_requirements() -> bool:
    """Check if analysis requirements are met"""
    
    print("🔍 Checking analysis requirements...")
    
    # Try to load features with labels
    df, has_labels = load_features_with_labels()
    
    if df is None:
        print("💡 Run the main pipeline first: ./train_final.sh")
        return False
    
    if not has_labels:
        return False
    
    # Check training results
    result_files = list(Path("results").glob("final_s*_ho777_*.json"))
    if not result_files:
        print("⚠️ No training results found - some analyses will be limited")
    else:
        print(f"✅ Found {len(result_files)} training result files")
    
    # Check target distribution
    target_col = 'target'
    class_dist = df[target_col].value_counts().to_dict()
    print(f"🎯 Target distribution: {class_dist}")
    
    # Calculate class balance
    pos_rate = df[target_col].mean()
    print(f"   ASD rate: {pos_rate:.1%} ({int(pos_rate * len(df))} of {len(df)})")
    
    if pos_rate < 0.1 or pos_rate > 0.9:
        print("   ⚠️ Highly imbalanced dataset - analysis may be affected")
    
    print("✅ Requirements check passed\n")
    return True

def run_analysis_tool(tool_name: str, description: str) -> bool:
    """Run an individual analysis tool"""
    
    tool_path = Path("analysis") / f"{tool_name}.py"
    
    if not tool_path.exists():
        print(f"❌ Tool not found: {tool_path}")
        return False
    
    print(f"🔄 Running {description}...")
    print(f"   Command: python {tool_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(tool_path)], 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            
            # Show key output lines
            output_lines = result.stdout.strip().split('\n')
            key_lines = [line for line in output_lines 
                        if any(symbol in line for symbol in ['✅', '❌', '📊', '🎯', '🚨'])]
            
            if key_lines:
                print("   Key results:")
                for line in key_lines[-3:]:  # Show last 3 key lines
                    print(f"   {line}")
            
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def analyze_current_performance() -> Dict:
    """Quick analysis of current model performance"""
    
    print("📊 Analyzing current performance gaps...")
    
    # Load recent training results
    result_files = list(Path("results").glob("final_s*_ho777_*.json"))
    
    if not result_files:
        print("   ⚠️ No training results available")
        return {}
    
    performances = []
    
    for file_path in sorted(result_files)[-3:]:  # Last 3 results
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract holdout performance
            holdout = data.get('holdout', {})
            if holdout:
                performances.append({
                    'file': file_path.name,
                    'sensitivity': holdout.get('sensitivity', 0),
                    'specificity': holdout.get('specificity', 0),
                    'auc': holdout.get('auc', 0)
                })
        
        except Exception:
            continue
    
    if not performances:
        print("   ⚠️ Could not extract performance metrics")
        return {}
    
    # Calculate averages
    avg_sens = sum(p['sensitivity'] for p in performances) / len(performances)
    avg_spec = sum(p['specificity'] for p in performances) / len(performances)
    avg_auc = sum(p['auc'] for p in performances) / len(performances)
    
    # Calculate gaps
    sens_gap = 0.86 - avg_sens
    spec_gap = max(0, 0.75 - avg_spec)  # Only if below target
    
    performance_summary = {
        'current_sensitivity': avg_sens,
        'current_specificity': avg_spec,
        'current_auc': avg_auc,
        'sensitivity_gap': sens_gap,
        'specificity_gap': spec_gap,
        'meets_targets': sens_gap <= 0 and spec_gap <= 0
    }
    
    print(f"   Current performance:")
    print(f"     Sensitivity: {avg_sens:.3f} (target: 0.860, gap: {sens_gap:+.3f})")
    print(f"     Specificity: {avg_spec:.3f} (target: 0.750, gap: {spec_gap:+.3f})")
    print(f"     AUC: {avg_auc:.3f}")
    
    if sens_gap > 0:
        print(f"   🚨 PRIORITY: Need {sens_gap:.3f} sensitivity improvement")
    
    if spec_gap > 0:
        print(f"   ⚠️ Need {spec_gap:.3f} specificity improvement")
    
    if performance_summary['meets_targets']:
        print(f"   ✅ Clinical targets achieved!")
    
    return performance_summary

def generate_improvement_plan(performance_summary: Dict) -> List[str]:
    """Generate specific improvement recommendations"""
    
    print("\n💡 Generating improvement plan...")
    
    plan = []
    
    # Performance-based recommendations
    if performance_summary:
        if performance_summary['sensitivity_gap'] > 0.05:
            plan.extend([
                "🚨 CRITICAL: Sensitivity improvement needed",
                "   - Lower classification thresholds",
                "   - Cost-sensitive training (higher FN penalty)",
                "   - Analyze false negative cases"
            ])
        
        if performance_summary['specificity_gap'] > 0.05:
            plan.extend([
                "⚠️ Specificity improvement needed", 
                "   - Feature selection to reduce overfitting",
                "   - Stronger regularization"
            ])
    
    # Variance reduction (always needed based on CV analysis)
    plan.extend([
        "📉 CRITICAL: Reduce CV variance (60-90% range)",
        "   - Current fold variance affects clinical reliability",
        "   - Implement feature selection for stability",
        "   - Try repeated cross-validation",
        "   - Consider ensemble methods"
    ])
    
    # Data quality
    if Path("data/raw").exists():
        plan.append("🔍 Data quality analysis recommended")
    
    return plan

def run_comprehensive_analysis() -> Dict:
    """Run all analysis tools in sequence"""
    
    print("🚀 Running comprehensive performance analysis...\n")
    
    analysis_tools = [
        ("data_quality_analysis", "Data Quality Analysis"),
        ("feature_analysis", "Feature Quality Analysis"), 
        ("cv_variance_analysis", "CV Variance Analysis"),
        ("feature_selection", "Robust Feature Selection")
    ]
    
    results = {}
    
    for tool_name, description in analysis_tools:
        print(f"\n{'='*60}")
        success = run_analysis_tool(tool_name, description)
        results[tool_name] = success
        
        if not success:
            print(f"⚠️ Continuing with remaining analyses...")
    
    return results

def summarize_results(analysis_results: Dict, performance_summary: Dict) -> None:
    """Summarize all analysis results"""
    
    print(f"\n{'='*60}")
    print("📊 ANALYSIS SUMMARY & NEXT STEPS")
    print(f"{'='*60}")
    
    # Tool execution summary
    print("\n🔧 Analysis Tools Executed:")
    for tool, success in analysis_results.items():
        status = "✅" if success else "❌"
        tool_display = tool.replace('_', ' ').title()
        print(f"   {status} {tool_display}")
    
    # Performance gaps
    if performance_summary:
        print("\n🎯 Performance Gaps:")
        sens_gap = performance_summary['sensitivity_gap']
        spec_gap = performance_summary['specificity_gap']
        
        if sens_gap > 0:
            print(f"   🚨 Sensitivity: {sens_gap:.3f} gap to reach 86%")
        else:
            print(f"   ✅ Sensitivity: Target achieved")
            
        if spec_gap > 0:
            print(f"   ⚠️ Specificity: {spec_gap:.3f} gap to reach 75%")
        else:
            print(f"   ✅ Specificity: Target achieved")
    
    # Files created
    output_dir = Path("analysis")
    if output_dir.exists():
        output_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.json"))
        if output_files:
            print(f"\n📄 Analysis Results Generated:")
            for file_path in sorted(output_files):
                print(f"   {file_path}")
    
    # Action plan
    print(f"\n🚀 RECOMMENDED ACTION SEQUENCE:")
    action_plan = [
        "1. Review feature_selection.py results",
        "2. Retrain using recommended features:",
        "   ./train_final.sh --features analysis/recommended_features.csv", 
        "3. Compare new CV variance vs current 60-90% range",
        "4. Optimize thresholds for 86% sensitivity target",
        "5. Validate on independent test set"
    ]
    
    for action in action_plan:
        print(f"   {action}")
    
    print(f"\n✅ Analysis complete! Ready for systematic improvement.")

def main():
    """Main analysis workflow"""
    
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot proceed with analysis")
        return 1
    
    # Analyze current performance
    performance_summary = analyze_current_performance()
    
    # Generate improvement plan  
    improvement_plan = generate_improvement_plan(performance_summary)
    
    print("\n📊 Identified Issues to Address:")
    for item in improvement_plan:
        print(f"   {item}")
    
    # Ask user if they want to run full analysis
    print(f"\n🚀 Ready to run comprehensive analysis?")
    print("   This will execute all analysis tools to diagnose and fix issues.")
    
    try:
        response = input("   Continue? [Y/n]: ").strip().lower()
        if response and response != 'y' and response != 'yes':
            print("\n👍 Analysis skipped. Run individual tools as needed:")
            print("   python analysis/debug_columns.py")
            print("   python analysis/data_quality_analysis.py")
            print("   python analysis/feature_analysis.py")
            print("   python analysis/cv_variance_analysis.py")
            print("   python analysis/feature_selection.py")
            return 0
    except KeyboardInterrupt:
        print("\n\n👍 Analysis cancelled.")
        return 0
    
    # Run comprehensive analysis
    analysis_results = run_comprehensive_analysis()
    
    # Summarize everything
    summarize_results(analysis_results, performance_summary)
    
    return 0


if __name__ == "__main__":
    exit(main())