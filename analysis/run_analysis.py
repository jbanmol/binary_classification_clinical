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
from typing import Dict, List

def print_header():
    """Print analysis header"""
    print("\n🔬 Clinical ML Performance Analysis & Improvement")
    print("=" * 55)
    print("🎯 Current Targets: 86% sensitivity, 75% specificity")
    print("⚠️  Current Issue: High CV variance (60-90% sensitivity range)")
    print("🚀 Goal: Systematic performance improvement\n")

def check_requirements() -> bool:
    """Check if analysis requirements are met"""
    
    print("🔍 Checking analysis requirements...")
    
    # Check for processed features
    feature_files = list(Path("results").glob("*features_aligned.csv"))
    if not feature_files:
        print("❌ No processed feature files found in results/")
        print("💡 Run the main pipeline first: ./train_final.sh")
        return False
    
    feature_file = sorted(feature_files)[-1]
    print(f"✅ Found features: {feature_file.name}")
    
    # Check for training results
    result_files = list(Path("results").glob("final_s*_ho777_*.json"))
    if not result_files:
        print("⚠️ No training results found - some analyses will be limited")
    else:
        print(f"✅ Found {len(result_files)} training result files")
    
    # Check data structure
    try:
        df = pd.read_csv(feature_file)
        print(f"📊 Dataset: {df.shape[0]} samples, {df.shape[1]} columns")
        
        if 'label' not in df.columns and 'target' not in df.columns:
            print("❌ No target column found")
            return False
        
        target_col = 'label' if 'label' in df.columns else 'target'
        class_dist = df[target_col].value_counts().to_dict()
        print(f"🎯 Target distribution: {class_dist}")
        
    except Exception as e:
        print(f"❌ Error reading features: {e}")
        return False
    
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
