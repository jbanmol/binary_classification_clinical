#!/usr/bin/env python3
"""
Data Quality Analysis for Clinical ML Performance Improvement
Analyzes session quality, data completeness, and identifies outliers

Usage: python analysis/data_quality_analysis.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class DataQualityAnalyzer:
    """Analyze clinical coloring session data quality"""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        
    def analyze_sessions(self) -> pd.DataFrame:
        """Analyze session-level data quality metrics"""
        
        print("🔍 Analyzing session quality...")
        
        session_metrics = []
        json_files = list(self.data_path.rglob("*.json"))
        
        print(f"Found {len(json_files)} session files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    session = json.load(f)
                
                child_id = file_path.parent.name
                metrics = self._calculate_metrics(session, child_id)
                session_metrics.append(metrics)
                
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue
        
        if not session_metrics:
            print("❌ No valid sessions found")
            return pd.DataFrame()
        
        df = pd.DataFrame(session_metrics)
        df['quality_score'] = self._quality_score(df)
        df['is_outlier'] = self._detect_outliers(df)
        
        print(f"✅ Analyzed {len(df)} sessions")
        print(f"Quality score: {df['quality_score'].mean():.1f} ± {df['quality_score'].std():.1f}")
        print(f"Outliers: {df['is_outlier'].sum()} ({df['is_outlier'].mean()*100:.1f}%)")
        
        return df
    
    def _calculate_metrics(self, session: dict, child_id: str) -> dict:
        """Calculate session quality metrics"""
        
        duration = session.get('duration', 0)
        touches = session.get('touch_points', [])
        zones = session.get('zones_visited', [])
        
        return {
            'child_id': child_id,
            'duration': duration,
            'n_touches': len(touches),
            'n_zones': len(set(zones)) if zones else 0,
            'touch_rate': len(touches) / duration if duration > 0 else 0,
            'zone_rate': len(set(zones)) / duration if duration > 0 and zones else 0,
            'completed': session.get('completion_status') == 'completed',
            'has_velocity': bool(session.get('velocity_data')),
            'data_complete': sum([
                bool(session.get('duration')),
                bool(session.get('touch_points')),
                bool(session.get('zones_visited'))
            ]) / 3
        }
    
    def _quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quality score 0-100"""
        
        score = pd.Series(0.0, index=df.index)
        
        # Duration score (25 points)
        score += np.clip(df['duration'] / 300 * 25, 0, 25)
        
        # Touch data score (25 points)
        score += np.clip(df['n_touches'] / 100 * 25, 0, 25)
        
        # Zone coverage (25 points)
        score += np.clip(df['n_zones'] / 10 * 25, 0, 25)
        
        # Completeness (25 points)
        score += df['data_complete'] * 25
        
        return score
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.Series:
        """Detect outlier sessions"""
        
        outliers = pd.Series(False, index=df.index)
        
        # Duration outliers
        outliers |= (df['duration'] < 10) | (df['duration'] > 600)
        
        # Touch outliers
        outliers |= df['n_touches'] < 5
        
        # Quality outliers
        outliers |= df['quality_score'] < 20
        
        return outliers


def main():
    """Run data quality analysis"""
    
    print("🔬 Clinical Data Quality Analysis")
    print("=" * 40)
    
    analyzer = DataQualityAnalyzer()
    
    if not analyzer.data_path.exists():
        print(f"❌ Data path not found: {analyzer.data_path}")
        print("💡 Make sure raw coloring session data is available")
        return
    
    # Run analysis
    results = analyzer.analyze_sessions()
    
    if results.empty:
        print("❌ No data to analyze")
        return
    
    # Save results
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    results.to_csv(output_dir / "session_quality_analysis.csv", index=False)
    
    # Summary report
    print("\n📊 Quality Summary:")
    print(f"  Sessions analyzed: {len(results)}")
    print(f"  Children: {results['child_id'].nunique()}")
    print(f"  Average quality: {results['quality_score'].mean():.1f}/100")
    print(f"  Completion rate: {results['completed'].mean()*100:.1f}%")
    print(f"  Outlier rate: {results['is_outlier'].mean()*100:.1f}%")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    low_quality = (results['quality_score'] < 40).sum()
    if low_quality > 0:
        print(f"  ⚠️ {low_quality} sessions have low quality (<40) - consider filtering")
    
    short_sessions = (results['duration'] < 60).sum()
    if short_sessions > 0:
        print(f"  ⏱️ {short_sessions} sessions are very short (<60s) - review engagement")
    
    if results['is_outlier'].mean() > 0.1:
        print(f"  🚨 High outlier rate - investigate data collection issues")
    
    if low_quality == 0 and short_sessions == 0:
        print("  ✅ Data quality looks good overall")
    
    print(f"\n📄 Detailed results saved: analysis/session_quality_analysis.csv")
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
