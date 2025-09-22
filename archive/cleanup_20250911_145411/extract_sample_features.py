#!/usr/bin/env python3
"""
Sample Feature Extraction - Process only first 20 children for demonstration
"""

from extract_coloring_features import ColoringFeatureExtractor
from pathlib import Path
from rag_system.config import config as RAG_CFG

# Override the process method to limit children
class SampleFeatureExtractor(ColoringFeatureExtractor):
    def process_all_children(self, max_children=20):
        """Process limited number of children for demonstration"""
        all_features = []
        
        # Get all child folders
        DATA_PATH = RAG_CFG.RAW_DATA_PATH
        child_folders = [f for f in DATA_PATH.iterdir() if f.is_dir()][:max_children]
        
        print(f"Processing {len(child_folders)} children (sample)...")
        
        for i, child_folder in enumerate(child_folders):
            child_id = child_folder.name
            
            # Skip if not in labels
            if child_id not in self.labels_dict:
                print(f"  Skipping {child_id} - no label found")
                continue
            
            # Get coloring files
            coloring_files = list(child_folder.glob("Coloring_*.json"))
            
            if not coloring_files:
                print(f"  No coloring files for {child_id}")
                continue
            
            print(f"  Processing {child_id} ({self.labels_dict[child_id]}) - {len(coloring_files)} files")
            
            # Extract features
            child_features = self.extract_all_features(child_id, coloring_files)
            all_features.extend(child_features)
        
        # Convert to DataFrame
        import pandas as pd
        features_df = pd.DataFrame(all_features)
        
        # Save features
        OUTPUT_PATH = RAG_CFG.PROJECT_PATH / 'extracted_features'
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_PATH / 'sample_coloring_features.csv'
        features_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(features_df)} session features to {output_file}")
        
        # Save feature summary
        self.save_feature_summary(features_df)
        
        return features_df

def main():
    """Main extraction function for sample"""
    extractor = SampleFeatureExtractor()
    features_df = extractor.process_all_children(max_children=20)
    
    # Print summary
    print(f"\nSample extraction complete!")
    print(f"Total sessions: {len(features_df)}")
    print(f"Feature columns: {len(features_df.columns)}")
    print(f"\nGroup distribution:")
    print(features_df['group'].value_counts())
    
    # Display some key features by group
    if len(features_df) > 0:
        print("\nKey feature means by group:")
        key_features = [
            'stroke_velocity_mean_mean', 
            'stroke_acceleration_std_mean',
            'out_of_bounds_ratio', 
            'zone_transition_rate',
            'final_completion'
        ]
        
        # Only show features that exist
        existing_features = [f for f in key_features if f in features_df.columns]
        
        if existing_features:
            summary = features_df.groupby('group')[existing_features].mean()
            print(summary)
    
    return features_df

if __name__ == "__main__":
    features_df = main()
