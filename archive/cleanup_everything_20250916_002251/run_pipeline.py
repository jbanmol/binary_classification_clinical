#!/usr/bin/env python3
"""
Main execution script for RAG + MCP Binary Classification Pipeline
Run this script to execute the complete pipeline for ASD vs TD classification
"""

import sys
import traceback
from pathlib import Path

# Add the project modules to path
sys.path.append(str(Path(__file__).parent / "rag_system"))
sys.path.append(str(Path(__file__).parent / "mcp_orchestrator"))

def run_full_pipeline():
    """Execute the complete RAG + MCP pipeline"""
    try:
        print("🚀 Initializing RAG + MCP Binary Classification Pipeline")
        print("="*70)
        
        # Import after adding to path
        from mcp_orchestrator.project_manager import orchestrator
        
        # Execute the full pipeline with a reasonable session limit for demo
        results = orchestrator.execute_full_pipeline(limit_sessions=100, target_sensitivity=0.87)
        
        print("\n" + "="*70)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*70)
        
        if 'error' not in results:
            print("✅ Pipeline completed successfully!")
            
            # Print summary of each phase
            for phase, phase_results in results.items():
                if isinstance(phase_results, dict) and phase_results.get('status') == 'completed':
                    print(f"\n✅ {phase.replace('_', ' ').title()}: Completed")
                    
                    # Print key metrics for each phase
                    if phase == 'data_preparation':
                        ds = phase_results.get('data_summary', {})
                        print(f"   📊 Sessions processed: {ds.get('total_sessions', 'N/A')}")
                        print(f"   👥 Unique children: {ds.get('unique_children', 'N/A')}")
                        print(f"   🏷️  Label distribution: {ds.get('label_distribution', 'N/A')}")
                    
                    elif phase == 'model_development':
                        print(f"   🏆 Best model: {phase_results.get('best_model', 'N/A')}")
                        best_score = phase_results.get('best_cv_score', 0)
                        if best_score > 0:
                            print(f"   📈 Best CV AUC: {best_score:.3f}")
                        
                        dataset_info = phase_results.get('dataset_info', {})
                        print(f"   🎯 Training samples: {dataset_info.get('total_samples', 'N/A')}")
                        print(f"   🔧 Features used: {dataset_info.get('n_features', 'N/A')}")
                        
                        # Show top features
                        feature_importance = phase_results.get('feature_importance', {})
                        if feature_importance:
                            print("   🔝 Top 5 features:")
                            for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                                print(f"      {i+1}. {feature}: {importance:.3f}")
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            
    except Exception as e:
        print(f"💥 Pipeline execution failed with error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")

def demo_rag_queries():
    """Demonstrate RAG research capabilities with example queries"""
    try:
        print("\n🔬 Demonstrating RAG Research Capabilities")
        print("="*50)
        
        from rag_system.research_engine import research_engine
        
        # Example research queries
        demo_queries = [
            "What motor control differences distinguish ASD from TD children?",
            "How do palm touches affect classification accuracy?",
            "What are the key behavioral patterns in task completion?",
            "How do tremor indicators vary between diagnostic groups?"
        ]
        
        for query in demo_queries:
            print(f"\n🔍 Query: {query}")
            result = research_engine.research_query(query, n_results=5)
            
            if 'error' not in result:
                print(f"   📊 Found {result['total_results']} relevant patterns")
                print(f"   🏷️  Distribution: ASD={result['label_distribution'].get('ASD', 0)}, TD={result['label_distribution'].get('TD', 0)}")
                print(f"   🎯 Similarity: {result['average_similarity']:.2f}")
                
                if result.get('research_insights'):
                    print("   💡 Key insights:")
                    for insight in result['research_insights'][:3]:
                        print(f"      • {insight}")
            else:
                print(f"   ❌ Query failed: {result['error']}")
                
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG + MCP Binary Classification Pipeline")
    parser.add_argument("--mode", choices=["full", "demo", "query"], default="full",
                      help="Execution mode: full pipeline, demo queries, or custom query")
    parser.add_argument("--query", type=str, 
                      help="Custom research query (use with --mode query)")
    parser.add_argument("--limit", type=int, default=100,
                      help="Limit number of sessions to process (for testing)")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "demo":
        demo_rag_queries()
    elif args.mode == "query":
        if args.query:
            try:
                from rag_system.research_engine import research_engine
                result = research_engine.research_query(args.query)
                print(f"Query: {args.query}")
                print(f"Results: {result}")
            except Exception as e:
                print(f"Query failed: {e}")
        else:
            print("Please provide a query with --query option")
    
    print("\n🎉 Execution completed!")
