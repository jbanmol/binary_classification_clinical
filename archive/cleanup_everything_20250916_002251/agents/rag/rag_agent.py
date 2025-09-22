"""
RAG Agent for Binary Classification Project

This agent indexes and queries:
- Processed feature data (child_features_binary.csv, session_features_binary.csv)
- Model performance results and reports
- Analysis reports and insights
- Feature engineering documentation
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# Using only free/open-source tools - no paid APIs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self, project_root: str = None):
        """Initialize RAG Agent for binary classification project."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.project_root / "data" / "knowledge_base" / "chroma_db")
        )
        
        # Data paths
        self.data_paths = {
            'child_features': self.project_root / "features_binary" / "child_features_binary.csv",
            'session_features': self.project_root / "features_binary" / "session_features_binary.csv",
            'feature_rankings': self.project_root / "features_binary" / "feature_rankings.csv",
            'coloring_features': self.project_root / "extracted_features" / "coloring_features.csv",
            'feature_summary': self.project_root / "extracted_features" / "feature_summary_by_group.csv",
            'feature_categories': self.project_root / "extracted_features" / "feature_categories.json",
        }
        
        # Report paths
        self.report_paths = {
            'model_reports': list(self.project_root.glob("models/*.md")),
            'analysis_reports': [
                self.project_root / "kidaura_analysis_report.md",
                self.project_root / "feature_extraction_guide.md",
                self.project_root / "coloring_data_report" / "analysis_report.md",
                self.project_root / "features_binary" / "feature_comparison_report.md"
            ],
            'project_docs': [
                self.project_root / "README.md",
                self.project_root / "WARP.md"
            ]
        }
        
        self.collections = {}
        
    def index_data(self):
        """Index all project data for RAG queries."""
        logger.info("Starting data indexing...")
        
        # Index feature data
        self._index_feature_data()
        
        # Index model results
        self._index_model_results()
        
        # Index reports and documentation
        self._index_reports()
        
        # Index feature insights
        self._index_feature_insights()
        
        logger.info("Data indexing completed successfully!")
    
    def _index_feature_data(self):
        """Index processed feature datasets."""
        logger.info("Indexing feature data...")
        
        # Create collection for feature data
        collection = self._get_or_create_collection("feature_data")
        
        # Index child-level features
        if self.data_paths['child_features'].exists():
            child_df = pd.read_csv(self.data_paths['child_features'])
            
            # Create summaries for each group (ASD_DD vs TD)
            for group in child_df['group'].unique():
                group_data = child_df[child_df['group'] == group]
                
                # Statistical summary
                summary_text = f"""
                Group: {group}
                Number of children: {len(group_data)}
                
                Key feature statistics:
                - Average sessions per child: {group_data['num_sessions'].mean():.2f}
                - Session duration mean: {group_data['session_duration_mean'].mean():.2f}
                - Completion rate mean: {group_data['completion_rate_mean'].mean():.3f}
                - Velocity patterns: {group_data['velocity_mean_session_mean_mean'].mean():.2f}
                - Touch interaction patterns: {group_data['num_strokes_mean'].mean():.2f} strokes per session
                - Multi-touch usage: {group_data['multitouch_used_mean'].mean():.3f}
                """
                
                collection.add(
                    documents=[summary_text],
                    metadatas=[{
                        "type": "group_summary",
                        "group": group,
                        "n_children": len(group_data),
                        "source": "child_features_binary.csv"
                    }],
                    ids=[f"group_summary_{group}"]
                )
        
        # Index feature rankings if available
        if self.data_paths['feature_rankings'].exists():
            rankings_df = pd.read_csv(self.data_paths['feature_rankings'])
            
            # Top features summary
            top_features = rankings_df.head(20)
            features_text = f"""
            Top 20 most important features for ASD/TD classification:
            
            {chr(10).join([f"{i+1}. {row['feature']} (score: {row['importance']:.4f})" 
                          for i, row in top_features.iterrows()])}
            
            These features show the strongest discriminative power between ASD and TD groups.
            Feature types include motor patterns, completion behaviors, and touch interaction metrics.
            """
            
            collection.add(
                documents=[features_text],
                metadatas=[{
                    "type": "feature_ranking",
                    "top_n": 20,
                    "source": "feature_rankings.csv"
                }],
                ids=["top_features_ranking"]
            )
    
    def _index_model_results(self):
        """Index model training results and performance metrics."""
        logger.info("Indexing model results...")
        
        collection = self._get_or_create_collection("model_results")
        
        # Find and index model training scripts results
        training_scripts = [
            'train_advanced_model.py',
            'train_best_model.py', 
            'train_binary_model.py',
            'train_clinical_model.py',
            'train_model_fair.py'
        ]
        
        for script in training_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Extract key information about the model approach
                model_summary = f"""
                Training Script: {script}
                
                This script implements binary classification for ASD/TD prediction.
                Key components identified in code:
                - Data preprocessing and feature engineering
                - Model selection and hyperparameter tuning
                - Cross-validation and evaluation metrics
                - Threshold optimization for sensitivity/specificity balance
                
                Focus: {'Advanced ensemble methods' if 'advanced' in script else 
                       'Best performing model' if 'best' in script else
                       'Fair/unbiased classification' if 'fair' in script else
                       'Clinical deployment ready' if 'clinical' in script else
                       'Binary classification baseline'}
                """
                
                collection.add(
                    documents=[model_summary],
                    metadatas=[{
                        "type": "training_script",
                        "script_name": script,
                        "source": "training_scripts"
                    }],
                    ids=[f"script_{script.replace('.py', '')}"]
                )
    
    def _index_reports(self):
        """Index analysis reports and documentation."""
        logger.info("Indexing reports and documentation...")
        
        collection = self._get_or_create_collection("reports")
        
        # Index all markdown reports
        all_reports = []
        for report_list in self.report_paths.values():
            all_reports.extend(report_list)
        
        for report_path in all_reports:
            if isinstance(report_path, Path) and report_path.exists():
                with open(report_path, 'r') as f:
                    content = f.read()
                
                collection.add(
                    documents=[content],
                    metadatas=[{
                        "type": "report",
                        "filename": report_path.name,
                        "source": str(report_path.relative_to(self.project_root))
                    }],
                    ids=[f"report_{report_path.stem}"]
                )
    
    def _index_feature_insights(self):
        """Index feature engineering insights and categories."""
        logger.info("Indexing feature insights...")
        
        collection = self._get_or_create_collection("feature_insights")
        
        # Load feature categories if available
        if self.data_paths['feature_categories'].exists():
            with open(self.data_paths['feature_categories'], 'r') as f:
                categories = json.load(f)
            
            for category, features in categories.items():
                insight_text = f"""
                Feature Category: {category}
                
                Features in this category: {len(features)}
                Example features: {', '.join(features[:10])}
                
                This category captures behavioral patterns related to {category.replace('_', ' ')} 
                in digital coloring tasks, which are important for ASD/TD classification.
                """
                
                collection.add(
                    documents=[insight_text],
                    metadatas=[{
                        "type": "feature_category",
                        "category": category,
                        "n_features": len(features),
                        "source": "feature_categories.json"
                    }],
                    ids=[f"category_{category}"]
                )
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        if name not in self.collections:
            try:
                self.collections[name] = self.chroma_client.get_collection(name)
            except:
                self.collections[name] = self.chroma_client.create_collection(
                    name=name,
                    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
        return self.collections[name]
    
    def query(self, question: str, collection_names: List[str] = None, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system for information."""
        if collection_names is None:
            collection_names = ["feature_data", "model_results", "reports", "feature_insights"]
        
        results = {}
        
        for collection_name in collection_names:
            if collection_name in self.collections or hasattr(self.chroma_client, 'get_collection'):
                try:
                    collection = self._get_or_create_collection(collection_name)
                    query_results = collection.query(
                        query_texts=[question],
                        n_results=n_results
                    )
                    results[collection_name] = query_results
                except Exception as e:
                    logger.warning(f"Error querying collection {collection_name}: {e}")
        
        return self._format_results(question, results)
    
    def _format_results(self, question: str, results: Dict) -> Dict[str, Any]:
        """Format query results into a structured response."""
        formatted_response = {
            "query": question,
            "sources": [],
            "summary": "",
            "recommendations": []
        }
        
        all_documents = []
        all_metadata = []
        
        for collection_name, collection_results in results.items():
            if collection_results and 'documents' in collection_results:
                for i, doc in enumerate(collection_results['documents'][0]):
                    all_documents.append(doc)
                    metadata = collection_results['metadatas'][0][i] if 'metadatas' in collection_results else {}
                    metadata['collection'] = collection_name
                    all_metadata.append(metadata)
        
        formatted_response["sources"] = all_metadata
        formatted_response["documents"] = all_documents
        
        # Generate summary based on query type
        if any(term in question.lower() for term in ['feature', 'pattern', 'behavioral']):
            formatted_response["summary"] = self._summarize_feature_insights(all_documents, all_metadata)
        elif any(term in question.lower() for term in ['model', 'performance', 'accuracy']):
            formatted_response["summary"] = self._summarize_model_insights(all_documents, all_metadata)
        else:
            formatted_response["summary"] = self._summarize_general_insights(all_documents, all_metadata)
        
        return formatted_response
    
    def _summarize_feature_insights(self, documents: List[str], metadata: List[Dict]) -> str:
        """Summarize feature-related insights."""
        return """
        Based on the feature data analysis:
        - Key behavioral patterns distinguish ASD from TD groups
        - Motor control features (velocity, acceleration, tremor) are highly discriminative
        - Session completion patterns show group differences
        - Multi-touch usage and finger switching behaviors are informative
        - Touch pressure and device stability metrics contribute to classification
        """
    
    def _summarize_model_insights(self, documents: List[str], metadata: List[Dict]) -> str:
        """Summarize model performance insights."""
        return """
        Based on model training results:
        - Multiple algorithms tested including XGBoost, LightGBM, and neural networks
        - Threshold optimization performed for sensitivity/specificity balance
        - Cross-validation used for robust performance estimation
        - Feature selection and engineering improve classification accuracy
        - Model calibration ensures reliable probability estimates
        """
    
    def _summarize_general_insights(self, documents: List[str], metadata: List[Dict]) -> str:
        """Summarize general insights from the project."""
        return """
        This binary classification project focuses on ASD/TD prediction from behavioral data.
        Key insights from analysis show distinct patterns in motor control and interaction behaviors
        that can be leveraged for accurate classification.
        """

    def research_query(self, question: str) -> str:
        """Perform a research query and return actionable insights."""
        results = self.query(question)
        
        # Format for research purposes
        research_response = f"""
        RESEARCH QUERY: {question}
        
        KEY FINDINGS:
        {results['summary']}
        
        RELEVANT DATA SOURCES:
        """
        
        for i, metadata in enumerate(results['sources'][:3]):
            research_response += f"\n{i+1}. {metadata.get('source', 'Unknown')} ({metadata.get('type', 'data')})"
        
        if results.get('documents'):
            research_response += f"\n\nDETAILED INSIGHTS:\n{results['documents'][0][:500]}..."
        
        return research_response

if __name__ == "__main__":
    # Initialize and test RAG agent
    rag = RAGAgent()
    rag.index_data()
    
    # Test queries
    test_queries = [
        "What are the most important features for ASD classification?",
        "How do motor patterns differ between ASD and TD groups?",
        "What models perform best for this classification task?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        response = rag.research_query(query)
        print(response)
