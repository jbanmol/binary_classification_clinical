"""
Agent Coordinator - Communication Interface between RAG and MCP Agents

This module coordinates the interaction between:
- RAG Agent (research and analysis)
- MCP Agent (task execution)

Provides unified interface for research-driven task execution.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys

# Add parent directories to path to import agents
sys.path.append(str(Path(__file__).parent.parent / "rag"))
sys.path.append(str(Path(__file__).parent.parent / "mcp"))

from rag_agent import RAGAgent
from mcp_agent import MCPAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentCoordinator:
    def __init__(self, project_root: str = None):
        """Initialize the agent coordinator."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        
        # Initialize both agents
        self.rag_agent = RAGAgent(project_root=str(self.project_root))
        self.mcp_agent = MCPAgent(project_root=str(self.project_root))
        
        # Conversation history for context
        self.conversation_history = []
        
        logger.info(f"AgentCoordinator initialized for project: {self.project_root}")
    
    def process_request(self, user_request: str, request_type: str = "auto") -> Dict[str, Any]:
        """
        Process a user request by coordinating RAG research and MCP execution.
        
        Args:
            user_request: Natural language request from user
            request_type: "research", "execute", or "auto" (determines which agent to use)
        
        Returns:
            Structured response with research findings and execution results
        """
        logger.info(f"Processing request: {user_request[:100]}...")
        
        response = {
            "request": user_request,
            "request_type": request_type,
            "research_phase": {},
            "execution_phase": {},
            "recommendations": [],
            "success": False
        }
        
        try:
            # Phase 1: Research (RAG Agent)
            if request_type in ["auto", "research"]:
                research_results = self._research_phase(user_request)
                response["research_phase"] = research_results
            
            # Phase 2: Execution (MCP Agent) 
            if request_type in ["auto", "execute"] and self._should_execute(user_request):
                execution_results = self._execution_phase(user_request, response["research_phase"])
                response["execution_phase"] = execution_results
            
            # Generate recommendations
            response["recommendations"] = self._generate_recommendations(
                user_request, response["research_phase"], response["execution_phase"]
            )
            
            response["success"] = True
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            response["error"] = str(e)
            response["success"] = False
        
        # Add to conversation history
        self.conversation_history.append(response)
        
        return response
    
    def _research_phase(self, user_request: str) -> Dict[str, Any]:
        """Execute research phase using RAG agent."""
        logger.info("Starting research phase...")
        
        research_results = {
            "queries_performed": [],
            "findings": [],
            "data_sources": [],
            "insights": ""
        }
        
        # Generate research queries based on request
        queries = self._generate_research_queries(user_request)
        
        for query in queries:
            logger.info(f"RAG Query: {query}")
            result = self.rag_agent.research_query(query)
            
            research_results["queries_performed"].append(query)
            research_results["findings"].append(result)
        
        # Synthesize research insights
        research_results["insights"] = self._synthesize_research(research_results["findings"])
        
        return research_results
    
    def _execution_phase(self, user_request: str, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks using MCP agent based on research findings."""
        logger.info("Starting execution phase...")
        
        execution_results = {
            "tasks_executed": [],
            "results": [],
            "status": "pending"
        }
        
        # Generate execution plan based on research
        execution_plan = self._generate_execution_plan(user_request, research_results)
        
        for task in execution_plan:
            logger.info(f"Executing task: {task['name']}")
            
            try:
                result = self._execute_task(task)
                execution_results["tasks_executed"].append(task["name"])
                execution_results["results"].append(result)
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                execution_results["results"].append({
                    "success": False,
                    "error": str(e),
                    "task": task["name"]
                })
        
        # Overall execution status
        successful_tasks = sum(1 for r in execution_results["results"] if r.get("success", False))
        total_tasks = len(execution_results["results"])
        
        if successful_tasks == total_tasks:
            execution_results["status"] = "completed"
        elif successful_tasks > 0:
            execution_results["status"] = "partial"
        else:
            execution_results["status"] = "failed"
        
        return execution_results
    
    def _generate_research_queries(self, user_request: str) -> List[str]:
        """Generate specific research queries based on user request."""
        request_lower = user_request.lower()
        queries = []
        
        # Feature-related queries
        if any(term in request_lower for term in ["feature", "pattern", "behavioral", "motor"]):
            queries.extend([
                "What are the most important features for ASD classification?",
                "How do behavioral patterns differ between ASD and TD groups?",
                "Which feature engineering techniques work best for this data?"
            ])
        
        # Model-related queries  
        if any(term in request_lower for term in ["model", "train", "classify", "predict", "algorithm"]):
            queries.extend([
                "What models perform best for ASD/TD classification?",
                "What are the optimal hyperparameters for this classification task?",
                "How should I handle class imbalance in this dataset?"
            ])
        
        # Performance/evaluation queries
        if any(term in request_lower for term in ["performance", "accuracy", "evaluate", "metrics"]):
            queries.extend([
                "What evaluation metrics are most important for clinical classification?",
                "How can I optimize sensitivity and specificity trade-offs?",
                "What threshold optimization strategies work best?"
            ])
        
        # Data analysis queries
        if any(term in request_lower for term in ["data", "analysis", "explore", "understand"]):
            queries.extend([
                "What insights can be drawn from the current data analysis?",
                "How is the data distributed between ASD and TD groups?",
                "What data quality issues should I be aware of?"
            ])
        
        # Default fallback query
        if not queries:
            queries.append(user_request)
        
        return queries[:3]  # Limit to 3 queries to avoid overwhelming
    
    def _generate_execution_plan(self, user_request: str, research_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate execution plan based on user request and research findings."""
        request_lower = user_request.lower()
        execution_plan = []
        
        # Model training tasks
        if any(term in request_lower for term in ["train", "model", "build"]):
            if "advanced" in request_lower or "ensemble" in request_lower:
                execution_plan.append({
                    "name": "train_advanced_model",
                    "type": "training",
                    "params": {"model_type": "advanced"}
                })
            elif "fair" in request_lower or "bias" in request_lower:
                execution_plan.append({
                    "name": "train_fair_model", 
                    "type": "training",
                    "params": {"model_type": "fair"}
                })
            else:
                execution_plan.append({
                    "name": "train_best_model",
                    "type": "training", 
                    "params": {"model_type": "best"}
                })
        
        # Feature extraction tasks
        if any(term in request_lower for term in ["extract", "feature", "process"]):
            execution_plan.append({
                "name": "extract_features",
                "type": "feature_extraction",
                "params": {"feature_type": "binary"}
            })
        
        # Analysis tasks
        if any(term in request_lower for term in ["analyze", "compare", "report"]):
            execution_plan.append({
                "name": "run_analysis",
                "type": "analysis",
                "params": {"script_name": "group_comparison"}
            })
        
        # Evaluation tasks  
        if any(term in request_lower for term in ["evaluate", "test", "performance"]):
            execution_plan.append({
                "name": "evaluate_model",
                "type": "evaluation",
                "params": {}
            })
        
        # MLflow tasks
        if any(term in request_lower for term in ["mlflow", "tracking", "experiment"]):
            execution_plan.append({
                "name": "start_mlflow",
                "type": "mlflow",
                "params": {"port": 5000}
            })
        
        return execution_plan
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task using the MCP agent."""
        task_type = task.get("type", "unknown")
        params = task.get("params", {})
        
        if task_type == "training":
            return self.mcp_agent.train_model(**params)
        elif task_type == "feature_extraction":
            return self.mcp_agent.extract_features(**params) 
        elif task_type == "analysis":
            return self.mcp_agent.run_analysis_script(**params)
        elif task_type == "evaluation":
            return self.mcp_agent.evaluate_model(**params)
        elif task_type == "mlflow":
            return self.mcp_agent.start_mlflow_ui(**params)
        elif task_type == "file_operation":
            return self.mcp_agent.manage_files(**params)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    def _should_execute(self, user_request: str) -> bool:
        """Determine if the request requires execution (vs just research)."""
        execution_keywords = [
            "train", "run", "execute", "start", "build", "create", "extract",
            "analyze", "evaluate", "test", "deploy", "install", "setup"
        ]
        
        return any(keyword in user_request.lower() for keyword in execution_keywords)
    
    def _synthesize_research(self, findings: List[str]) -> str:
        """Synthesize research findings into actionable insights."""
        if not findings:
            return "No research findings available."
        
        # Simple synthesis - in a real implementation, this could use an LLM
        synthesis = "Based on the research findings:\n\n"
        
        for i, finding in enumerate(findings, 1):
            key_points = finding.split("KEY FINDINGS:")[-1].split("RELEVANT DATA SOURCES:")[0].strip()
            synthesis += f"{i}. {key_points[:200]}...\n\n"
        
        return synthesis
    
    def _generate_recommendations(self, user_request: str, research_results: Dict[str, Any], 
                                 execution_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on research and execution results."""
        recommendations = []
        
        # Research-based recommendations
        if research_results.get("insights"):
            recommendations.append("Consider the research insights when planning next steps")
        
        # Execution-based recommendations  
        if execution_results.get("status") == "completed":
            recommendations.append("All tasks completed successfully - review results")
        elif execution_results.get("status") == "partial":
            recommendations.append("Some tasks failed - investigate and retry failed operations")
        elif execution_results.get("status") == "failed":
            recommendations.append("Execution failed - check error logs and requirements")
        
        # General recommendations based on request type
        request_lower = user_request.lower()
        if "train" in request_lower:
            recommendations.append("After training, evaluate model performance and optimize thresholds")
        if "feature" in request_lower:
            recommendations.append("Consider feature selection and engineering based on importance rankings")
        if "analyze" in request_lower:
            recommendations.append("Review analysis reports and visualizations for insights")
        
        return recommendations or ["Continue with your binary classification project development"]
    
    def get_project_overview(self) -> Dict[str, Any]:
        """Get comprehensive project overview using both agents."""
        logger.info("Generating project overview...")
        
        # Get technical status from MCP agent
        mcp_status = self.mcp_agent.get_project_status()
        
        # Get research insights from RAG agent
        research_overview = self.rag_agent.research_query(
            "What are the key insights and current status of this binary classification project?"
        )
        
        return {
            "timestamp": mcp_status["timestamp"],
            "project_root": mcp_status["project_root"],
            "technical_status": mcp_status,
            "research_insights": research_overview,
            "conversation_history_length": len(self.conversation_history),
            "capabilities": {
                "rag_agent": "Research, analysis, and insight generation from project data",
                "mcp_agent": "Task execution, model training, and file operations", 
                "coordinator": "Unified research-driven task execution"
            }
        }

if __name__ == "__main__":
    # Test the agent coordinator
    coordinator = AgentCoordinator()
    
    print("="*60)
    print("AGENT COORDINATOR TEST")
    print("="*60)
    
    # Test project overview
    overview = coordinator.get_project_overview()
    print("Project Overview:")
    print(f"- Models available: {len(overview['technical_status']['models'])}")
    print(f"- Data files: {len(overview['technical_status']['data_files'])}")
    print(f"- Research capabilities: {overview['capabilities']['rag_agent']}")
    
    # Test sample requests
    test_requests = [
        "What are the most important features for ASD classification?",
        "Train an advanced model for binary classification",
        "Analyze the performance differences between groups"
    ]
    
    for request in test_requests:
        print(f"\n{'-'*50}")
        print(f"Test Request: {request}")
        print('-'*50)
        
        response = coordinator.process_request(request, request_type="research")  # Research only for testing
        print(f"Success: {response['success']}")
        print(f"Research queries: {len(response['research_phase'].get('queries_performed', []))}")
        print(f"Recommendations: {len(response['recommendations'])}")
        
        if response['recommendations']:
            print("Top recommendation:", response['recommendations'][0])
