"""
Main Orchestrator for RAG + MCP Agent System

This is the main entry point for coordinating RAG research and MCP execution
for the binary classification project.

Usage:
    python agents/main_orchestrator.py --query "Train an advanced model for ASD classification"
    python agents/main_orchestrator.py --interactive
    python agents/main_orchestrator.py --index-data  # Index data for RAG system
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from shared.agent_coordinator import AgentCoordinator

def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG + MCP Agent System for Binary Classification Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python agents/main_orchestrator.py --interactive
  
  # Single query
  python agents/main_orchestrator.py --query "What are the best features for ASD classification?"
  
  # Index data for RAG system
  python agents/main_orchestrator.py --index-data
  
  # Get project overview
  python agents/main_orchestrator.py --overview
  
  # Research only (no execution)
  python agents/main_orchestrator.py --query "Train a model" --research-only
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive mode"
    )
    
    parser.add_argument(
        "--index-data",
        action="store_true", 
        help="Index project data for RAG system"
    )
    
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Get project overview"
    )
    
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Perform research only (no task execution)"
    )
    
    parser.add_argument(
        "--execute-only", 
        action="store_true",
        help="Execute tasks only (no research)"
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to project root directory"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format for responses"
    )
    
    return parser

def print_response(response: Dict[str, Any], format: str = "text"):
    """Print response in specified format."""
    if format == "json":
        print(json.dumps(response, indent=2, default=str))
        return
    
    # Text format
    print("=" * 60)
    print(f"REQUEST: {response.get('request', 'Unknown')}")
    print("=" * 60)
    
    # Research Phase
    research_phase = response.get('research_phase', {})
    if research_phase:
        print("\n🔍 RESEARCH PHASE:")
        print(f"   Queries performed: {len(research_phase.get('queries_performed', []))}")
        
        if research_phase.get('insights'):
            print(f"\n   Key Insights:")
            insights = research_phase['insights'][:300] + "..." if len(research_phase['insights']) > 300 else research_phase['insights']
            print(f"   {insights}")
    
    # Execution Phase
    execution_phase = response.get('execution_phase', {})
    if execution_phase:
        print(f"\n⚡ EXECUTION PHASE:")
        print(f"   Status: {execution_phase.get('status', 'Unknown')}")
        print(f"   Tasks executed: {len(execution_phase.get('tasks_executed', []))}")
        
        if execution_phase.get('tasks_executed'):
            print(f"   Tasks: {', '.join(execution_phase['tasks_executed'])}")
    
    # Recommendations
    recommendations = response.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Success status
    success = response.get('success', False)
    status_emoji = "✅" if success else "❌"
    print(f"\n{status_emoji} STATUS: {'SUCCESS' if success else 'FAILED'}")
    
    if not success and 'error' in response:
        print(f"   Error: {response['error']}")
    
    print("\n" + "=" * 60)

def interactive_mode(coordinator: AgentCoordinator, output_format: str):
    """Run interactive mode for continuous queries."""
    print("=" * 60)
    print("🤖 RAG + MCP AGENT SYSTEM - INTERACTIVE MODE")
    print("=" * 60)
    print("Enter queries to get research insights and execute tasks.")
    print("Available commands:")
    print("  - Any natural language query about your binary classification project")
    print("  - 'overview' - Get project overview")
    print("  - 'status' - Get current project status") 
    print("  - 'help' - Show this help message")
    print("  - 'quit' or 'exit' - Exit interactive mode")
    print("\nExamples:")
    print("  > What are the most important features for ASD classification?")
    print("  > Train an advanced model using ensemble methods")
    print("  > Analyze performance differences between ASD and TD groups")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🤖 Query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Any natural language query")
                print("  - 'overview' - Project overview") 
                print("  - 'status' - Project status")
                print("  - 'quit' - Exit")
                continue
                
            elif query.lower() == 'overview':
                overview = coordinator.get_project_overview()
                print_response({"request": "Project Overview", "overview": overview}, output_format)
                continue
                
            elif query.lower() == 'status':
                status = coordinator.mcp_agent.get_project_status()
                print("\n📊 PROJECT STATUS:")
                print(f"   Models: {len(status['models'])}")
                print(f"   Data files: {len(status['data_files'])}")
                print(f"   Recent results: {len(status['recent_results'])}")
                continue
            
            # Process regular query
            print(f"\n🔄 Processing query: {query[:50]}{'...' if len(query) > 50 else ''}")
            response = coordinator.process_request(query)
            print_response(response, output_format)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            continue

def index_data(coordinator: AgentCoordinator):
    """Index project data for RAG system."""
    print("=" * 60)
    print("📚 INDEXING PROJECT DATA FOR RAG SYSTEM")
    print("=" * 60)
    
    try:
        print("🔄 Starting data indexing...")
        coordinator.rag_agent.index_data()
        print("✅ Data indexing completed successfully!")
        
        # Test a sample query to verify indexing worked
        print("\n🧪 Testing RAG system with sample query...")
        test_response = coordinator.rag_agent.research_query(
            "What are the key features in this binary classification project?"
        )
        
        if test_response:
            print("✅ RAG system is working correctly!")
            print(f"Sample response preview: {test_response[:200]}...")
        else:
            print("⚠️  RAG system test returned empty response")
            
    except Exception as e:
        print(f"❌ Error during data indexing: {e}")
        return False
    
    print("\n" + "=" * 60)
    return True

def get_project_overview(coordinator: AgentCoordinator, output_format: str):
    """Get and display project overview."""
    print("=" * 60)  
    print("📊 PROJECT OVERVIEW")
    print("=" * 60)
    
    try:
        overview = coordinator.get_project_overview()
        
        if output_format == "json":
            print(json.dumps(overview, indent=2, default=str))
        else:
            technical_status = overview.get('technical_status', {})
            
            print(f"🏠 Project Root: {overview.get('project_root', 'Unknown')}")
            print(f"📅 Timestamp: {overview.get('timestamp', 'Unknown')}")
            print(f"\n📊 Technical Status:")
            print(f"   Models: {len(technical_status.get('models', []))}")
            print(f"   Data Files: {len(technical_status.get('data_files', []))}")
            print(f"   Recent Results: {len(technical_status.get('recent_results', []))}")
            
            print(f"\n🧠 Capabilities:")
            capabilities = overview.get('capabilities', {})
            for agent, capability in capabilities.items():
                print(f"   {agent}: {capability}")
            
            print(f"\n💬 Conversation History: {overview.get('conversation_history_length', 0)} interactions")
        
    except Exception as e:
        print(f"❌ Error getting project overview: {e}")
    
    print("\n" + "=" * 60)

def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    try:
        # Initialize coordinator
        print("🚀 Initializing RAG + MCP Agent System...")
        coordinator = AgentCoordinator(project_root=args.project_root)
        print("✅ Agent system initialized successfully!")
        
        # Handle different modes
        if args.index_data:
            index_data(coordinator)
            
        elif args.overview:
            get_project_overview(coordinator, args.output_format)
            
        elif args.interactive:
            interactive_mode(coordinator, args.output_format)
            
        elif args.query:
            # Determine request type
            request_type = "auto"
            if args.research_only:
                request_type = "research"
            elif args.execute_only:
                request_type = "execute"
            
            print(f"🔄 Processing query: {args.query}")
            response = coordinator.process_request(args.query, request_type=request_type)
            print_response(response, args.output_format)
            
        else:
            # No specific mode - show help and start interactive
            parser.print_help()
            print(f"\n🤖 Starting interactive mode...")
            interactive_mode(coordinator, args.output_format)
    
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
