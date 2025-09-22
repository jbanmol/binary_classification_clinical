"""
MCP (Model Context Protocol) Agent for Binary Classification Project

This agent handles task execution including:
- Model training and evaluation
- Data processing and feature extraction
- File operations and experiment management
- MLflow integration and logging
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import pickle
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPAgent:
    def __init__(self, project_root: str = None):
        """Initialize MCP Agent for binary classification project."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.current_experiment = None
        
        # Key project paths
        self.paths = {
            'models': self.project_root / "models",
            'data': self.project_root / "data",
            'features_binary': self.project_root / "features_binary",
            'extracted_features': self.project_root / "extracted_features",
            'results': self.project_root / "results",
            'logs': self.project_root / "logs"
        }
        
        # Available training scripts
        self.training_scripts = {
            'advanced': 'train_advanced_model.py',
            'best': 'train_best_model.py',
            'binary': 'train_binary_model.py',
            'clinical': 'train_clinical_model.py',
            'fair': 'train_model_fair.py'
        }
        
        # Feature extraction scripts
        self.feature_scripts = {
            'binary': 'extract_features_binary.py',
            'coloring': 'extract_coloring_features.py',
            'sample': 'extract_sample_features.py'
        }
    
    def execute_command(self, command: Union[str, List[str]], cwd: str = None, capture_output: bool = True) -> Dict[str, Any]:
        """Execute a shell command and return results."""
        if isinstance(command, str):
            command = command.split()
        
        work_dir = Path(cwd) if cwd else self.project_root
        
        try:
            logger.info(f"Executing command: {' '.join(command)} in {work_dir}")
            
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=capture_output,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout if capture_output else '',
                'stderr': result.stderr if capture_output else '',
                'command': ' '.join(command)
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return {
                'success': False,
                'error': 'Command timed out',
                'command': ' '.join(command)
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': ' '.join(command)
            }
    
    def train_model(self, model_type: str = 'best', **kwargs) -> Dict[str, Any]:
        """Execute model training with specified parameters."""
        if model_type not in self.training_scripts:
            return {
                'success': False,
                'error': f"Unknown model type: {model_type}. Available: {list(self.training_scripts.keys())}"
            }
        
        script = self.training_scripts[model_type]
        command = ['python', script]
        
        # Add command line arguments from kwargs
        for key, value in kwargs.items():
            if value is not None:
                command.extend([f"--{key.replace('_', '-')}", str(value)])
        
        logger.info(f"Training {model_type} model with command: {' '.join(command)}")
        
        # Start MLflow run if not already active
        if not mlflow.active_run():
            experiment_name = f"binary_classification_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            self.current_experiment = experiment_name
        
        result = self.execute_command(command)
        
        if result['success']:
            logger.info(f"Model training completed successfully: {model_type}")
            result['model_type'] = model_type
            result['experiment'] = self.current_experiment
        else:
            logger.error(f"Model training failed: {result.get('stderr', 'Unknown error')}")
        
        return result
    
    def extract_features(self, feature_type: str = 'binary', **kwargs) -> Dict[str, Any]:
        """Execute feature extraction with specified parameters."""
        if feature_type not in self.feature_scripts:
            return {
                'success': False,
                'error': f"Unknown feature type: {feature_type}. Available: {list(self.feature_scripts.keys())}"
            }
        
        script = self.feature_scripts[feature_type]
        command = ['python', script]
        
        # Add command line arguments from kwargs
        for key, value in kwargs.items():
            if value is not None:
                command.extend([f"--{key.replace('_', '-')}", str(value)])
        
        logger.info(f"Extracting {feature_type} features with command: {' '.join(command)}")
        
        result = self.execute_command(command)
        
        if result['success']:
            logger.info(f"Feature extraction completed successfully: {feature_type}")
            result['feature_type'] = feature_type
        else:
            logger.error(f"Feature extraction failed: {result.get('stderr', 'Unknown error')}")
        
        return result
    
    def evaluate_model(self, model_path: str = None, data_path: str = None) -> Dict[str, Any]:
        """Evaluate a trained model on test data."""
        if not model_path:
            # Find most recent model
            model_files = list(self.paths['models'].glob("*.pkl"))
            if not model_files:
                return {'success': False, 'error': 'No model files found'}
            model_path = str(max(model_files, key=os.path.getctime))
        
        if not data_path:
            # Use default test data
            data_path = str(self.paths['features_binary'] / "child_features_binary.csv")
        
        command = ['python', 'main.py', '--mode', 'evaluate', '--model-path', model_path, '--data-path', data_path]
        
        logger.info(f"Evaluating model: {model_path}")
        
        result = self.execute_command(command)
        
        if result['success']:
            logger.info("Model evaluation completed successfully")
            result['model_path'] = model_path
            result['data_path'] = data_path
        else:
            logger.error(f"Model evaluation failed: {result.get('stderr', 'Unknown error')}")
        
        return result
    
    def start_mlflow_ui(self, port: int = 5000) -> Dict[str, Any]:
        """Start MLflow UI server."""
        mlruns_path = self.project_root / "mlruns"
        command = ['mlflow', 'ui', '--backend-store-uri', str(mlruns_path), '--port', str(port)]
        
        logger.info(f"Starting MLflow UI on port {port}")
        
        # Run in background
        result = self.execute_command(command, capture_output=False)
        
        if result['success']:
            logger.info(f"MLflow UI started at http://localhost:{port}")
            result['url'] = f"http://localhost:{port}"
        
        return result
    
    def manage_files(self, action: str, source: str, destination: str = None) -> Dict[str, Any]:
        """Manage project files (copy, move, delete, etc.)."""
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = self.project_root / source
        
        try:
            if action == 'copy':
                if not destination:
                    return {'success': False, 'error': 'Destination required for copy operation'}
                
                dest_path = Path(destination)
                if not dest_path.is_absolute():
                    dest_path = self.project_root / destination
                
                if source_path.is_file():
                    shutil.copy2(source_path, dest_path)
                else:
                    shutil.copytree(source_path, dest_path)
                
                logger.info(f"Copied {source_path} to {dest_path}")
                return {'success': True, 'action': 'copy', 'source': str(source_path), 'destination': str(dest_path)}
            
            elif action == 'move':
                if not destination:
                    return {'success': False, 'error': 'Destination required for move operation'}
                
                dest_path = Path(destination)
                if not dest_path.is_absolute():
                    dest_path = self.project_root / destination
                
                shutil.move(source_path, dest_path)
                logger.info(f"Moved {source_path} to {dest_path}")
                return {'success': True, 'action': 'move', 'source': str(source_path), 'destination': str(dest_path)}
            
            elif action == 'delete':
                if source_path.is_file():
                    source_path.unlink()
                else:
                    shutil.rmtree(source_path)
                
                logger.info(f"Deleted {source_path}")
                return {'success': True, 'action': 'delete', 'source': str(source_path)}
            
            elif action == 'create_dir':
                source_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory {source_path}")
                return {'success': True, 'action': 'create_dir', 'path': str(source_path)}
            
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
        
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return {'success': False, 'error': str(e), 'action': action}
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get current project status including models, data, and results."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'models': [],
            'data_files': [],
            'recent_results': [],
            'active_experiment': self.current_experiment
        }
        
        # Check for models
        if self.paths['models'].exists():
            for model_file in self.paths['models'].glob("*.pkl"):
                stat = model_file.stat()
                status['models'].append({
                    'name': model_file.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Check for data files
        for data_dir in [self.paths['features_binary'], self.paths['extracted_features']]:
            if data_dir.exists():
                for data_file in data_dir.glob("*.csv"):
                    stat = data_file.stat()
                    status['data_files'].append({
                        'name': data_file.name,
                        'path': str(data_file.relative_to(self.project_root)),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # Check for recent results
        if self.paths['results'].exists():
            for result_file in self.paths['results'].rglob("*"):
                if result_file.is_file() and result_file.suffix in ['.png', '.pdf', '.txt', '.md', '.json']:
                    stat = result_file.stat()
                    status['recent_results'].append({
                        'name': result_file.name,
                        'path': str(result_file.relative_to(self.project_root)),
                        'type': result_file.suffix,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # Sort by modification time
        status['models'].sort(key=lambda x: x['modified'], reverse=True)
        status['recent_results'].sort(key=lambda x: x['modified'], reverse=True)
        status['recent_results'] = status['recent_results'][:10]  # Keep only 10 most recent
        
        return status
    
    def run_analysis_script(self, script_name: str, **kwargs) -> Dict[str, Any]:
        """Run analysis scripts like group comparison, data analysis, etc."""
        analysis_scripts = {
            'coloring_analysis': 'analyze_coloring_data.py',
            'feature_comparison': 'compare_old_new_features.py',
            'group_comparison': 'group_feature_comparison.py'
        }
        
        if script_name not in analysis_scripts:
            return {
                'success': False,
                'error': f"Unknown analysis script: {script_name}. Available: {list(analysis_scripts.keys())}"
            }
        
        script = analysis_scripts[script_name]
        command = ['python', script]
        
        # Add command line arguments from kwargs
        for key, value in kwargs.items():
            if value is not None:
                command.extend([f"--{key.replace('_', '-')}", str(value)])
        
        logger.info(f"Running analysis script: {script}")
        
        result = self.execute_command(command)
        
        if result['success']:
            logger.info(f"Analysis completed successfully: {script_name}")
            result['script_name'] = script_name
        else:
            logger.error(f"Analysis failed: {result.get('stderr', 'Unknown error')}")
        
        return result
    
    def setup_environment(self) -> Dict[str, Any]:
        """Set up the project environment (install dependencies, create directories)."""
        logger.info("Setting up project environment...")
        
        results = []
        
        # Install requirements
        req_files = ['requirements.txt', 'requirements_advanced.txt']
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                result = self.execute_command(['pip', 'install', '-r', str(req_path)])
                results.append({
                    'task': f'install_{req_file}',
                    'success': result['success'],
                    'details': result.get('stdout', '')[:200] + '...' if result.get('stdout') else ''
                })
        
        # Create directories using config.py
        result = self.execute_command(['python', 'config.py'])
        results.append({
            'task': 'create_directories',
            'success': result['success'],
            'details': result.get('stdout', '')
        })
        
        # Initialize git if not already done
        git_dir = self.project_root / '.git'
        if not git_dir.exists():
            result = self.execute_command(['git', 'init'])
            results.append({
                'task': 'git_init',
                'success': result['success'],
                'details': result.get('stdout', '')
            })
        
        overall_success = all(r['success'] for r in results)
        
        return {
            'success': overall_success,
            'tasks': results,
            'message': 'Environment setup completed' if overall_success else 'Environment setup had some issues'
        }

if __name__ == "__main__":
    # Initialize and test MCP agent
    mcp = MCPAgent()
    
    # Test project status
    print("="*50)
    print("PROJECT STATUS")
    print("="*50)
    status = mcp.get_project_status()
    print(json.dumps(status, indent=2))
    
    # Test command execution
    print("\n" + "="*50)
    print("TESTING COMMAND EXECUTION")
    print("="*50)
    result = mcp.execute_command(['ls', '-la'])
    print(f"Command success: {result['success']}")
    if result['success']:
        print("Directory listing completed")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
