#!/usr/bin/env python3
"""
RAG System Configuration for Binary Classification Project
ASD (including DD) vs TD classification using coloring game data
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _resolve_project_path() -> Path:
    """Resolve the project root path.
    Priority: $PROJECT_PATH or $RAG_PROJECT_PATH env var -> repo root (two levels up from this file).
    """
    env_path = os.getenv("PROJECT_PATH") or os.getenv("RAG_PROJECT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # rag_system/config.py -> rag_system -> project root (parent of parent)
    return Path(__file__).resolve().parent.parent


def _resolve_raw_data_path() -> Path:
    """Resolve raw data path with env override, fallback to previous absolute path."""
    default = "/Users/jbanmol/Desktop/git_projects/binary-classification-project/data"
    return Path(os.getenv("RAW_DATA_PATH", default)).expanduser().resolve()


def _resolve_labels_path() -> Path:
    """Resolve labels CSV path with env override, fallback to data/processed/labels.csv."""
    default = str(Path(__file__).resolve().parent.parent / "data" / "processed" / "labels.csv")
    return Path(os.getenv("LABELS_PATH", default)).expanduser().resolve()


@dataclass
class RAGConfig:
    """Configuration for the RAG system"""

    # Data paths (env-overridable)
    PROJECT_PATH: Path = field(default_factory=_resolve_project_path)
    RAW_DATA_PATH: Path = field(default_factory=_resolve_raw_data_path)
    LABELS_PATH: Path = field(default_factory=_resolve_labels_path)

    # RAG local storage
    RAG_DATA_PATH: Path = field(
        default_factory=lambda: (
            Path(os.getenv("PROJECT_PATH") or os.getenv("RAG_PROJECT_PATH") or Path(__file__).resolve().parent.parent)
            / "rag_system"
            / "data"
        ).expanduser().resolve()
    )
    VECTOR_DB_PATH: Path = field(
        default_factory=lambda: (
            Path(os.getenv("PROJECT_PATH") or os.getenv("RAG_PROJECT_PATH") or Path(__file__).resolve().parent.parent)
            / "rag_system"
            / "vector_db"
        ).expanduser().resolve()
    )

    # Binary classification mapping
    LABEL_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        'ASD_DD': 'ASD',
        'ASD': 'ASD',
        'DD': 'ASD',  # Combine DD with ASD
        'TD': 'TD'
    })

    # Vector database settings
    VECTOR_DB_NAME: str = "coloring_behavior_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Local, lightweight
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Template types from the data
    TEMPLATE_TYPES: List[str] = field(default_factory=lambda: ["hut", "cake"])

    # Touch zones observed in data
    TOUCH_ZONES: List[str] = field(default_factory=lambda: [
        "Super outside", "Outside", "Bound", "Wall", "Roof"
    ])

    # Touch phases
    TOUCH_PHASES: List[str] = field(default_factory=lambda: [
        "Began", "Moved", "Stationary", "Ended", "Canceled"
    ])

    # Feature categories for indexing
    FEATURE_CATEGORIES: Dict[str, List[str]] = field(default_factory=lambda: {
        "motor_control": [
            "velocity_mean", "velocity_std", "acceleration_mean",
            "smoothness", "tremor_features", "path_length"
        ],
        "planning_executive": [
            "zone_transitions", "color_switches", "completion_rate",
            "session_duration", "stroke_count"
        ],
        "multi_touch": [
            "palm_touches", "finger_count", "multi_touch_complexity",
            "touch_phases"
        ],
        "spatial_temporal": [
            "coverage_area", "stroke_timing", "pause_patterns",
            "template_accuracy"
        ]
    })

    # RAG Research Capabilities
    RAG_CAPABILITIES: List[str] = field(default_factory=lambda: [
        "data_ingestion",           # Process raw JSON files
        "feature_extraction",       # Extract behavioral features
        "pattern_analysis",         # Identify ASD vs TD patterns
        "statistical_analysis",     # Perform statistical tests
        "model_insights",           # Provide modeling recommendations
        "research_synthesis"        # Synthesize findings
    ])

    # MCP Workflow Management
    PROJECT_PIPELINE: List[str] = field(default_factory=lambda: [
        "data_preparation",         # Organize and validate data
        "feature_engineering",      # RAG-guided feature extraction
        "exploratory_analysis",     # RAG-powered data exploration
        "model_development",        # Build classification models
        "validation_testing",       # Validate model performance
        "results_reporting"         # Generate final reports
    ])

    def __post_init__(self):
        """Create necessary directories"""
        self.RAG_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = RAGConfig()


def get_binary_label(original_label: str) -> str:
    """Convert original labels to binary ASD/TD classification"""
    return config.LABEL_MAPPING.get(original_label, "UNKNOWN")


def is_coloring_file(filepath: Path) -> bool:
    """Check if file is a coloring game session file"""
    return filepath.name.startswith("Coloring_") and filepath.suffix == ".json"


def extract_child_id(filepath: Path) -> str:
    """Extract child ID from file path or filename"""
    if filepath.parent.name != "fileKeys":
        return filepath.parent.name
    # Fallback: extract from filename
    return filepath.name.split("_")[-1].replace(".json", "")


def extract_session_timestamp(filepath: Path) -> Optional[str]:
    """Extract timestamp from coloring filename"""
    try:
        # Format: Coloring_YYYY-MM-DD HH:MM:SS.microseconds_childid.json
        parts = filepath.name.replace("Coloring_", "").replace(".json", "").split("_")
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1].split('.')[0]}"
    except Exception:
        pass
    return None


def get_template_from_features(session_features: Dict) -> str:
    """Extract template type from session features"""
    # This will be implemented based on your feature extraction
    # For now, return from the first column of your CSV data
    return session_features.get("template", "unknown")
