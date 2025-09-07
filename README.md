# Binary Classification Project

A high-accuracy binary classification model with emphasis on sensitivity and specificity optimization.

## Project Overview

This project implements a binary classification model designed to achieve high accuracy with particular focus on:
- **Sensitivity (True Positive Rate)**: Correctly identifying positive cases
- **Specificity (True Negative Rate)**: Correctly identifying negative cases
- **Balanced performance**: Optimizing for both metrics simultaneously

## Project Structure

```
binary-classification-project/
│
├── data/                    # Data directory
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── external/            # External data sources
│
├── src/                     # Source code
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── model.py             # Model training and selection
│   └── evaluation.py        # Model evaluation utilities
│
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── docs/                    # Documentation
│
├── config.py                # Configuration parameters
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd binary-classification-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training a Model

```bash
python main.py --mode train --data-path data/train.csv
```

#### Evaluating a Model

```bash
python main.py --mode evaluate --model-path models/best_model.pkl --data-path data/test.csv
```

#### Making Predictions

```bash
python main.py --mode predict --model-path models/best_model.pkl --data-path data/new_data.csv
```

## Model Approach

### Key Features:
1. **Class Imbalance Handling**: Techniques to handle imbalanced datasets
2. **Feature Engineering**: Domain-specific feature creation
3. **Model Selection**: Comparison of multiple algorithms
4. **Hyperparameter Optimization**: Systematic tuning for optimal performance
5. **Threshold Optimization**: Finding the best decision threshold for sensitivity/specificity trade-off

### Evaluation Metrics:
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score
- AUC-ROC
- AUC-PR
- Matthews Correlation Coefficient (MCC)

## Configuration

Edit `config.py` to modify:
- Model hyperparameters
- Data paths
- Evaluation metrics
- Training parameters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- List any references, papers, or resources used
- Credit any contributors or inspirations
