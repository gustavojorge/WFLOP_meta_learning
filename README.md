# Meta-Learning for Algorithm Selection in Multi-Objective Optimization

This project implements a meta-learning system for automatic selection among three multi-objective optimization algorithms (MOEA/D, NSGA-II, and COMOLSD) using problem meta-features and quality indicators of Pareto fronts.

## Overview

The system automates the choice among MOEA/D, NSGA-II, and COMOLSD based on meta-features and trains predictive models for different indicators: additive epsilon (ε), hypervolume, and IGD.

### Evaluation Strategies

- **SBS (Single Best Solver)**: Globally best fixed algorithm
- **VBS (Virtual Best Solver)**: Oracle that always selects the best per instance (theoretical lower bound on loss)
- **AS (Algorithm Selector)**: Predictive ML model (multi-output Random Forest) attempting to approximate VBS

### Merit Metric

Primary metric: `m = (AS - VBS) / (SBS - VBS)`

Interpretation:
- `m = 0`: AS is perfect (equals VBS)
- `m = 1`: AS is useless (equals SBS)
- `0 < m < 1`: AS improves over SBS but not optimal

## Project Structure (Main Folders)

```
meta_learning/
├── pipeline.py                    # Main pipeline orchestration
├── components/
│   ├── build_indices_dict.py      # Dynamic generation of best-algorithm indices per instance
│   ├── merit_table_builder.py     # Merit table construction (calls models_and_merit_builder)
│   ├── models_and_merit_builder.py# Training + merit computation per configuration
│   ├── regression_metrics.py      # Regression metrics (AS vs VBS)
│   └── figure_builder.py          # Visualizations (feature importance)
└── utils/
    ├── test_model.py              # Testing saved models
    ├── build_datasets/            # Dataset construction
    └── ParamsSearch/              # Hyperparameter search
```

## Installation

1. Install Poetry (if not installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd meta_learning
poetry install
```

## Usage

### Run the full pipeline (merits, metrics, figures, indices JSON):
```bash
# Default indicator (epsilon)
poetry run python pipeline.py

# Other indicators
poetry run python pipeline.py --label hipervolume
poetry run python pipeline.py --label igd
```

During execution the pipeline also generates a JSON listing instances where each algorithm is best (criterion: minimal indicator value):
```
result/<label>/indices/<label>_dict.json
```

### Run individual components:
```bash
poetry run python components/merit_table_builder.py epsilon   # Merit table

poetry run python components/regression_metrics.py epsilon    # AS vs VBS metrics

poetry run python components/figure_builder.py epsilon        # Feature importance figures

# Manually generate indices JSON
poetry run python components/build_indices_dict.py --print-python
poetry run python components/build_indices_dict.py --json result/epsilon/indices/epsilon_dict.json
```

### Test a saved model:
```bash
poetry run python utils/test_model.py
```

## Output Structure

The pipeline produces the following:

1. `result/<label>/merit_<label>.csv` – merit table (m) per configuration (ℓ, r, Adaptive Walk)
2. `result/<label>/models/theoretical_models/` – VBS, SBS, AS predictions (CSV per configuration)
3. `result/<label>/models/pickle_models/` – saved Random Forest models
4. `result/<label>/features_importance/` – feature importances per configuration
5. `result/<label>/regression_metrics/` – metrics (MSE, RMSE, MAE, MAPE, R²)
6. `result/<label>/indices/<label>_dict.json` – instances where each algorithm is best
7. `result/<label>/logs/` – subprocess execution logs
8. `result/<label>/figures/` – aggregated figures (baseline vs best configuration)

## Key Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **scikit-learn**: Machine learning models
- **scipy**: Scientific utilities
- **matplotlib / seaborn**: Visualization

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request