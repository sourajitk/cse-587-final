# RECAP: Reaction Completion for Biochemical Pathways

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**RECAP** (REaction Completion for Automated Pathways) is a deep learning framework for predicting missing components in biochemical reactions. Built on ReactionT5v2 and fine-tuned on MetaNetX, RECAP can complete partial reaction equations by predicting missing substrates, products, or cofactors.

## Key Features

- **Reaction Completion**: Predict missing reactants, products, or cofactors
- **Multiple Completion Modes**: Support for product prediction, retrosynthesis, and bidirectional completion
- **Biochemical Focus**: Trained on MetaNetX biochemical reaction database
- **Flexible Input**: Accept SMILES, reaction SMARTS, or enzyme EC numbers
- **Confidence Scoring**: Uncertainty quantification for predictions

## Installation

### From PyPI (recommended)

```bash
pip install recap-biochem
```

### From Source

```bash
git clone https://github.com/yourusername/recap.git
cd recap
pip install -e .
```

### With GPU Support

```bash
pip install recap-biochem[gpu]
```

## Quick Start

### Command Line

```bash
# Predict products from substrates
recap complete --substrates "CC(=O)SCoA.O=C([O-])CC(=O)[O-]" --mode forward

# Retrosynthesis - predict substrates from products
recap complete --products "CC(=O)CC(=O)SCoA" --mode retro

# Complete reaction with partial information
recap complete --partial "CC(=O)SCoA.? >> CC(=O)CC(=O)SCoA" --mode fill
```

### Python API

```python
from recap import ReactionCompleter

# Initialize model
completer = ReactionCompleter.from_pretrained("recap-base")

# Forward prediction (substrates → products)
result = completer.complete(
    substrates=["CC(=O)SCoA", "O=C([O-])CC(=O)[O-]"],
    mode="forward"
)
print(result.products)  # Predicted products
print(result.confidence)  # Prediction confidence

# Retrosynthesis (products → substrates)
result = completer.complete(
    products=["CC(=O)CC(=O)SCoA"],
    mode="retro"
)
print(result.substrates)

# Fill missing components
result = completer.complete(
    partial_reaction="ATP.? >> ADP.?",
    mode="fill"
)
print(result.complete_reaction)
```

### With EC Number Context

```python
# Use enzyme context for better predictions
result = completer.complete(
    substrates=["pyruvate", "NAD+"],
    ec_number="1.2.4.1",  # Pyruvate dehydrogenase
    mode="forward"
)
```

## Model Variants

| Model | Parameters | Description |
|-------|-----------|-------------|
| `recap-base` | 250M | Base model, good balance of speed and accuracy |
| `recap-large` | 780M | Higher accuracy, slower inference |
| `recap-fast` | 60M | Optimized for speed, suitable for screening |

## Training

### Prepare Data

```bash
# Download and preprocess MetaNetX data
python scripts/prepare_metanetx.py --version 4.4 --output data/metanetx

# Create training splits
python scripts/create_splits.py --input data/metanetx --output data/splits
```

### Fine-tune Model

```bash
# Fine-tune ReactionT5v2 on MetaNetX
python scripts/train.py \
    --config configs/train_base.yaml \
    --data_dir data/splits \
    --output_dir checkpoints/recap-base
```

### Evaluate

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint checkpoints/recap-base \
    --test_data data/splits/test.csv \
    --output results/evaluation.json
```

## Evaluation Metrics

RECAP is evaluated on multiple metrics:

- **Top-k Accuracy**: Correct prediction in top-k candidates
- **SMILES Validity**: Percentage of chemically valid predictions
- **Reaction Balance**: Atom conservation in predicted reactions
- **Biochemical Plausibility**: Consistency with known biochemistry

## Project Structure

```
recap/
├── recap/                  # Main package
│   ├── models/            # Model architectures
│   │   ├── reaction_t5.py # ReactionT5v2 wrapper
│   │   ├── completer.py   # Main completion model
│   │   └── heads.py       # Prediction heads
│   ├── data/              # Data processing
│   │   ├── metanetx.py    # MetaNetX loader
│   │   ├── tokenizer.py   # Reaction tokenizer
│   │   └── dataset.py     # PyTorch datasets
│   ├── evaluation/        # Evaluation metrics
│   │   ├── metrics.py     # Core metrics
│   │   └── benchmarks.py  # Benchmark suites
│   └── utils/             # Utilities
│       ├── chemistry.py   # RDKit utilities
│       └── io.py          # I/O helpers
├── scripts/               # Training and evaluation scripts
├── configs/               # Configuration files
├── notebooks/             # Example notebooks
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Citation

If you use RECAP in your research, please cite:

```bibtex
@article{recap2025,
  title={RECAP: Deep Learning for Biochemical Reaction Completion},
  author={Santra, Somtirtha and Maranas, Costas D.},
  journal={},
  year={2025}
}
```

## Related Work

- [ReactionT5](https://github.com/sagawatatsuya/ReactionT5) - Base model architecture
- [MetaNetX](https://www.metanetx.org/) - Biochemical reaction database
- [RXNMapper](https://github.com/rxn4chemistry/rxnmapper) - Atom mapping

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Maranas Group at Penn State for support and feedback
- ReactionT5 authors for the base model
- MetaNetX team for the comprehensive reaction database
