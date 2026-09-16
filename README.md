# LightAtlas

[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://hsynylmz-umhg.github.io/lightatlas/)
[![CI](https://github.com/hsynylmz-umhg/lightatlas/actions/workflows/ci.yml/badge.svg)](https://github.com/hsynylmz-umhg/lightatlas/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-throughput astronomical light curve analysis and photometric classification engine.

## Overview

LightAtlas provides scalable, reproducible tooling for processing time-series photometry, extracting variability features, and training machine learning / deep learning classifiers for astronomical transients and variable stars.

## Features

- **PEP 621 Standard**: Modern packaging with setuptools-scm dynamic versioning.
- **High Performance**: Optimized tabular data and array workflows with NumPy, SciPy, and PyArrow.
- **Flexible Extras**: Modular dependency groups for astrophysics tooling (`astro`), deep learning (`torch`), clustering (`cluster`), and development (`dev`).
- **Comprehensive Quality Gates**: Enforced linting and formatting via Ruff and rigorous test coverage.

## Installation

### From Source (Development Mode)

```bash
git clone https://github.com/hsynylmz-umhg/lightatlas.git
cd lightatlas
uv venv
uv pip install -e ".[dev]"
```

To install optional extras (e.g., astrophysics or torch support):

```bash
uv pip install -e ".[astro,torch,cluster,dev]"
```

## Quick Start

```python
import lightatlas

print(f"LightAtlas version: {lightatlas.__version__}")
```

## Running Tests & Checks

```bash
# Linting
ruff check .

# Formatting check
ruff format --check .

# Test suite with coverage
pytest -q
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
