# Contributing to LightAtlas

Thank you for your interest in contributing to LightAtlas! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hsynylmz-umhg/lightatlas.git
   cd lightatlas
   ```

2. **Set up virtual environment with development dependencies:**
   ```bash
   uv venv
   uv pip install -e ".[dev,torch]"
   ```

## Code Quality Standards

We enforce strict quality gates using `ruff` and `pytest`:

- **Formatting:** Run `ruff format .` before submitting changes.
- **Linting:** Run `ruff check .` and ensure all checks pass without warnings.
- **Tests & Coverage:** Run `pytest -q`. All tests must pass and code coverage must remain >= 80%.

## Pull Request Workflow

1. Fork the repository and create a feature branch from `master`.
2. Commit your changes using conventional commit messages (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
3. Ensure all local quality checks pass:
   ```bash
   ruff check .
   ruff format --check .
   pytest -q
   ```
4. Push your branch and open a Pull Request. CI will run automated tests across Python 3.10-3.12 and PyTorch.
