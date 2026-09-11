# Contributing to rocm-qlora

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Bala-Mosay/ROCM-QLORA.git
cd ROCM-QLORA
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full test suite (212 tests)
pytest tests/ -v

# Quick smoke test (no GPU needed)
python smoke_test.py

# V5 HIP kernel tests
python smoke_test_v5.py
```

## Code Style

- Follow existing code conventions
- Add tests for new features
- Keep functions focused and small
- Use type hints where practical

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b fix/my-bug`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit with a clear message
6. Push and open a PR

## Reporting Issues

Open an issue on [GitHub Issues](https://github.com/Bala-Mosay/ROCM-QLORA/issues) with:
- What you expected
- What actually happened
- Steps to reproduce
- Your environment (ROCm version, GPU, Python version)
