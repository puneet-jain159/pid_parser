# Tests

This directory contains unit tests and integration tests for the PID Parser project.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pid_parser --cov-report=html

# Run specific test file
pytest tests/test_specific.py
```

## Test Structure

- Unit tests for individual functions
- Integration tests for pipeline components
- Mock tests for external API calls

