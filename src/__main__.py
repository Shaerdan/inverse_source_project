#!/usr/bin/env python
"""
Entry point for running the package as a module.

Usage:
    python -m inverse_source --help
    python -m inverse_source compare --domains disk ellipse
    python -m inverse_source demo --type bem
"""

from .cli import main

if __name__ == "__main__":
    main()
