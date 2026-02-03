"""
Setup script for Inverse Source Localization Package

This file is kept for backward compatibility with older pip versions.
The canonical configuration is in pyproject.toml.

For development installation:
    pip install -e .

For full installation with all optional dependencies:
    pip install -e ".[full]"

Note: FEniCSx and cyipopt are best installed via conda:
    conda install -c conda-forge fenics-dolfinx cyipopt
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
