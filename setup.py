"""
Setup script for inverse_source package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inverse_source",
    version="1.0.0",
    author="Inverse Source Project",
    description="Inverse source localization for Poisson equation using BEM and FEM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shaerdan/inverse_source_project",
    packages=find_packages(),
    package_dir={"inverse_source": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
    ],
    extras_require={
        "fem": [
            "fenics-dolfinx>=0.8",
            "mpi4py",
            "petsc4py",
            "gmsh",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
)
