"""
Setup script for Inverse Source Localization Package
"""

from setuptools import setup, find_packages

setup(
    name="inverse-source",
    version="0.1.0",
    author="Serdan",
    description="Inverse source localization using BEM and FEM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shaerdan/inverse_source_project",
    packages=["inverse_source"],
    package_dir={"inverse_source": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "cvxpy>=1.4",
        "ecos>=2.0",
        "gmsh>=4.12",
        "meshio>=5.3",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "inverse-source=inverse_source.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
