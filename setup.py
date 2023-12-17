#!/usr/bin/env python
"""Setup script for the package."""
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ppg_project",
    version="0.1.0",
    author="Bastien Orset",
    author_email="bastien.orset@gmail.com",
    description="PPG challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "xgboost",
        "heartpy",
        "PyWavelets",
        "tqdm",
        "torch",
        "torchvision",
        "torchsummary"
    ],
    python_requires=">=3.10",
)
