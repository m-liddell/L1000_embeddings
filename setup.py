#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent


def read(rel_path):
    with open(here / rel_path) as fh:
        return fh.read()


setup(
    name="deep_embeddings",
    version="0.0.1", 
    description="Deep embeddings from L1000 dataset",
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
    author="you",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6, <4",
)
