"""
Build & install script for the PistaDB Python package.

Usage:
    # From the project root, after building the C library with CMake:
    pip install -e python/
    # or
    pip install python/
"""
from setuptools import setup, find_packages

setup(
    name="pistadb",
    version="1.0.0",
    description="Lightweight embedded vector database – Python bindings",
    author="PistaDB",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    extras_require={
        "dev": ["pytest", "scipy"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
