# setup.py

from setuptools import setup, find_packages

setup(
    name="lagrar",
    version="0.1.0",
    description="LaGRAR: Task-aware Latent Graph Rewiring Can Robustly Solve Oversquashing-Oversmoothing Dilemma",
    author="Siddhant Saxena",
    author_email="mrsiddy.py@gmail.com",
    url="https://github.com/mr-siddy/LaGRAR",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "torch-scatter>=2.0.9",
        "torch-sparse>=0.6.13",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "POT>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
        ],
        "experiments": [
            "optuna>=2.10.0",
            "wandb>=0.12.0",
        ],
    },
)