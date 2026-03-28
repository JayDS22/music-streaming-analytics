from setuptools import setup, find_packages

setup(
    name="music-streaming-analytics",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0", "numpy>=1.24.0", "scipy>=1.10.0",
        "scikit-learn>=1.3.0", "pyyaml>=6.0", "loguru>=0.7.0"
    ],
    entry_points={"console_scripts": ["music-analytics=main:main"]},
)
