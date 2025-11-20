"""Setup configuration for PV Circularity Simulator."""

from setuptools import setup, find_packages

setup(
    name="pv-circularity-simulator",
    version="1.0.0",
    description="Comprehensive PV system lifecycle and circularity simulator",
    author="PV Circularity Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "altair>=5.0.0",
        "pvlib>=0.10.0",
        "numpy-financial>=1.0.0",
        "fastapi>=0.104.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.6.0",
            "ruff>=0.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
