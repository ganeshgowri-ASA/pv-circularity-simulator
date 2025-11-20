"""
Setup script for PV Circularity Simulator.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="pv-circularity-simulator",
    version="0.1.0",
    author="PV Circularity Simulator Team",
    author_email="team@pvcs.example.com",
    description="Comprehensive photovoltaic system lifecycle analysis platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/pv-circularity-simulator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pydantic>=2.5.0",
        "pvlib-python>=0.10.0",
        "pyvista>=0.42.0",
        "plotly>=5.18.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "streamlit>=1.29.0",
        "tzdata>=2023.3",
        "python-dateutil>=2.8.2",
        "trimesh>=4.0.0",
        "scikit-optimize>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pylint>=3.0.0",
            "ipython>=8.18.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pvcs-shade-analysis=pv_circularity_simulator.batch4.b05_system_design.s03_helioscope_shade_analysis.ui:main",
        ],
    },
)
