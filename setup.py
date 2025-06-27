"""
Setup script for ζ-Regularization Framework
==========================================

Installation script for the complete ζ-regularization package including
core algebra, field theory simulations, and interactive GUI.

Usage:
    pip install -e .
    python setup.py install
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ζ-Regularization Framework: Information-preserving quantum field theory"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['numpy>=1.19.0', 'matplotlib>=3.3.0', 'scipy>=1.5.0']

setup(
    name="zeta-regularization",
    version="1.0.0",
    author="ζ-Regularization Framework Team",
    author_email="zeta-physics@example.com",
    description="Information-preserving regularization for fundamental physics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zeta-physics/zeta-regularization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.7",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=3.0",
            "jupyter>=1.0",
            "notebook>=6.0"
        ],
        "gui": [
            "tkinter"  # Usually included with Python
        ],
        "examples": [
            "jupyter>=1.0",
            "notebook>=6.0",
            "plotly>=5.0",
            "seaborn>=0.11"
        ]
    },
    
    entry_points={
        "console_scripts": [
            "zeta-gui=zeta_gui:main",
            "zeta-demo=examples.basic_demo:main",
            "zeta-benchmark=tools.benchmark:main",
        ],
    },
    
    package_data={
        "": ["*.md", "*.txt", "*.json"],
        "docs": ["*.md", "*.rst"],
        "examples": ["*.ipynb", "*.py"],
    },
    
    include_package_data=True,
    zip_safe=False,
    
    project_urls={
        "Bug Reports": "https://github.com/zeta-physics/zeta-regularization/issues",
        "Source": "https://github.com/zeta-physics/zeta-regularization",
        "Documentation": "https://zeta-physics.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/zeta-regularization",
    },
    
    keywords=[
        "physics", "mathematics", "quantum field theory", "regularization", 
        "tropical geometry", "holography", "information theory", "singularities",
        "divergences", "semiring", "symbolic computation"
    ],
    
    # Test configuration
    test_suite="tests",
    
    # Documentation
    command_options={
        'build_sphinx': {
            'project': ('setup.py', 'ζ-Regularization'),
            'version': ('setup.py', '1.0'),
            'release': ('setup.py', '1.0.0'),
            'source_dir': ('setup.py', 'docs'),
        }
    },
)
