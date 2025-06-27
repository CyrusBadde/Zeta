# ζ-Regularization Framework - Installation & Usage Guide

## 🚀 Quick Start (30 seconds)

```bash
# 1. Clone or download the repository
git clone https://github.com/zeta-physics/zeta-regularization.git
cd zeta-regularization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive GUI
python main.py
```

**That's it!** The GUI will open and you can start exploring ζ-regularization immediately.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.7 or higher
- **RAM**: 512 MB
- **Storage**: 100 MB
- **OS**: Windows 7+, macOS 10.12+, Linux (any distribution)

### Recommended
- **Python**: 3.9 or higher
- **RAM**: 2 GB (for large field simulations)
- **Storage**: 500 MB (including examples and documentation)
- **Display**: 1280x720 or higher (for GUI)

## 📦 Installation Options

### Option 1: Quick Setup (Recommended)

Download the complete package and install dependencies:

```bash
# Download (choose one):
git clone https://github.com/zeta-physics/zeta-regularization.git
# OR download ZIP from GitHub and extract

cd zeta-regularization
pip install -r requirements.txt
```

### Option 2: Full Development Setup

For contributors and advanced users:

```bash
git clone https://github.com/zeta-physics/zeta-regularization.git
cd zeta-regularization

# Install in development mode with all extras
pip install -e .[dev,gui,examples]

# Run tests to verify installation
python main.py --test
```

### Option 3: Minimal Installation

Just the core algebra without GUI or examples:

```bash
# Create minimal setup
mkdir zeta-minimal
cd zeta-minimal

# Copy only essential files:
# - src/zeta_core/algebra.py
# - requirements.txt (numpy and matplotlib only)

pip install numpy matplotlib
```

## 🗂️ Repository Structure

Here's what you get:

```
zeta-regularization/
├── main.py                      # 🚀 Main launcher - START HERE
├── zeta_gui.py                  # 🖥️ Interactive GUI application
├── requirements.txt             # 📋 Dependencies
├── setup.py                     # 📦 Package installation
├── README.md                    # 📖 Main documentation
├── INSTALL.md                   # 📋 This installation guide
├── LICENSE                      # ⚖️ MIT license
├── src/                         # 💻 Source code
│   ├── zeta_core/               # 🧮 Core algebraic structures
│   │   ├── algebra.py           #     ZetaSymbol, ZetaExtendedNumber
│   │   └── semiring.py          #     Mathematical verification
│   ├── field_theory/            # 🌊 Field theory simulations
│   │   ├── zeta_fields.py       #     Multi-dimensional evolution
│   │   ├── qft_calculations.py  #     Loop integrals
│   │   └── conservation.py      #     Conservation laws
│   ├── holography/              # 🔗 Holographic applications
│   │   └── ads_cft.py           #     Tropical AdS/CFT
│   └── experiments/             # 🧪 Experimental predictions
│       └── predictions.py       #     Testable signatures
├── examples/                    # 📚 Tutorials and examples
│   ├── basic_tutorial.ipynb     #     Interactive introduction
│   ├── qft_calculations.ipynb   #     QFT applications
│   ├── field_simulations.ipynb  #     Field theory examples
│   └── holographic_apps.ipynb   #     Holography examples
├── tests/                       # 🧪 Comprehensive test suite
│   ├── test_algebra.py          #     Core algebra tests
│   ├── test_fields.py           #     Field theory tests
│   └── test_reconstruction.py   #     ζ-reconstruction tests
├── docs/                        # 📖 Documentation
│   ├── mathematical_background.md
│   ├── physical_interpretation.md
│   ├── computational_guide.md
│   └── api_reference.md
└── paper/                       # 📄 Research paper
    ├── main.tex                 #     Complete mathematical development
    └── figures/                 #     Paper figures and plots
```

## 🖥️ Usage Options

The framework provides multiple ways to interact with ζ-regularization:

### 1. Interactive GUI (Recommended for beginners)

```bash
python main.py
# OR
python main.py --gui
```

**Features:**
- Visual ζ-algebra operations
- Real-time field theory simulations
- Interactive semiring verification
- Educational tutorials
- Export results and plots

### 2. Command Line Demo

```bash
python main.py --demo
```

**What it does:**
- Demonstrates basic ζ-algebra
- Verifies semiring properties
- Runs a field theory simulation
- Shows conservation laws

### 3. Jupyter Tutorials

```bash
python main.py --tutorial
# OR manually:
jupyter notebook examples/basic_tutorial.ipynb
```

**Available tutorials:**
- `basic_tutorial.ipynb` - Introduction to ζ-algebra
- `qft_calculations.ipynb` - Loop integrals and renormalization
- `field_simulations.ipynb` - Multi-dimensional field evolution
- `holographic_applications.ipynb` - AdS/CFT correspondence

### 4. Python API (For advanced users)

```python
from zeta_core.algebra import zeta, finite
from field_theory.zeta_fields import ZetaField, FieldConfiguration

# Basic ζ-algebra
a = finite(2.5)
za = zeta(3.0, "physics_source")
result = a + za  # Tropical addition

# Field theory
config = FieldConfiguration(dimensions=2, grid_size=128)
field = ZetaField(config)
field.set_initial_condition("gaussian", add_zeta=True)
field.evolve(total_time=5.0)
```

### 5. Testing and Benchmarks

```bash
# Run all tests
python main.py --test

# Performance benchmark
python main.py --benchmark

# Specific test file (if pytest installed)
pytest tests/test_algebra.py -v
```

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
# Ensure you're in the right directory
cd zeta-regularization
ls  # Should see main.py, src/, etc.

# Check Python path
python -c "import sys; print(sys.path)"
```

**2. GUI won't start**
```bash
# Check tkinter installation
python -c "import tkinter; print('✓ tkinter available')"

# On Ubuntu/Debian, may need:
sudo apt-get install python3-tk

# Try command line demo instead:
python main.py --demo
```

**3. Matplotlib display issues**
```bash
# Set backend for headless systems
export MPLBACKEND=Agg
python main.py --demo

# Or install GUI backend:
pip install PyQt5  # or TkAgg
```

**4. Permission errors**
```bash
# Use virtual environment
python -m venv zeta_env
source zeta_env/bin/activate  # Linux/macOS
# OR
zeta_env\Scripts\activate  # Windows

pip install -r requirements.txt
```

**5. Jupyter notebook won't start**
```bash
# Install Jupyter
pip install jupyter notebook

# Launch manually
cd examples
jupyter notebook basic_tutorial.ipynb
```

### Dependency Conflicts

If you have dependency conflicts:

```bash
# Create clean environment
python -m venv fresh_env
source fresh_env/bin/activate

# Install only what's needed
pip install numpy>=1.19.0 matplotlib>=3.3.0

# Test core functionality
python -c "from src.zeta_core.algebra import zeta, finite; print('✓ Core works')"
```

### Performance Issues

For large simulations:

```python
# Reduce grid size for testing
config = FieldConfiguration(
    dimensions=1,        # Start with 1D
    grid_size=32,       # Smaller grid
    time_step=0.01      # Larger time step
)

# Use efficient methods
field.evolve(1.0, method="euler", save_history=False)
```

## 🧪 Verification

Verify your installation:

```bash
# Quick verification
python main.py --demo

# Comprehensive verification
python main.py --test

# Expected output:
# ✓ All semiring properties verified
# ✓ ζ-reconstruction working perfectly
# ✓ Field evolution stable
# ✓ Conservation laws maintained
```

## 🎓 Learning Path

**New to ζ-regularization?** Follow this learning path:

1. **Start here**: `python main.py` (GUI exploration)
2. **Mathematics**: Read `docs/mathematical_background.md`
3. **Hands-on**: `jupyter notebook examples/basic_tutorial.ipynb`
4. **Physics**: `jupyter notebook examples/qft_calculations.ipynb`
5. **Advanced**: `jupyter notebook examples/field_simulations.ipynb`
6. **Research**: Read `paper/main.tex`

**Already familiar with tropical geometry or QFT?** Jump to:
- `examples/qft_calculations.ipynb` for immediate applications
- `src/field_theory/zeta_fields.py` for implementation details
- `paper/main.tex` for complete mathematical development

## 🔄 Updates

Keep your installation current:

```bash
# If using git
git pull origin main
pip install -r requirements.txt

# Check for updates
python main.py --help  # Shows current version
```

## 🆘 Getting Help

**Quick help:**
```bash
python main.py --help
```

**Documentation:**
- `README.md` - Overview and features
- `docs/` - Detailed documentation
- `examples/` - Interactive tutorials

**Community:**
- GitHub Issues: Report bugs or ask questions
- GitHub Discussions: General discussion and ideas
- Email: cyrusbadde@protonmail.com
**Bug reports should include:**
- Python version: `python --version`
- Operating system
- Error message (full traceback)
- Steps to reproduce

## ✅ Success Checklist

Your installation is successful if:

- [ ] `python main.py` launches without errors
- [ ] GUI opens and displays ζ-algebra tab
- [ ] `python main.py --demo` runs and shows semiring verification
- [ ] `python main.py --test` passes all tests
- [ ] Basic algebra works: `python -c "from src.zeta_core.algebra import zeta; print(zeta(2.5, 'test'))"`

If all items are checked, you're ready to explore ζ-regularization!

## 🎉 Welcome!

You now have a complete, working installation of the ζ-regularization framework. This represents a new approach to fundamental physics that preserves information traditionally lost in regularization procedures.

**What's next?**
- Explore the GUI to get familiar with ζ-algebra
- Try the tutorials to understand the physics
- Run your own field theory simulations
- Contribute to the project on GitHub

**Remember:** "Preserving the infinite to understand the finite"

Happy exploring! 🌟
