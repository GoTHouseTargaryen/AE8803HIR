# AE8803 Hamiltonian Integration Research

This repository contains implementations and documentation for high-order symplectic integration methods for Hamiltonian systems, specifically applying Yoshida composition methods to the simple harmonic oscillator.

## 🚀 Quick Start

**No setup required!** Just run the script with Python 3.7+:

### Option 1: Interactive Quick Start (Recommended for First Time)
```powershell
python QUICKSTART.py
```
This will:
- Check your Python version
- Verify all files are present
- Optionally test the automatic setup
- Show you example commands

### Option 2: Direct Run
```powershell
# Windows PowerShell
python hw4_problem3.py --demo --plot --energy
```

```bash
# Linux/macOS
python3 hw4_problem3.py --demo --plot --energy
```

The script will automatically:
1. Create a virtual environment (`.venv`)
2. Install required packages (numpy, matplotlib)
3. Run the simulation and show results

## 📁 Repository Contents

### Getting Started
- **`QUICKSTART.py`** - Interactive setup verification and quick start guide
  - Checks Python version compatibility
  - Verifies all required files
  - Tests automatic environment setup
  - Shows example commands

### Main Scripts
- **`hw4_problem3.py`** - Main implementation with Yoshida integrators (orders 4, 6, 8) and RK4
  - Self-bootstrapping virtual environment
  - Interactive plotting with zoom controls
  - Comprehensive energy statistics
  - See [README_hw4_problem3.md](README_hw4_problem3.md) for detailed usage

- **`generate_plots_for_latex.py`** - Generates publication-quality plots for LaTeX
  - Creates individual 1×2 plots for each method
  - 300 DPI PNG output
  - Normalized relative energy error

### Documentation
- **`hw4_problem3_methodology.tex`** - Complete LaTeX document including:
  - Mathematical formulation
  - Yoshida composition theory
  - Implementation details with equations
  - Comprehensive results and analysis
  - Quantitative comparisons

- **`README_hw4_problem3.md`** - Detailed script documentation
  - Usage examples
  - Command-line flags
  - Output interpretation
  - Implementation notes

### Support Files
- **`requirements.txt`** - Python package dependencies (auto-managed)
- **`.gitignore`** - Excludes virtual environments and generated files

## 🎯 What This Does

Implements and compares numerical integration methods for the simple harmonic oscillator:

**Harmonic Oscillator:**
```
H(q, p) = 0.5(p² + q²)
dq/dt = p
dp/dt = -q
```

**Integration Methods:**
1. **RK4** - Classical 4th-order Runge-Kutta (non-symplectic baseline)
2. **Yoshida-4** - 4th-order symplectic integrator (3 stages)
3. **Yoshida-6** - 6th-order symplectic integrator (9 stages)
4. **Yoshida-8** - 8th-order symplectic integrator (27 stages)

**Key Results** (Δt=0.1, 10,000 steps, ~159 orbits):
- RK4: Max relative energy error = 1.4×10⁻⁴ (linear drift)
- Yoshida-4: Max relative energy error = 7.7×10⁻⁶ (bounded, 18× better)
- Yoshida-6: Max relative energy error = 9.2×10⁻⁸ (bounded, 1,500× better)
- Yoshida-8: Max relative energy error = 7.2×10⁻¹¹ (bounded, 1.9M× better)

## 📊 Example Usage

### Quick Demo (All Methods)
```powershell
python hw4_problem3.py --demo --plot --energy
```
Shows all 4 methods in a comparison grid with position and energy error plots.

### Single Method with Energy Stats
```powershell
python hw4_problem3.py --method y6 --dt 0.1 --steps 10000 --plot --energy
```

### Show Yoshida Coefficients
```powershell
python hw4_problem3.py --show-coefficients
```
Displays analytically computed composition weights and verifies order conditions.

### Generate LaTeX Plots
```powershell
python generate_plots_for_latex.py
```
Creates `rk4_results.png`, `yoshida4_results.png`, `yoshida6_results.png`, `yoshida8_results.png`

### Compile LaTeX Document
```powershell
pdflatex hw4_problem3_methodology.tex
pdflatex hw4_problem3_methodology.tex  # Run twice for references
```

## 🔬 Key Features

### Automatic Environment Setup
- ✅ No manual virtual environment creation
- ✅ No manual package installation
- ✅ Works on Windows, Linux, macOS
- ✅ Just requires Python 3.7+ with pip

### Symplectic Integration
- ✅ Preserves Hamiltonian structure
- ✅ Bounded energy errors (no secular drift)
- ✅ Long-term stability for conservative systems
- ✅ Analytical coefficient computation (no hardcoding)

### Comprehensive Analysis
- ✅ Normalized relative energy error (|ΔH/E₀|)
- ✅ Statistical metrics (mean, std dev, max, RMS)
- ✅ Interactive plots with zoom and pan
- ✅ Publication-quality figure generation

## 📚 Mathematical Background

Yoshida methods use **operator splitting** to compose symplectic operators:

```
S₂ₖ(h) = S₂ₖ₋₂(w₁h) ∘ S₂ₖ₋₂(w₀h) ∘ S₂ₖ₋₂(w₁h)
```

Where composition weights satisfy:
- **Consistency:** Σwᵢ = 1
- **Order conditions:** Σwᵢ^(2j+1) = 0 for j=1,2,...,k-1

This recursive composition achieves arbitrarily high even orders while maintaining symplecticity.

See the LaTeX document for complete mathematical derivation.

## 🎓 Educational Use

This code demonstrates:
- **Geometric numerical integration** - Structure-preserving methods
- **Operator splitting** - Decomposing Hamiltonian evolution
- **Symplectic vs non-symplectic** - Why structure preservation matters
- **High-order methods** - Recursive composition techniques
- **Long-term stability** - Bounded vs unbounded error accumulation

## 🔧 System Requirements

- **Python**: 3.7 or higher
- **pip**: Package installer (usually included with Python)
- **Internet**: First run only (to download numpy and matplotlib)

That's it! No other dependencies or configuration needed.

## 📖 References

1. H. Yoshida, "Construction of higher order symplectic integrators", *Physics Letters A* **150**, 262–268 (1990)
2. M. Suzuki, "Fractal decomposition of exponential operators", *Physics Letters A* **146**, 319–323 (1990)
3. E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2nd ed., Springer (2006)

## 📝 License

Educational use for AE8803 coursework.

---

**Author**: Alan Y  
**Course**: AE8803 - Hamiltonian Integration Research  
**Last Updated**: November 20, 2025
