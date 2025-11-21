# LaTeX Document Generation Instructions

## Overview

A comprehensive LaTeX document (`hw4_problem3_methodology.tex`) has been created that details the methodology, equations, and implementation of the Yoshida symplectic integrators.

## Files Created

1. **hw4_problem3_methodology.tex** - Main LaTeX document with:
   - Complete mathematical formulation
   - Harmonic oscillator equations
   - Leapfrog/velocity-Verlet method
   - Yoshida recursive composition (orders 4, 6, 8)
   - Analytical coefficient derivations
   - RK4 comparison
   - Placeholders for plots and analysis sections

2. **generate_plots_for_latex.py** - Script to generate publication-quality PNG plots

## Steps to Generate Complete Document

### Step 1: Generate Plots

Run the plot generation script:

```powershell
python generate_plots_for_latex.py
```

This will create:
- `position_vs_time.png` - Position vs time comparison
- `energy_error_vs_time.png` - Energy error vs time comparison

The script will also print energy statistics for the table in the LaTeX document.

### Step 2: Fill in Analysis Sections

The LaTeX document has blank sections marked with horizontal lines for you to fill in:

1. **Position Plot Analysis** (Section 6.1)
2. **Energy Error Plot Analysis** (Section 6.2)
3. **Energy Statistics Table** (Section 6.3) - Fill in the numerical values
4. **Discussion Sections** (Section 7.1-7.3)
5. **Conclusions** (Section 8)

### Step 3: Compile LaTeX Document

Using your LaTeX distribution (TeXLive, MiKTeX, etc.):

```bash
pdflatex hw4_problem3_methodology.tex
pdflatex hw4_problem3_methodology.tex  # Run twice for references
```

Or use your preferred LaTeX editor (Overleaf, TeXstudio, etc.)

## Document Structure

### Complete Sections (No Editing Required)

- Section 1: Introduction
- Section 2: Problem Formulation (Hamiltonian, equations of motion)
- Section 3: Symplectic Integration (leapfrog method)
- Section 4: Yoshida Composition Method
  - Recursive formula
  - Order conditions
  - Construction of Y4, Y6, Y8 with exact coefficients
  - Implementation details
- Section 5: RK4 Comparison
- References

### Sections Requiring Your Input

1. **Section 6.1: Position Plot Analysis**
   - Describe what you observe in the position plot
   - Comment on oscillation behavior
   - Note any differences between methods (if visible)

2. **Section 6.2: Energy Error Plot Analysis**
   - Compare energy conservation for all four methods
   - Note the behavior of RK4 (linear drift)
   - Note the behavior of Yoshida methods (bounded oscillation)
   - Compare Y4 vs Y6 vs Y8 error magnitudes

3. **Section 6.3: Energy Statistics Table**
   - Fill in the table with values from the output of `generate_plots_for_latex.py`
   - The script prints: Mean, Std Dev, Max |ΔH|, Final ΔH for each method

4. **Section 7.1: Symplectic vs Non-Symplectic Discussion**
   - Explain why symplectic integrators preserve energy better
   - Contrast bounded oscillations vs linear drift

5. **Section 7.2: Order vs Computational Cost**
   - Discuss trade-offs between accuracy and computation
   - Comment on when higher-order methods are worth the cost

6. **Section 7.3: Practical Considerations**
   - When to use which method
   - Timestep selection guidelines

7. **Section 8: Conclusions**
   - Summarize key findings
   - Practical recommendations

## Key Equations Included

The document includes all critical equations:

- Hamiltonian: H(q,p) = 0.5(p² + q²)
- Equations of motion: dq/dt = p, dp/dt = -q
- Leapfrog steps (kick, drift, kick)
- Yoshida composition: S₂ₖ(h) = S₂ₖ₋₂(w₁h) ∘ S₂ₖ₋₂(w₀h) ∘ S₂ₖ₋₂(w₁h)
- Composition weights: w₁ = 1/(2 - 2^(1/(2k-1))), w₀ = 1 - 2w₁
- Numerical coefficients for orders 4, 6, 8

## Customization

You can modify:
- Plot parameters in `generate_plots_for_latex.py` (DPI, figure size, colors)
- Integration parameters (dt, steps) to generate different comparisons
- LaTeX formatting (fonts, margins, etc.) in the document preamble

## Required LaTeX Packages

The document uses standard packages (should be in most LaTeX distributions):
- amsmath, amssymb (math symbols)
- graphicx (images)
- float (figure placement)
- geometry (margins)
- hyperref (hyperlinks)
- listings, xcolor (code formatting, if needed)

## Questions?

Refer to the main implementation in `hw4_problem3.py` for computational details.
