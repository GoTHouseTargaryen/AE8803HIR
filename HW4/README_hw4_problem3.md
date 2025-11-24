# HW4 Problem 3 – Yoshida High-Order Symplectic Integrators

This repository contains `hw4_problem3.py`, a Python script implementing Yoshida high-order symplectic integration methods (orders 4, 6, and 8) for the simple harmonic oscillator with unit mass and unit angular frequency. A classical (non-symplectic) RK4 method is also provided for comparison.

## Physical Model
Harmonic oscillator (all constants = 1):
\[ H(q,p) = \tfrac{1}{2} (p^2 + q^2), \quad \dot q = p, \quad \dot p = -q. \]

## Base Second-Order Method
We use the symmetric velocity-Verlet (leapfrog) integrator `S(h)`:
```
p_half = p + 0.5*h*force(q)
q_new  = q + h*p_half
p_new  = p_half + 0.5*h*force(q_new)
```
This method is time-reversible and symplectic (second order).

## Yoshida Composition
Higher-order symplectic schemes are obtained by composing `S(h)` with scaled substeps `S(w_i * h)` where the coefficients `w_i` satisfy algebraic order conditions (Yoshida 1990).

### Analytical Coefficient Calculation
The script computes coefficients **analytically** using recursive composition (Suzuki fractal decomposition):

```
S_2k(h) = S_{2k-2}(w1*h) · S_{2k-2}(w0*h) · S_{2k-2}(w1*h)
```

where:
```
w1 = 1 / (2 - 2^(1/(2k-1)))
w0 = 1 - 2*w1
```

This recursion ensures:
- **Order condition 1**: Σw_i = 1 (time-step consistency)
- **Order conditions 2k**: Σw_i^(2j+1) = 0 for j = 1,2,...,k-1 (cancels odd-order error terms)

**Stage counts**:
- Order 4: 3 stages
- Order 6: 9 stages (3×3 composition)
- Order 8: 27 stages (3×3×3 composition)

### Computed Coefficients (Sample - Order 4)
```
w1 =  1.351207191959657772
w0 = -1.702414383919315544
w1 =  1.351207191959657772
```
Negative coefficients represent **backward substeps**, which are mathematically valid in symplectic integrators.

Use `--show-coefficients` to print all orders and verify order conditions:
```powershell
python hw4_problem3.py --show-coefficients
```

## RK4 Reference
The classical 4th-order Runge–Kutta method integrates the ODE system but is not symplectic; long-time energy conservation is inferior compared to symplectic schemes.

## Script Usage
PowerShell examples:
```powershell
# Show analytically computed coefficients and verify order conditions
python hw4_problem3.py --show-coefficients

# Run single method
python hw4_problem3.py --method y4 --dt 0.01 --steps 10000 --q0 1.0 --p0 0.0 --plot --energy

# Compare all methods (demo mode)
python hw4_problem3.py --demo --dt 0.01 --steps 5000 --plot --energy

# RK4 comparison
python hw4_problem3.py --method rk4 --dt 0.01 --steps 2000 --energy

# With custom axis limits (optional)
python hw4_problem3.py --demo --dt 0.1 --steps 10000 --plot --tlim 0,100 --delim -1e-6,1e-6
```

### Core Flags
- `--method {y4,y6,y8,rk4}`: choose integrator (omit when using `--demo`).
- `--dt`: time step.
- `--steps`: number of integration steps.
- `--q0`, `--p0`: initial conditions (default: q0=1.0, p0=0.0).
- `--energy`: print energy statistics.
- `--plot`: generate plots (position vs time and energy error vs time).
- `--demo`: run all Yoshida orders plus RK4 and compare.
- `--show-coefficients`: display computed coefficients and order condition residuals.
- `--log-error`: plot energy error on logarithmic scale (for visualizing small errors).

### Zoom/Limit Options (Optional)
Fine-tune plot axes (format: `min,max` or `min:max`):
- `--tlim`: time axis limits
- `--qylim`: position y-axis limits
- `--delim`: energy error y-axis limits (overrides per-method auto-scaling in demo mode)

Note: Phase space and momentum plots have been removed; only position and energy error are shown.

## Output Metrics
For each run the script can report:
- `E0`: initial energy.
- `mean`, `std`: statistics of sampled energies.
- `max|dE/E0|`: maximum **normalized relative energy error** (dimensionless).
- `rms(dE/E0)`: root mean square normalized relative energy error.

**Important**: All energy error calculations use **normalized relative error**:
```
dE/E0 = |E(t) - E0| / |E0|
```
This dimensionless quantity allows fair comparison across different initial conditions and represents the fractional energy deviation (e.g., 1.0e-4 = 0.01% error).

## Plotting Features
The script generates interactive plots using Matplotlib:

### Plot Layout
- **Demo mode**: N×2 grid (one row per method)
  - Column 1: Position q(t) vs time
  - Column 2: Relative energy error |ΔH/E₀| vs time (normalized, dimensionless)
- **Single-method mode**: 1×2 layout
  - Left: Position q(t) vs time
  - Right: Relative energy error |ΔH/E₀| vs time (normalized, dimensionless)

### Interactive Controls
- **Zoom/Pan**: Use the Matplotlib toolbar (magnifying glass for zoom, hand for pan)
- **Linked axes**: Time axes are shared across position and error plots; zooming one updates both
- **Reset hotkey**: Press `r` to reset all axes to initial limits
- **Per-method scaling**: In demo mode, each method's energy error plot auto-scales to its own error envelope (unless overridden with `--delim`)

## Observations (Guidance for Report)
1. **Energy Conservation**: Symplectic Yoshida methods show bounded, quasi-periodic **relative** energy error that does not drift secularly; RK4 exhibits a **linear drift** over long times (not exponential—the harmonic oscillator is stable and integrable).

2. **Normalized Relative Error**: All energy errors are reported as |ΔH/E₀|, a dimensionless quantity representing fractional energy deviation. For example:
   - RK4 (dt=0.1, 10k steps): max |ΔH/E₀| ≈ 1.4×10⁻⁴ (0.014% error)
   - Yoshida-4: max |ΔH/E₀| ≈ 7.7×10⁻⁶ (0.00077% error) — 18× better
   - Yoshida-6: max |ΔH/E₀| ≈ 9.2×10⁻⁸ (0.0000092% error) — 1,500× better
   - Yoshida-8: max |ΔH/E₀| ≈ 7.2×10⁻¹¹ (near machine precision) — 1.9M× better

3. **Why RK4 drifts linearly (not exponentially)**:
   - The harmonic oscillator is stable; errors don't amplify exponentially
   - RK4 introduces a small systematic energy bias per oscillation period
   - Over N periods, this accumulates linearly: ΔH ≈ C·N ≈ C·t
   - Exponential growth would require an unstable system or numerical instability (neither applies here)

3. **Order vs. Stability**: Higher-order Yoshida methods (6, 8) allow larger step sizes for a target accuracy but require more force evaluations per step (number of sub-stages). Cost vs. accuracy trade-off should be discussed.

4. **Symplectic Structure**: Symplectic integrators preserve a "shadow Hamiltonian" H* close to the true H, resulting in bounded oscillatory error rather than secular drift.

5. **Coefficient Signs**: Negative coefficients (e.g., in order 6 and 8) imply backward substeps; mathematically valid for symplectic schemes but can amplify rounding errors in some contexts.

6. **Computational Cost**: Stage counts: 3 (order 4), 9 (order 6), 27 (order 8). Effective force evaluations per global step scale accordingly.

## Extending
- Add adaptive step size (not typical for symplectic splitting but possible with caution).
- Implement other symplectic schemes (Forest–Ruth, Suzuki, Omelyan). 
- Generalize to multi-dimensional oscillators / non-quadratic potentials.
- Add convergence study mode to plot error vs timestep on log-log scale.

## LaTeX Documentation
The repository includes:
- `hw4_problem3_methodology.tex`: Complete LaTeX document with mathematical formulation, implementation details, results, and analysis
- `generate_plots_for_latex.py`: Generates publication-quality individual method plots (300 DPI PNG files):
  - `rk4_results.png`: RK4 position and relative energy error
  - `yoshida4_results.png`: Yoshida-4 position and relative energy error
  - `yoshida6_results.png`: Yoshida-6 position and relative energy error
  - `yoshida8_results.png`: Yoshida-8 position and relative energy error

To generate LaTeX plots:
```powershell
python generate_plots_for_latex.py
```

All plots use normalized relative energy error (|ΔH/E₀|) for consistent comparison.

## Virtual Environment
The script automatically bootstraps a Python virtual environment (`.venv`) and installs required packages (`numpy`, `matplotlib`) on first run. No manual setup needed—just run:
```powershell
python hw4_problem3.py --demo --plot
```

## Reference
H. Yoshida, "Construction of higher order symplectic integrators", Physics Letters A, 150 (1990) 262–268.

## License
Educational use for HW4; no external license specified.

---
Last updated: 2025-11-20
