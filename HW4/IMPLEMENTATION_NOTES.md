# Implementation Notes: Analytical Yoshida Coefficients & Interactive Plotting

## Summary of Changes
1. The script computes Yoshida symplectic integrator coefficients **analytically** using the recursive composition method (Suzuki fractal decomposition) rather than hard-coding numerical values.
2. Interactive plotting with GUI zoom/pan controls and linked axes across subplots.
3. Streamlined plot layout showing only position q(t) and energy error ΔH(t) (removed momentum and phase space plots).
4. Per-method energy error auto-scaling in demo mode for clear visualization of differences.

## Mathematical Foundation

### Symmetric Composition Formula
For order 2k, we recursively compose the order (2k-2) scheme:

```
S_2k(h) = S_{2k-2}(w1·h) · S_{2k-2}(w0·h) · S_{2k-2}(w1·h)
```

where the composition weights are:
```
w1 = 1 / (2 - 2^(1/(2k-1)))
w0 = 1 - 2·w1
```

### Order Conditions
The coefficients {w_i} must satisfy:
1. **Normalization**: Σw_i = 1
2. **Odd-power cancellation**: Σw_i^(2j+1) = 0 for j = 1, 2, ..., k-1

These conditions ensure the local truncation error is O(h^(2k+1)).

## Implementation Details

### Function: `compute_yoshida_coefficients(order)`
- **Order 2**: Returns `[1.0]` (base leapfrog)
- **Order 4**: Direct computation using w1, w0, w1
- **Order 6**: Recursively composes order-4 scheme → 9 stages (3×3)
- **Order 8**: Recursively composes order-6 scheme → 27 stages (3×3×3)

### Caching
Computed coefficients are cached in `_YOSHIDA_COEFFS_CACHE` to avoid recomputation.

### Verification Function: `verify_yoshida_order_conditions(order, coeffs)`
Computes residuals for:
- `sum_w`: should be ~0 (actually 1 - Σw_i)
- `sum_w3`, `sum_w5`, `sum_w7`: should be ~0 for respective orders

## Validation Results

### Coefficient Verification (via `--show-coefficients`)
All order conditions satisfied to **machine precision** (~1e-13 to 1e-15).

### Energy Conservation (Example: 10000 steps, dt=0.1)
| Method   | max\|ΔE\|   | RMS(ΔE)     | Behavior             |
|----------|-------------|-------------|----------------------|
| Yoshida4 | ~1e-9       | ~5e-10      | Bounded oscillatory  |
| Yoshida6 | ~1e-13      | ~5e-14      | Near machine ε       |
| Yoshida8 | ~1e-14      | ~5e-15      | At machine precision |
| RK4      | ~1e-5       | ~5e-6       | Linear drift         |

**Key Observations**:
- **Symplectic methods**: Energy error remains bounded and oscillatory (preserves modified Hamiltonian H*)
- **RK4**: Shows linear drift ΔH ≈ C·t, **not exponential**
  - The harmonic oscillator is stable and integrable
  - RK4 introduces small systematic bias per period
  - Accumulation is linear in time for this system
  - Exponential growth would require unstable dynamics or numerical instability

### Why RK4 Energy Drift is Linear
For the harmonic oscillator:
- True solution is periodic and bounded
- RK4 is 4th-order accurate (O(h⁴) global error)
- Each oscillation period introduces a small energy change δE ≈ constant
- Over N periods: ΔE_total ≈ N·δE ≈ (t/T)·δE ∝ t
- Result: **linear drift**, not exponential blowup

Exponential error growth only occurs for:
- Chaotic/unstable systems (positive Lyapunov exponents)
- Numerical instability (timestep too large)
- Neither applies to the harmonic oscillator with moderate dt

## Key Advantages of Analytical Computation

1. **Transparency**: Order conditions are explicit in code (educational value)
2. **Flexibility**: Easy to extend to arbitrary even orders 2k
3. **Reproducibility**: No reliance on external coefficient tables
4. **Numerical Accuracy**: Computed in full double precision without transcription errors

## Plotting Implementation

### Layout Changes
- **Removed**: Momentum p(t) and phase space (q vs p) plots
- **Kept**: Position q(t) and energy error ΔH(t) only
- **Demo mode**: N×2 grid (N methods, 2 columns)
- **Single-method mode**: 1×2 layout

### Interactive Features
1. **Linked axes**: Time axes shared across q(t) and ΔH(t) plots
   - Zoom/pan in one subplot updates both
   - Demo mode: all time-series plots linked across rows
2. **GUI zoom controls**: Standard Matplotlib toolbar (magnifying glass, pan)
3. **Reset hotkey**: Press `r` to restore initial axis limits
4. **Per-method scaling**: Each method's ΔH plot auto-scales to its error envelope
   - Makes order differences clearly visible
   - Can override with `--delim` CLI flag

### Technical Details
- Uses `ax.sharex()` for cross-version Matplotlib compatibility
- Stores default limits in closure for reset functionality
- Keyboard event handler via `fig.canvas.mpl_connect('key_press_event', on_key)`

## Command-Line Zoom Options
Optional axis limit overrides (format: `min,max` or `min:max`):
- `--tlim`: time axis
- `--qylim`: position y-axis
- `--delim`: energy error y-axis (overrides per-method auto-scaling)

Example:
```powershell
python hw4_problem3.py --demo --dt 0.1 --steps 10000 --plot --tlim 0,100 --delim -1e-6,1e-6
```

## Virtual Environment Bootstrap
The script auto-creates `.venv` and installs dependencies on first run:
1. Checks for `numpy` and `matplotlib`
2. Creates virtual environment if packages missing
3. Installs requirements
4. Re-executes script inside venv
No manual `pip install` needed.

## Reference
H. Yoshida, "Construction of higher order symplectic integrators", *Physics Letters A*, **150** (1990) 262–268.

---
Last updated: 2025-11-13
