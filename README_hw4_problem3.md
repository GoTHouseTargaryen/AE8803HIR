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
```
Flags:
- `--method {y4,y6,y8,rk4}`: choose integrator (omit when using `--demo`).
- `--dt`: time step.
- `--steps`: number of integration steps.
- `--q0`, `--p0`: initial conditions.
- `--energy`: print energy statistics.
- `--plot`: generate phase space, q(t), energy and energy deviation plots.
- `--demo`: run all Yoshida orders plus RK4 and summarize.
- `--show-coefficients`: display computed coefficients and order condition residuals.

## Output Metrics
For each run the script can report:
- `E0`: initial energy.
- `mean`, `std`: statistics of sampled energies.
- `max|dE|`: maximum absolute deviation from initial energy.
- `rms(dE)`: root mean square energy deviation.

## Observations (Guidance for Report)
1. **Energy Conservation**: Symplectic Yoshida methods show bounded, quasi-periodic energy error that does not drift secularly; RK4 exhibits a slow drift over long times (depending on `dt`).
2. **Order vs. Stability**: Higher-order Yoshida methods (6, 8) allow larger step sizes for a target accuracy but require more force evaluations per step (number of sub-stages). Cost vs. accuracy trade-off should be discussed.
3. **Phase Space**: Symplectic integrators preserve the elliptical orbit shape over long times; RK4 slowly deforms the ellipse (amplitude drift).
4. **Coefficient Signs**: Negative coefficients (e.g., in 6th and 8th order) imply backward substeps; mathematically fine but can amplify rounding errors for stiff problems.
5. **Computational Cost**: Stage counts: 3 (order 4), 7 (order 6), 15 (order 8). Effective force evaluations per global step scale accordingly.

## Extending
- Add adaptive step size (not typical for symplectic splitting but possible with caution).
- Implement other symplectic schemes (Forest–Ruth, Suzuki, Omelyan). 
- Generalize to multi-dimensional oscillators / non-quadratic potentials.

## Reference
H. Yoshida, "Construction of higher order symplectic integrators", Physics Letters A, 150 (1990) 262–268.

## License
Educational use for HW4; no external license specified.
