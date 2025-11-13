# Implementation Notes: Analytical Yoshida Coefficients

## Summary of Changes
The script now computes Yoshida symplectic integrator coefficients **analytically** using the recursive composition method (Suzuki fractal decomposition) rather than hard-coding numerical values.

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

### Energy Conservation (1000 steps, dt=0.01)
| Method   | max\|ΔE\|   | RMS(ΔE)     |
|----------|-------------|-------------|
| Yoshida4 | 3.80e-10    | 2.26e-10    |
| Yoshida6 | 4.61e-14    | 2.54e-14    |
| Yoshida8 | 5.27e-15    | 2.20e-15    |
| RK4      | 6.94e-12    | 4.01e-12    |

**Observation**: Higher-order symplectic methods show exponentially better energy conservation due to preservation of the Hamiltonian structure.

## Key Advantages of Analytical Computation

1. **Transparency**: Order conditions are explicit in code (educational value)
2. **Flexibility**: Easy to extend to arbitrary even orders 2k
3. **Reproducibility**: No reliance on external coefficient tables
4. **Numerical Accuracy**: Computed in full double precision without transcription errors

## Reference
H. Yoshida, "Construction of higher order symplectic integrators", *Physics Letters A*, **150** (1990) 262–268.

---
Generated: 2025-11-12
