"""Quick stability test for the fixed integrator"""
import HW5.hw5_simulation as hw5_simulation

print("Testing fixed integrator with 1-year simulation...")

# Locate a simulation entry point in the hw5_simulation module and call it.
# Try common names to be robust if the function was named differently.
_sim_fn = None
for name in ("run_simulation", "simulate", "run", "runSimulation"):
    if hasattr(hw5_simulation, name):
        _sim_fn = getattr(hw5_simulation, name)
        break

if _sim_fn is None:
    raise AttributeError(
        "HW5.hw5_simulation does not expose a simulation function; "
        "expected one of: run_simulation, simulate, run, runSimulation"
    )

results = _sim_fn(duration_years=1.0, dt_days=0.001)

print(f"\n=== STABILITY TEST RESULTS ===")
print(f"Final relative energy error: {results['final_dE']:.3e}")
print(f"Final relative angular momentum errors: L1={results['final_dL1']:.3e}, L2={results['final_dL2']:.3e}, L3={results['final_dL3']:.3e}")

if results['final_dE'] < 1e-3:
    print("\n✓ PASS: Energy error is acceptable (< 1e-3)")
else:
    print(f"\n✗ FAIL: Energy error too large")
