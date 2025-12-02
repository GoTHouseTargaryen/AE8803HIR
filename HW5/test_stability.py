"""Quick stability test for the fixed integrator"""
import hw5_simulation

print("Testing fixed integrator with 1-year simulation...")
results = hw5_simulation.run_simulation(duration_years=1.0, dt_days=0.001)
print(f"\n=== STABILITY TEST RESULTS ===")
print(f"Final relative energy error: {results['final_dE']:.3e}")
print(f"Final relative angular momentum errors: L1={results['final_dL1']:.3e}, L2={results['final_dL2']:.3e}, L3={results['final_dL3']:.3e}")

if results['final_dE'] < 1e-3:
    print("\n✓ PASS: Energy error is acceptable (< 1e-3)")
else:
    print(f"\n✗ FAIL: Energy error too large")
