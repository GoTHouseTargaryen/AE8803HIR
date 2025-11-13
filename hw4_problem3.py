"""
HW4 Problem 3: Yoshida high-order symplectic integrators (orders 4, 6, 8)
for the simple harmonic oscillator, with comparison to classical RK4.

Harmonic oscillator (set all physical constants to unity):
    H(q, p) = 0.5 * (p**2 + q**2)
Equations of motion:
    dq/dt =  p
    dp/dt = -q

We build higher-order symplectic schemes by composing the second-order
velocity-Verlet (leapfrog) integrator S(h) using the Yoshida recursive
composition method (Suzuki fractal decomposition).

Coefficients are computed ANALYTICALLY using the formula:
    S_2k(h) = S_{2k-2}(w1*h) · S_{2k-2}(w0*h) · S_{2k-2}(w1*h)
where:
    w1 = 1 / (2 - 2^(1/(2k-1)))
    w0 = 1 - 2*w1

This recursion ensures order conditions are satisfied:
    - Σw_i = 1 (time-step consistency)
    - Σw_i^(2j+1) = 0 for j=1,2,...,k-1 (odd-order error cancellation)

Usage examples (PowerShell):
    python hw4_problem3.py --show-coefficients
    python hw4_problem3.py --method y4 --dt 0.01 --steps 10000 --q0 1.0 --p0 0.0 --plot
    python hw4_problem3.py --demo   # runs all Yoshida orders + RK4 comparison
    python hw4_problem3.py --method rk4 --dt 0.01 --steps 1000 --energy

Outputs: prints summary of energy conservation statistics; optional plots.

Reference: H. Yoshida, Phys. Lett. A 150, 262–268 (1990)
"""
from __future__ import annotations
import argparse
import math
import sys
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ------------------------------------------------------------------------- #
# Automatic virtual environment bootstrap
# Creates .venv, installs required packages, and re-executes script inside it
# the first time if dependencies are missing. This allows "python hw4_problem3.py"
# to just work on a fresh machine (with base Python available).
# ------------------------------------------------------------------------- #

REQUIRED_PACKAGES = ["numpy", "matplotlib"]
ENV_DIR = Path(__file__).parent / ".venv"

def in_venv() -> bool:
    return sys.prefix != sys.base_prefix  # standard heuristic

def ensure_virtualenv():
    if os.environ.get("HW4_ENV_BOOTSTRAPPED"):
        return  # already relaunched inside venv
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            missing.append(pkg)
    if not missing:
        return  # everything present
    # Create venv if needed
    if not ENV_DIR.exists():
        print(f"[bootstrap] Creating virtual environment at {ENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(ENV_DIR)])
    # Determine python/pip executables inside venv (Windows layout assumed)
    vpy = ENV_DIR / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    pip_exe = [str(vpy), "-m", "pip"]
    print("[bootstrap] Upgrading pip ...")
    subprocess.check_call(pip_exe + ["install", "--upgrade", "pip"])  # upgrade pip
    # Write requirements.txt if not present
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        req_file.write_text("\n".join(REQUIRED_PACKAGES) + "\n")
    print(f"[bootstrap] Installing missing packages: {missing}")
    subprocess.check_call(pip_exe + ["install"] + missing)
    # Relaunch script inside venv
    print("[bootstrap] Relaunching inside virtual environment ...")
    os.environ["HW4_ENV_BOOTSTRAPPED"] = "1"
    os.execv(str(vpy), [str(vpy), __file__, *sys.argv[1:]])

ensure_virtualenv()

import numpy as np  # after bootstrap ensures availability

try:
    import matplotlib.pyplot as plt
    _HAVE_PLOT = True
except Exception:  # pragma: no cover
    _HAVE_PLOT = False

# ---------------------------- Physics helpers ---------------------------- #

def hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute harmonic oscillator Hamiltonian H = 0.5*(p^2 + q^2)."""
    return 0.5 * (p*p + q*q)

# Force: F = -dV/dq = -q when V = 0.5*q^2

def force(q: float) -> float:
    return -q

# ---------------------- Symplectic splitting operators ------------------- #

def drift_step(q: float, p: float, h: float) -> Tuple[float, float]:
    """Drift step: advance position using current momentum.
    This is the exp(h * p * d/dq) operator for harmonic oscillator.
    q_new = q + h*p
    p_new = p (unchanged)
    """
    return q + h * p, p

def kick_step(q: float, p: float, h: float) -> Tuple[float, float]:
    """Kick step: advance momentum using force at current position.
    This is the exp(h * F(q) * d/dp) operator.
    q_new = q (unchanged)
    p_new = p + h*force(q)
    """
    return q, p + h * force(q)

def leapfrog_step(q: float, p: float, h: float) -> Tuple[float, float]:
    """One velocity-Verlet / leapfrog step of size h.
    
    This is a symmetric composition: Kick(h/2) - Drift(h) - Kick(h/2)
    which is second-order accurate and symplectic.
    """
    q, p = kick_step(q, p, 0.5 * h)
    q, p = drift_step(q, p, h)
    q, p = kick_step(q, p, 0.5 * h)
    return q, p

# ---------------------- Yoshida compositions ----------------------------- #

def compute_yoshida_coefficients(order: int) -> List[float]:
    """
    Compute Yoshida symplectic integrator coefficients analytically using
    recursive composition based on Lie operator expansion order conditions.
    
    For order 2k, we recursively compose lower-order schemes S_{2k-2}(w*h)
    to cancel error terms up to order 2k.
    
    The key relation (Yoshida 1990, Suzuki fractal decomposition):
        S_2k(h) = S_{2k-2}(w1*h) * S_{2k-2}(w0*h) * S_{2k-2}(w1*h)
    where w1 = 1/(2 - 2^(1/(2k-1))) and w0 = 1 - 2*w1.
    
    This satisfies the order conditions: sum(w_i) = 1 and sum(w_i^(2j+1)) = 0
    for j=1,2,...,k-1 (symmetric schemes have only odd-order error terms).
    
    Reference: H. Yoshida, Phys. Lett. A 150, 262 (1990)
    """
    if order == 2:
        # Base second-order symmetric scheme (single leapfrog step)
        return [1.0]
    elif order == 4:
        # Order 4: Triple composition S_2(w1) S_2(w0) S_2(w1)
        # Order condition: w1 + w0 + w1 = 1 and w1^3 + w0^3 + w1^3 = 0
        # Solution: w1 = 1/(2 - 2^(1/3)), w0 = 1 - 2*w1
        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w0 = 1.0 - 2.0 * w1
        return [w1, w0, w1]
    elif order == 6:
        # Order 6: Recursively compose order-4 scheme
        # S_6(h) = S_4(w1*h) S_4(w0*h) S_4(w1*h)
        # w1 = 1/(2 - 2^(1/5)), w0 = 1 - 2*w1
        w1 = 1.0 / (2.0 - 2.0**(1.0/5.0))
        w0 = 1.0 - 2.0 * w1
        base = compute_yoshida_coefficients(4)
        # Compose: scale each base coefficient by w1, w0, w1 in sequence
        result = []
        for w in [w1, w0, w1]:
            result.extend([w * c for c in base])
        return result
    elif order == 8:
        # Order 8: Recursively compose order-6 scheme
        # S_8(h) = S_6(w1*h) S_6(w0*h) S_6(w1*h)
        # w1 = 1/(2 - 2^(1/7)), w0 = 1 - 2*w1
        w1 = 1.0 / (2.0 - 2.0**(1.0/7.0))
        w0 = 1.0 - 2.0 * w1
        base = compute_yoshida_coefficients(6)
        result = []
        for w in [w1, w0, w1]:
            result.extend([w * c for c in base])
        return result
    else:
        raise ValueError(f"Unsupported Yoshida order: {order}. Supported: 2, 4, 6, 8")

# Cache computed coefficients to avoid recomputation
_YOSHIDA_COEFFS_CACHE: Dict[int, List[float]] = {}

def get_yoshida_coefficients(order: int) -> List[float]:
    """Get Yoshida coefficients, using cache if available."""
    if order not in _YOSHIDA_COEFFS_CACHE:
        _YOSHIDA_COEFFS_CACHE[order] = compute_yoshida_coefficients(order)
    return _YOSHIDA_COEFFS_CACHE[order]

def verify_yoshida_order_conditions(order: int, coeffs: List[float]) -> Dict[str, float]:
    """
    Verify that coefficients satisfy the symplectic order conditions.
    
    For order 2k symmetric scheme:
    - sum(w_i) = 1 (time step consistency)
    - sum(w_i^3) = 0 (order 4 requires this)
    - sum(w_i^5) = 0 (order 6)
    - sum(w_i^7) = 0 (order 8)
    
    Returns dictionary of residuals for each condition.
    """
    w = np.array(coeffs)
    conditions = {
        'sum_w': np.sum(w) - 1.0,  # should be 0
        'sum_w3': np.sum(w**3) if order >= 4 else 0.0,
        'sum_w5': np.sum(w**5) if order >= 6 else 0.0,
        'sum_w7': np.sum(w**7) if order >= 8 else 0.0,
    }
    return conditions

@dataclass
class IntegrationResult:
    t: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray

# ---------------------------- Integrators -------------------------------- #

def integrate_yoshida(order: int, q0: float, p0: float, dt: float, steps: int) -> IntegrationResult:
    """Integrate using Yoshida symplectic method.
    
    The Yoshida composition applies to the symplectic splitting operators.
    For a separable Hamiltonian H = T(p) + V(q), we use:
    - Drift operator: exp(h*T) which advances q using p
    - Kick operator: exp(h*V) which advances p using force(q)
    
    The second-order base method is: Kick(h/2) - Drift(h) - Kick(h/2)
    
    Higher-order Yoshida methods compose these with weighted timesteps.
    For symmetric composition with coefficients [w1, w2, ..., wn]:
    We apply: Drift(w1*dt) - Kick(w2*dt) - ... pattern
    
    For odd number of coefficients (symmetric), we use pattern:
    Kick(w[0]*h/2) - Drift(w[0]*h) - Kick((w[0]+w[1])*h/2) - Drift(w[1]*h) - ...
    """
    coeffs = get_yoshida_coefficients(order)
    n_stages = len(coeffs)
    
    q = np.empty(steps+1)
    p = np.empty(steps+1)
    q[0], p[0] = q0, p0
    
    for n in range(steps):
        qn, pn = q[n], p[n]
        
        # Apply Yoshida composition using kick-drift-kick pattern
        # First half-kick
        qn, pn = kick_step(qn, pn, 0.5 * coeffs[0] * dt)
        
        # Middle stages: full drift, then combined half-kicks
        for i in range(n_stages - 1):
            qn, pn = drift_step(qn, pn, coeffs[i] * dt)
            qn, pn = kick_step(qn, pn, 0.5 * (coeffs[i] + coeffs[i+1]) * dt)
        
        # Final drift and half-kick
        qn, pn = drift_step(qn, pn, coeffs[-1] * dt)
        qn, pn = kick_step(qn, pn, 0.5 * coeffs[-1] * dt)
        
        q[n+1], p[n+1] = qn, pn
    
    t = np.linspace(0.0, dt*steps, steps+1)
    energy = hamiltonian(q, p)
    return IntegrationResult(t, q, p, energy)

# Classical RK4 (NOT symplectic)

def rk4_step(q: float, p: float, h: float) -> Tuple[float, float]:
    def f(q_, p_):
        return p_, -q_
    k1q, k1p = f(q, p)
    k2q, k2p = f(q + 0.5*h*k1q, p + 0.5*h*k1p)
    k3q, k3p = f(q + 0.5*h*k2q, p + 0.5*h*k2p)
    k4q, k4p = f(q + h*k3q, p + h*k3p)
    q_new = q + (h/6.0)*(k1q + 2*k2q + 2*k3q + k4q)
    p_new = p + (h/6.0)*(k1p + 2*k2p + 2*k3p + k4p)
    return q_new, p_new

def integrate_rk4(q0: float, p0: float, dt: float, steps: int) -> IntegrationResult:
    q = np.empty(steps+1)
    p = np.empty(steps+1)
    q[0], p[0] = q0, p0
    for n in range(steps):
        q[n+1], p[n+1] = rk4_step(q[n], p[n], dt)
    t = np.linspace(0.0, dt*steps, steps+1)
    energy = hamiltonian(q, p)
    return IntegrationResult(t, q, p, energy)

# ------------------------- Utility / diagnostics -------------------------- #

def energy_metrics(res: IntegrationResult) -> Dict[str, float]:
    E0 = res.energy[0]
    dE = res.energy - E0
    return {
        "E0": E0,
        "E_mean": float(res.energy.mean()),
        "E_std": float(res.energy.std()),
        "max_abs_dE": float(np.max(np.abs(dE))),
        "rms_dE": float(math.sqrt(np.mean(dE*dE))),
    }

def print_metrics(label: str, metrics: Dict[str, float]):
    print(f"[{label}] E0={metrics['E0']:.6e} mean={metrics['E_mean']:.6e} std={metrics['E_std']:.3e} "
          f"max|dE|={metrics['max_abs_dE']:.3e} rms(dE)={metrics['rms_dE']:.3e}")

# ----------------------------- Plotting ---------------------------------- #

def plot_results(results: Dict[str, IntegrationResult], show_energy: bool = True, demo_mode: bool = False):  # pragma: no cover
    if not _HAVE_PLOT:
        print("matplotlib not available; skipping plots.")
        return
    import matplotlib.pyplot as plt
    
    if demo_mode and len(results) > 1:
        # Demo mode: Create separate subplot for each method (4 methods x 4 plot types)
        n_methods = len(results)
        fig, axs = plt.subplots(n_methods, 4, figsize=(16, 3*n_methods))
        
        # First pass: compute global limits for consistent scaling
        all_t = []
        all_q = []
        all_p = []
        all_dE = []
        for label, res in results.items():
            E0 = res.energy[0]
            dE = res.energy - E0
            all_t.append(res.t)
            all_q.append(res.q)
            all_p.append(res.p)
            all_dE.append(dE)
        
        t_min, t_max = np.min([t.min() for t in all_t]), np.max([t.max() for t in all_t])
        q_min, q_max = np.min([q.min() for q in all_q]), np.max([q.max() for q in all_q])
        p_min, p_max = np.min([p.min() for p in all_p]), np.max([p.max() for p in all_p])
        dE_min, dE_max = np.min([dE.min() for dE in all_dE]), np.max([dE.max() for dE in all_dE])
        
        # Add 5% padding to limits
        q_range = q_max - q_min
        p_range = p_max - p_min
        dE_range = dE_max - dE_min
        q_lim = [q_min - 0.05*q_range, q_max + 0.05*q_range]
        p_lim = [p_min - 0.05*p_range, p_max + 0.05*p_range]
        dE_lim = [dE_min - 0.05*dE_range, dE_max + 0.05*dE_range]
        
        # Second pass: plot with consistent limits
        for idx, (label, res) in enumerate(results.items()):
            E0 = res.energy[0]
            dE = res.energy - E0
            
            # Position vs time
            axs[idx, 0].plot(res.t, res.q, color='C0')
            axs[idx, 0].set_ylabel('Position (q)')
            axs[idx, 0].set_title(f'{label}: Position vs Time')
            axs[idx, 0].grid(True, alpha=0.3)
            axs[idx, 0].set_xlim([t_min, t_max])
            axs[idx, 0].set_ylim(q_lim)
            if idx == n_methods - 1:
                axs[idx, 0].set_xlabel('Time (t)')
            
            # Momentum vs time
            axs[idx, 1].plot(res.t, res.p, color='C1')
            axs[idx, 1].set_ylabel('Momentum (p)')
            axs[idx, 1].set_title(f'{label}: Momentum vs Time')
            axs[idx, 1].grid(True, alpha=0.3)
            axs[idx, 1].set_xlim([t_min, t_max])
            axs[idx, 1].set_ylim(p_lim)
            if idx == n_methods - 1:
                axs[idx, 1].set_xlabel('Time (t)')
            
            # Phase space
            axs[idx, 2].plot(res.q, res.p, color='C2')
            axs[idx, 2].set_ylabel('Momentum (p)')
            axs[idx, 2].set_title(f'{label}: Phase Space')
            axs[idx, 2].grid(True, alpha=0.3)
            axs[idx, 2].set_xlim(q_lim)
            axs[idx, 2].set_ylim(p_lim)
            axs[idx, 2].set_aspect('equal', adjustable='box')
            if idx == n_methods - 1:
                axs[idx, 2].set_xlabel('Position (q)')
            
            # Energy error vs time
            axs[idx, 3].plot(res.t, dE, color='C3')
            axs[idx, 3].set_ylabel('Energy Error (ΔH)')
            axs[idx, 3].set_title(f'{label}: Energy Error vs Time')
            axs[idx, 3].grid(True, alpha=0.3)
            axs[idx, 3].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            axs[idx, 3].set_xlim([t_min, t_max])
            axs[idx, 3].set_ylim(dE_lim)
            axs[idx, 3].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
            if idx == n_methods - 1:
                axs[idx, 3].set_xlabel('Time (t)')
        
        fig.tight_layout()
        plt.show()
    else:
        # Single method mode: 2x2 layout
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        ax_q = axs[0, 0]  # Position q vs time
        ax_p = axs[0, 1]  # Momentum p vs time
        ax_phase = axs[1, 0]  # Phase space (q vs p)
        ax_error = axs[1, 1]  # Energy error vs time
        
        # Plot position vs time
        for label, res in results.items():
            ax_q.plot(res.t, res.q, label=label, alpha=0.8)
        ax_q.set_xlabel('Time (t)')
        ax_q.set_ylabel('Position (q)')
        ax_q.set_title('Position vs Time')
        ax_q.legend()
        ax_q.grid(True, alpha=0.3)
        
        # Plot momentum vs time
        for label, res in results.items():
            ax_p.plot(res.t, res.p, label=label, alpha=0.8)
        ax_p.set_xlabel('Time (t)')
        ax_p.set_ylabel('Momentum (p)')
        ax_p.set_title('Momentum vs Time')
        ax_p.legend()
        ax_p.grid(True, alpha=0.3)
        
        # Plot phase space (q vs p)
        for label, res in results.items():
            ax_phase.plot(res.q, res.p, label=label, alpha=0.7)
        ax_phase.set_xlabel('Position (q)')
        ax_phase.set_ylabel('Momentum (p)')
        ax_phase.set_title('Phase Space (q vs p)')
        ax_phase.legend()
        ax_phase.grid(True, alpha=0.3)
        ax_phase.axis('equal')  # Equal aspect ratio to show circular orbit
        
        # Plot energy error vs time
        for label, res in results.items():
            E0 = res.energy[0]
            dE = res.energy - E0
            ax_error.plot(res.t, dE, label=label, alpha=0.8)
        ax_error.set_xlabel('Time (t)')
        ax_error.set_ylabel('Energy Error (ΔH)')
        ax_error.set_title('Energy Error vs Time')
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        ax_error.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Use scientific notation for small error values
        ax_error.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
        
        fig.tight_layout()
        plt.show()

# ------------------------------- CLI ------------------------------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Yoshida high-order symplectic integrators for harmonic oscillator")
    ap.add_argument('--method', choices=['y4','y6','y8','rk4'], help='Integration method')
    ap.add_argument('--dt', type=float, default=0.01, help='Time step size')
    ap.add_argument('--steps', type=int, default=1000, help='Number of steps')
    ap.add_argument('--q0', type=float, default=1.0, help='Initial position q(0)')
    ap.add_argument('--p0', type=float, default=0.0, help='Initial momentum p(0)')
    ap.add_argument('--plot', action='store_true', help='Generate plots')
    ap.add_argument('--energy', action='store_true', help='Show energy statistics')
    ap.add_argument('--demo', action='store_true', help='Run demo comparing all methods')
    ap.add_argument('--show-coefficients', action='store_true', help='Print computed Yoshida coefficients and verify order conditions')
    return ap.parse_args(argv)

# ------------------------------ Main logic -------------------------------- #

def run_single(method: str, q0: float, p0: float, dt: float, steps: int) -> Tuple[str, IntegrationResult]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if method.startswith('y'):
        order = int(method[1:])
        res = integrate_yoshida(order, q0, p0, dt, steps)
        label = f"Yoshida{order}"
    elif method == 'rk4':
        res = integrate_rk4(q0, p0, dt, steps)
        label = 'RK4'
    else:
        raise ValueError(f"Unknown method: {method}")
    return label, res

def demo(dt: float, steps: int, q0: float, p0: float) -> Dict[str, IntegrationResult]:
    methods = ['y4','y6','y8','rk4']
    results = {}
    for m in methods:
        label, res = run_single(m, q0, p0, dt, steps)
        results[label] = res
    return results

def main(argv: List[str] | None = None):
    ns = parse_args(argv or sys.argv[1:])
    
    if ns.show_coefficients:
        for order in [4, 6, 8]:
            coeffs = get_yoshida_coefficients(order)
            print(f"\n=== Yoshida Order {order} ===")
            print(f"Number of stages: {len(coeffs)}")
            print("Coefficients:")
            for i, c in enumerate(coeffs, 1):
                print(f"  w[{i:2d}] = {c:23.18f}")
            # Verify order conditions
            conditions = verify_yoshida_order_conditions(order, coeffs)
            print("Order condition residuals (should be ~0):")
            for key, val in conditions.items():
                print(f"  {key:8s} = {val:12.3e}")
        return
    
    if ns.demo:
        results = demo(ns.dt, ns.steps, ns.q0, ns.p0)
        if ns.energy:
            for label, res in results.items():
                print_metrics(label, energy_metrics(res))
        if ns.plot:
            plot_results(results, show_energy=True, demo_mode=True)
        else:
            # Print concise summary even without --energy for demo
            for label, res in results.items():
                print_metrics(label, energy_metrics(res))
        return

    if not ns.method:
        print("No --method specified. Use --demo for a full comparison.")
        return

    label, res = run_single(ns.method, ns.q0, ns.p0, ns.dt, ns.steps)
    if ns.energy:
        print_metrics(label, energy_metrics(res))
    if ns.plot:
        plot_results({label: res}, show_energy=True)

if __name__ == '__main__':  # pragma: no cover
    main()
