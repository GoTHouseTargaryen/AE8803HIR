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
    
    The Yoshida composition applies the base second-order leapfrog integrator
    recursively with weighted timesteps. Each coefficient w_i indicates a 
    full leapfrog step S(w_i * dt).
    
    For order 2k, we have coefficients [w_0, w_1, ..., w_{n-1}] and apply:
        S_2k(dt) = S(w_0*dt) · S(w_1*dt) · ... · S(w_{n-1}*dt)
    
    where S(h) is the second-order symmetric leapfrog (velocity-Verlet):
        Kick(h/2) - Drift(h) - Kick(h/2)
    """
    coeffs = get_yoshida_coefficients(order)
    
    q = np.empty(steps+1)
    p = np.empty(steps+1)
    q[0], p[0] = q0, p0
    
    for n in range(steps):
        qn, pn = q[n], p[n]
        
        # Apply each leapfrog substep with scaled timestep
        for w in coeffs:
            qn, pn = leapfrog_step(qn, pn, w * dt)
        
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
    dE = np.abs(res.energy - E0)/np.abs(E0)
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

def plot_results(results: Dict[str, IntegrationResult], show_energy: bool = True, demo_mode: bool = False, zoom: Dict[str, Tuple[float, float] | None] | None = None, log_error: bool = False):  # pragma: no cover
    if not _HAVE_PLOT:
        print("matplotlib not available; skipping plots.")
        return
    import matplotlib.pyplot as plt
    
    def _pick_lim(default_lim: Tuple[float, float], key: str) -> Tuple[float, float]:
        """Return zoom override if provided; otherwise default_lim.
        Ensures non-degenerate span if min==max.
        """
        val = None
        if zoom is not None:
            val = zoom.get(key)
        if val is not None:
            a, b = val
            if a == b:
                eps = 1e-12 if a == 0 else abs(a) * 1e-12
                return (a - eps, b + eps)
            return (a, b)
        return default_lim
    
    if demo_mode and len(results) > 1:
        # Demo mode: Create separate subplot for each method (2 columns: q(t) and ΔH)
        n_methods = len(results)
        fig, axs = plt.subplots(n_methods, 2, figsize=(12, 3*n_methods))
        
        # First pass: compute global limits for consistent scaling
        all_t = []
        all_q = []
        all_p = []
        all_dE = []  # kept for potential future global stats; not used for per-method scaling
        for label, res in results.items():
            E0 = res.energy[0]
            dE = res.energy - E0
            all_t.append(res.t)
            all_q.append(res.q)
            all_p.append(res.p)
            all_dE.append(dE)
        
        # Global time span and symmetric limits that encompass all data
        t_min = np.min([t.min() for t in all_t])
        t_max = np.max([t.max() for t in all_t])
        q_abs = np.max([np.max(np.abs(q)) for q in all_q])
        p_abs = np.max([np.max(np.abs(p)) for p in all_p])
        
        pad = 0.05  # 5% padding
        q_lim = _pick_lim((-q_abs * (1 + pad), q_abs * (1 + pad)), 'qylim')
        p_lim = _pick_lim((-p_abs * (1 + pad), p_abs * (1 + pad)), 'pylim')
        t_lim = _pick_lim((t_min, t_max), 'tlim')
        # Energy error limits are set per-method below
        
        # Second pass: plot with consistent limits
        for idx, (label, res) in enumerate(results.items()):
            E0 = res.energy[0]
            dE = np.abs(res.energy - E0)/np.abs(E0)
            
            # Position vs time
            axs[idx, 0].plot(res.t, res.q, color='C0')
            axs[idx, 0].set_ylabel('Position (q)')
            axs[idx, 0].set_title(f'{label}: Position vs Time')
            axs[idx, 0].grid(True, alpha=0.3)
            axs[idx, 0].set_xlim(t_lim)
            axs[idx, 0].set_ylim(q_lim)
            if idx == n_methods - 1:
                axs[idx, 0].set_xlabel('Time (t)')
            
            # Energy error vs time
            print(dE)
            axs[idx, 1].plot(res.t, dE, color='C3')
            axs[idx, 1].set_ylabel('Relative Energy Error (|dE/E0|)')
            axs[idx, 1].set_title(f'{label}: Energy Error vs Time')
            axs[idx, 1].grid(True, alpha=0.3)
            axs[idx, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            axs[idx, 1].set_xlim(t_lim)
            # Per-method symmetric y-limits for energy error (overridden by --delim if provided)
            if zoom and zoom.get('delim') is not None:
                axs[idx, 1].set_ylim(_pick_lim((-1.0, 1.0), 'delim'))
            else:
                dE_max_local = float(np.max(dE))
                if dE_max_local == 0.0:
                    dE_max_local = 1e-16
                dE_lim_local = (0, dE_max_local * (1 + pad))
                axs[idx, 1].set_ylim(dE_lim_local)
            axs[idx, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
            if idx == n_methods - 1:
                axs[idx, 1].set_xlabel('Time (t)')
        
        # Link time axes (share x across q and ΔH)
        time_axes = []
        for i in range(n_methods):
            time_axes.extend([axs[i, 0], axs[i, 1]])
        if time_axes:
            base = time_axes[0]
            for ax in time_axes[1:]:
                ax.sharex(base)

        # Store default limits and bind reset hotkey 'r'
        defaults = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in fig.axes}

        def on_key(event):
            if event.key == 'r':
                for ax, (xl, yl) in defaults.items():
                    ax.set_xlim(xl)
                    ax.set_ylim(yl)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)

        fig.tight_layout()
        plt.show()
    else:
        # Single method mode: 1x2 layout (q vs time, energy error vs time)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        ax_q = axs[0]  # Position q vs time
        ax_error = axs[1]  # Energy error vs time
        
        # Plot position vs time
        for label, res in results.items():
            ax_q.plot(res.t, res.q, label=label, alpha=0.8)
        # Time and y-limits
        t_all = np.concatenate([res.t for res in results.values()])
        t_lim = _pick_lim((t_all.min(), t_all.max()), 'tlim')
        ax_q.set_xlim(t_lim)
        if zoom and zoom.get('qylim') is not None:
            ax_q.set_ylim(_pick_lim((0.0, 1.0), 'qylim'))
        ax_q.set_xlabel('Time (t)')
        ax_q.set_ylabel('Position (q)')
        ax_q.set_title('Position vs Time')
        ax_q.legend()
        ax_q.grid(True, alpha=0.3)
        
        # Plot energy error vs time
        for label, res in results.items():
            E0 = res.energy[0]
            dE = np.abs(res.energy - E0)/np.abs(E0)
            if log_error:
                dE_pos = np.where(dE > 0, dE, 1e-16)  # Avoid log(0)
                ax_error.semilogy(res.t, dE_pos, label=label, alpha=0.8)
            else:
                ax_error.plot(res.t, dE, label=label, alpha=0.8)
        ax_error.set_xlim(t_lim)
        if not log_error and zoom and zoom.get('delim') is not None:
            ax_error.set_ylim(_pick_lim((0.0, 1.0), 'delim'))
        ax_error.set_xlabel('Time (t)')
        if log_error:
            ax_error.set_ylabel('|Energy Error| (|dE/E0|)')
        else:
            ax_error.set_ylabel('Relative Energy Error (|dE/E0|)')
            ax_error.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            # Use scientific notation for small error values
            ax_error.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
        ax_error.set_title('Energy Error vs Time')
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        
        # Link time axes (share x across q and ΔH)
        ax_error.sharex(ax_q)

        # Store default limits and bind reset hotkey 'r'
        defaults = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in fig.axes}

        def on_key(event):
            if event.key == 'r':
                for ax, (xl, yl) in defaults.items():
                    ax.set_xlim(xl)
                    ax.set_ylim(yl)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)

        fig.tight_layout()
        plt.show()

# ------------------------------- CLI ------------------------------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Yoshida high-order symplectic integrators for harmonic oscillator")
    ap.add_argument('--method', nargs='+', choices=['y4','y6','y8','rk4'], help='Integration method(s) - can specify multiple')
    ap.add_argument('--dt', type=float, default=0.01, help='Time step size')
    ap.add_argument('--steps', type=int, default=1000, help='Number of steps')
    ap.add_argument('--q0', type=float, default=1.0, help='Initial position q(0)')
    ap.add_argument('--p0', type=float, default=0.0, help='Initial momentum p(0)')
    ap.add_argument('--plot', action='store_true', help='Generate plots')
    ap.add_argument('--energy', action='store_true', help='Show energy statistics')
    ap.add_argument('--log-error', action='store_true', help='Plot energy error on log scale (absolute value)')
    ap.add_argument('--demo', action='store_true', help='Run demo comparing all methods')
    ap.add_argument('--show-coefficients', action='store_true', help='Print computed Yoshida coefficients and verify order conditions')
    # Zoom/limit options: pass as "min,max" (comma or colon separated)
    ap.add_argument('--tlim', type=str, help='Time axis limits: min,max')
    ap.add_argument('--qylim', type=str, help='q(t) y-limits: min,max')
    ap.add_argument('--pylim', type=str, help='p(t) y-limits: min,max')
    ap.add_argument('--delim', type=str, help='Energy error y-limits: min,max')
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

    def _parse_lim(s: str | None) -> Tuple[float, float] | None:
        if not s:
            return None
        # Accept separators comma or colon
        if ':' in s:
            parts = s.split(':', 1)
        else:
            parts = s.split(',', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid limit format '{s}'. Use 'min,max'.")
        a = float(parts[0].strip())
        b = float(parts[1].strip())
        return (a, b)

    zoom = {
        'tlim': _parse_lim(ns.tlim),
        'qylim': _parse_lim(ns.qylim),
        'pylim': _parse_lim(ns.pylim),
        'delim': _parse_lim(ns.delim),
    }
    
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
            plot_results(results, show_energy=True, demo_mode=True, zoom=zoom, log_error=ns.log_error)
        else:
            # Print concise summary even without --energy for demo
            for label, res in results.items():
                print_metrics(label, energy_metrics(res))
        return

    if not ns.method:
        print("No --method specified. Use --demo for a full comparison.")
        return

    # Run specified method(s)
    results = {}
    for method in ns.method:
        label, res = run_single(method, ns.q0, ns.p0, ns.dt, ns.steps)
        results[label] = res
    
    # Print energy metrics if requested
    if ns.energy:
        for label, res in results.items():
            print_metrics(label, energy_metrics(res))
    
    # Plot results
    if ns.plot:
        # Use demo_mode if multiple methods specified
        use_demo_mode = len(results) > 1
        plot_results(results, show_energy=True, demo_mode=use_demo_mode, zoom=zoom, log_error=ns.log_error)

if __name__ == '__main__':  # pragma: no cover
    main()
