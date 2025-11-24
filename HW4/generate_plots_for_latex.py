"""
Generate publication-quality plots for LaTeX document.
Saves position and energy error plots separately as PNG files.
"""
import sys
import os
import subprocess
from pathlib import Path

# Bootstrap virtual environment (same logic as hw4_problem3.py)
_VENV = Path(__file__).parent / ".venv"
_IN_VENV = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if not _IN_VENV:
    # Create venv if missing
    if not _VENV.exists():
        print("[bootstrap] Creating virtual environment at", _VENV)
        subprocess.check_call([sys.executable, "-m", "venv", str(_VENV)])
    
    # Check for missing packages
    pip_exe = _VENV / ("Scripts" if os.name == 'nt' else "bin") / ("pip.exe" if os.name == 'nt' else "pip")
    py_exe = _VENV / ("Scripts" if os.name == 'nt' else "bin") / ("python.exe" if os.name == 'nt' else "python")
    
    # Upgrade pip
    print("[bootstrap] Upgrading pip ...")
    subprocess.check_call([str(py_exe), "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
    
    # Install required packages
    print("[bootstrap] Installing required packages ...")
    subprocess.check_call([str(pip_exe), "install", "numpy", "matplotlib"], stdout=subprocess.DEVNULL)
    
    # Re-exec inside venv
    print("[bootstrap] Relaunching inside virtual environment ...")
    os.execv(str(py_exe), [str(py_exe)] + sys.argv)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import from hw4_problem3
from hw4_problem3 import demo, energy_metrics

def generate_plots(dt=0.1, steps=10000, dpi=300):
    """Generate and save plots for LaTeX document."""
    
    # Run all methods
    print("Running simulations...")
    results = demo(dt, steps, q0=1.0, p0=0.0)
    
    # Print energy statistics
    print("\nEnergy Statistics:")
    print("="*80)
    for label, res in results.items():
        E0 = res.energy[0]
        dE = np.abs(res.energy - E0)/np.abs(E0)
        metrics = energy_metrics(res)
        print(f"\n{label}:")
        print(f"  E0:          {metrics['E0']:12.6e}")
        print(f"  Mean E:      {metrics['E_mean']:12.6e}")
        print(f"  Std Dev E:   {metrics['E_std']:12.6e}")
        print(f"  Mean dE/E0:  {np.mean(dE):12.6e}")
        print(f"  Std Dev dE/E0: {np.std(dE):12.6e}")
        print(f"  Max |dE/E0|: {np.max(dE):12.6e}")
        print(f"  Final dE/E0: {dE[-1]:12.6e}")
        print(f"  RMS dE/E0:   {metrics['rms_dE']:12.6e}")
    
    # Color map for consistent colors
    colors = {'RK4': 'C0', 'Yoshida4': 'C1', 'Yoshida6': 'C2', 'Yoshida8': 'C3'}
    
    # Generate separate plots for each method
    method_order = ['RK4', 'Yoshida4', 'Yoshida6', 'Yoshida8']
    saved_files = []
    
    print("\nGenerating individual method plots...")
    for label in method_order:
        if label not in results:
            continue
            
        res = results[label]
        E0 = res.energy[0]
        dE = np.abs(res.energy - E0)/np.abs(E0)
        color = colors.get(label, 'black')
        
        # Create 1x2 subplot for this method (position and energy error)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Position vs Time
        ax1.plot(res.t, res.q, color=color, linewidth=1.5)
        ax1.set_xlabel('Time (t)', fontsize=11)
        ax1.set_ylabel('Position (q)', fontsize=11)
        ax1.set_title(f'{label}: Position vs Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim((0, res.t[-1]))
        
        # Energy Error vs Time
        ax2.plot(res.t, dE, color=color, linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('Time (t)', fontsize=11)
        ax2.set_ylabel('Relative Energy Error (|dE/E0|)', fontsize=11)
        ax2.set_title(f'{label}: Energy Error vs Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
        ax2.set_xlim((0, res.t[-1]))
        
        fig.tight_layout()
        filename = f'{label.lower().replace("-", "")}_results.png'
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_files.append(filename)
        print(f"  Saved: {filename}")
        plt.close(fig)
    
    print("\n" + "="*80)
    print("Plot generation complete!")
    print(f"Files saved: {', '.join(saved_files)}")
    print("="*80)

if __name__ == '__main__':
    generate_plots()
