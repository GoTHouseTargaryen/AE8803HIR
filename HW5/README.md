# HW5: Symplectic Integration of CRTBP and TFRBP

This project implements 6th-order Yoshida symplectic integrators for:
- **CRTBP**: Circular Restricted Three-Body Problem (Earth-Moon L4 station)
- **TFRBP**: Torque-Free Rigid Body Problem (rotating ellipsoid)

## Quick Start (Any System)

### Option 1: Direct Run (Auto-Bootstrap)
The simulation script automatically installs dependencies if missing:

```bash
python hw5_simulation.py
```

or

```bash
python3 hw5_simulation.py
```

### Option 2: Manual Setup with Virtual Environment

#### Windows (PowerShell)
```powershell
.\bootstrap.ps1
python hw5_simulation.py
```

#### Linux/macOS
```bash
chmod +x bootstrap.sh
./bootstrap.sh
python hw5_simulation.py
```

### Option 3: Manual Installation
```bash
pip install -r requirements.txt
python hw5_simulation.py
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

The script will auto-install these if missing when run directly.

## Output

The simulation generates:
- `hw5_crtbp_traj.png`: CRTBP trajectory in (ξ, η) coordinates
- `hw5_crtbp_energy.png`: Energy conservation (log scale)
- `hw5_tfbrp_L.png`: Inertial angular momentum components
- `hw5_tfbrp_E.png`: TFRBP energy evolution
- `hw5_tfbrp_dL.png`: Angular momentum conservation (log scale)

## Features

- **Auto-Bootstrap**: Automatically installs dependencies on first run
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Virtual Environment Support**: Optional isolation via bootstrap scripts
- **6th-Order Accuracy**: Yoshida composition method (S2→S4→S6)
- **Symplectic Integration**: Preserves phase space structure and conserved quantities

## Methodology

### CRTBP
- Normalized rotating frame coordinates
- Hamiltonian splitting: Drift (kinetic) + Kick (potential) + Rotation (Coriolis)
- L4 initialization with canonical momentum

### TFRBP
- Kinematic/Dynamic splitting
- Rodrigues formula for orientation updates
- Exact axis flows via Lie-Poisson structure

### Yoshida Recursion
- **S2**: Symmetric 2nd-order base step
- **S4**: `S2(a·τ) ∘ S2(b·τ) ∘ S2(a·τ)` with `a = 1/(2-2^(1/3))`
- **S6**: `S4(w1·τ) ∘ S4(w0·τ) ∘ S4(w1·τ)` with `w1 = 1/(2-2^(1/5))`

## Simulation Parameters

- **Duration**: 25 years
- **Time Step**: 6 hours (0.25 days)
- **Energy Error Target**: < 10⁻¹² (CRTBP), < 10⁻⁸ (TFRBP)

## Documentation

See `hw5_compare.pdf` for full methodology, derivations, and results.

## License

Academic use only - AE 8803 coursework.
