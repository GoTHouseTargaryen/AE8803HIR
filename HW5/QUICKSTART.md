# Quick Reference Card

## Run Simulation (All Platforms)

```bash
python hw5_simulation.py
```

That's it! Dependencies auto-install if missing.

## Bootstrap Scripts (Optional Virtual Environment)

| Platform | Command |
|----------|---------|
| Windows PowerShell | `.\bootstrap.ps1` |
| Windows CMD | `run_simulation.bat` |
| Linux/macOS | `./bootstrap.sh` |

## Output Files

- `hw5_crtbp_traj.png` - CRTBP trajectory
- `hw5_crtbp_energy.png` - Energy conservation
- `hw5_tfbrp_L.png` - Angular momentum
- `hw5_tfbrp_E.png` - Rigid body energy  
- `hw5_tfbrp_dL.png` - Angular momentum error
- `hw5_combined.png` - Combined plots

## Troubleshooting One-Liners

```bash
# Check Python version (need 3.7+)
python --version

# Try python3 instead
python3 hw5_simulation.py

# Manual dependency install
pip install numpy matplotlib

# Check if packages installed
python -c "import numpy, matplotlib; print('OK')"

# Make executable (Linux/macOS)
chmod +x bootstrap.sh hw5_simulation.py
```

## Clean Virtual Environment

```bash
# Remove .venv folder
rm -rf .venv                           # Linux/macOS
Remove-Item -Recurse -Force .venv     # PowerShell
```

## Documentation

- `README.md` - Project overview
- `SETUP.md` - Detailed setup guide
- `hw5_compare.pdf` - Full methodology

## Runtime

~30-60 seconds on modern hardware
