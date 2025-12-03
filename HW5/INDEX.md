# HW5 Documentation Index

## Quick Links

- **Just want to run it?** → See `QUICKSTART.md`
- **Need detailed setup help?** → See `SETUP.md`
- **Platform issues?** → See `COMPATIBILITY.md`
- **Project overview?** → See `README.md`
- **Full methodology?** → See `hw5_compare.pdf`

## File Organization

### Executable Files

| File | Platform | Purpose |
|------|----------|---------|
| `hw5_simulation.py` | All | Main simulation with auto-bootstrap |
| `bootstrap.sh` | Linux/macOS | Setup virtual environment (Bash) |
| `bootstrap.ps1` | Windows | Setup virtual environment (PowerShell) |
| `run_simulation.bat` | Windows | Double-click runner (Batch) |
| `test_bootstrap.py` | All | Verify bootstrap system works |

### Documentation Files

| File | Audience | Content |
|------|----------|---------|
| `README.md` | Everyone | Project overview & quick start |
| `QUICKSTART.md` | New users | One-page reference card |
| `SETUP.md` | Detailed guide | Complete setup instructions |
| `COMPATIBILITY.md` | Platform-specific | OS compatibility matrix |
| `INDEX.md` | Navigation | This file - documentation map |

### Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git version control exclusions |

### Academic Files

| File | Content |
|------|---------|
| `hw5_compare.pdf` | Full report with methodology & results |
| `hw5_compare.tex` | LaTeX source for report |

### Output Files (Generated)

| File | Content |
|------|---------|
| `hw5_crtbp_traj.png` | CRTBP trajectory plot |
| `hw5_crtbp_energy.png` | CRTBP energy conservation |
| `hw5_tfbrp_L.png` | Rigid body angular momentum |
| `hw5_tfbrp_E.png` | Rigid body energy evolution |
| `hw5_tfbrp_dL.png` | Angular momentum error |
| `hw5_combined.png` | All plots combined |

## Usage Scenarios

### Scenario 1: First-Time User (5 seconds)
```bash
python hw5_simulation.py
```
See: `QUICKSTART.md`

### Scenario 2: Want Isolated Environment (30 seconds)
**Windows:**
```powershell
.\bootstrap.ps1
python hw5_simulation.py
```

**Linux/macOS:**
```bash
chmod +x bootstrap.sh
./bootstrap.sh
python hw5_simulation.py
```
See: `SETUP.md` → "Using Bootstrap Scripts"

### Scenario 3: Having Issues (check list)
1. Read error message
2. Check `COMPATIBILITY.md` → "Known Issues"
3. Try `python3` instead of `python`
4. Run `test_bootstrap.py` to diagnose
5. Check platform-specific notes in `COMPATIBILITY.md`

See: `SETUP.md` → "Troubleshooting"

### Scenario 4: Understanding the Math
1. Open `hw5_compare.pdf`
2. Read methodology sections:
   - CRTBP: Pages 2-4
   - TFRBP: Pages 5-7
   - Yoshida recursion: Embedded in each section
3. Review results plots: Pages 4, 7-8

See: `hw5_compare.pdf`

## Bootstrap System Overview

```
User runs: python hw5_simulation.py
    ↓
Script checks: numpy installed?
    ↓ No
Auto-install: pip install numpy
    ↓ Yes
Script checks: matplotlib installed?
    ↓ No
Auto-install: pip install matplotlib
    ↓ Yes
Run simulation → Generate plots → Done!
```

**Key feature:** Zero manual setup required!

## Recommended Reading Order

1. **New Users:**
   - `README.md` (2 min)
   - `QUICKSTART.md` (1 min)
   - Run: `python hw5_simulation.py`

2. **Setup Issues:**
   - `SETUP.md` → Troubleshooting section
   - `COMPATIBILITY.md` → Your platform
   - Run: `python test_bootstrap.py`

3. **Understanding Code:**
   - `hw5_simulation.py` (comments explain structure)
   - `hw5_compare.pdf` (methodology)
   - Code matches LaTeX section by section

4. **Development/Modification:**
   - `hw5_simulation.py` → Bootstrap section (lines 1-55)
   - Class structure (lines 60-200)
   - Integration logic (lines 200-300)
   - Plotting (lines 300-380)

## File Size Summary

| Category | Total Size |
|----------|------------|
| Documentation | ~15 KB |
| Scripts | ~33 KB |
| Dependencies (when installed) | ~100 MB |
| Output plots | ~5 MB |
| **Total (with deps)** | **~105 MB** |

## Platform-Specific Quick Start

### Windows 10/11
```powershell
# Method 1: Direct (recommended)
python hw5_simulation.py

# Method 2: PowerShell script
.\bootstrap.ps1

# Method 3: Double-click
run_simulation.bat
```

### Ubuntu / Debian
```bash
# Usually need python3
python3 hw5_simulation.py

# Or with virtual env
chmod +x bootstrap.sh
./bootstrap.sh
```

### macOS
```bash
# Usually need python3
python3 hw5_simulation.py

# Or with Homebrew Python
/usr/local/bin/python3 hw5_simulation.py
```

### WSL / Remote Linux
```bash
# Set backend for headless
export MPLBACKEND=Agg
python3 hw5_simulation.py
```

## Technical Support Decision Tree

```
Problem?
├─ Python not found
│  └─ See SETUP.md → "Python Not Installed"
│
├─ Import errors
│  └─ Run: python test_bootstrap.py
│     ├─ Packages missing → Will auto-install on next run
│     └─ Other error → See COMPATIBILITY.md
│
├─ Permission denied
│  └─ Linux/macOS: chmod +x bootstrap.sh hw5_simulation.py
│     Windows: Set-ExecutionPolicy (see SETUP.md)
│
├─ Slow downloads
│  └─ See COMPATIBILITY.md → "Slow pip downloads"
│
└─ Corporate firewall
   └─ See COMPATIBILITY.md → "SSL certificate verification"
```

## Version History

- **v1.0** - Initial release with basic simulation
- **v2.0** - Added auto-bootstrap functionality
- **v2.1** - Added cross-platform scripts & comprehensive docs

## Dependencies

### Runtime
- Python 3.7+
- NumPy (any recent version)
- Matplotlib (any recent version)

### Optional
- Virtual environment (venv module, included with Python)
- LaTeX (to rebuild hw5_compare.pdf from .tex)

## License & Credits

- **Course:** AE 8803 - Advanced Orbital Mechanics
- **Topic:** Symplectic Integration
- **Method:** 6th-order Yoshida composition
- **Author:** Alan Yang
- **Date:** December 2025

Academic use only.

## Getting Help

1. **Check documentation** in this order:
   - QUICKSTART.md
   - SETUP.md  
   - COMPATIBILITY.md

2. **Run diagnostics:**
   ```bash
   python test_bootstrap.py
   ```

3. **Check file exists:**
   ```bash
   ls hw5_simulation.py
   ```

4. **Check Python works:**
   ```bash
   python --version
   python -c "print('Hello')"
   ```

5. **Manual install:**
   ```bash
   pip install numpy matplotlib
   ```

---

**Bottom line:** Just run `python hw5_simulation.py` and it should work! 🚀
