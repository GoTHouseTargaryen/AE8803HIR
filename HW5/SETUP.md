# Setup and Bootstrap Guide for HW5

This guide explains how to run the HW5 simulation on any computer, regardless of the operating system or Python setup.

## Prerequisites

- Python 3.7 or higher
- Internet connection (for auto-installing dependencies)

## Three Ways to Run

### 1. Direct Run (Recommended - Zero Setup)

The simulation script has built-in bootstrap functionality that automatically installs missing dependencies:

```bash
python hw5_simulation.py
```

**How it works:**
- The script checks if numpy and matplotlib are installed
- If missing, it automatically runs `pip install` for you
- Then runs the simulation
- Works on Windows, Linux, and macOS

**First-time users:** Just run the command above!

### 2. Using Bootstrap Scripts (Virtual Environment)

If you prefer isolated environments:

#### Windows PowerShell
```powershell
.\bootstrap.ps1
python hw5_simulation.py
```

#### Windows Command Prompt / Double-Click
Double-click `run_simulation.bat` or run:
```cmd
run_simulation.bat
```

#### Linux / macOS
```bash
chmod +x bootstrap.sh
./bootstrap.sh
python hw5_simulation.py
```

**Benefits:**
- Creates isolated virtual environment in `.venv/`
- Doesn't affect system Python packages
- Easy to clean up (just delete `.venv/` folder)

### 3. Manual Installation

For users who want full control:

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python hw5_simulation.py
```

## Troubleshooting

### "python: command not found"

Try `python3` instead:
```bash
python3 hw5_simulation.py
```

### "Permission denied" on Linux/macOS

Make script executable:
```bash
chmod +x bootstrap.sh hw5_simulation.py
./hw5_simulation.py
```

### Behind a Corporate Firewall

If pip fails, download packages manually:
```bash
pip install --user numpy matplotlib
```

Or use a proxy:
```bash
pip install --proxy=http://proxy.example.com:8080 numpy matplotlib
```

### Python Not Installed

Download from:
- **Windows/macOS**: https://www.python.org/downloads/
- **Linux**: Use package manager
  ```bash
  sudo apt install python3 python3-pip  # Debian/Ubuntu
  sudo dnf install python3 python3-pip  # Fedora
  sudo pacman -S python python-pip      # Arch
  ```

### Virtual Environment Won't Activate

**Windows PowerShell** may block scripts. Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry `.\bootstrap.ps1`

## Expected Output

The simulation will:
1. Print progress updates
2. Generate 6 PNG files:
   - `hw5_crtbp_traj.png`: CRTBP trajectory
   - `hw5_crtbp_energy.png`: Energy conservation
   - `hw5_tfbrp_L.png`: Angular momentum
   - `hw5_tfbrp_E.png`: Rigid body energy
   - `hw5_tfbrp_dL.png`: Angular momentum error
   - `hw5_combined.png`: All plots combined
3. Print "Done."

**Runtime:** ~30-60 seconds on modern hardware

## Verification

After running, check that output files exist:

**Windows:**
```powershell
ls hw5_*.png
```

**Linux/macOS:**
```bash
ls hw5_*.png
```

You should see 6 PNG files.

## Advanced: Running on Remote Systems

### SSH without Display
If running on a remote server without X11:

```bash
export MPLBACKEND=Agg
python hw5_simulation.py
```

This forces matplotlib to save files without opening windows.

### Running in Jupyter Notebook

```python
%run hw5_simulation.py
```

Or copy-paste the code into cells.

### Docker Container

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt hw5_simulation.py ./
RUN pip install -r requirements.txt
CMD ["python", "hw5_simulation.py"]
```

Build and run:
```bash
docker build -t hw5-sim .
docker run -v $(pwd):/app/output hw5-sim
```

## Files Explained

| File | Purpose |
|------|---------|
| `hw5_simulation.py` | Main simulation with auto-bootstrap |
| `requirements.txt` | Python dependencies list |
| `bootstrap.ps1` | Windows PowerShell setup script |
| `bootstrap.sh` | Linux/macOS Bash setup script |
| `run_simulation.bat` | Windows double-click runner |
| `README.md` | Project overview and quick start |
| `SETUP.md` | This file - detailed setup guide |

## Getting Help

1. **Check Python version**: `python --version` (need 3.7+)
2. **Check pip works**: `pip --version`
3. **Try verbose output**: `python -v hw5_simulation.py`
4. **Manual install**: `pip install numpy matplotlib --user`

## Clean Up

To remove virtual environment:

**All platforms:**
```bash
rm -rf .venv  # Linux/macOS
rmdir /s .venv  # Windows CMD
Remove-Item -Recurse -Force .venv  # Windows PowerShell
```

To uninstall packages (if installed globally):
```bash
pip uninstall numpy matplotlib
```

## Summary

**For 99% of users:**
```bash
python hw5_simulation.py
```

That's it! The script handles everything else automatically.
