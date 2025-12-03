@echo off
REM Bootstrap and run HW5 simulation on Windows
echo === HW5 Simulation Runner ===
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH!
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo Running simulation with auto-bootstrap...
echo.

REM Run the simulation (it will auto-install dependencies)
python hw5_simulation.py

echo.
echo Simulation complete!
echo Output files generated in current directory.
pause
