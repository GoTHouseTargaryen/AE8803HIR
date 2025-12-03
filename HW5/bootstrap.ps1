# Bootstrap script for HW5 - Yoshida Symplectic Integrator
# This script creates a virtual environment and installs required packages

Write-Host "=== HW5 Bootstrap Script ===" -ForegroundColor Cyan

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    # Install essential packages if requirements.txt doesn't exist
    pip install numpy matplotlib scipy
    # Create requirements.txt
    pip freeze > requirements.txt
}

Write-Host "Bootstrap complete!" -ForegroundColor Green
Write-Host "Virtual environment is activated and ready to use." -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the simulation:" -ForegroundColor Yellow
Write-Host "  python hw5_simulation.py" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate the virtual environment:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
