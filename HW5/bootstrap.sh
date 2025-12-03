#!/bin/bash
# Bootstrap script for HW5 - Yoshida Symplectic Integrator (Linux/macOS)
# This script creates a virtual environment and installs required packages

echo "=== HW5 Bootstrap Script ==="

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment!"
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install essential packages if requirements.txt doesn't exist
    pip install numpy matplotlib scipy
    # Create requirements.txt
    pip freeze > requirements.txt
fi

echo "Bootstrap complete!"
echo "Virtual environment is activated and ready to use."
echo ""
echo "To run the simulation:"
echo "  python hw5_simulation.py"
echo ""
echo "To activate the virtual environment later:"
echo "  source .venv/bin/activate  (Linux/macOS)"
echo "  .venv\\Scripts\\activate  (Windows)"
