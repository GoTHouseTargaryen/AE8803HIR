#!/usr/bin/env python3
"""
Quick Start Guide and Verification
===================================
This script verifies the repository is ready to use and shows quick-start examples.
"""
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✓ Python version is compatible (3.7+)")
        return True
    else:
        print("✗ Python 3.7 or higher required")
        return False

def check_files():
    """Check that required files exist."""
    print_header("Checking Repository Files")
    
    required_files = [
        "hw4_problem3.py",
        "generate_plots_for_latex.py",
        "hw4_problem3_methodology.tex",
        "README.md",
        "README_hw4_problem3.md",
        "requirements.txt"
    ]
    
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_present = False
    
    return all_present

def show_quick_start():
    """Show quick start commands."""
    print_header("Quick Start Commands")
    
    print("\n1. Show Yoshida coefficients:")
    print("   python hw4_problem3.py --show-coefficients")
    
    print("\n2. Run demo (all methods comparison):")
    print("   python hw4_problem3.py --demo --plot --energy")
    
    print("\n3. Run single method:")
    print("   python hw4_problem3.py --method y6 --dt 0.1 --steps 10000 --plot --energy")
    
    print("\n4. Generate LaTeX plots:")
    print("   python generate_plots_for_latex.py")
    
    print("\n5. View detailed usage:")
    print("   python hw4_problem3.py --help")

def verify_bootstrap():
    """Test that bootstrap works."""
    print_header("Testing Bootstrap (First-Time Setup)")
    print("Running: python hw4_problem3.py --show-coefficients")
    print("(This will create .venv and install packages if needed)\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "hw4_problem3.py", "--show-coefficients"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and "Order 4" in result.stdout:
            print("✓ Bootstrap successful!")
            print("✓ Coefficients displayed correctly")
            
            # Check if venv was created
            if Path(".venv").exists():
                print("✓ Virtual environment created at .venv/")
            
            return True
        else:
            print("✗ Bootstrap failed")
            if result.stderr:
                print(f"\nError output:\n{result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Bootstrap timed out (may need longer on slow connections)")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run verification."""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  AE8803HIR Repository - Quick Start & Verification".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Check Python version
    if not check_python_version():
        print("\n⚠ Please install Python 3.7 or higher")
        return 1
    
    # Check files
    if not check_files():
        print("\n⚠ Some required files are missing")
        print("   Make sure you've cloned/downloaded the complete repository")
        return 1
    
    # Show quick start
    show_quick_start()
    
    # Ask user if they want to test bootstrap
    print("\n" + "─"*70)
    response = input("\nTest automatic setup now? (y/n): ").strip().lower()
    
    if response == 'y':
        if verify_bootstrap():
            print("\n" + "─"*70)
            print("\n✓ SUCCESS! Repository is ready to use.")
            print("\nYou can now run any of the commands shown above.")
            print("The virtual environment (.venv) has been created and")
            print("all required packages have been installed.")
            return 0
        else:
            print("\n✗ Setup test failed. Please check the error messages above.")
            return 1
    else:
        print("\n✓ File check passed. Run this script again with 'y' to test setup.")
        print("  Or directly run: python hw4_problem3.py --demo --plot")
        return 0

if __name__ == "__main__":
    sys.exit(main())
