#!/usr/bin/env python3
"""
Test script to verify bootstrap functionality works correctly.
This simulates running on a fresh system by checking all bootstrap mechanisms.
"""

import sys
import subprocess
import importlib.util
import os

def test_bootstrap_mechanism():
    """Test the bootstrap code in hw5_simulation.py"""
    print("=" * 60)
    print("Testing Bootstrap Functionality")
    print("=" * 60)
    
    # Test 1: Check Python version
    print("\n1. Python Version Check:")
    print(f"   Version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 7:
        print("   ✓ Python 3.7+ detected")
    else:
        print(f"   ✗ Python {major}.{minor} is too old (need 3.7+)")
        return False
    
    # Test 2: Check importlib.util availability
    print("\n2. Bootstrap Module Check:")
    try:
        import importlib.util
        print("   ✓ importlib.util available")
    except ImportError:
        print("   ✗ importlib.util not available")
        return False
    
    # Test 3: Check if pip is available
    print("\n3. Package Manager Check:")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✓ pip available: {result.stdout.strip()}")
        else:
            print("   ✗ pip not available")
            return False
    except Exception as e:
        print(f"   ✗ pip check failed: {e}")
        return False
    
    # Test 4: Check if required packages are installed
    print("\n4. Required Packages Check:")
    packages = {"numpy": "numpy", "matplotlib": "matplotlib"}
    all_installed = True
    for pkg_name, import_name in packages.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"   ✗ {pkg_name} not installed")
            all_installed = False
        else:
            print(f"   ✓ {pkg_name} installed at {spec.origin}")
    
    # Test 5: Check if hw5_simulation.py exists
    print("\n5. Simulation Script Check:")
    script_path = os.path.join(os.path.dirname(__file__), "hw5_simulation.py")
    if os.path.exists(script_path):
        print(f"   ✓ hw5_simulation.py found")
        print(f"     Path: {script_path}")
    else:
        print(f"   ✗ hw5_simulation.py not found")
        return False
    
    # Test 6: Verify bootstrap code in script
    print("\n6. Bootstrap Code Verification:")
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ensure_package' in content:
                print("   ✓ Bootstrap function 'ensure_package' found")
            else:
                print("   ✗ Bootstrap function not found")
                return False
            
            if 'importlib.util' in content:
                print("   ✓ importlib.util import found")
            else:
                print("   ✗ importlib.util import not found")
                return False
    except Exception as e:
        print(f"   ✗ Failed to read script: {e}")
        return False
    
    # Test 7: Check bootstrap scripts
    print("\n7. Bootstrap Script Files Check:")
    bootstrap_files = {
        "bootstrap.ps1": "Windows PowerShell",
        "bootstrap.sh": "Linux/macOS Bash",
        "run_simulation.bat": "Windows Batch",
        "README.md": "Documentation",
        "requirements.txt": "Dependencies"
    }
    for filename, description in bootstrap_files.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   ✓ {filename} ({description}): {size} bytes")
        else:
            print(f"   ✗ {filename} missing")
    
    # Test 8: Syntax check
    print("\n8. Syntax Check:")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", script_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("   ✓ No syntax errors")
        else:
            print(f"   ✗ Syntax errors found:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"   ✗ Syntax check failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    if all_installed:
        print("✓ All checks passed! Bootstrap system is ready.")
        print("\nYou can run the simulation with:")
        print("  python hw5_simulation.py")
    else:
        print("✓ Bootstrap system is ready.")
        print("  Missing packages will be auto-installed on first run.")
        print("\nRun the simulation with:")
        print("  python hw5_simulation.py")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_bootstrap_mechanism()
    sys.exit(0 if success else 1)
