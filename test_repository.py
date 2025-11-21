#!/usr/bin/env python3
"""
Quick verification script to test that the repository is ready for download and use.
This script tests the basic functionality without opening plot windows.
"""
import sys
import subprocess
from pathlib import Path

def test_basic_import():
    """Test that hw4_problem3 can be imported."""
    print("Testing hw4_problem3 import...")
    try:
        import hw4_problem3
        print("✓ hw4_problem3.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import hw4_problem3: {e}")
        return False

def test_coefficient_display():
    """Test showing coefficients (no plotting)."""
    print("\nTesting coefficient display...")
    try:
        result = subprocess.run(
            [sys.executable, "hw4_problem3.py", "--show-coefficients"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and "Order 4" in result.stdout:
            print("✓ Coefficient display works")
            return True
        else:
            print(f"✗ Coefficient display failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to run coefficient test: {e}")
        return False

def test_single_method():
    """Test running a single method without plotting."""
    print("\nTesting single method execution (Y4, no plots)...")
    try:
        result = subprocess.run(
            [sys.executable, "hw4_problem3.py", "--method", "y4", "--dt", "0.1", 
             "--steps", "100", "--energy"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and "E0:" in result.stdout:
            print("✓ Single method execution works")
            return True
        else:
            print(f"✗ Single method execution failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to run single method test: {e}")
        return False

def test_generate_plots():
    """Test generating LaTeX plots."""
    print("\nTesting LaTeX plot generation...")
    try:
        result = subprocess.run(
            [sys.executable, "generate_plots_for_latex.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        # Check if plot files were created
        plots_exist = all([
            Path("rk4_results.png").exists(),
            Path("yoshida4_results.png").exists(),
            Path("yoshida6_results.png").exists(),
            Path("yoshida8_results.png").exists()
        ])
        if result.returncode == 0 and plots_exist:
            print("✓ LaTeX plot generation works")
            print("  Generated: rk4_results.png, yoshida4_results.png, yoshida6_results.png, yoshida8_results.png")
            return True
        else:
            print(f"✗ LaTeX plot generation failed")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to generate plots: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Repository Verification Test")
    print("="*60)
    
    tests = [
        test_basic_import,
        test_coefficient_display,
        test_single_method,
        test_generate_plots
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Unexpected error in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n✓ All tests passed! Repository is ready for distribution.")
        print("\nUsers can download and run:")
        print("  python hw4_problem3.py --demo --plot --energy")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
