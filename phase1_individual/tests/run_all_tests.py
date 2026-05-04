"""
Test Runner: Run all UDE equation recovery tests
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def run_all_tests():
    """Run all tests and report results"""
    
    print("\n" + "="*70)
    print("UDE EQUATION RECOVERY TEST SUITE")
    print("="*70)
    print()
    
    tests = [
        ("Linear Equation", "test_01_linear_equation"),
        ("Nonlinear Polynomial", "test_02_nonlinear_equation"),
        ("Stress Equation", "test_03_stress_equation"),
        ("Lotka-Volterra", "test_04_lotka_volterra"),
    ]
    
    results = []
    
    for test_name, test_module in tests:
        print(f"\nRunning: {test_name}")
        print("-"*70)
        
        try:
            # Import and run test
            module = __import__(test_module)
            
            if test_module == "test_01_linear_equation":
                success = module.test_linear_equation()
            elif test_module == "test_02_nonlinear_equation":
                success = module.test_nonlinear_equation()
            elif test_module == "test_03_stress_equation":
                success = module.test_stress_equation()
            elif test_module == "test_04_lotka_volterra":
                success = module.test_lotka_volterra()
            
            results.append((test_name, success))
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
