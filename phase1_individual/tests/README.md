# UDE Tests

## Test Files

### Individual Tests:
- `test_01_linear_equation.py` - Linear equation recovery
- `test_02_nonlinear_equation.py` - Nonlinear polynomial
- `test_03_stress_equation.py` - Stress dynamics equation
- `test_04_lotka_volterra.py` - Predator-prey system

### Run All Tests:
```bash
python tests/run_all_tests.py
```

### Run Individual Test:
```bash
python tests/test_01_linear_equation.py
```

## Expected Results:
All tests should PASS (✅)

## Test Purposes:
1. **Linear** - Verify basic coefficient recovery
2. **Nonlinear** - Verify neural network captures complex terms
3. **Stress** - Verify application to stress modeling
4. **Lotka-Volterra** - Verify handling of coupled systems
