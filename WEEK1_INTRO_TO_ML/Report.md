# Week 1 
**Assignment 1 | Introduction to Machine Learning**


---

## Objective

Build a user-interactive mathematical function calculator that accepts two arbitrary functions `f1(x)` and `f2(x)` as string inputs, evaluates them numerically over a defined domain, handles real-world boundary conditions, and visualizes results including arithmetic operations, differentiation, and integration.

---

## Key Concepts Implemented

- Dynamic function parsing using `eval()` with safe wrappers
- Numerical differentiation via `numpy.gradient`
- Numerical integration via `numpy.trapz` (area between curves)
- Boundary condition detection and NaN-safe computation
- Multi-panel matplotlib visualization

---

## Boundary Conditions Handled

| Condition | Handling |
|-----------|----------|
| Division by zero (`f2 = 0`) | Result replaced with `NaN` |
| Negative base with fractional exponent | Flagged and excluded |
| Exponent `> 10` | Restricted to prevent overflow |
| Log of non-positive values | Set to `NaN` |
| Overflow (`> 1e10`) | Cleaned with `np.isinf` / threshold filter |

---

## Operations Computed

- Addition, Subtraction, Multiplication, Division
- Power, Modulo, Log-ratio
- Derivative `dy1/dx`, `dy2/dx`
- Area between curves (definite integral)

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python 3 | Core language |
| NumPy | Numerical computation |
| Matplotlib | Visualization |
| `math` module | Built-in math functions in eval |

---

## Files

```
Assignment1_240751_PoonamGupta_Intro_to_ML.ipynb
```

---

## Sample Output

- Individual function plots with NaN regions highlighted in red
- Per-operation plots (add, sub, mul, div, pow, mod, log-ratio)
- Derivative plots
- Printed area between curves

---

## Learning Outcomes

- Practical understanding of numerical methods vs symbolic computation
- Boundary-aware programming for mathematical operations
- Clean data pipelines using NaN masking
