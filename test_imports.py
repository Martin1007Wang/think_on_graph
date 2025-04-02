from fractions import Fraction

def f(x):
    return x**3 - 5*x**2 + 2*x

def f_prime(x):
    return 3*x**2 - 10*x + 2

def newton_method(x0, max_iter=100):
    x = Fraction(x0, 1)
    for n in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_next = x - fx / fpx
        # Simplify the fraction (though Fraction should already be simplified)
        x_next = Fraction(x_next.numerator, x_next.denominator)
        # Check denominator digits
        denominator_digits = len(str(x_next.denominator))
        print(f"Iteration {n+1}: x = {x_next}, Denominator digits = {denominator_digits}")
        if denominator_digits == 25:
            return n + 1, x_next
        x = x_next
    raise ValueError("Maximum iterations reached without finding a 25-digit denominator.")

# Initial guess
x0 = 1
try:
    n, x_n = newton_method(x0)
    print(f"The smallest n is {n}, with x_n = {x_n} and denominator {x_n.denominator}")
except ValueError as e:
    print(e)
