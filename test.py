from math import sin
from scipy.optimize import minimize_scalar


def f1(x):
    return sin(x) * x**3


def test(minimizer, f, a, b, eps):
    return abs(minimize_scalar(f, bounds=(a, b)).x - minimizer(f, a, b, eps).minimize().x) < eps
