import numpy as np
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

def p3(epsilon, filename="P3.png"):
    n = 100
    (a, b) = (0, 2*math.pi)
    x = np.linspace(a, b, n) + np.random.normal(0, epsilon, n)
    y = np.sin(x)

    x_downsample = x[0:99:n//10]
    y_downsample = y[0:99:n//10]
    poly = lagrange(x_downsample, y_downsample) 

    poly_coefs = Polynomial(poly.coef[::-1]).coef

    step = 0.1
    plt.clf()
    plt.scatter(x, y, label='Input Data', color="blue")

    x_new = np.sort(x)
    y_new = Polynomial(poly.coef[::-1])(x_new)
    err   = np.dot(y - y_new, y - y_new) / y.shape[0]
    errstr = "0.000" if err < 0.0005 else str(err)[0:5]
    plt.plot(x_new, y_new, label='Polynomial; Error = ' + errstr, color="red")
    plt.legend(loc="upper right")
    plt.title("Lagrange Interpolation of Sinusoidal Function with Epsilon=" + str(epsilon))
    plt.savefig("P3.s=" + str(epsilon) + ".png")

    print("For S=", str(epsilon), "Error is \t", err)

# p3(0)
# p3(0.1)
# p3(0.3)
# p3(0.5)
# p3(1)
p3(2)