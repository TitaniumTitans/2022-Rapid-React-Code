import numpy as np
from sympy import S, symbols, printing
import matplotlib.pyplot as plt
import csv
array = []
# with open("dist_vs_rpm.txt", newline="") as data:
with open("5.csv", newline="") as data:
    # data_reader = csv.reader(data, delimiter="\t")
    data_reader = csv.reader(data)
    array = np.array([[float(point[0]), float(point[1])] for point in data_reader])

x = array[:,0]
y = array[:,1]

plt.scatter(x, y)

# Best squares fit of 3rd degree polynomial
z = np.polyfit(array[:,0], array[:,1], 2)
p = np.poly1d(z)

d = np.poly1d([0.0584559726, -10.31606462, 3932.2829871])

x_new = np.linspace(x[0], x[-1], 50)
y_new = p(x_new)
y_d = d(x_new)



print(z)


# def dans (x):
#     return 3932.2829871-10.31606462*x+0.0584559726*x*x

x = symbols("x")
poly = sum(S("{:6.3f}".format(v))*x**i for i, v in enumerate(z[::-1]))
eq_latex = printing.latex(poly)


plt.xlabel("Distance (inches)")
plt.ylabel("Setpoint RPM")
plt.plot(x_new, y_new, label="${}$".format(eq_latex))
a = '$0.058x^2-10.316x+3932.283$'
plt.plot(x_new, y_d, label=a)
plt.legend(fontsize="small")
plt.show()


# 0    Out,Short
# 1    Out,Front
# 2    In,Near Front Rim
# 3    In,Center
# 4    Out,Bounced
# 5    In,Hit Back Rim
# 6    Out,Near Back Rim
# 7    Out,Long