# from this import s
from sympy import S, symbols, printing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from scipy.interpolate import lagrange
from scipy.stats import gaussian_kde
array = []
# with open("dist_vs_rpm.txt", newline="") as data:
# with open("filtered_data_copy_2.csv", newline="") as data:
with open("combined.csv", newline="") as data:
    data_reader = csv.reader(data)
    # data_reader = csv.reader(data,delimiter="\t")
    array = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in data_reader])
    # array = np.array([[float(point[0]), float(point[1]), float(point[2])] for point in data_reader])
    # array = np.array([[[float(p[0]), float(p[1])], float(p[2])] for p in data_reader])
# keys = np.unique(array[:,0])
# for key in keys:
    
# h = np.unique(array[:,:2], axis=0)
# [[(a[0],a[1]), a[2]] for a in array]
out = np.unique(array, axis=0, return_counts=True)
# out = np.unique(array[:,0], axis=0, return_counts=True)

out[0][:,2]=out[1]
final = out[0]
# out = np.array([[k, *np.unique(array[array[:,0] == k,1], return_counts=True)[1]] for k in np.unique(array[:,0])])
# print(final)

# # 0    Out,Short
# # 1    Out,Front
# # 2    In,Near Front Rim
# # 3    In,Center
# # 4    Out,Bounced
# # 5    In,Hit Back Rim
# # 6    Out,Near Back Rim
# # 7    Out,Long

x = array[:,0]
y = array[:,1]
z = array[:,2]

xy = np.vstack([final[:,0],final[:,1]])
g=gaussian_kde(xy)(xy)
idx = g.argsort()
ix, iy, iz = final[:,0][idx], final[:,1][idx], g[idx]
# sorted = array[array[:, 2].argsort()]
# segmented = np.split(sorted, np.where(np.diff(sorted[:,2]))[0]+1)

# print(len(segmented))
# plt.scatter(final[:,0], final[:,1], c="black", s=50)
# plt.scatter(final[:,0], final[:,1], c=final[:,2], s=10, cmap="inferno")
# plt.scatter(final[:,0], final[:,1], c=g, s=100, cmap="inferno")
plt.scatter(ix, iy, c=iz, s=100, cmap="inferno")
# plt.scatter(segmented[0][:,0], segmented[0][:,1], label='0')
# plt.scatter(segmented[1][:,0], segmented[1][:,1], label='1')
# plt.scatter(segmented[2][:,0], segmented[2][:,1], label='2')
# plt.scatter(segmented[3][:,0], segmented[3][:,1], label='3')
# plt.scatter(segmented[4][:,0], segmented[4][:,1], label='4')
# plt.scatter(segmented[5][:,0], segmented[5][:,1], label='5')
# plt.scatter(segmented[6][:,0], segmented[6][:,1], label='6')
# plt.scatter(segmented[7][:,0], segmented[7][:,1], label='7')

# # Best squares fit of 3rd degree polynomial
# z = np.polyfit(array[:,0], array[:,1], 2)
p2 = np.poly1d([0.0613688,-10.7311,3931.47])
p = np.poly1d([1.77919115e-02, 2.82493682e+00, 3.04914857e+03])

# d = np.poly1d([0, 9.0340148, 2506.2936])
d = lagrange([137.878, 221.240, 189.908, 161.817, 107.099],[3650.77,4648.52,4109.96,3781.16,3596.92])

d2 = np.poly1d([-0.000234425,0.226605,-46.2009,6316.95])

x_new = np.linspace(x[0], x[-1], 50)
y_new = p(x_new)
y_new2 = d2(x_new)
y_d = d(x_new)
y_d2 = p2(x_new)

# print(z)


# # def dans (x):
# #     return 3932.2829871-10.31606462*x+0.0584559726*x*x

# x = symbols("x")
# poly = sum(S("{:6.3f}".format(v))*x**i for i, v in enumerate(z[::-1]))
# eq_latex = printing.latex(poly)


plt.xlabel("Distance (inches)")
plt.ylabel("Setpoint RPM")
# plt.plot(x_new, y_new)
# plt.plot(x_new, y_new2)
# a = '$0.058x^2-10.316x+3932.283$'
# plt.plot(x_new, y_d)
# plt.plot(x_new, y_d2)
plt.legend(fontsize="small")
plt.show()

