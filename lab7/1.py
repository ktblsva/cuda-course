import matplotlib.pyplot as plt
import numpy as np

ax=plt.subplot(111)
fn = open("x.txt")
n = fn.readlines()
n = [int(i) for i in n]

N = 20
for j in range(N):
	name = "y" + str(j) + ".txt"
	ft =  open(name)
	T = ft.readlines()
	T = [float (i) for i in T]
	lab = "f[" + str(j) + "]"
	ax.plot(n, T, label=lab)

ax.legend()
plt.show()
