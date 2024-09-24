import matplotlib.pyplot as plt
import numpy as np

ax=plt.subplot(111)
fn = open("x.txt")
n = fn.readlines()
n = [int(i) for i in n]

ft =  open("y.txt")
T = ft.readlines()
T = [float (i) for i in T]

ft1 =  open("y1.txt")
T1 = ft1.readlines()
T1 = [float (i) for i in T1]
#ax.plot(n, T)
ax.plot(T1, T)
ax.legend()
plt.show()
