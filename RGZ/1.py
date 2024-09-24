import matplotlib.pyplot as plt
import numpy as np

ax=plt.subplot(111)
fn = open("N.dat")
n = fn.readlines()
n = [int(i) for i in n]

thrustF = open("dataThrust.dat")
thrust = thrustF.readlines()
thrust =  [float(i) for i in thrust]
ax.plot(n, thrust, label = "thrust")


cublasF = open("dataCublas.dat")
cublas = cublasF.readlines()
cublas =  [float(i) for i in cublas]
ax.plot(n, cublas, label = "cublas")


cudaF = open("dataCuda.dat")
cuda = cudaF.readlines()
cuda =  [float(i) for i in cuda]
ax.plot(n, cuda, label = "cuda")

og = open("dataOpengl.dat")
ogl = og.readlines()
ogl =  [float(i) for i in ogl]
ax.plot(n, ogl, label = "opengl glsl using std::chrono")

og1 = open("dataOpengl1.dat")
ogl1 = og1.readlines()
ogl1 =  [float(i) for i in ogl1]
ax.plot(n, ogl1, label = "opengl glsl using queries")

plt.xlim([0, 8388608])  # диапазон горизонтальной оси от 0 до 200
plt.ylim([0, 12])
ax.legend()
plt.show()
