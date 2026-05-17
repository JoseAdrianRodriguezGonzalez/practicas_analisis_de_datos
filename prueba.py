import numpy as np 

altura=np.array([1.62,1.72,1.85,1.82,1.72,1.55,1.65,1.77,1.83,1.53])
peso= np.array([53,71,85,86,76,62,68,77,97,65])
edad=np.array([20,30,25,24,23,25,26,20,33,24])
matrix_xy=np.corrcoef(altura,peso)
matrix_xz=np.corrcoef(altura,edad)
matrix_yz=np.corrcoef(peso,edad)
r_xy=matrix_xy[0,1]
r_xz=matrix_xz[0,1]
r_yz=matrix_yz[0,1]
print(r_xy,r_xz,r_yz)
r_partial=(r_xy-(r_xz*r_yz))/((1-r_xz**2)*(1-r_yz**2))**(1/2)
print(r_partial)

