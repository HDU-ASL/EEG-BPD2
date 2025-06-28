

import os  
from PIL import Image
import numpy as np  




def read_pose(filename0):  
    poses = []  
    timestamp0= []

    with open(filename0, 'r') as file0:  

        lines0 = file0.readlines()  # 读取所有行到列表中，以便通过索引访问  
  
    for i in range(len(lines0)): 
            if i==2760:
                break
            line=lines0[i]
            data = line.split()  
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])  
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])  
            q = np.array([qw, qx, qy, qz])  
            t = np.array([tx, ty, tz, 1.0])  
            T=np.array([tx, ty, tz, qw, qx, qy, qz])  
            poses.append(T)  

  
    return poses  
def quaternion_to_rotation_matrix(q):  
    """  
    Convert quaternion coefficients (in w, x, y, z order) to a rotation matrix.  
    """  
    w, x, y, z = q  
    Nq = np.dot(q, q)  
    s = 2.0 / Nq  
    X = x * s  
    Y = y * s  
    Z = z * s  
    wX = w * X  
    wY = w * Y  
    wZ = w * Z  
    xX = x * X  
    xY = x * Y  
    xZ = x * Z  
    yY = y * Y  
    yZ = y * Z  
    zZ = z * Z  
  
    rotation_matrix = np.array([  
        [1 - (yY + zZ), xY - wZ, xZ + wY],  
        [xY + wZ, 1 - (xX + zZ), yZ - wX],  
        [xZ - wY, yZ + wX, 1 - (xX + yY)]  
    ], dtype=np.float64)  
  
    return rotation_matrix  
