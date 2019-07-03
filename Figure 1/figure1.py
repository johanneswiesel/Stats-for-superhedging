#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johannes Wiesel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

N = 10**3
one = np.ones((N,2))

points = np.random.rand(N, 3)*2
points[:, 2] = np.linalg.norm(points[:, 0:2]-one, axis=1)

points2 = points.copy()
points2[:, 2] = np.linalg.norm(points[:, 0:2] - one, axis=1) * (np.linalg.norm(points[:,0:2] - one,axis=1) <= 0.5) + \
      (np.linalg.norm(points[:,0:2] - one,axis=1) > 0.5) * (1 - np.linalg.norm(points[:,0:2] - one, axis=1))

points[0,:] = points2[0,:] = [0,0,-1]
points[1,:] = points2[0,:] = [2,0,-1]
points[2,:]= points2[0,:] = [0,2,-1]
points[3,:]= points2[0,:] = [2,2,-1]

#Compute convex hull
hull = ConvexHull(points-[[1,1,0]])
hull.close()

hull2 = ConvexHull(points2-[[1,1,0]])
hull.close()

#Plot 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = points[:,0]
y = points[:,1]
z = np.sqrt((x-1)**2+(y-1)**2)
z[0:4] = [0,0,0,0]
ax.plot_trisurf(x,y,z)

ax.scatter(points[4:,0], points[4:,1], points[4:,2], c=points[4:,2])
ax.view_init(20, 75)
ax.set_xticks([0, 0.5,1,1.5, 2])
ax.set_yticks([0, 0.5,1,1.5, 2])

#Plot 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = points[:,0]
y = points[:,1]
z = np.sqrt((x - 1)** 2 + (y - 1)** 2) * (np.sqrt((x-1)**2 + (y - 1)** 2) < 0.5) + \
    (np.sqrt((x - 1)** 2 + (y - 1)** 2) > 0.5)*(1 - np.sqrt((x - 1)** 2 + (y - 1)**2))
ax.plot_trisurf(x,y,z)

ax.scatter(points2[4:,0], points2[4:,1], points2[4:,2], c=points2[4:,2])
ax.view_init(20, 75)
ax.set_xticks([ 0, 0.5,1,1.5, 2])
ax.set_yticks([-0.5, 0, 0.5,1,1.5, 2])
