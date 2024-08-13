"""This program is for visualizing a downsampeled point cloud and visualize the points that counts as cables
It can be used to tune the cable width factor"""



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys
import open3d as o3d
import cv2
from geomdl import fitting
from model_util_dlos import dlosDatasetConfig

DC = dlosDatasetConfig()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util



idx = 0

# Get pointcloud from depth image at idx
processedPointCloud = np.load("processedDlosData/" + str(idx) + "_pointCloud.npy")
bSplinePoints = np.load("processedDlosData/" + str(idx) + "_bSplinePoints.npy")
pointVotes = np.load("processedDlosData/" + str(idx) + "_pointVotes.npy")



Votes = o3d.geometry.PointCloud()
Votes.points = o3d.utility.Vector3dVector(pointVotes[:, 1:4])

Seeds = o3d.geometry.PointCloud()
Seeds.points = o3d.utility.Vector3dVector(processedPointCloud[(pointVotes[:, 0] == 1), :])
print(processedPointCloud[(pointVotes[:, 0] == 1), :].shape)

bspline = o3d.geometry.PointCloud()
bspline.points = o3d.utility.Vector3dVector(np.vstack((bSplinePoints[0, :, :], bSplinePoints[1, :, :])))

pointCloud = o3d.geometry.PointCloud()
pointCloud.points = o3d.utility.Vector3dVector(processedPointCloud)


o3d.io.write_point_cloud(str(idx) + "_Votes.ply", Votes)
o3d.io.write_point_cloud(str(idx) + "_Seeds.ply", Seeds)
o3d.io.write_point_cloud(str(idx) + "_bspline.ply", bspline)
o3d.io.write_point_cloud(str(idx) + "_scene.ply", pointCloud)