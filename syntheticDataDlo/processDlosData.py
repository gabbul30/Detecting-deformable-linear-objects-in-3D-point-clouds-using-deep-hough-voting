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

import pc_util


MAX_NUM_OBJ = 4


fieldOfView = 70
aspectRatio = 1.0

image_size = [512,512]

def rgbd_to_pcd(img_depth):
    global aspectRatio, fieldOfView
    
    fov_Y = fieldOfView / 180 * np.pi
    aspect_ratio = aspectRatio
    fov_X = 2 * np.arctan(np.tan(fov_Y * 0.5) * aspect_ratio)

    scale = 1000

    # depth_arr = img_depth*1./scale
    depth_arr = img_depth
    depth = o3d.geometry.Image((depth_arr).astype(np.int16))


    width = image_size[0]
    height = image_size[1]
    cx = width//2
    cy = height//2

    fx = width*0.5/np.tan(fov_X/2)
    fy = height*0.5/np.tan(fov_Y/2)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth,
        intrinsic=camera_intrinsic,
        # depth_scale=scale,
    )

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

rawDataPath = "dlos_data_0430/"

for idx in range(30000): # Hardcoded for the size of the dataset
    depthNum = idx % 300
    fileNum = str(int(idx/300)) # maybe change filnum to string
    print("Currently processing pointcloud:", idx, "| In folder:",fileNum, "| at folderindex:", depthNum)

    # Load depth image and create point cloud
    depth_image = cv2.imread(rawDataPath + fileNum + '/depth_' + str(depthNum) + '.png', cv2.IMREAD_UNCHANGED)
    point_cloud = rgbd_to_pcd(depth_image)
    processedPointCloud = np.asarray(point_cloud.points)
    print("pointCloud shape:", processedPointCloud.shape)
            
    # Calculating center, bounding box, and b-spline points at idx.
    Cable1 = np.load(rawDataPath + fileNum +'/dlo_positions_1.npy')[depthNum, :, :] # 31, 3
    Cable2 = np.load(rawDataPath + fileNum +'/dlo_positions_2.npy')[depthNum, :, :]


    # Bspline points
    labelPointsCable1 = np.asarray(fitting.approximate_curve(Cable1.tolist(), degree=2, ctrlpts_size=5).ctrlpts)
    labelPointsCable2 = np.asarray(fitting.approximate_curve(Cable2.tolist(), degree=2, ctrlpts_size=5).ctrlpts)
    bSplinePoints = np.dstack((labelPointsCable1, labelPointsCable2))
    bSplinePoints = bSplinePoints.transpose(2, 0, 1)
    print("bSplinePoints shape:", bSplinePoints.shape)
    # Centers
    labelCenterCable1 = np.mean(labelPointsCable1, axis=0)
    labelCenterCable2 = np.mean(labelPointsCable2, axis=0)

    
    # Vote label
    pointVotes = np.zeros((processedPointCloud.shape[0], 4))

    #Check if close to cable points
    for point in range(processedPointCloud.shape[0]):                  # 0.7 is a factor for the distance to check for points, this can be used for wider cables where the representing points are closer to eachother then to the radius of the cable
        controlDistance1 = np.linalg.norm(Cable1[0, :] - Cable1[1, :]) * 0.7 # The distance to check whether a point is so close to a "cable representation point" that it would belong to that cable
        for cablePoint in range(Cable1.shape[0]):
            if controlDistance1 > np.linalg.norm(Cable1[cablePoint, :] - processedPointCloud[point, :]):
                pointVotes[point, 0] = 1
                pointVotes[point, 1:4] = labelCenterCable1

                
        controlDistance2 = np.linalg.norm(Cable2[0, :] - Cable2[1, :]) * 0.7
        for cablePoint in range(Cable2.shape[0]):
            if controlDistance2 > np.linalg.norm(Cable2[cablePoint, :] - processedPointCloud[point, :]):
                pointVotes[point, 0] = 1
                pointVotes[point, 1:4] = labelCenterCable2
    print("pointVotes shape:", pointVotes.shape)

    np.save("processedDlosData/" + str(idx) + "_pointCloud.npy", processedPointCloud)
    np.save("processedDlosData/" + str(idx) + "_bSplinePoints.npy", bSplinePoints)
    np.save("processedDlosData/" + str(idx) + "_pointVotes.npy", pointVotes)


