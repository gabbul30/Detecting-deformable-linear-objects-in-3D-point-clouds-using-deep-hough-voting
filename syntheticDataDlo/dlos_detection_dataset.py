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



class dlosDetectionDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, use_v1=False,
        augment=False, scan_idx_list=None):

        self.raw_data_path = os.path.join(ROOT_DIR, 'syntheticDataDlo/dlos_data')
        self.num_points = num_points
        self.augment = augment
        
        self.scanNumbers = np.load(os.path.join(self.raw_data_path, (split_set + ".npy")))
       
    def __len__(self):
        return len(self.scanNumbers)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        idx = self.scanNumbers[idx]
        depthNum = idx % 300
        fileNum = str(int(idx/300)) # maybe change filnum to string

        # Get pointcloud from depth image at idx
        # point_cloud_depth = np.load(os.path.join(self.data_path, fileNum)+'depth_'+ depthNum +'.png')# Nx3 was Nx6 with color
        depth_image = cv2.imread(os.path.join(self.raw_data_path, fileNum)+'depth_'+ depthNum +'.png', cv2.IMREAD_UNCHANGED)
        point_cloud = rgbd_to_pcd(depth_image)
        processedPointCloud = np.asarray(point_cloud.points)
        
        
        # Calculating center, bounding box, and b-spline points at idx.
        Cable1 = np.load(os.path.join(self.raw_data_path, fileNum)+'dlo_positions_1.npy')[depthNum, :, :] # 31, 3
        Cable2 = np.load(os.path.join(self.raw_data_path, fileNum)+'dlo_positions_2.npy')[depthNum, :, :]
        # Bspline points
        labelPointsCable1 = np.asarray(fitting.approximate_curve(Cable1.tolist(), degree=2, ctrlpts_size=5).ctrlpts)
        labelPointsCable2 = np.assaray(fitting.approximate_curve(Cable2.tolist(), degree=2, ctrlpts_size=5).ctrlpts)
        # Centers
        labelCenterCable1 = np.mean(labelPointsCable1, axis=0)
        labelCenterCable2 = np.mean(labelPointsCable2, axis=0)
        labelCenters = np.vstack((labelPointsCable1, labelPointsCable2))
        # Bounding boxes
        maxCorner1 = np.max(labelPointsCable1, axis=0)
        minCorner1 = np.min(labelPointsCable1, axis=0)
        size1 = maxCorner1 - minCorner1
        bbox1 = np.concatenate((labelCenterCable1, size1, 0, 0), axis=1) # angle forced to 0 and class to 0 (for class cable)
        
        maxCorner2 = np.max(labelPointsCable2, axis=0)
        minCorner2 = np.min(labelPointsCable2, axis=0)
        size2 = maxCorner2 - minCorner2
        bbox2 = np.concatenate((labelCenterCable2, size2, 0, 0), axis=1) # angle forced to 0 and class to 0 (for class cable)

        bboxes = np.hstack((bbox1, bbox2))

        # Size class
        sizeClasses = np.zeros((MAX_NUM_OBJ,))
        sizeResiduals = np.zeros((MAX_NUM_OBJ, 3))
        sizeClasses[0], sizeResiduals[0] = DC.size2class(size1, DC.class2type[0])
        sizeClasses[1], sizeResiduals[1] = DC.size2class(size2, DC.class2type[0])

        # Box label mask
        labelMask = np.zeros((MAX_NUM_OBJ))
        labelMask[0:centers.shape[0]] = 1

        # Vote label (not done)
        pointCloud, choices pc_util.random_sampling(processedPointCloud, self.num_points, return_choices=True)
        pointVotes = np.zeros((processedPointCloud.shape(0), 10))
        # Here is a check if a point is in a bounding box.
        for point in range(processedPointCloud.shape(0)):
            for box in range(bboxes.shape(0)):
                tempPoint = processedPointCloud[point] - bboxes[box][:3]
                if not np.any(tempPoint < (-bboxes[box][3:6])/2):
                    if not np.any(tempPoint > bboxes[box][3:6]/2):
                        pointVotes[point][0] = 1
                        pointVotes[point][1:4] = bboxes[box][:3]
        pointVotes = np.tile(pointVotes, (1, 3)) # From scannet dataloader
        pointVotesMask = pointVotes[choices, 0] # From sunrgbd dataloader
        pointVotes = pointVotes[choices, 1:] # From sunrgbd dataloader

        # Max boxes
        maxBboxes = np.zeros((MAX_NUM_OBJ, 8))
        maxBboxes[0:bboxes.shape[0], :] = bboxes

        # done
        ret_dict = {}
        ret_dict['point_clouds'] = pointCloud.astype(np.float32)
        ret_dict['center_label'] = labelCenters.astype(np.float32)
        ret_dict['heading_class_label'] = np.zeros((MAX_NUM_OBJ,)).astype(np.int64)
        ret_dict['heading_residual_label'] = np.zeros((MAX_NUM_OBJ,)).astype(np.float32)
        ret_dict['size_class_label'] = sizeClasses.astype(np.int64)
        ret_dict['size_residual_label'] = sizeResiduals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # classes are only cables now
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = labelMask.astype(np.float32)
        ret_dict['vote_label'] = pointVotes.astype(np.float32)
        ret_dict['vote_label_mask'] = pointVotesMask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = maxBboxes
        
        return ret_dict