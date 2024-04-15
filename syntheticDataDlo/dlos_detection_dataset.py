import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys
import open3d as o3d
import cv2
from geomdl import fitting



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))




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
       
    def __len__(self):
        return 30000

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

        depthNum = idx % 300
        fileNum = int(idx/300)
        
        # Get pointcloud from depth image at idx
        # point_cloud_depth = np.load(os.path.join(self.data_path, fileNum)+'depth_'+ depthNum +'.png')# Nx3 was Nx6 with color
        depth_image = cv2.imread(os.path.join(self.data_path, fileNum)+'depth_'+ depthNum +'.png', cv2.IMREAD_UNCHANGED)
        point_cloud = rgbd_to_pcd(depth_image)
        processedPointCloud = np.asarray(point_cloud.points)
        
        
        # Calculating center, bounding box, and b-spline points at idx.
        Cable1 = np.load(os.path.join(self.data_path, fileNum)+'dlo_positions_1.npy')[depthNum, :, :] # 300, 31, 3
        Cable2 = np.load(os.path.join(self.data_path, fileNum)+'dlo_positions_2.npy')[depthNum, :, :]
        # Bspline points (Label)
        labelPointsCable1 = fitting.aproximate_curve(Cable1.tolist(), degree=2, ctrlpts_size=5).evalpts
        labelPointsCable2 = fitting.aproximate_curve(Cable2.tolist(), degree=2, ctrlpts_size=5).evalpts
        # Centers
        labelCenterCable1 = np.sum(labelPointsCable1, axis=0)
        labelCenterCable2 = np.sum(labelPointsCable2, axis=0)
        # Bounding box
        



        point_votes = np.load(os.path.join(self.data_path, fileNum)+'_votes.npz')['point_votes'] # Nx10

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:,0] = -1 * bboxes[:,0]
                bboxes[:,6] = np.pi - bboxes[:,6]
                point_votes[:,[1,4,7]] = -1 * point_votes[:,[1,4,7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:,3:6] + MEAN_COLOR_RGB
                rgb_color *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                rgb_color += (0.1*np.random.random(3)-0.05) # color shift for each channel
                rgb_color += np.expand_dims((0.05*np.random.random(point_cloud.shape[0])-0.025), -1) # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0])>0.3,-1)
                point_cloud[:,3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random()*0.3+0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0)
            point_cloud[:,0:3] *= scale_ratio
            bboxes[:,0:3] *= scale_ratio
            bboxes[:,3:6] *= scale_ratio
            point_votes[:,1:4] *= scale_ratio
            point_votes[:,4:7] *= scale_ratio
            point_votes[:,7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio[0,0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0],:] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6]*2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size

        target_bboxes_mask = label_mask 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes
        return ret_dict