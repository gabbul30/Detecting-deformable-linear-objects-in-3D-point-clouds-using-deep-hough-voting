# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert(CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
        augment=False, use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
elif FLAGS.dataset == 'dlos':
    sys.path.append(os.path.join(ROOT_DIR, 'syntheticDataDlo'))
    from dlos_detection_dataset import dlosDetectionDataset, MAX_NUM_OBJ
    from model_util_dlos import dlosDatasetConfig
    DATASET_CONFIG = dlosDatasetConfig()
    TEST_DATASET = dlosDetectionDataset('test', num_points=NUM_POINT,
        augment=False, use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset, num_workers=4, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}
# ------------------------------------------------------------------------- GLOBAL CONFIG END

# Objectness prob from dump helper, softmax
def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def evaluate_one_epoch():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval() # set model to eval mode (for bn and dp)

    # Used for the accuracy of detecting DLOs
    totalObjects = 0
    detectedObjects = 0
    allDetectedDistances = np.zeros(5)
    forwardPassTimes = []

    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}

        if inputs['point_clouds'].shape[0] == BATCH_SIZE:
            forwardPassTime = time.time()

        with torch.no_grad():
            end_points = net(inputs)

        if inputs['point_clouds'].shape[0] == BATCH_SIZE:
            forwardPassTimes.append(time.time() - forwardPassTime)
            print("forward pass time:", forwardPassTimes[batch_idx])


        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    
        # Dump evaluation results for visualization
        if batch_idx == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

        # calculate dlos accuracy
        confidence_threshold = 0.50
        objectness_scores = end_points['objectness_scores'].detach().cpu().numpy()
        pred_bspline = end_points['bSplinePoints'].detach().cpu().numpy()
        control_points = end_points['controlPoints'].detach().cpu().numpy()
        objectLabel = end_points['box_label_mask'].detach().cpu().numpy()
        pred_mask = end_points['pred_mask']

        for scene in range(control_points.shape[0]): # For every scene in a batch
            # From dump helper
            objectness_prob = softmax(objectness_scores[scene,:,:])[:,1]

            # Debug
            # print("Bspline confident shape:", pred_bspline[scene,np.logical_and(objectness_prob>confidence_threshold, pred_mask[scene,:]==1),:].shape)

            # Used to track if an object is detected, 1 if an object has a detected status, 0 if not detected.
            numObjectsInScene = int(np.sum(objectLabel[scene,:]))
            detected = np.zeros(numObjectsInScene)

            confident_predictions = pred_bspline[scene,np.logical_and(objectness_prob>confidence_threshold, pred_mask[scene,:]==1),:]
            for pred in range(confident_predictions.shape[0]):# For each prediction, check distances to ground truths.

                closestToGT = 0 # To keep track of which ground truth the prediction is closest to
                closestDistance = np.zeros(5) # Distances from predicted points to gt

                for gt in range(numObjectsInScene): # To iterate through all objects in a scene.
                    distance = np.zeros(5)
                    revDistance = np.zeros(5)
                    
                    prediction = confident_predictions[pred, :].reshape((5,3))
                    revPrediction = np.flip(prediction, axis=0)

                    groundtruth = control_points[scene, gt, :].reshape((5,3))

                    for point in range(5): # for each point in a DLO
                        distance[point] = np.linalg.norm(prediction[point,:] - groundtruth[point,:]) # Test one direction of the DLO
                        revDistance[point] = np.linalg.norm(revPrediction[point,:] - groundtruth[point,:]) # Test the other direction

                    if np.sum(distance) > np.sum(revDistance): # The shortest distance stored in "distance"
                        distance = revDistance

                    if gt == 0:
                        closestDistance = distance
                    else:
                        if np.sum(distance) < np.sum(closestDistance):
                            closestToGT = gt
                            closestDistance = distance
                
                detected[closestToGT] = 1
                if batch_idx == 0 and scene == 0 and pred == 0:
                    allDetectedDistances = closestDistance
                else:
                    allDetectedDistances = np.vstack((allDetectedDistances,closestDistance))
            
            totalObjects = totalObjects + numObjectsInScene
            detectedObjects = detectedObjects + int(np.sum(detected))
        #     print("Total objects:", totalObjects) For debugging
        #     print("Detected objects", detectedObjects)
        #     print("Shape of allDetectedDistances", allDetectedDistances.shape)
        # print("All allDetectedDistances after scene", allDetectedDistances)
                

    # Print accuracy!
    print(f"\nOut of {totalObjects} DLOs, {detectedObjects} DLOs were detected from {allDetectedDistances.shape[0]} predictions\ngiving the model a DLO detection rate of {detectedObjects/totalObjects}")
    print(f"\n========Statistics from predicted points========")
    print("\nDistance from predicted points to control points is")
    print("The mean:", np.mean(allDetectedDistances))
    print("The standard deviation:", np.std(allDetectedDistances))
    print("The median:", np.median(allDetectedDistances))

    plt.scatter(np.arange(allDetectedDistances.shape[0]),np.mean(allDetectedDistances, axis=1),c="green")
    plt.title("Mean")
    plt.show()

    plt.scatter(np.arange(allDetectedDistances.shape[0]),np.sort(np.mean(allDetectedDistances, axis=1)),c="green")
    plt.title("Mean sorted")
    plt.show()
    
    print("\nDistance for each closest control point prediction for each DLO is")
    print("The mean:", np.mean(np.min(allDetectedDistances,axis=1)))
    print("The standard deviation:", np.std(np.min(allDetectedDistances,axis=1)))
    print("The median:", np.median(np.min(allDetectedDistances,axis=1)))

    plt.hist(np.mean(allDetectedDistances, axis=1),bins=40)
    plt.title("Mean histogram")
    plt.show()

    print("Avarage time for a forward pass with a batch size of", BATCH_SIZE ,":", np.mean(forwardPassTimes))



    #  For standard object detection
    # Log statistics
    # for key in sorted(stat_dict.keys()):
    #     log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # # Evaluate average precision
    # for i, ap_calculator in enumerate(ap_calculator_list):
    #     print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
    #     metrics_dict = ap_calculator.compute_metrics()
    #     for key in metrics_dict:
    #         log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    eval()
