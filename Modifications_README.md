# Detecting deformable linear objects in 3D point clouds using Deep Hough Voting


This repository includes the modifications done to VoteNet in order to both detect and represent deformable linear objects using NURBS. The modifications was done as a part of my bachelors thesis, to investigate if Deep Hough Voting could be used in order to detect and represent deformable linear objects.

The paper for the project:

https://urn.kb.se/resolve?urn=urn:nbn:se:oru:diva-115235

## Setup
For the setup and dependencies a docker file is included in the "docker" folder which contains all the libraries nessesary to run this repository. In the same folder, all python modules with their versions are also listed in "pythonModulesAndVersions.txt". For tensorboard to work, the latest version of tensorboard was used during the project.

For the rest of the setup, I suggest following the original README, which is included.

## Data preparation
The input data for training and testing needs to be prepared different from the original VoteNet.
This includes three main inputs:
* The point cloud of a scene. (NumPoints, 3)
* The "bSplinePoints" (NumObj, 15, 3) which are the control points generated from the curve of a deformable linear object.
* The "pointVotes" (NumPoints, 4). For each point in the point cloud, the first feature (pointVotes[:,0]) is a ZERO if the point is not located on an object and a ONE if it is located on an object. The rest of the features pointVotes[:,1:] is the centroid coordinate of the object the point belongs to. Note that the centroid of a DLO in this project is the mean of all control points.

These can all be generated with the "processDlosData.py" script which has DLO segments and a depth image as input. Note that in the project, 2 DLOs were used in each scene and if there are not 2 DLOs in each scene, some further modifications would be required to at least the preparation script for the data and the dataloader.

There would also have to be a train/test split on the data where I would refer to the dataloader (dlos_detection_dataset.py + model_util_dlos.py) and also the train test split (trainTestSplit.py).

## Training and testing
To train a model, use the following command:

    CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset dlos --log_dir logDir --no_height

The height feature was not used in this project, so the dataloader will not take height into account. That is why the flag "--no_height" is used.

To test and evaluate the model, use the following command:

    python3 eval.py --dataset dlos --checkpoint_path logDir/checkpoint.tar --dump_dir dumpDir --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal --no_height

These commands only differ from the original commands included in the original README by using the --no_height flag and using "dlos" as dataset. "dlos" uses the custom dataloader made for the project. "sunrgbd" and "scannet" will not work because they are not compatible with the modifications done to the algorithm. The accuracy used in the evaluation is described in the paper for this project.

For visualizing and comparing predictions to groud truth dlos from the dumpDir, a python script is included too in the "evals" folder.