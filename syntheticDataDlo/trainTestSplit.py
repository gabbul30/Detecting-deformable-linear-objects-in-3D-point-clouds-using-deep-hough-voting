import numpy as np
# This program splits the dlos data into train and test and shuffles the data
SIZE_DATASET = 30000
USE_EVERY_NUM_SCENE = 4 # Make sure the dataset size can be devided by this number
TRAIN_SIZE = 0.8


if __name__ == "__main__":
    dataset = np.arange(int(SIZE_DATASET/USE_EVERY_NUM_SCENE))
    dataset = dataset * USE_EVERY_NUM_SCENE
    print(f"Size of dataset {SIZE_DATASET} | Size of train {int(len(dataset)*TRAIN_SIZE)} | Size of test {int(len(dataset)-len(dataset)*TRAIN_SIZE)}")
    print("Ordered:\n",dataset)
    np.random.shuffle(dataset)
    print("Un-ordered:\n",dataset)
    train = dataset[:int(len(dataset)*TRAIN_SIZE)]
    test = dataset[int(len(dataset)*TRAIN_SIZE):]
    print("Train shape:", train.shape, "Test shape:", test.shape)
    np.save("processedDlosData/train.npy",train)
    np.save("processedDlosData/test.npy", test)

    for cable in range(len(train)): # To calculate the mean size of a cable (ONLY in training set)
        trainSetIdx = np.load("processedDlosData/train.npy")
        # Bspline points
        bSplinePoints = np.load("processedDlosData/" + str(trainSetIdx[cable]) + "_bSplinePoints.npy")
        labelPointsCable1 = bSplinePoints[0, :, :]
        labelPointsCable2 = bSplinePoints[1, :, :]
        # Bounding boxes
        maxCorner1 = np.max(labelPointsCable1, axis=0)
        minCorner1 = np.min(labelPointsCable1, axis=0)
        size1 = maxCorner1 - minCorner1

        maxCorner2 = np.max(labelPointsCable2, axis=0)
        minCorner2 = np.min(labelPointsCable2, axis=0)
        size2 = maxCorner2 - minCorner2

        sizes = np.vstack((size1, size2))
        if cable == 0:
            allSizes = sizes
        else:
            allSizes = np.vstack((allSizes, sizes))
    print("How many cables are in the mean size calculation: ", allSizes.shape[0])
    print("Mean size of the training set: ", np.mean(allSizes, axis=0))
    print("Include the above as the mean size of the class in \"model_util_dlos\"")