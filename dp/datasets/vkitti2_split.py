import os
import numpy as np

if __name__ == "__main__":
    data_root = "/mnt/storage8t/minghanz/Datasets/vKITTI2"
    list_train_name = "lists/vkitti2_train.list"
    list_val_name = "lists/vkitti2_trainval.list"
    list_test_name = "lists/vkitti2_test.list"
    
    path_list = []
    ### vKITTI2/Scene01/15-deg-left/frames/rgb/Camera_0
    for scene in os.listdir(data_root):
        variations = ["clone", "overcast", "morning", "fog"]
        for variation in variations:
            cameras = ["Camera_0", "Camera_1"]
            for camera in cameras:
                path = os.path.join(data_root, scene, variation, "frames", "rgb", camera)
                path_list_cur = [os.path.join(path, image).replace(data_root+"/", "") for image in os.listdir(path)]
                path_list = path_list + path_list_cur

    train_ratio = 0.90
    val_ratio = 0.05

    n_data = len(path_list)
    n_train = int(n_data * train_ratio)
    n_val = int(n_data * val_ratio)
    n_test = n_data - n_train - n_val

    path_list_random = np.random.permutation(path_list)

    path_train = path_list_random[:n_train]
    path_val = path_list_random[n_train:n_train+n_val]
    path_test = path_list_random[n_train+n_val:]

    with open(list_train_name, "w") as f:
        for path in path_train:
            path_dep = path.replace("rgb", "depth").replace("jpg", "png")
            f.write(path + " " + path_dep +"\n")

    with open(list_val_name, "w") as f:
        for path in path_val:
            path_dep = path.replace("rgb", "depth").replace("jpg", "png")
            f.write(path + " " + path_dep +"\n")

    with open(list_test_name, "w") as f:
        for path in path_test:
            path_dep = path.replace("rgb", "depth").replace("jpg", "png")
            f.write(path + " " + path_dep +"\n")

    