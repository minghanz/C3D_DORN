# Monocular Depth Prediction through Continuous 3D Loss (DORN version)
This is the implementation of the paper [Monocular Depth Prediction Through Continuous 3D Loss](https://arxiv.org/abs/2003.09763) based on DORN network. The DORN implementation is based on this [repo](https://github.com/dontLoveBugs/SupervisedDepthPrediction), which is an unofficial Pytorch implementation. 

# Get started
The code is tested under under python 3.6, PyTorch 1.2.0, CUDA 10.1 on Ubuntu 18.04. 

## 1. Install C3D library
The key contribution of this work is the Continuous 3D (C3D) Loss. Install this library following the instuctions in [C3D](https://github.com/minghanz/c3d).

## 2. Data preparation (KITTI)
### a. KITTI raw dataset preparation (for training, borrowed from [Monodepth2](https://github.com/nianticlabs/monodepth2))
You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i datasets/lists/kitti_archives_to_download.txt -P <kitti_data>
```
Then unzip with
```shell
$ cd <kitti_data>
$ unzip "*.zip"
$ cd ..
```
The target folder `<kitti_data>` can be set to any path you prefer. **Warning:** the data weighs about **175GB**, so make sure to choose a path you have enough space to unzip too!

### b. KITTI depth dataset preparation (for evaluation)
Download the KITI depth dataset from this link [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).\
Then,
```shell
$ cd scripts
$ python unzip_kitti_depth_to_raw.py -s <path_to_data_depth_annotated.zip> -t <kitti_data>
```

### c. Interpolating KITTI raw LIDAR depth to dense depth images
This part is optional, if you want to train the network with raw KITTI LIDAR as depth supervison without the C3D loss as a baseline for comparison. Following the [DORN paper](https://arxiv.org/abs/1806.02446), we do interpolation on the raw LIDAR depth. 

Interpolation of the raw LIDAR depth is done using the tool in NYU-depth. The process is time-consuming for the whole KITTI dataset. Therefore we provided a link to download the interpolated files: [link](https://drive.google.com/drive/folders/11yv9FM5UV54yQrnj0DTd4Gn7abKa0f5J?usp=sharing). 

With the zip files downloaded, we can now extract them to merge with the raw KITTI folder: 
```shell
$ cd scripts
$ ./unzip_filled_depth.sh
```
Remember to edit the path to the downloaded zip files and target folder in `scripts/unzip_filled_depth.sh`.

## 3. Training
```shell
$ ./train.sh <num_of_gpu_to_use> -c <path_to_config_file>
```
For example: 
- Training using original DORN setup with **KITTI (densified) depth** as ground truth: 
```shell
$ ./train.sh <num_of_gpu_to_use> -c config/dorn_kitti.yaml
```
- Training using original DORN setup with interpolated **KITTI raw LIDAR depth** as ground truth: 
```shell
$ ./train.sh <num_of_gpu_to_use> -c config/dorn_kitti_filled.yaml
```
- Training using sparse **KITTI raw LIDAR depth** with our proposed **Continuous 3D (C3D) Loss**: 
```shell
$ ./train.sh <num_of_gpu_to_use> -c config/dorn_kitti_sparse.yaml
```
In all these examples, remember to change the `path` in config file to the `<kitti_data>` path you choose. 

## 4. Evaluation
```shell
$ cd scripts
$ python eval.py -c config/dorn_eval_kitti.yaml -r <path_to_checkpoint>
```
This script provides quantitative results on KITTI eigen test set using improved depth images. Optionally you can generate qualitative outputs (depth predictions, normal images, etc. ) using this script.  

Remember to change the `path` in `config/dorn_eval_kitti.yaml` to the `<kitti_data>` path you choose. A pretrained model can be downloaded at: [link](https://drive.google.com/drive/folders/1w2B9QvS9-1DZuLJ4K_tv8-qAtgpLR7Ea?usp=sharing). 