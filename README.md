# MV3D(In progress)

This is an experimental Tensorflow implementation of MV3D - a ConvNet for object detection with Lidar and Mono-camera. 
And this work based on the code of [hengck23](https://github.com/hengck23/didi-udacity-2017) 

For details about MV3D please refer to the paper [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/abs/1611.07759) by [Xiaozhi Chen](https://arxiv.org/find/cs/1/au:+Chen_X/0/1/0/all/0/1), [Huimin Ma](https://arxiv.org/find/cs/1/au:+Ma_H/0/1/0/all/0/1), [Ji Wan](https://arxiv.org/find/cs/1/au:+Wan_J/0/1/0/all/0/1), [Bo Li](https://arxiv.org/find/cs/1/au:+Li_B/0/1/0/all/0/1), [Tian Xia](https://arxiv.org/find/cs/1/au:+Xia_T/0/1/0/all/0/1).

### Requirements: software

1. Requirements for Tensorflow 1.1  (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`, `mayavi (for visualization)` 

### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with two ResNet50, 8G of GPU memory is required (using CUDNN)

### Installation 
1. 使用https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh ；ref : https://www.continuum.io/downloads#linux install
```Shell
    conda create -n tensorflow python=3.5
    source activate tensorflow
    conda install -c menpo opencv3=3.2.0
    conda install matplotlib simplejson pandas

    cuda install: https://developer.nvidia.com/cuda-downloads

    按照python35 tensorflow gpu: ref https://www.tensorflow.org/install/install_linux 安装这里要求的cudnn版本 https://developer.nvidia.com/rdp/cudnn-download
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-linux_x86_64.whl
    
    conda install -c menpo mayavi
    conda install pyqt=4
    pip install easydict
    pip install pillow

    roi_pooling.so不同机器需重新编译；
    cd $MV3D/net/roipooling_op/
    目录下有make.sh 编译;查看 make.sh 文件；使用 /usr/local/cuda/bin/nvcc 80版本nvcc编译;
```

2. Downloads KITTI object datasets.
```Shell
 % Specify KITTI data path so that the structure is like
 % {kitti_dir}/object/training/image_2
 %                            /image_3
 %                            /calib
 %                            /lidar_bv
 %							 /velodyne   
 % {kitti_dir}/object/testing/image_2
 %                           /image_3
 %                           /calib
 %                           /lidar_bv
 %							/velodyne
```

3. Make Lidar top View data
%Edit your data path:
```shell
vim $MV3D/net/configuration.py:

    CFG.PATH.TRAIN.KITTI = "{kitti_dir}/object/training"
    CFG.PATH.TRAIN.TARGET = 'your path to save the top view data'  
```
%Make data
```shell
cd $MV3D/data
python generate_top_view_data.py
```
% Generate groundtruth file
```shell
cd $MV3D/data
python generate_gt.py

```

4. Download pre-trained ImageNet models
Download the pre-trained ImageNet models [tensorflow checkpoint vgg16 and ResNet V1 50](https://github.com/tensorflow/models/tree/master/slim)
```Shell
    mv resnet_v1_50.ckpt $MV3D/outputs/check_points/resnet_v1_50.ckpt
    mv vgg_16.ckpt $MV3D/outputs/check_points/vgg_16.ckpt
```

5. Run script to train model 

```Shell
 cd $MV3D
 python ./tools/train_ResNet_vgg_double_up_c.py
```

6. Run script to test model 

```Shell
 cd $MV3D
 python ./tools/train_val.py
```

### 代码细节


####1. 大多数的网络超参数可以在 ./net/configuration里面配置
-  CFG.TRAIN.RCNN_JITTER：rcnn ground truth 抖动开关，用来增加rcnn训练过程中的正样本数量
-  CFG.TRAIN.RCNN_JITTER_NUMS ：每个GT 生成抖动样本的个数
-  CFG.TRAIN.RPN_OHEM ：rpn ohem开关
- CFG.TRAIN.RPN_BALANCE_POS_NUMS ; rpn 阶段均匀采样正样本开关

####2.通过修改./tool/train.py 285到298行的注释可以选择不同的预训练模型进行初始化

####3.跑test_val.py测试网络的时候也要修改对应的checkpoint文件，网络才能正确加载模型

####4.网络定义在network.py。

####5.代码的结构和faster rcnn 基本一样，只不过是rpn回归的是3D proposals ，由于代码实现过程中的历史原因，3D proposal 的时候分别回归了bird view 的2D proposals 和高度的下标和上标。

####6. ./data/generate_gt.py 是转换KITTI的3D bounding box groundtruth的脚本，转换后会得到8个顶点的坐标按顺序分别是（车后往前看）车后背左下角，车后背右下角，车前脸右下角，车前脸左下角，车后背左上角，车后背右上角，车前脸右上角，车前脸左上角。另外新做的标注数据工具也是按照这个顺序标注的。

####7.如果想测试视频片段，先修改./data/generate_top_view_data的注释转换数据，然后修改./tools/test_tracklet.py 的相应路径，运行该测试代码即可


### Examples

Image and corresponding Lidar map 

1.

![figure_20](examples/result_video_img_00056.png)

2.

![figure_20](examples/result_video_img_00111.png)

3.

![figure_20](examples/result_video_img_00190.png)

### References

[Lidar Birds Eye Views](http://ronny.rest/blog/post_2017_03_26_lidar_birds_eye/)

[part.2: Didi Udacity Challenge 2017 — Car and pedestrian Detection using Lidar and RGB](https://medium.com/@hengcherkeng/part-1-didi-udacity-challenge-2017-car-and-pedestrian-detection-using-lidar-and-rgb-fff616fc63e8)

[Faster_RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[TFFRCNN](https://github.com/CharlesShang/TFFRCNN)

