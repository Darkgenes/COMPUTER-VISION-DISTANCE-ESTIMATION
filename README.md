# COMPUTER-VISION-DISTANCE-ESTIMATION
Distance estimation using YOLOv7 Pose estimation and POINT CLOUD method by Camera ZED.
This project is based on https://github.com/WongKinYiu/yolov7 and https://github.com/stereolabs/zed-examples

Steps:

1. First of all we need to install ZED SDK:
https://www.stereolabs.com/developers/release/

This project was only tested by using ZED SDK for L4T 32.7 (Jetson Tx2) with CUDA 10.2, but probably it will work for other last versions of CUDA and ZED SDK.

2. Install python 3.8 in your Jetson Tx2 (recommended)
3. Download this project.
4. Move COMPUTER-VISION-DISTANCE-ESTIMATION to usr/local/zed/samples/
5. Install requirements for YOLOv7 pose (pip || pip3 || your_method || install -r requierements.txt
6. Copy the ogl viewer (It's necesary for using zed libraries) into COMPUTER-VISION-DISTANCE-ESTIMATION. This folder is in usr/local/zed/samples/deoth sensing/python/ogl_viewer
7. And that's all
