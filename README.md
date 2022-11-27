# COMPUTER-VISION-DISTANCE-ESTIMATION
Distance estimation using YOLOv7 Pose estimation and POINT CLOUD method by Camera ZED.
This project is based on https://github.com/WongKinYiu/yolov7/tree/pose and https://github.com/stereolabs/zed-examples.


Steps:
0. We use L4T 32.7.1 Jetson linux for this project. Jetpack link for download: https://developer.nvidia.com/embedded/jetpack-sdk-462
1. Jetson Tx2 have problems with memory (eMMC) space limitation. We recommend buy a micro SD card of 64Gb at least. This video will be helpful (probably): https://www.youtube.com/watch?v=uPpVoX8fumA 
2. We tested on python 3.6.9 (default for L4T 32.7.1 Jetson linux), but a higher version would be more recommended. (

3. For this project, we use a ZED stereo camera. Please install ZED SDK: https://www.stereolabs.com/developers/release/

#This project was only tested by using ZED SDK for L4T 32.7.1 (with Jetpack 4.6.2 on Jetson Tx2 with CUDA 10.2).

4. Download this project.
5. Move COMPUTER-VISION-DISTANCE-ESTIMATION to usr/local/zed/samples/
6. Install requirements for YOLOv7 pose (pip || pip3 || your_method || install -r requierements.txt (
7. Copy the ogl viewer (It's necesary for using zed libraries) into COMPUTER-VISION-DISTANCE-ESTIMATION. This folder is in usr/local/zed/samples/depth sensing/python/ogl_viewer
8. And that's all.
