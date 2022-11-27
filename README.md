# COMPUTER-VISION-DISTANCE-ESTIMATION
Distance estimation using YOLOv7 Pose estimation and POINT CLOUD method by Camera ZED.
This project is based on https://github.com/WongKinYiu/yolov7/tree/pose and https://github.com/stereolabs/zed-examples.


Steps:
0. We use L4T 32.7.2 Jetson linux for this project. Jetpack link for download: https://developer.nvidia.com/embedded/jetpack-sdk-462
1. Jetson Tx2 have problems with memory (eMMC) space limitation. We recommend buy a micro SD card of 64Gb at least. This video will be helpful (probably): https://www.youtube.com/watch?v=uPpVoX8fumA 
2. We tested on python 3.6.9 (default for L4T 32.7.1 Jetson linux), but a higher version would be more recommended.

3. For this project, we use a ZED stereo camera. Please install ZED SDK: https://www.stereolabs.com/developers/release/

#This project was only tested by using ZED SDK for L4T 32.7.2 (with Jetpack 4.6.2 on Jetson Tx2 with CUDA 10.2).

4. Download this project.

5. Move COMPUTER-VISION-DISTANCE-ESTIMATION to usr/local/zed/samples/

6. Install requirements for YOLOv7 pose (pip || pip3 || your_method || install -r requierements.txt. 

#It's probably ypu need to install pytorch apart. For that, you need to asegure the compatibility for versions first: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

#In this case, we use 32.7.2 (is the same as 32.7.1 + security fixes). It would be PyTorch v1.9.0, torchvision v0.10.0, torchaudio v0.9.0 then. 
#This link will be helpful for you (maybe):

https://medium.com/hackers-terminal/installing-pytorch-torchvision-on-nvidias-jetson-tx2-81591d03ce32#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjI3Yjg2ZGM2OTM4ZGMzMjdiMjA0MzMzYTI1MGViYjQzYjMyZTRiM2MiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2NjkzMjg0NDIsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExMjcwODI0MDc0MTk4OTE1NTIxMyIsImVtYWlsIjoiY3Jpc3RoaWFtLjA2MDIwMUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IkNyaXN0aGlhbSBGZWxpcGUgR29uesOhbGV6IE3DqW5kZXoiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUxtNXd1M0tCZEFidlFkbHBid0s2NE9XUDJIWG5TZHlKZTBqSzdCZG9FSHBvQT1zOTYtYyIsImdpdmVuX25hbWUiOiJDcmlzdGhpYW0gRmVsaXBlIiwiZmFtaWx5X25hbWUiOiJHb256w6FsZXogTcOpbmRleiIsImlhdCI6MTY2OTMyODc0MiwiZXhwIjoxNjY5MzMyMzQyLCJqdGkiOiIzNDkzMTU1MzY2ZDE5MzI2MGI0ZTRjMjRkNGY4YjY2ZjI4ZTA0YzQ3In0.nlxARpTRNTzJfIFiH4BVfhbe60T8iIggatixmgSsS2tQ0qAL3mnLoBP257mtiqqmcYdDZR2ABgySy7QLclektdxncTEDc_2-eMZUSSZXIC2ug4UNiHPNudqDG72ieuEI_7J_XPTBtOges4RMxVf1HcihAZARpq_Wuff1HACn0-_QJpFZOL7IHdLE9FQhSNpuF2ltZVQAakN4Dm0I6qfGoyP3yo28RwMzxO-YK_aEdRB1rIA9-MBp27nxlJihWDbyLt0B5oXprLxuqKtuOfVdr9_R4sBTXVHJxiKCKxeQYamSBWu7DOc1vmHLfpESq-HmqD41dI6k_bKDT8RX8aJung

#Or this another link:

https://qengineering.eu/install-pytorch-on-jetson-nano.html

#Another thing to consider is numpy version. Probably you will need to downgrade the version to 1.19.4 instead of 1.19.5 (last version)

7. Copy the ogl viewer (It's necesary for using zed libraries) into COMPUTER-VISION-DISTANCE-ESTIMATION. This folder is in usr/local/zed/samples/depth sensing/python/ogl_viewer

8. And that's all.
