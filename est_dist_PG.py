########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
Codes example: 
https://towardsdatascience.com/implementing-real-time-object-detection-system-using-pytorch-and-opencv-70bac41148f7 (explanation)
https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Drone_Human_Detection_Model.py
Yolov7 source:
https://github.com/WongKinYiu/yolov7
#Code edited by: Cristhiam González && Dorymar Gómez  
Distance estimation using a CAMERA ZED with pose estimation (YOLOv7-POSE) and distance estimation (using point cloud 3D coord)  
"""

from pathlib import Path
import cv2
import math
import numpy as np
from numpy import random
import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import torch
import torch.backends.cudnn as cudnn
#MQTT THINGSPEAK
import paho.mqtt.publish as publish
#import psutil
#import string

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#Code for point cloud method (distance estimation)
#Real time (camera)
#Camera info: 
#https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-#:~:text=The%20ZED%20and%20ZED%202,have%20a%206.3cm%20baseline

opt  = {
    
    "res_zed": 720,
    "weights": "yolov7x-pose.pt", # Path to weights file default weights are for nano model
    "img_size": 1080, # default image size (640) Other values 960
    "conf_thres": 0.35, # confidence threshold for inference. #0.25
    "iou_thres" : 0.3, # NMS IoU threshold for inference.
    "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : 0,  # list of classes to filter or None
    "kpt_label": True, #For keypoints pose-estimation
    "soc_distance": 200, #For social distancing
}

#Distance Estimation Code used:
#Características del texto
font = cv2.FONT_HERSHEY_TRIPLEX
tamañoLetra = 1
color_verde = (111, 252, 0) #Verde
color_rojo = (51, 71, 255) #Rojo
color_azul = (255, 128, 51) #Azul
grosorLetra = 2

# The ThingSpeak Channel ID.
# Replace <YOUR-CHANNEL-ID> with your channel ID.
channel_ID = "<YOUR-CHANNEL-ID>"

# The hostname of the ThingSpeak MQTT broker.
mqtt_host = "mqtt3.thingspeak.com"

# Your MQTT credentials for the device, enter yours
mqtt_client_ID = ""
mqtt_username  = ""
mqtt_password  = ""

t_transport = "websockets"
t_port = 80

# Create the topic string.
topic = "channels/" + channel_ID + "/publish"


def main():
    
    #Preparación yolov7_pose
    torch.cuda.empty_cache()
    
    colors = [random.randint(0, 255) for _ in range(3)]    #Dar un color random al box y labels in range [3] => B   G   R
    
    #Initialize
    device = select_device(opt['device'])
    print("\n\nDevice Used: ", device) #Mostrar en primera instancia que dispositivo se está usando
    half = False
    #half = False if device.type != 'cpu' else True  # half precision only supported on CUDA
    print("\n\nHalf bool: ", half) #Mostrar en primera instancia que dispositivo se está usando
    
    # Load model
    model = attempt_load(opt['weights'], map_location = device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    print("\n\nStride Value: ", stride) #Mostrar stride óptimo 
    
    if isinstance(opt['img_size'], (list, tuple)):
        assert len(opt['img_size']) == 2; "height and width of image has to be specified"
        opt['img_size'][0] = check_img_size(opt['img_size'][0], s = stride)
        opt['img_size'][1] = check_img_size(opt['img_size'][1], s = stride)
    else:
        opt['img_size'] = check_img_size(opt['img_size'], s = stride)  # check img_size
     
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.half().to(device) if half == True else model.to(device)
    
    # Directories
    #Para ir añadiendo según un path
    save_dir_2 = increment_path(Path('runs_depth/detect_avi/') / 'exp', exist_ok = False)  # increment run
    (save_dir_2).mkdir(parents = True, exist_ok = True)  # make dir
    save_path_2 = str(save_dir_2 / 'depth_est.avi')  # Dirección + nombre del archivo original
    
    # Initializing camera
    cudnn.benchmark = True  # set True to speed up constant image size inference
    print("Running camera ZEDv1...")
    
     # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt['img_size'], opt['img_size']).to(device).type_as(next(model.parameters())))  # run once
    
    ## Create a ZED camera object
    zed = sl.Camera()
    
    #Parámetros cámara 
    #Camera control
    init = sl.InitParameters() #Por defecto inicia en HD720, con fps = 0, svo_real_time_mode_ = false
    
    if (opt['res_zed'] == 720):
        
        init.camera_resolution = sl.RESOLUTION.HD720  # Set the resolution 720, etc
        init.camera_fps = 60  # Set the resolution 720, etc
        ##Camera settings (-1 es auto)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 5)       
        zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 3)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 6)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 4)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 6)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
    
    elif (opt['res_zed'] == 1080):
        init.camera_resolution = sl.RESOLUTION.HD1080  # Set the resolution 1080, etc
        init.camera_fps = 30  # Set the resolution 1080, etc
        ##Camera settings (-1 es auto)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 3)       
        zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 5)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 3)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 6)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
    
    #Sobre pŕofundidad
    init.depth_mode = sl.DEPTH_MODE.ULTRA  # Set the depth mode to ULTRA, PERFORMANCE, QUALITY, NEURAL (NO)
    init.coordinate_units = sl.UNIT.METER  #Set the units to show
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP #SIstema de cordenadas mano derecha ((Neg) z profundidad, derecha de la zed x y y hacía arriba de la zed)
    init.depth_minimum_distance = 0.5      # Set the minimum depth perception distance to 50cm
    init.depth_maximum_distance = 10       # Set the maximum depth perception distance to 10m
    
    if not zed.is_opened():
        print("Opening ZED Camera...") #Esperamos que la cámara ZED abrá
    status = zed.open(init)    
    
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(-1)
    
    
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)
    
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # #Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100  #Umbral editable (modo estandar)
    runtime_parameters.textureness_confidence_threshold = 100 #Umbral editable (modo estandar)
    
    #Para prueba videografíca
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #Para formato de video AVI
    
    #Para side by side *2
    vid_writer = cv2.VideoWriter(save_path_2, fourcc, 15, ((zed.get_camera_information().camera_resolution.width*2), (zed.get_camera_information().camera_resolution.height)))
    torch.cuda.empty_cache() 
    
    #Obtener imagen 
    image_est_dis = sl.Mat()
    #método #1, nube de puntos
    #point_cloud = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1) #Obtener profundidad apartir de la nube de puntos
    point_cloud = sl.Mat()
    print_camera_information(zed)

    print("Recording started...")
    print("Press Ctrl + C to stop recording: ")
    
    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)       # Create window with freedom of dimensions
    
    #Para registro de distancias:
    bool_dis_min = False
    dist_min = 0.0
    dist_max = 0.0
    dist = 0.0
    x_actual_1 = 0
    y_actual_1 = 0
    z_actual_1 = 0
    x_actual_2 = 0
    y_actual_2 = 0
    z_actual_2 = 0
    alerta = 0
    
    while True: 
    
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            #zed.retrieve_image(image_est_dis, sl.VIEW.LEFT)
            #zed.retrieve_image(image, sl.VIEW.RIGHT)
            zed.retrieve_image(image_est_dis, sl.VIEW.SIDE_BY_SIDE)
            x_svo = image_est_dis.get_data() 
            x_avi = cv2.cvtColor(x_svo, cv2.COLOR_RGBA2RGB)
            
            img = letterbox(x_avi, opt['img_size'], stride = stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB, to 3x416x416, transpose ordena a 2 0 1 (cambia el orden de la información)
            #OpenCV img = cv2.imread(path) loads an image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.
            #print("Transpose:", img)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment = False)[0]
            
            # Apply NMS, Identifica entidades duplicadas de la salida
            pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes = opt['classes'], agnostic = False, kpt_label = opt['kpt_label'])
            #print(f"Pred Info: {pred}")     #Mostrar los fps

            t2 = time_synchronized()

            # Print FPS (inference + NMS)
            fps_act = np.round((1/(t2 - t1)), 3)   #1/tiempo trascurrido entre cuadro, con una sensibilidad de 3 décimas
            print(f"FPS_processed: {fps_act}")     #Mostrar los fps
            
            # Process detections
            for det in pred:  # detections per image, se recorre las predicciones hechas
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    scale_coords(img.shape[2:], det[:, :4], x_avi.shape, kpt_label = False)
                    scale_coords(img.shape[2:], det[:, 6:], x_avi.shape, kpt_label = opt['kpt_label'], step = 3)
                        
            #Referencia desde la cabeza (Mitad de las dos orejas), #ref => (3, 4)
            #Cabeza
            xyxy = det[: , 15:20].cpu().numpy()  # xyxy are the box coordinates, sin normalizar y tiene todos los valores creo  # xyxy are the box coordinates, sin normalizar y tiene todos los valores creo-
            # # # Get and print distance value in mm at the center of the image
            color = (0, 255, 0)
            colors = ['green']*len(xyxy)
            for i in range(len(xyxy)): #Comparamos cuadro por cuadro (1 hasta el últ)
                for j in range(i+1, len(xyxy)):  #El 1 con 2, con 3 ..., el 2 con el 3, 4 .. etc.
                # Calculate distance of the centers
                    #Encontramos ambos centros
                    x, y = centro(xyxy[i]) #Obtenemos el centro de los dos puntos de referencia de cada persona 
                    x2, y2 = centro(xyxy[j]) 
                    
                    if (abs(x) < zed.get_camera_information().camera_resolution.width - 1) & ( abs(x2) < zed.get_camera_information().camera_resolution.width - 1): #Para solo obtener los valores en una "pantalla", vista izquierda se estima y la derecha se van actualizando datos de estimación
                        
                        #Método dos
                        #zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res) #Obtenemos el valor de puntos de nube de esa coordenanda 
                        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) #Obtenemos el valor de puntos de nube de esa coordenanda 
                        info_2, point_cloud_value = point_cloud.get_value(x, y) 
                        
                        #Obtenemos los 3 valores de la nube
                        x_actual_1 = point_cloud_value[0]
                        y_actual_1 = point_cloud_value[1] 
                        z_actual_1 = point_cloud_value[2]
                        
                        info_2, point_cloud_value_2 = point_cloud.get_value(x2, y2) 
                        
                        #Obtenemos los 3 valores de la nube
                        x_actual_2 = point_cloud_value_2[0]
                        y_actual_2 = point_cloud_value_2[1]
                        z_actual_2 = point_cloud_value_2[2]

                        if (((z_actual_2 > -5.5)&(z_actual_1 > -5.5))):   #Esto sirve para indicar que ha detectado una profundidad (es negativa debido al sistema de coordenadas actual)
                            
                            #Primer método    
                            #Euclidian vectorial equation
                            a = round(x_actual_2 - x_actual_1, 3)
                            b = round(y_actual_2 - y_actual_1, 3)
                            c = round(z_actual_2 - z_actual_1, 3)
                            #Obtención de la distancia
                            dist = euclidian_eq_vect (a*100, b*100, c*100) #A cm
                            
                            if dist < opt["soc_distance"]: #200cm para mostrar valores númericos si la distancia es menor a 200 cm
                            #if ((dist > 0)):     #para mostrar valores númericos si la distancia es mayor a 0 cm
                                colors[i] = 'red'
                                colors[j] = 'red'
                                alerta = 1
                                #Lineas
                                cv2.line(x_avi, (x, y), (x2, y2), (0, 0, 255), 2) #Dibujamos la linea
                                #Escribir texto
                                cv2.putText(x_avi, str(round(dist, 2)) + " cm", (int(np.mean([x, x2])), int(np.mean([y, y2]))), font, tamañoLetra, color_rojo, grosorLetra)
                            else: 
                                alerta = 0
            
            #Par ir actualizando la información de interés    
            if (bool_dis_min == False):
                dist_min = dist
                bool_dis_min = True
                
            if (dist_max == 0):
                dist_min = dist
            
            if ((dist < dist_min) & (bool_dis_min == True)):
                dist_min = dist
                
            if (dist > dist_max):
                dist_max = dist
            
            #Personas incumpliendo el protocolo
            contador = 0
            aforo = 0
            
            for i, (x1, y1, x2, y2) in enumerate(det[:, :4]):
                aforo += 1
                if (x1 < zed.get_camera_information().camera_resolution.width - 1) & ( x2 < zed.get_camera_information().camera_resolution.width - 1):
                    # Draw the boxes
                    if colors[i] == 'green':
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                        contador += 1
                    
                    cv2.rectangle(x_avi, (int(x1), int(y1)), (int(x2), int(y2)), color, 2) #Dibujar la caja
            #Escribir Info:
            aforox = round(aforo/2) #Aproximamos a arriba
            cv2.putText(x_avi, "Aforo aula: " + str(aforox), (((zed.get_camera_information().camera_resolution.width*2) - 500), 50), font, tamañoLetra, color_azul, grosorLetra)
            cv2.putText(x_avi, "Personas en riesgo: " + str(contador), (((zed.get_camera_information().camera_resolution.width*2) - 500), 100), font, tamañoLetra, color_azul, grosorLetra)    
            cv2.putText(x_avi, "Distancia max: " + str(round(dist_max, 2)), (((zed.get_camera_information().camera_resolution.width*2) - 500), 150), font, tamañoLetra, color_azul, grosorLetra)
            cv2.putText(x_avi, "Distancia min: " + str(round(dist_min, 2)), (((zed.get_camera_information().camera_resolution.width*2) - 500), 200), font, tamañoLetra, color_azul, grosorLetra)
                        
        vid_writer.write(x_avi)
        cv2.imshow("ZED", x_avi)
        cv2.waitKey(1)
        # build the payload string.
        payload = "field1=" + str(dist) + "&field2=" + str(aforox) + "&field3=" + str(contador) + "&field4=" + str(dist_max) + "&field5=" + str(dist_min) + "&field6=" + str(alerta) #Campo 1: Distancias, 2: Aforo , 3: Personas incumpliendo el distanciamiento, 4: Dis min, 5: Dis max y 6: Alerta
        # attempt to publish this data to the topic.
        try:
            #print ("Writing Payload = ", payload," to host: ", mqtt_host, " clientID= ", mqtt_client_ID, " User ", mqtt_username, " PWD ", mqtt_password)
            publish.single(topic, payload, hostname = mqtt_host, transport = t_transport, port = t_port, client_id = mqtt_client_ID, auth = {'username':mqtt_username,'password':mqtt_password})
        except:
            print ("There was an error while publishing the data.")


def euclidian_eq_vect (x1, y1, z1):
    # Get and print distance value in mm at the center of the image
    # We measure the distance camera - object using Euclidean distance
    
    dist = math.sqrt((x1 * x1) + (y1 * y1) + (z1 * z1))
    return dist 
 
def prom (rel_w1, rel_w2, x1, x2):
    
    a_x = abs( (((rel_w2 + rel_w1)/2))*(4*(x2-x1)) )
    return a_x 
            
def centro(xyxy):
        '''Calculate la mitad de ambos puntos escogidos'''
        a, b, c, d, e = xyxy #c es la confianza
        
        x1 = int(np.mean([a, d])) #Promedio
        y1 = int(np.mean([b, e])) #Promedio

        return x1, y1
        

def dist_to_cam_est (point_cloud_value):
    # Get and print distance value in mm at the center of the image
    # We measure the distance camera - object using Euclidean distance
    
    distance_z = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                        point_cloud_value[1] * point_cloud_value[1] +
                        point_cloud_value[2] * point_cloud_value[2])
    
    return distance_z

def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
    print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
    print("Firmware: {0}.".format(cam.get_camera_information().camera_firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))

if __name__ == "__main__":
    with torch.no_grad():
        main()
