#importing UI Libraries
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename 
from PIL import ImageTk, Image

#importing backend Libraries
import tensorflow as tf
from numpy import expand_dims
import keras
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
import cv2
import numpy as np
import keras.backend as K
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import pandas as pd
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import random
import os

########################################################################
#backend 
model = load_model('models/model.h5', compile=False)


input_w, input_h = 416, 416

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


#importing capsnet model and functions
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))

custom_objects = { 'CapsuleLayer': CapsuleLayer, 
                   'Mask': Mask, 
                   'Length': Length, 
                   'PrimaryCap': PrimaryCap,
                   'margin_loss': margin_loss
                 }
model_name = 'models/capsnet_1600_0005LW.h5'
capsNet_model = load_model(model_name, custom_objects=custom_objects)
IMG_SIZE = 128


#functions
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    nb_box = 3 # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

#intersection over union        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    #Union(A,B) = A + B - Inter(A,B)
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):    #boxes from correct_yolo_boxes and  decode_netout
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    # image = load_img(filename) #load_img() Keras function to load the image .
    # print(image)
    height, width, unknown = filename.shape
    #print(filename.shape)
    # load the image with the required size
    #image = load_img(filename, target_size=shape) # target_size argument to resize the image after loading
    filename = cv2.resize(filename, shape)
    # convert to numpy array
    image = img_to_array(filename)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0  #rescale the pixel values from 0-255 to 0-1 32-bit floating point values.
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
#     print(v_boxes.shape)
    return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores, pred):
    capsnet_label = {0:'With Mask', 1:"Mask Incorrect", 2:"Other Covering", 3:"no mask"}
    #load the image
    # img = cv2.imread(filename)
   
    img = filename.copy()
    for i in range(len(v_boxes)):
        # retrieving the coordinates from each bounding box
        box = v_boxes[i]
        label = capsnet_label[pred[i]]
        # get coordinates
        
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        start_point = (x1, y1) 
        # Ending coordinate
        # represents the bottom right corner of rectangle
        
        end_point = (x2, y2) 
        # Red color in BGR 
        color = (0, 255, 0)
        if pred[i] == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # Line thickness of 2 px 
        thickness = 2
        # font 
        font = cv2.FONT_HERSHEY_PLAIN 
        # fontScale 
        fontScale = 1.5
        #create the shape
        img = cv2.rectangle(img, start_point, end_point, color, thickness) 
        # draw text and score in top left corner
        #label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        cv2.putText(img, label, (x1,y1), font,  
                      fontScale, color, thickness, cv2.LINE_AA)
    return img, label
        
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]  

# define the probability threshold for detected objects
class_threshold = 0.6

labels = ["face"]

def detect_mask(file):
    imagepath = file
    image_name = os.path.basename(imagepath)
    root_dir = os.path.abspath(".")
    save_path = os.path.join(root_dir, "Image_Results/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(imagepath):
        print("[INFO]: IMAGE FOUND")
    else:
        print("[ERROR]: IMAGE DOES NOT EXIST")

    image = cv2.imread(imagepath) #read input

    input_w, input_h= 128,128 #set input height and width for yolov3 model 
     
    height, width, layers = image.shape
    size = (width, height)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = np.stack((gray_image,)*3, axis=-1)

    frame_pred, image_w, image_h = load_image_pixels(frame, (input_w, input_h))
    yhat = model.predict(frame_pred)
    v_boxes = 0
    boxes = list()

    for i in range(len(yhat)):
                    # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
                    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

                # suppress non-maximal boxes
    do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

                # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)


    # draw what we found
    if v_boxes:
        pred = list()
        for i in range (len(v_boxes)):
            box = v_boxes[i]
            padding = 0.10
            if box.xmin < box.xmin-round(padding*(box.xmax-box.xmin)):
                box.xmin = box.xmin
            else: 
                box.xmin -= round(padding*(box.xmax-box.xmin))
            if box.xmin < 0:
                box.xmin = 0
            if box.ymin <  box.ymin-round(padding*(box.ymax-box.ymin)):
                box.ymin = box.ymin
            else: 
                box.ymin -= round(padding*(box.ymax-box.ymin))
            if box.ymin < 0:
                box.ymin = 0
            box.xmax += round(padding*(box.xmax-box.xmin))
            box.ymax += round(padding*(box.ymax-box.ymin))
            print(box.xmax, box.ymax, box.xmin, box.ymin)
            detected_face = frame[box.ymin:box.ymax, box.xmin:box.xmax]

            temp = []
            img = detected_face
            img = img.astype('float32')
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            temp.append(img)

            train_x = np.stack(temp)
            train_x = np.repeat(train_x, 16, axis=0)
            train_x /= 255.0
            train_x = train_x.astype('float32')
            train_y = np.random.randint(0,3,(16,4))
            y, x= capsNet_model.predict([train_x, train_y])
            pred.append(np.bincount(np.argmax(y, axis=1)).argmax()) 
        main_frame, label = draw_boxes(image, v_boxes, v_labels, v_scores, pred) #error, needs to draw boxes to faces individually  
    else:
        main_frame = image
        label = "No Face is Detected!"
    cv2.imwrite(save_path+"DFS_"+image_name, main_frame)
    detected_image_path = save_path+"DFS_"+image_name
    return main_frame, label, detected_image_path

#######################################################################
photo_lbl = None
class_lbl = None
size = (300, 300)

def classify(path):
    # run model here
    img_frame, label, detected_path = detect_mask(path)

    img = Image.open(detected_path) # output image put here)
    size = img.size
    img = img.resize(size, Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    photo_lbl.configure(
        text = "", 
        image = tk_img,
        width = size[0],
        height = size[1]
    )
    photo_lbl.image = tk_img
    photo_lbl.text = ""
    class_lbl.configure(text = label)


def uploadFile():
    file_path = askopenfilename(filetypes=[('Image Files', "*.jpg *.png *jpeg")])
    if file_path is not None:
        classify(file_path)

window = tk.Tk()
window.title("DFS - CapsNet Mask and Usage Classification Prototype")

header_txt = "Enhancing CNN-based Face Mask Detection  \n and Usage Classification using Capsule Networks"
names_txt  = "Dela Cruz, Federez, Samonte"
year_txt   = "BSCS - Thesis 3 4Q2122"

header_lbl = tk.Label(
    text = header_txt, 
    font = ('Arial', 18)
    ).pack()
names_lbl  = tk.Label(
    text = names_txt, 
    font = ('Arial', 14)
    ).pack()
year_lbl   = tk.Label(text = year_txt).pack()

photo_lbl = tk.Label(
    text = "Image appears here",
    width = 25,
    height = 25
    )
class_lbl = tk.Label(
    text = "CLASSIFICATION",
    fg = "blue",
    font = ("Arial", 18)
    )

upload_btn = tk.Button(
    text = "Upload and Run",
    width = 15,
    command = uploadFile
    ).pack()


photo_lbl.pack()
class_lbl.pack()



window.mainloop()

