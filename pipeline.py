import os
import cv2
from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import difflib


def get_bounding_boxes_yolov8(img_path, model):
    detections = model(img_path)
    confs = detections[0].boxes.conf
    classes = detections[0].boxes.cls
    boxes = detections[0].boxes.xyxy
    conf_thr = 0.0
    bounding_boxes = []
    for elem in zip(boxes, classes, confs):
        top_left = (int(elem[0][0]), int(elem[0][1]))
        bottom_right = (int(elem[0][2]), int(elem[0][3]))
        label = str(int(elem[1]))
        conf = float(elem[2])
        # Convert int value labels to their corresponding classes:
        label = "license"
        # Filter low-confidence detections:
        if conf > conf_thr:
            bounding_boxes.append(([top_left, bottom_right], label, conf))
    return bounding_boxes

def get_acuracy(a, b):
    bclean = ''.join(c for c in b if c.isalnum())
    seq=difflib.SequenceMatcher(None, a[:-4], bclean)
    acuracy=seq.ratio()*100
    return acuracy

model = YOLO("./best.pt")

for name in os.listdir("./images"):
    img = cv2.imread("./images/"+name)
    bbs = get_bounding_boxes_yolov8("./images/"+name, model)

    if len(bbs) == 0:
        continue
    
    #segmentação
    img = img[bbs[0][0][0][1]: bbs[0][0][1][1], bbs[0][0][0][0]: bbs[0][0][1][0]]

    #Resize
    resized_img = img
    """     
    scale_percent = 150 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    """
    #Binarização
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    adjusted_brightness_img = cv2.convertScaleAbs(gray_img, alpha=0.03, beta=0)
    blur_img = cv2.medianBlur(gray_img,5)
    
    #Brightness
    binary_otsu_brightness = cv2.threshold(adjusted_brightness_img,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary_brightness = cv2.adaptiveThreshold(adjusted_brightness_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    #Blur
    binary_blur = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    binary_otsu_blur = cv2.threshold(blur_img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #OCR
    config = r"--oem 3 --psm 1"
    text_on_original = pytesseract.image_to_string(resized_img, config=config)
    text_on_otsu_blur = pytesseract.image_to_string(binary_otsu_blur[1], config=config)
    text_on_otsu_brightness = pytesseract.image_to_string(binary_otsu_brightness[1], config=config)
    
    text_on_binary_blur = pytesseract.image_to_string(binary_blur, config=config)
    text_on_binary_brightness = pytesseract.image_to_string(binary_brightness, config=config)

    cv2.imshow('text_on_original', adjusted_brightness_img)
    cv2.imshow('text_on_otsu_brightness', binary_otsu_brightness[1])
    cv2.imshow('text_on_otsu_blur', binary_otsu_blur[1])
    cv2.imshow('text_on_binary_brightness', binary_brightness)
    cv2.imshow('text_on_binary_blur', binary_blur)

    print('ESPERADO = ', name)
    print('OTSU BRIGHTNESS = ', text_on_otsu_brightness, 
        ' || Acuracy: ', get_acuracy(name, text_on_otsu_brightness), '%')
    print('BINARY BRIGHTNESS = ', text_on_binary_brightness, 
        ' || Acuracy: ', get_acuracy(name, text_on_binary_brightness), '%')
    print('OTSU BLUR = ', text_on_otsu_blur, 
    ' || Acuracy: ', get_acuracy(name, text_on_otsu_blur), '%')
    print('BINARY BLUR = ', text_on_binary_blur, 
    ' || Acuracy: ', get_acuracy(name, text_on_binary_blur), '%')

    cv2.waitKey()

