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

avg_acuracy_original = 0
avg_acuracy_otsu_brightness = 0
avg_acuracy_binary_brightness = 0
avg_acuracy_otsu_blur = 0
avg_acuracy_binary_blur = 0
avg_acuracy_otsu_blur_brightness = 0
avg_acuracy_binary_blur_brightness = 0
match_total_original = 0
match_total_otsu_brightness = 0
match_total_binary_brightness = 0
match_total_otsu_blur = 0
match_total_binary_blur = 0
match_total_otsu_blur_brightness = 0
match_total_binary_blur_brightness = 0
num_images = 0

model = YOLO("./best.pt")

for name in os.listdir("./images"):
    
    image = cv2.imread("./images/"+name)
    bbs = get_bounding_boxes_yolov8("./images/"+name, model)

    if len(bbs) == 0:
        continue
    
    #segmentação
    resized_img = image[bbs[0][0][0][1]: bbs[0][0][1][1], bbs[0][0][0][0]: bbs[0][0][1][0]]

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
    blur_img = cv2.medianBlur(gray_img,3)
    blur_with_brigh_img = cv2.medianBlur(adjusted_brightness_img,3)

    #Brightness
    binary_otsu_brightness = cv2.threshold(adjusted_brightness_img,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary_brightness = cv2.adaptiveThreshold(adjusted_brightness_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    #Blur
    binary_otsu_blur = cv2.threshold(blur_img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary_blur = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    #Brightness + blur
    binary_otsu_blur_brightness = cv2.threshold(blur_with_brigh_img,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary_blur_brightness = cv2.adaptiveThreshold(blur_with_brigh_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    
    #OCR
    config = r"--oem 3 --psm 6"
    text_on_original = pytesseract.image_to_string(resized_img, config=config)

    text_on_otsu_blur = pytesseract.image_to_string(binary_otsu_blur[1], config=config)
    text_on_otsu_brightness = pytesseract.image_to_string(binary_otsu_brightness[1], config=config)
    text_on_otsu_blur_brightness = pytesseract.image_to_string(binary_otsu_blur_brightness[1], config=config)
    
    text_on_binary_blur = pytesseract.image_to_string(binary_blur, config=config)
    text_on_binary_brightness = pytesseract.image_to_string(binary_brightness, config=config)
    text_on_binary_blur_brightness = pytesseract.image_to_string(binary_blur_brightness, config=config)
    
    #Show
    cv2.imshow('car', image)
    cv2.imshow('text_on_original', resized_img)
    cv2.imshow('text_on_otsu_brightness', binary_otsu_brightness[1])
    cv2.imshow('text_on_otsu_blur', binary_otsu_blur[1])
    cv2.imshow('text_on_binary_brightness', binary_brightness)
    cv2.imshow('text_on_binary_blur', binary_blur)
    cv2.imshow('text_on_otsu_blur_brightness', binary_otsu_blur_brightness[1])
    cv2.imshow('text_on_binary_blur_brightness', binary_blur_brightness)

    #Acuracy Total    
    avg_acuracy_original += get_acuracy(name, text_on_original)
    avg_acuracy_otsu_brightness += get_acuracy(name, text_on_otsu_brightness)
    avg_acuracy_binary_brightness += get_acuracy(name, text_on_binary_brightness)
    avg_acuracy_otsu_blur += get_acuracy(name, text_on_otsu_blur)
    avg_acuracy_binary_blur += get_acuracy(name, text_on_binary_blur)
    avg_acuracy_otsu_blur_brightness += get_acuracy(name, text_on_otsu_blur_brightness)
    avg_acuracy_binary_blur_brightness += get_acuracy(name, text_on_binary_blur_brightness)

    #Match Total
    if get_acuracy(name, text_on_original) == 100:
        match_total_original += 1
    if get_acuracy(name, text_on_otsu_brightness) == 100:
        match_total_otsu_brightness += 1
    if get_acuracy(name, text_on_binary_brightness) == 100:
        match_total_binary_brightness += 1
    if get_acuracy(name, text_on_otsu_blur) == 100:
        match_total_otsu_blur += 1
    if get_acuracy(name, text_on_binary_blur) == 100:
        match_total_binary_blur += 1
    if get_acuracy(name, text_on_otsu_blur_brightness) == 100:
        match_total_otsu_blur_brightness += 1
    if get_acuracy(name, text_on_binary_blur_brightness) == 100:
        match_total_binary_blur_brightness += 1

    num_images += 1

    print('ESPERADO = ', name)
    print('OTSU BRIGHTNESS = ', text_on_otsu_brightness, 
        ' || Acuracy: ', get_acuracy(name, text_on_otsu_brightness), '%')
    print('BINARY BRIGHTNESS = ', text_on_binary_brightness, 
        ' || Acuracy: ', get_acuracy(name, text_on_binary_brightness), '%')
    print('OTSU BLUR = ', text_on_otsu_blur, 
    ' || Acuracy: ', get_acuracy(name, text_on_otsu_blur), '%')
    print('BINARY BLUR = ', text_on_binary_blur, 
    ' || Acuracy: ', get_acuracy(name, text_on_binary_blur), '%')
    print('OTSU BLUR BRIGHT = ', text_on_otsu_blur_brightness, 
    ' || Acuracy: ', get_acuracy(name, text_on_otsu_blur_brightness), '%')
    print('BINARY BLUR BRIGHT = ', text_on_binary_blur_brightness, 
    ' || Acuracy: ', get_acuracy(name, text_on_binary_blur_brightness), '%')

    cv2.waitKey()
# Acuracy Avarage 
avg_acuracy_original /= num_images
avg_acuracy_otsu_brightness /= num_images
avg_acuracy_binary_brightness /= num_images
avg_acuracy_otsu_blur /= num_images
avg_acuracy_binary_blur /= num_images
avg_acuracy_otsu_blur_brightness /= num_images
avg_acuracy_binary_blur_brightness /= num_images

# Match Totals Avarage
match_total_original_rate =  match_total_original / num_images * 100
match_total_otsu_brightness_rate = match_total_otsu_brightness / num_images * 100
match_total_binary_brightness_rate = match_total_binary_brightness / num_images * 100
match_total_otsu_blur_rate = match_total_otsu_blur / num_images * 100
match_total_binary_blur_rate = match_total_binary_blur / num_images * 100
match_total_otsu_blur_brightness_rate = match_total_otsu_blur_brightness / num_images * 100
match_total_binary_blur_brightness_rate = match_total_binary_blur_brightness / num_images * 100

#Results
print("Average Acuracy (Original):", avg_acuracy_original, "%")
print("Average Acuracy (OTSU Brightness):", avg_acuracy_otsu_brightness, "%")
print("Average Acuracy (Binary Brightness):", avg_acuracy_binary_brightness, "%")
print("Average Acuracy (OTSU Blur):", avg_acuracy_otsu_blur, "%")
print("Average Acuracy (Binary Blur):", avg_acuracy_binary_blur, "%")
print("Average Acuracy (OTSU BlurBright):", avg_acuracy_otsu_blur_brightness, "%")
print("Average Acuracy (Binary BlurBright):", avg_acuracy_binary_blur_brightness, "%")
print("Match Total Rate (Original):", match_total_original_rate, "%")
print("Match Total Rate (OTSU Brightness):", match_total_otsu_brightness_rate, "%")
print("Match Total Rate (Binary Brightness):", match_total_binary_brightness_rate, "%")
print("Match Total Rate (OTSU Blur):", match_total_otsu_blur_rate, "%")
print("Match Total Rate (Binary Blur):", match_total_binary_blur_rate, "%")
print("Match Total Rate (OTSU BlurBrightness):", match_total_otsu_blur_brightness_rate, "%")
print("Match Total Rate (Binary BlurBrightness):", match_total_binary_blur_brightness_rate, "%")

