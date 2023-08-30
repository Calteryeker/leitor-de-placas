import os
import cv2
from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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

def compare_plates(real, read):
    real = real[:-4]
    total_points = 0
    if len(real) == len(read):
        total_points+=1
    if real in read.upper():
        total_points+=1
        return total_points
    else:
        char_count = 0
        for c in real:
            if c in read: 
                char_count+=1
        
        total_points += char_count/len(real)

    return total_points

model = YOLO("./best.pt")

pts_original_total = 0
pts_binary_total = 0
pts_otsu_total = 0

nao_lidas_original = 0
nao_lidas_binary = 0
nao_lidas_otsu = 0

for name in os.listdir("./images_test"):
    img = cv2.imread("./images_test/"+name)
    bbs = get_bounding_boxes_yolov8("./images_test/"+name, model)

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
    binary_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.medianBlur(binary_img,5)
    binary_normal = cv2.adaptiveThreshold(binary_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    binary_otsu = cv2.threshold(binary_img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #OCR
    config = r"--oem 3 --psm 1"
    text_on_original = pytesseract.image_to_string(resized_img, config=config)
    text_on_binary = pytesseract.image_to_string(binary_normal, config=config)
    text_on_otsu = pytesseract.image_to_string(binary_otsu[1], config=config)

    #Avaliando resultados
    #Pontuação para imagem original
    if text_on_original == "":
        text_on_original = "nao foi possivel ler a placa"
        nao_lidas_original += 1
        pts_original = 0
    else:
        pts_original = compare_plates(name, text_on_original)

    pts_original_total += pts_original

    #Potuação para imagem binarizada com limiar adaptativo
    if text_on_binary == "":
        text_on_binary = "nao foi possivel ler a placa"
        nao_lidas_binary += 1
        pts_binary = 0
    else:
        pts_binary = compare_plates(name, text_on_binary)

    pts_binary_total += pts_binary

    #Pontuação para imagem binarizada com OTSU
    if text_on_otsu == "":
        text_on_otsu = "nao foi possivel ler a placa"
        nao_lidas_otsu +=1
        pts_otsu = 0
    else:
        pts_otsu = compare_plates(name, text_on_otsu)
    
    pts_otsu_total += pts_otsu
    
    # cv2.namedWindow(text_on_original, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(text_on_original, 200,150)
    """ cv2.imshow('text_on_original', binary_img)
    cv2.imshow('text_on_binary', binary_normal)
    cv2.imshow('text_on_otsu', binary_otsu[1]) """
    print('ESPERADO', name[:-4])
    print('ORIGINAL', text_on_original.upper(), 'Pontuação:', pts_original)
    print('ADAPTATIVE TRESHOLD', text_on_binary.upper(), 'Pontuação:', pts_binary)
    print('OTSU', text_on_otsu.upper(), 'Pontuação:', pts_otsu)

    """ if cv2.waitKey() == 27:
        break """

print('\n\nPontuação Final:')
print('ORIGINAL: ', pts_original_total)
print('ADAPTATIVE TRESHOLD', pts_binary_total)
print('OTSU', pts_otsu_total)

print('\n\nQuantidade de Placas não lidas:')
print('ORIGINAL: ', nao_lidas_original)
print('ADAPTATIVE TRESHOLD', nao_lidas_binary)
print('OTSU', nao_lidas_otsu)