import os
import cv2
from ultralytics import YOLO
import pytesseract


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
    binary_img = resized_img

    #OCR
    config = r"--oem 3 --psm 6"
    text_on_img = pytesseract.image_to_string(binary_img, config=config)

    #mostrando resultado
    if text_on_img == "":
        text_on_img = "nao foi possivel ler a placa"
    
    cv2.namedWindow(text_on_img, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(text_on_img, 200,150)
    cv2.imshow(text_on_img, img)
    
    if cv2.waitKey() == 27:
        break
