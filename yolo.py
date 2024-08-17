import numpy as np
import cv2
import os

def load_yolo_model():
    labelsPath = 'yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = 'yolo-coco/yolov3.weights'
    configPath = 'yolo-coco/yolov3.cfg'

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return net, LABELS, COLORS

def detect_objects(image_path, net, LABELS, COLORS):
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()

    if isinstance(unconnected_layers[0], list) or isinstance(unconnected_layers[0], np.ndarray):
        ln = [ln[i[0] - 1] for i in unconnected_layers]
    else:
        ln = [ln[i - 1] for i in unconnected_layers]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join('static/uploads', 'output.jpg')
    cv2.imwrite(output_path, image)

    return output_path
