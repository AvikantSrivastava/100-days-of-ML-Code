# import part
import numpy as np
# import argparse
import time
import cv2
import os

# setting confidence and threshold values
thresh = 0.3
confi = 0.5

# loading the labels
labelsPath = './assets/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(69)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# importing the weights
weightsPath = './assets/yolov3.weights'
configPath = './assets/yolov3.cfg'

print("yolo load ho raha hai")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# loading the images
image_path = './images/4.png'
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# saving the boxes confidences and classes in a list
boxes = []
confidences = []
classIDs = []



# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > confi:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi,
	thresh)

colror = (255,0,0)
(71,99,255)
# [148,0,211] 


# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		# draw a bounding box rectangle and label on the image
		# color = [int(c) for c in COLORS[classIDs[i]]]
        # color = colror

		cv2.rectangle(image, (x, y), (x + w, y + h), colror, 3)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.6, colror, 2)
# show the output image
cv2.imwrite('output.jpg',image)
cv2.imshow("Image", image)
cv2.waitKey(0)