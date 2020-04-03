import numpy as np
# import argparse
import random
import time
import cv2
import os

image = ''
mask_rcnn = ''
visualize = False
confi = 0.5
threshold = 0.3

#defining the labels path and 
# loading the coco class labels
labelsPath = 'assets/object_detection_classes_coco.txt'
# f = open("demofile.txt", "r")
LABELS = open(labelsPath , ).read().strip().split("\n")

# loading the set of colors that will be used to 
# visualize instance segmentation
colorsPath = 'assets/colors.txt'
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

# loading the model

weightsPath = 'assets/frozen_inference_graph.pb'
configPath = 'assets/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

print("loading the model........")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load the image
image = cv2.imread('images/2.jpg')
(H, W) = image.shape[:2]

# constructing blob now
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
end = time.time()

print('Timing information')
print("Mask R-CNN took {:.6f} seconds".format(end - start))
print("boxes shape: {}".format(boxes.shape))
print("masks shape: {}".format(masks.shape))

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
	# extract the class ID of the detection along with the confidence
	# (i.e., probability) associated with the prediction
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]
	# filter out weak predictions by ensuring the detected probability
	# is greater than the minimum probability
	if confidence > confi:
		# clone our original image so we can draw on it
		clone = image.copy()
		# scale the bounding box coordinates back relative to the
		# size of the image and then compute the width and the height
		# of the bounding box
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY


        		# extract the pixel-wise segmentation for the object, resize
		# the mask such that it's the same dimensions of the bounding
		# box, and then finally threshold to create a *binary* mask
		mask = masks[i, classID]
		mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_NEAREST)
		mask = (mask > threshold)
		# extract the ROI of the image
		roi = clone[startY:endY, startX:endX]



		# check to see if are going to visualize how to extract the
		# masked region itself
		if visualize:
			# convert the mask from a boolean to an integer mask with
			# to values: 0 or 255, then apply the mask
			visMask = (mask * 255).astype("uint8")
			instance = cv2.bitwise_and(roi, roi, mask=visMask)
			# show the extracted ROI, the mask, along with the
			# segmented instance
			cv2.imshow("ROI", roi)
			cv2.imshow("Mask", visMask)
			cv2.imshow("Segmented", instance)


		# now, extract *only* the masked region of the ROI by passing
		# in the boolean mask array as our slice condition
		roi = roi[mask]
		# randomly select a color that will be used to visualize this
		# particular instance segmentation then create a transparent
		# overlay by blending the randomly selected color with the ROI
		color = random.choice(COLORS)
		blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
		# store the blended ROI in the original image
		clone[startY:endY, startX:endX][mask] = blended


		# draw the bounding box of the instance on the image
		color = [int(c) for c in color]
		cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)
		# draw the predicted label and associated probability of the
		# instance segmentation on the image
		text = "{}: {:.4f}".format(LABELS[classID], confidence)
		cv2.putText(clone, text, (startX, startY - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		# show the output image
		cv2.imshow("Output", clone)
		cv2.waitKey(0)