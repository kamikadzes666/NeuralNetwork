import os
import cv2
import numpy as np
# from pyagender.pyagender.pyagender import PyAgender

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


if not os.path.exists('updated_images'):
	print("New directory created")
	os.makedirs('updated_images')

if not os.path.exists('faces'):
	print("New directory created")
	os.makedirs('faces')

for file in os.listdir(base_dir + '/images'):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg']):
		print("Image path: {}".format(base_dir + 'images/' + file))

		image = cv2.imread(base_dir + '/images/' + file)

		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		#
		model.setInput(blob)
		detections = model.forward()

		# Create frame around face
		for i in range(0, detections.shape[2]):
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			confidence = detections[0, 0, i, 2]

			# If confidence > 0.4, show box around face
			if (confidence > 0.4):
				cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

		cv2.imwrite(base_dir + '/updated_images/' + file, image)
		print("Image " + file + " converted successfully")

		# Identify each face
		for i in range(0, detections.shape[2]):
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			confidence = detections[0, 0, i, 2]

			# If confidence > 0.4, save it as a separate file
			if (confidence > 0.4):
				frame = image[startY:endY, startX:endX]
				cv2.imwrite(base_dir + '/faces/' + str(i) + '_' + file, frame)