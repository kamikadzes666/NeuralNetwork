import os
import cv2
# from pyagender.pyagender.pyagender import PyAgender

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/bvlc_googlenet.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/bvlc_googlenet.caffemodel')

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