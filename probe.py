import os
import cv2
import dlib
import numpy as np

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from deepface import DeepFace
# from pyagender.pyagender.pyagender import PyAgender



def load_models(model_path, caffemodel, prototxt):
    # caffemodel_path = os.path.join(model_path, caffemodel)
    # prototxt_path = os.path.join(model_path, prototxt)
    caffemodel_path = model_path + caffemodel
    prototxt_path = model_path + prototxt
    model = cv2.dnn.readNet(prototxt_path, caffemodel_path)
    return model

def predict(model, img, height, width):
    face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
    model.setInput(face_blob)
    predictions = model.forward()
    class_num = predictions[0].argmax()
    confidence = predictions[0][class_num]

    return class_num, confidence


input_height = 224
input_width = 224

# load gender model
gender_model_path = 'model_data/gender'
gender_caffemodel = '/gender2.caffemodel'
gender_prototxt = '/gender2.prototxt'
gender_model = load_models(gender_model_path, gender_caffemodel, gender_prototxt)

# load age model
age_model_path = 'model_data/age'
age_caffemodel = '/dex_chalearn_iccv2015.caffemodel'
age_prototxt = '/age2.prototxt'
age_model = load_models(age_model_path, age_caffemodel, age_prototxt)


#Load emotions model
# emo_classifire = cv2.CascadeClassifier('/model_data/haarcascade_frontalface_default.xml')
emo_classifier = load_model('model_data/Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']



detector = dlib.get_frontal_face_detector()
font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2


######################
#	Работа с видео  #
#####################

# face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    # img = cv2.imread('images/11.jpg')
    # predictions = DeepFace.analyze(img,actions=['emotion'])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = detector(img_RGB, 1)



    for d in faces:
        roi_gray = gray[d.top():d.bottom(), d.left():d.right()]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

            preds = emo_classifier.predict(roi)[0]
            print("\nprediction = ", preds)
            label = class_labels[preds.argmax()]
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            label_position = (d.left(), d.bottom())
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


        left = int(0.6 * d.left())     # + 40% margin
        top = int(0.6 * d.top())       # + 40% margin
        right = int(1.4 * d.right())   # + 40% margin
        bottom = int(1.4 * d.bottom()) # + 40% margin
        face_segm = img_RGB[top:bottom, left:right]
        gender, gender_confidence = predict(gender_model, face_segm, input_height, input_width)
        age, age_confidence = predict(age_model, face_segm, input_height, input_width)
        gender = 'man' if gender == 1 else 'woman'
        text = '{} ({:.2f}%) {} ({:.2f}%)'.format(gender, gender_confidence*100, age-5, age_confidence*100)
        cv2.putText(img, text, (d.left(), d.top() - 20), font, fontScale, fontColor, lineType)
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)

    # faces = face_cascade_db.detectMultiScale(img_gray,1.1,19)
    # for (x,y,w,h) in faces:
    # 	cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    #
    cv2.imshow('rez',img)
    # cv2.waitKey()
    if cv2.waitKey(1) & 0xff ==ord ('q'):
        break
#
cap.release()
cv2.destroyAllWindows()
#
##############################
#	Работа с изображениями   #
##############################
#
# base_dir = os.path.dirname(__file__)
# prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
# caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')
#
# model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# if not os.path.exists('updated_images'):
# 	print("New directory created")
# 	os.makedirs('updated_images')
#
# if not os.path.exists('faces'):
# 	print("New directory created")
# 	os.makedirs('faces')


# for file in os.listdir(base_dir + '/images'):
# 	file_name, file_extension = os.path.splitext(file)
# 	if (file_extension in ['.png','.jpg']):
# 		print("Image path: {}".format(base_dir + 'images/' + file))
#
# 		image = cv2.imread(base_dir + '/images/' + file)
#
# 		cap = cv2.VideoCapture(0)
#
# 		(h, w) = image.shape[:2]
# 		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
# 		#
# 		model.setInput(blob)
# 		detections = model.forward()
#
# 		# Create frame around face
# 		for i in range(0, detections.shape[2]):
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			(startX, startY, endX, endY) = box.astype("int")
#
# 			confidence = detections[0, 0, i, 2]
#
# 			# If confidence > 0.4, show box around face
# 			if (confidence > 0.4):
# 				cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
#
# 		cv2.imwrite(base_dir + '/updated_images/' + file, image)
# 		print("Image " + file + " converted successfully")
#
# 		# Identify each face
# 		for i in range(0, detections.shape[2]):
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			(startX, startY, endX, endY) = box.astype("int")
#
# 			confidence = detections[0, 0, i, 2]
#
# 			# If confidence > 0.4, save it as a separate file
# 			if (confidence > 0.4):
# 				frame = image[startY:endY, startX:endX]
# 				cv2.imwrite(base_dir + '/faces/' + str(i) + '_' + file, frame)