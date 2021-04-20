import cv2
import dlib
from pprint import pprint
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np


def load_models(model_path, caffemodel, prototxt):
    caffemodel_path = model_path + caffemodel
    prototxt_path = model_path + prototxt
    model = cv2.dnn.readNet(prototxt_path, caffemodel_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return model

def predict(model, img, height, width):
    face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
    model.setInput(face_blob)
    predictions = model.forward()
    class_num = predictions[0].argmax()
    confidence = predictions[0][class_num]

    return class_num, confidence

def predict_emo(model, img, height, width):
    face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
    model.setInput(face_blob)
    predictions = model.forward()
    class_num = predictions[0]
    confidence = predictions[0]

    return class_num, confidence

input_height = 224
input_width = 224
totalFrames = 0

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
emo_model_path = 'model_data/emotions'
emo_caffemodel = '/EmotiW_VGG_S.caffemodel'
emo_prototxt = '/deploy.prototxt'
emo_model = load_models(emo_model_path, emo_caffemodel, emo_prototxt)

categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

detector = dlib.get_frontal_face_detector()
font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2



######################
#	Работа с видео  #
#####################

cap = cv2.VideoCapture(0)
# out = cv2.VideoWriter('output.mp4', -1, 20.0, (960,540))


ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}


skipFrames = 30

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = detector(img_RGB, 1)

    rects = []

    for d in faces:
        left = int(0.6 * d.left())     # + 40% margin
        top = int(0.6 * d.top())       # + 40% margin
        right = int(1.4 * d.right())   # + 40% margin
        bottom = int(1.4 * d.bottom()) # + 40% margin
        # Cutting face
        face_segm = img_RGB[top:bottom, left:right]
        face_segm2 = img_RGB[int(d.top()):int(d.bottom()),int(d.left()):int(d.right())]

        # cv2.imshow('face',img_RGB[int(d.top()):int(d.bottom()),int(d.left()):int(d.right())])
        # cv2.waitKey(0)

        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        rects.append((int(d.left()), int(d.top()), int(d.right()), int(d.bottom())))

        tracker.start_track(face_segm, rect)
        trackers.append(tracker)



        # for tracker in trackers:
        #
        #     tracker.update(face_segm)
        #     pos = tracker.get_position()
        #
        #     startX = int(pos.left())
        #     startY = int(pos.top())
        #     endX = int(pos.right())
        #     endY = int(pos.bottom())

        # rects.append((int(d.left()), int(d.top()), int(d.right()), int(d.bottom())))



        # pprint(objects.items()[0],)


        # Get predictions
        gender, gender_confidence = predict(gender_model, face_segm, input_height, input_width)
        age, age_confidence = predict(age_model, face_segm, input_height, input_width)
        emo, emo_confidence = predict_emo(emo_model,face_segm,input_height, input_width)

        # Correcting predictions
        # gender = 'man' if gender == 1 else 'woman'

        # emo=list(emo)
        # print(emo)
        # print(type(emo))
        # emo_index = []
        # for i in range(len(emo)):
            # emo_index = emo.index[i]

        # text = '{} ({:.2f}%) {} ({:.2f}%)'.format(gender, gender_confidence*100, age-5, age_confidence*100)
        # text_emo = '{} ({:.2f}%)'.format(emo, emo_confidence*100)
        # cv2.putText(img, text, (d.left(), d.top() - 20), font, fontScale, fontColor, lineType)
        # cv2.putText(img, text_emo, (d.left(), d.bottom()+25), font, fontScale, fontColor, lineType)
        # pprint(emo_index)


        emo_confidence=list(emo_confidence)
        emo_percent_list = []
        for _ in range(7):
            emo_percent = round(emo_confidence[_]*100,2)
            emo_percent_list.append(emo_percent)
        emo_dict = zip(categories,emo_percent_list)
        # pprint(list(emo_dict))

        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)
        # out.write(img)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow('rez',img)
    if cv2.waitKey(1) & 0xff ==ord ('q'):
        break
    totalFrames += 1
#
cap.release()
# out.release()
cv2.destroyAllWindows()