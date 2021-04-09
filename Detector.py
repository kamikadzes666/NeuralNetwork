import cv2
import numpy as np
import dlib
from pprint import pprint
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from Person import Person


class Detector:
    input_height = 224
    input_width = 224

    def __init__(self, cam_id=0):
        # init trackers data
        self.ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}
        self.persons = []
        self.rects = []
        # self.skip_frame_flag = True

        # load gender model
        gender_model_path = 'model_data/gender'
        gender_caffemodel = '/gender2.caffemodel'
        gender_prototxt = '/gender2.prototxt'
        self.gender_model = Detector._load_models(gender_model_path, gender_caffemodel, gender_prototxt)

        # load age model
        age_model_path = 'model_data/age'
        age_caffemodel = '/dex_chalearn_iccv2015.caffemodel'
        age_prototxt = '/age2.prototxt'
        self.age_model = Detector._load_models(age_model_path, age_caffemodel, age_prototxt)

        # Load emotions model
        emo_model_path = 'model_data/emotions'
        emo_caffemodel = '/EmotiW_VGG_S.caffemodel'
        emo_prototxt = '/deploy.prototxt'
        self.emo_model = Detector._load_models(emo_model_path, emo_caffemodel, emo_prototxt)

        self.categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.cam_id = cam_id

    def _load_models(model_path, caffemodel, prototxt):
        caffemodel_path = model_path + caffemodel
        prototxt_path = model_path + prototxt
        model = cv2.dnn.readNet(prototxt_path, caffemodel_path)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return model

    def predict_emo(model, img, height, width):
        face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
        model.setInput(face_blob)
        predictions = model.forward()
        class_num = predictions[0]
        confidence = predictions[0]

        return class_num, confidence

    def predict(model, img, height, width):
        face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
        model.setInput(face_blob)
        predictions = model.forward()
        class_num = predictions[0].argmax()
        confidence = predictions[0][class_num]

        return class_num, confidence

    def get_data(self):
        self.emo_confidence = list(self.emo_confidence)
        emo_percent_list = []
        for _ in range(7):
            emo_percent = round(self.emo_confidence[_] * 100, 2)
            emo_percent_list.append(emo_percent)
        emo_dict = zip(self.categories, emo_percent_list)

        print('===================================')
        print('{} ({:.2f}%)'.format(self.gender, self.gender_confidence * 100))
        print('{} ({:.2f}%)'.format(self.age - 5, self.age_confidence * 100))
        pprint(list(emo_dict))
        print('===================================')

    def tracker_add(self, face_segm, left, top, right, botom):
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(left, top, right, botom)
        self.rects.append((left, top, right, botom))

        tracker.start_track(face_segm, rect)
        self.trackers.append(tracker)

    def track(self):
        pass



    def cam_process(self):
        detector = dlib.get_frontal_face_detector()
        font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2

        self.cap = cv2.VideoCapture(self.cam_id)
        # out = cv2.VideoWriter('output.mp4', -1, 20.0, (960,540))



        while True:
            success, img = self.cap.read()
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(img_RGB, 1)



            self.rects.clear()

            self.age_list = []
            self.age_conf_list = []
            self.gender_list = []
            self.gender_conf_list = []
            self.emo_list = []



            for d in faces:
                left = int(0.6 * d.left())  # + 40% margin
                top = int(0.6 * d.top())  # + 40% margin
                right = int(1.4 * d.right())  # + 40% margin
                bottom = int(1.4 * d.bottom())  # + 40% margin

                # Cutting face
                face_segm = img_RGB[top:bottom, left:right]

                Detector.tracker_add(self,face_segm,int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))

                # Get predictions
                gender, gender_confidence = Detector.predict(self.gender_model, face_segm,
                                                                       Detector.input_height, Detector.input_width)
                age, age_confidence = Detector.predict(self.age_model, face_segm, Detector.input_height,
                                                                 Detector.input_width)
                emo, emo_confidence = Detector.predict_emo(self.emo_model, face_segm, Detector.input_height,
                                                                     Detector.input_width)

                # print

                # Correcting predictions
                gender = 'man' if gender == 1 else 'woman'
                age -=5
                emo_confidence = list(emo_confidence)
                emo_percent_list = []
                for _ in range(7):
                    emo_percent = round(emo_confidence[_] * 100, 2)
                    emo_percent_list.append(emo_percent)
                emo_dict = zip(self.categories, emo_percent_list)

                self.age_list.append(age)
                self.age_conf_list.append(age_confidence)
                self.gender_list.append(gender)
                self.gender_conf_list.append(gender_confidence)
                self.emo_list.append(emo_dict)


                text = '{} ({:.2f}%) {} ({:.2f}%)'.format(gender, gender_confidence * 100, age,
                                                          age_confidence * 100)
                # text_emo = '{} ({:.2f}%)'.format(emo, emo_confidence*100)
                cv2.putText(img, text, (d.left(), d.top() - 20), font, fontScale, fontColor, lineType)
                # cv2.putText(img, text_emo, (d.left(), d.bottom()+25), font, fontScale, fontColor, lineType)
                cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)

                # Detector.get_data(self)

                # out.write(img)

            objects = self.ct.update(self.rects)
            counter = 0

            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = self.trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    # if self.skip_frame_flag:
                    #     self.skip_frame_flag = False
                    #     break
                    to = TrackableObject(objectID, centroid,
                                         self.emo_list[counter],self.age_list[counter],self.gender_list[counter])
                    counter += 1

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
                self.trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            else: self.skip_frame_flag = True
            # for()

            cv2.namedWindow('rez', cv2.WINDOW_NORMAL)
            cv2.imshow('rez', img)

            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break


    def __del__(self):
        self.cap.release()
        # out.release()
        cv2.destroyAllWindows()

