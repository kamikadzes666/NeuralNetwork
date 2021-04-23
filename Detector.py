import cv2
import dlib
import time

# import cetroid and trackable object's files
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from CamStreamer import CamStreamer

# load any models and add gpu acceleration
def load_models(model_path, caffemodel, prototxt):
    caffemodel_path = model_path + caffemodel
    prototxt_path = model_path + prototxt
    model = cv2.dnn.readNet(prototxt_path, caffemodel_path)

    #gpu acceleration
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return model

# Emotions prediction function
def predict_emo(model, img, height, width):
    face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
    model.setInput(face_blob)
    predictions = model.forward()
    class_num = predictions[0]
    confidence = predictions[0]
    return class_num, confidence

# Ages and genders prediction function
def predict(model, img, height, width):
    face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
    model.setInput(face_blob)
    predictions = model.forward()
    class_num = predictions[0].argmax()
    confidence = predictions[0][class_num]
    return class_num, confidence


class Detector:
    input_height = 224
    input_width = 224

    def __init__(self, path2cam, has_event, event_buff, parameters):
        # init trackers data
        self.ct = CentroidTracker(maxDisappeared=parameters['maxDisappeared'],
                                  maxDistance=parameters['maxDistance'])
        self.trackers = []
        self.trackableObjects = {}
        self.persons = []
        self.rects = []

        self.age_list = []
        self.gender_list = []
        self.emo_list = []

        # logging.info('INIT')

        # load gender model
        gender_model_path = 'model_data/gender'
        gender_caffemodel = '/gender2.caffemodel'
        gender_prototxt = '/gender2.prototxt'
        self.gender_model = load_models(gender_model_path, gender_caffemodel, gender_prototxt)

        # load age model
        age_model_path = 'model_data/age'
        age_caffemodel = '/dex_chalearn_iccv2015.caffemodel'
        age_prototxt = '/age2.prototxt'
        self.age_model = load_models(age_model_path, age_caffemodel, age_prototxt)

        # Load emotions model
        emo_model_path = 'model_data/emotions'
        emo_caffemodel = '/EmotiW_VGG_S.caffemodel'
        emo_prototxt = '/deploy.prototxt'
        self.emo_model = load_models(emo_model_path, emo_caffemodel, emo_prototxt)

        # init configuration variables
        self.categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.cam_id = path2cam
        self.has_event = has_event
        self.event_buff = event_buff



        print('detector start')
        # event_buff.put({'process': 'detector', 'status': 'error'})
        # print(type(self.cam_id))
        self.cam_process()




    # add dlib trackers
    def tracker_add(self, face_segm, left, top, right, botom):
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(left, top, right, botom)
        self.rects.append((left, top, right, botom))
        tracker.start_track(face_segm, rect)
        self.trackers.append(tracker)


    def get_data(self, data):
        # print("DETECTOR:", data)
        self.event_buff.put(data)
        self.has_event.release()

    def cam_process(self):
        detector = dlib.get_frontal_face_detector()
        font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2

        self.stream = CamStreamer(0)
        # self.cap = cv2.VideoCapture(self.cam_id)
        # cap = CamStreamer(self.cam_id)
        while True:
            # img = cap.read()
            # success, img = self.cap.read()
            img = self.stream.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(img_rgb, 1)

            # Clearing buff lists
            self.rects.clear()
            self.age_list.clear()
            self.gender_list.clear()
            self.emo_list.clear()
            face_counter = 0
            for d in faces:
                left = int(0.6 * d.left())  # + 40% margin
                top = int(0.6 * d.top())  # + 40% margin
                right = int(1.4 * d.right())  # + 40% margin
                bottom = int(1.4 * d.bottom())  # + 40% margin

                # Cutting face
                face_segm = img_rgb[top:bottom, left:right]

                Detector.tracker_add(self, face_segm, int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))

                # Get predictions
                gender, gender_confidence = predict(self.gender_model, face_segm,
                                                    Detector.input_height, Detector.input_width)
                age, age_confidence = predict(self.age_model, face_segm, Detector.input_height,
                                              Detector.input_width)
                emo, emo_confidence = predict_emo(self.emo_model, face_segm, Detector.input_height,
                                                  Detector.input_width)

                # Correcting predictions
                gender = 'man' if gender == 1 else 'woman'
                age -= 5
                emo_confidence = list(emo_confidence)
                emo_percent_list = []
                for _ in range(7):
                    emo_percent = round(emo_confidence[_] * 100, 2)
                    emo_percent_list.append(emo_percent)
                emo_dict = zip(self.categories, emo_percent_list)

                # Update buff lists
                self.age_list.append(age)
                self.gender_list.append(gender)
                self.emo_list.append(emo_dict)
                face_counter += 1
                # Add text to img for test Detectors work (can be deleted)
                text = '{} ({:.2f}%) {} ({:.2f}%)'.format(gender, gender_confidence * 100, age,
                                                          age_confidence * 100)
                cv2.putText(img, text, (d.left(), d.top() - 20), font, fontScale, fontColor, lineType)
                cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)

            # Update trackable objects
            objects = self.ct.update(self.rects)
            # Counter to index objects list
            counter = 0
            prev_face_id = []
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current object ID
                to = self.trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid,
                                         self.emo_list[counter], self.age_list[counter], self.gender_list[counter])
                    # set time and sending data to MQTT server
                    to.set_time(time.time())
                    self.get_data(to.send_data())
                    counter += 1

                # otherwise, there is a trackable object so we can utilize it
                else:
                    to.centroids.append(centroid)
                    to.counted = True
                    if self.emo_list and counter < face_counter:
                        # change emotion data,set time and sending data to MQTT server

                        current_emo = self.emo_list[counter]
                        to.change_emo(current_emo)
                        to.set_time(time.time())
                        self.get_data(to.send_data())

                    counter += 1

                # store the trackable object in our dictionary
                self.trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                # (can be deleted)
                text = "ID {}".format(objectID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # visualization
            cv2.namedWindow('rez', cv2.WINDOW_NORMAL)
            cv2.imshow('rez', img)

            # wait 'q' button
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xff == ord('q'):
                self.event_buff.put({'process': 'detector', 'status': 'close'})
                self.has_event.release()
                self.stream.stop()
                break

    def __del__(self):
        # self.cap.release()
        self.stream.stop()
        # cv2.destroyAllWindows()
        # self.has_event.release()
