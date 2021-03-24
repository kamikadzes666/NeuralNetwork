import cv2
import numpy as np
import dlib
from pprint import pprint

class Detector:

    input_height = 224
    input_width = 224

    def __init__(self, cam_id = 0):


        # load gender model
        gender_model_path = 'model_data/gender'
        gender_caffemodel = '/gender2.caffemodel'
        gender_prototxt = '/gender2.prototxt'
        self.gender_model = Detector.load_models(gender_model_path, gender_caffemodel, gender_prototxt)

        # load age model
        age_model_path = 'model_data/age'
        age_caffemodel = '/dex_chalearn_iccv2015.caffemodel'
        age_prototxt = '/age2.prototxt'
        self.age_model = Detector.load_models(age_model_path, age_caffemodel, age_prototxt)

        # Load emotions model
        emo_model_path = 'model_data/emotions'
        emo_caffemodel = '/EmotiW_VGG_S.caffemodel'
        emo_prototxt = '/deploy.prototxt'
        self.emo_model = Detector.load_models(emo_model_path, emo_caffemodel, emo_prototxt)

        self.categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.cam_id = cam_id




    def load_models(model_path, caffemodel, prototxt):
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


    def cam_process(self):
        detector = dlib.get_frontal_face_detector()
        font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2

        self.cap = cv2.VideoCapture(self.cam_id)
        # out = cv2.VideoWriter('output.mp4', -1, 20.0, (960,540))

        while True:
            success, img = self.cap.read()
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(img_RGB, 1)

            for d in faces:
                left = int(0.6 * d.left())  # + 40% margin
                top = int(0.6 * d.top())  # + 40% margin
                right = int(1.4 * d.right())  # + 40% margin
                bottom = int(1.4 * d.bottom())  # + 40% margin

                # Cutting face
                face_segm = img_RGB[top:bottom, left:right]

                # Get predictions
                # gender, gender_confidence = predict(gender_model, face_segm, input_height, input_width)
                # age, age_confidence = predict(age_model, face_segm, input_height, input_width)
                emo, emo_confidence = Detector.predict_emo(self.emo_model, face_segm, Detector.input_height, Detector.input_width)

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
                emo_confidence = list(emo_confidence)
                emo_percent_list = []
                for _ in range(7):
                    emo_percent = round(emo_confidence[_] * 100, 2)
                    emo_percent_list.append(emo_percent)
                emo_dict = zip(self.categories, emo_percent_list)
                pprint(list(emo_dict))
                cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)
                # out.write(img)

            cv2.imshow('rez', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    def __del__(self):
        self.cap.release()
        # out.release()
        cv2.destroyAllWindows()



