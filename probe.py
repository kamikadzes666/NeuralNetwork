import cv2
import dlib


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

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = detector(img_RGB, 1)

    for d in faces:
        left = int(0.6 * d.left())     # + 40% margin
        top = int(0.6 * d.top())       # + 40% margin
        right = int(1.4 * d.right())   # + 40% margin
        bottom = int(1.4 * d.bottom()) # + 40% margin

        # Cutting face
        face_segm = img_RGB[top:bottom, left:right]

        # Get predictions
        gender, gender_confidence = predict(gender_model, face_segm, input_height, input_width)
        age, age_confidence = predict(age_model, face_segm, input_height, input_width)
        emo, emo_confidence = predict(emo_model,face_segm,input_height, input_width)

        # Correcting predictions
        gender = 'man' if gender == 1 else 'woman'
        emo = categories[emo]

        text = '{} ({:.2f}%) {} ({:.2f}%)'.format(gender, gender_confidence*100, age-5, age_confidence*100)
        text_emo = '{} ({:.2f}%)'.format(emo, emo_confidence*100)

        cv2.putText(img, text, (d.left(), d.top() - 20), font, fontScale, fontColor, lineType)
        cv2.putText(img, text_emo, (d.left(), d.bottom()+25), font, fontScale, fontColor, lineType)

        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)

    cv2.imshow('rez',img)
    if cv2.waitKey(1) & 0xff ==ord ('q'):
        break
#
cap.release()
cv2.destroyAllWindows()