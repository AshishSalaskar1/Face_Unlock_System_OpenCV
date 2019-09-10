import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

file_path = "userSamples/samples/"
data_set = [ f for f in listdir(file_path) if isfile(join(file_path,f))]

print(data_set)

training_data = []
labels = []

for i,files in enumerate(data_set):
    image_path = file_path + data_set[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    # print(image_path)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)

labels = np.asarray(labels,dtype=np.int32)

#training model wuth data sets
model = cv2.face.LBPHFaceRecognizer_create()

#train model
model.train(np.asarray(training_data),np.asarray(labels))
print("Model Trained Succeful with user data set")

# Model Prediction
face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        regionInterest = img[y:y+h,x:x+w] 
        regionInterest = cv2.resize(regionInterest,(200,200))

    return img,regionInterest

#start video capture((
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        #check correctness confidence
        if result[1] < 500:
            confidence = int(100* (1 - (result[1]/300)))
            confidence_msg = str(confidence)+"% correctness confidence"
        
        #display confidence on webcam
        cv2.putText(image,confidence_msg,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        if confidence>75:
            cv2.putText(image,"System Unlocked",(150,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            cv2.imshow("Face cropper",image)
        else:
            cv2.putText(image,"System Locked",(150,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
            cv2.imshow("Face cropper",image)


    #no face found
    except:
        cv2.putText(image,"Face not found in frame",(150,350),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        cv2.imshow("Face cropper",image)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

