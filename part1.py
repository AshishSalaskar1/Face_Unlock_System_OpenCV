import cv2
import numpy as np
import os

def delte_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def face_extractor(img):
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayImage,1.3,5)

    if faces is():
        return None
    
    for (x,y,w,h) in faces:
        croppedFace = img[y:y+h,x:x+w]
        
    return  croppedFace 


cap = cv2.VideoCapture(0)
count = 0
delte_folder("userSamples/samples/")

while True:
    ret,frame = cap.read()

    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        filePath = "userSamples/samples/user"+str(count)+".jpg"
        cv2.imwrite(filePath,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Face Cropper",face)

    else:
        print("Face Not found")
        pass

    if cv2.waitKey(1) == 13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()

print("All samples created")