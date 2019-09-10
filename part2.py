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
    print(image_path)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)

labels = np.asarray(labels,dtype=np.int32)

#training model wuth data sets
model = cv2.face.LBPHFaceRecognizer_create()

#train model
model.train(np.asarray(training_data),np.asarray(labels))
print("Model Trained Succeful with user data set")