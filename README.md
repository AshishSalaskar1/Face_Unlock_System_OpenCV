## **Face Unlock System Using OpenCV**

### Usage
 - Execute the `user_read.py` python script. It will then open the webcam and
   take 100 sample photos of face being detected in the webcam. Press
   Enter to abort the Process.
 - After collecting the samples, `train.py` can be executed. It trains
   the program with the collected samples to detect the given face. A
   webcam feed is then opened after training.
 - The program then unlocks the system in face matches with more than
   80% accuracy else it stays locked
   
 ### Features implemented
 - Model can be trained for a single face. Training examples are
   automatically captured from the webcam feed using OpenCV.
 - Program can validate whether the same face appears in the webcam feed
   to unlock the sytem Haar cascade classifier is being used for facial
   detection and recognition.
   
### Features yet to be implemented
 - Multiple face recognition
 - A identifier to be associated with each face detected ex Name or a
   unique id

   
   

