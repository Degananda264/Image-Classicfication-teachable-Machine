# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:06:58 2020

@author: degananda.reddy
"""
import cv2
import numpy as np
import time

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
'''def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face'''


if __name__=='__main__':
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0
    name=input("Enter you name:")
    if os.path.exists("./Datasets/train/{}".format(name)):
        path, dirs, files = next(os.walk("Datasets/train/{}".format(name)))
        file_count = len(files)
    if not os.path.exists("./Datasets/train/{}".format(name)):
        os.makedirs("Datasets/train/{}".format(name))
        file_count=0
    # Collect 100 samples of your face from webcam input
    while True:
    
        ret, frame = cap.read()
        if frame is not None:
            count += 1
            time.sleep(0.6)
            face = cv2.resize(frame, (400, 400))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)fbeukifushus
    
            # Save file in specified directory with unique name
        
            file_name_path = './Datasets/train/'+name+'/'+ str(file_count+count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)
            
        else:
            print("Face not found")
            
    
        if cv2.waitKey(1) == 13 or count == 10: #13 is the Enter Key
            break
            
    cap.release()
    cv2.destroyAllWindows()      
    print("Collecting Samples Complete")
    
