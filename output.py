# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
model = load_model('output_model.h5')

#train_names=enumerate(os.listdir(r"Datasets\train"))
#trained_names=list(train_names)
#img_width, img_height = 224, 224
#img = image.load_img(r'C:\Users\degananda.reddy\Desktop\images.jpg', target_size=(img_width, img_height))

#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)

#red=model.predict(x)
#arg=np.argmax(red)
#name=trained_names[arg][1]
#print("Image:",name)  


# Loading the cascades
'''face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face'''
train_names=enumerate(os.listdir(r"Datasets\train"))
trained_names=list(train_names)

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    #print(frame)
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    #face=face_extractor(frame)
    if frame is not None:
        face = cv2.resize(frame, (400, 400))
        cv2.imwrite('image.jpg', face)
        img_width, img_height = 224, 224
        img = image.load_img('image.jpg', target_size=(img_width, img_height))

        #img = image.load_img(frame)
        #im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
        #img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
        #img_array = np.expand_dims(img_array, axis=0)
        #pred = model.predict(img_array)
        #print(pred)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        red=model.predict(x)
        arg=np.argmax(red[0])
        if red[0][arg]==1:
            name=trained_names[arg][1]
        else:
            indices=red.argsort()[-3:][::-1]
            fname=[]
            for i in indices:
                for j in i[:3]:
                     name1=trained_names[j][1]
                     fname.append(name1)
            name="-".join(fname)
        #print("Image:",name)            
        #name="None matching"
        #arg=np.argmax(pred)
        #if(pred[0][arg]>0.5):
            #name=trained_names[arg][1]
        #else:
            #name="Not Found"
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()