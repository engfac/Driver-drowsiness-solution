##Author : W.S.D. Fernando / N.E.K.A.S. Dias / H.B.A.D. Maduranga
##Date : 2018 / 10 / 05
## Purpose : Detection of face and ee region

## Add inbuilt libraries
import numpy as np
import os
import cv2
from pygame import mixer

## Add created Xml files
face_cascade = cv2.CascadeClassifier('face.xml')

eye_cascade = cv2.CascadeClassifier('eye.xml')

## Alarm function
def sound():
    mixer.init()
    mixer.music.load('s.wav')
    mixer.music.play()

## Recording the video using web cam. this is the input command for opening the webcam.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

## Identify the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)

## Identify the Eyes        
        for (ex,ey,ew,eh) in eyes:
             font = cv2.FONT_HERSHEY_SIMPLEX
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
             cv2.putText(frame,'EYE RECOGNIZED',(x-w,y-h), font, 0.5, (0,255,0), 2, cv2.LINE_AA)
             sound()

    cv2.imshow('frame',frame)   
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

