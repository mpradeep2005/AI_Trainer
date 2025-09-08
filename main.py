import  cv2 as cv
import numpy as np
import mediapipe as mp
import pose_module as pm

vid=cv.VideoCapture("E:/AI_Trainer/res/gym1.mp4")
dec=pm.detector()
while vid.isOpened():
    isTrue,img=vid.read()
    img=cv.resize(img,(img.shape[1]//2,img.shape[0]//2))
    img=dec.find_pose(img)
    cv.imshow("img",img)

    if cv.waitKey(10) & 0xFF==27:
        break