import  cv2 as cv
import numpy as np
import mediapipe as mp
import pose_module as pm

vid=cv.VideoCapture(0)#"E:/AI_Trainer/res/gym1.mp4")
dec=pm.detector()
count=0
dir=0

while vid.isOpened():
    isTrue,img=vid.read()

    height = img.shape[0]  # y-axis
    width = img.shape[1]  # x-axis


    img=cv.resize(img,(img.shape[1]*2,img.shape[0]*2))
    img=dec.find_pose(img,False)
    lmlist=dec.get_conn(img,False)

    top_margin = int(height * 0.1)  # 10% from top
    bottom_margin = int(height * 0.05)  # 5% from bottom

    if lmlist !=0:
        angle=dec.find_angle(12,14,16,img,True)

        per=np.interp(angle,(210,330),(0,100))
        bar = np.interp(angle, (220, 330), (650, 100))


        if per==100:
            if dir==1:
                count+=0.5
                dir=0
        if per==0:
            if dir==0:
                count+=0.5
                dir=1
        bar_top = top_margin
        bar_bottom = height - bottom_margin
        bar_y = int(np.interp(count, [0, 100], [bar_bottom, bar_top]))

        # 4. Draw outer bar
        cv.rectangle(img, (width - 80, top_margin), (width - 30, height - bottom_margin),
                      (255, 0, 255), 3)

        # 5. Draw filled bar
        cv.rectangle(img, (width - 80, bar_y), (width - 30, height - bottom_margin),
                      (255, 0, 255), cv.FILLED)

        # 6. Show percentage text
        cv.putText(img, f"{int(count)}%", (width - 150, top_margin - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Draw Curl Count
        cv.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv.FILLED)

    cv.imshow("img",img)
    if cv.waitKey(10) & 0xFF==27:
        break