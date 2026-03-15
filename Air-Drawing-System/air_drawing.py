import cv2
import mediapipe as mp
import time
import numpy as np
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

canvas=None
prev_x,prev_y=0,0

color=(255,0,0)
brush_size=6
eraser_size=60


def count_fingers(points):

    fingers=0

    if points[4][0] > points[3][0]:
        fingers+=1

    if points[8][1] < points[6][1]:
        fingers+=1

    if points[12][1] < points[10][1]:
        fingers+=1

    if points[16][1] < points[14][1]:
        fingers+=1

    if points[20][1] < points[18][1]:
        fingers+=1

    return fingers


while True:

    success,frame=cap.read()
    if not success:
        break

    frame=cv2.flip(frame,1)

    if canvas is None:
        canvas=np.zeros_like(frame)

    # color palette
    cv2.rectangle(frame,(0,0),(200,70),(255,0,0),-1)
    cv2.rectangle(frame,(200,0),(400,70),(0,255,0),-1)
    cv2.rectangle(frame,(400,0),(600,70),(0,0,255),-1)
    cv2.rectangle(frame,(600,0),(800,70),(0,255,255),-1)
    cv2.rectangle(frame,(800,0),(1000,70),(0,0,0),-1)

    cv2.putText(frame,"CLEAR",(820,45),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    timestamp=int(time.time()*1000)

    result=landmarker.detect_for_video(mp_image,timestamp)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            h,w,_=frame.shape

            points=[]

            for lm in hand:
                px=int(lm.x*w)
                py=int(lm.y*h)
                points.append((px,py))

            # skeleton
            connections=[
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
            ]

            for c in connections:
                cv2.line(frame,points[c[0]],points[c[1]],(255,255,255),2)

            for p in points:
                cv2.circle(frame,p,5,(0,0,255),-1)

            fingers=count_fingers(points)

            index_tip=points[8]
            middle_tip=points[12]

            x_index,y_index=index_tip

            # ERASER MODE (4 or 5 fingers)
            if fingers >= 4:

                palm_x,palm_y=points[9]

                cv2.circle(frame,(palm_x,palm_y),eraser_size,(255,255,255),2)

                if prev_x==0 and prev_y==0:
                    prev_x,prev_y=palm_x,palm_y

                cv2.line(canvas,(prev_x,prev_y),
                         (palm_x,palm_y),
                         (0,0,0),
                         eraser_size)

                prev_x,prev_y=palm_x,palm_y


            # COLOR SELECTION MODE
            elif index_tip[1] < points[6][1] and middle_tip[1] < points[10][1]:

                prev_x,prev_y=0,0

                if y_index < 70:

                    if x_index < 200:
                        color=(255,0,0)

                    elif x_index < 400:
                        color=(0,255,0)

                    elif x_index < 600:
                        color=(0,0,255)

                    elif x_index < 800:
                        color=(0,255,255)

                    elif x_index < 1000:
                        canvas=np.zeros_like(frame)


            # DRAW MODE
            elif index_tip[1] < points[6][1]:

                if prev_x==0 and prev_y==0:
                    prev_x,prev_y=x_index,y_index

                dist=math.hypot(x_index-prev_x,y_index-prev_y)

                steps=int(dist/5)+1

                for i in range(steps):

                    ix=int(prev_x+(x_index-prev_x)*i/steps)
                    iy=int(prev_y+(y_index-prev_y)*i/steps)

                    cv2.circle(canvas,(ix,iy),
                               brush_size,
                               color,
                               -1)

                prev_x,prev_y=x_index,y_index

            else:
                prev_x,prev_y=0,0


    frame=cv2.add(frame,canvas)

    cv2.putText(frame,f"Brush:{brush_size}",
                (1050,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,255,255),2)

    cv2.imshow("Air Drawing System",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()