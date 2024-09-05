from moviepy.editor import VideoFileClip
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


image=mpimg.imread('img.jpg')
print('görüntü',type(image),'görüntü şekli',image.shape)
plt.imshow(image)


def caNNy(frame): #fotoğraf griye boyandı gauss blur canny filtresi uygulandı

    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def Segment(frame):
    height=frame.shape[0]
    polygons=np.array([
                            [(0,height),(800,height),(360,1080)]
    ])

    mask=np.zeros_like(frame)
    cv2.fillPoly(mask,polygons,255)
    segment=cv2.bitwise_and(frame,mask)
    return segment

def cizgi_hesap(frame,lines):
    left=[]
    right=[]


    for line in lines:


        x1 , y1, x2 , y2 = line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        y_intercept=parameters[1]
        cv2.line(parameters(x1,y1),(x2,y2),(0,0,255),10)


        if slope<0:
            left.append((slope,y_intercept))
        else:
            right.append((slope,y_intercept))


    left_avg=np.average(left,axis=0)
    right_avg=np.average(right,axis=0)

    left_line =cizgi_hesap(frame,left_avg)
    right_line=cizgi_hesap(frame,right_avg)
    return np.array([left_line,right_line])

def koordinat_hesap(frame,parameters):
    slope,intercept=parameters
    y1=frame.shape[0]
    y2=int(y1-150)

    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)

    return np.array([x1,x2,y1,y2])

def cizgi_cizdirme(frame,lines):

    lines_visualize=np.zeros_like(frame)

    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(lines_visualize,(x1,y1),(x2,y2),(0,255,0),5)
            return lines_visualize
cap=cv2.VideoCapture("xyz.mp4")



while(cap.isOpened()):
    ret,frame=cap.read()
    canny=caNNy(frame)
    cv2.imshow("canny_filtresi",canny)



    segment=Segment(canny)
    hough=cv2.HoughLinesP(segment,2,np.pi/180,100,np.array([]),minLineLength=100,maxLineGap=50)

    lines=cizgi_hesap(frame,hough)
    lines_visualize=cizgi_cizdirme(frame,lines)
    #cv2.imshow("hough_cizgileri",lines_visualize)

    #output=cv2.addWeighted(frame,0.9,lines_visualize,1,1)
    #cv2.imshow("output",output)

    if cv2.waitKey(100) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







