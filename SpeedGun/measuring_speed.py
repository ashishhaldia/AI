import cv2
import numpy as np 
import time 
object_list=[]
y1=None
y2=None
t1=None
t2=None
def nothing(x):
    pass
import pandas as pd 
data=pd.DataFrame(columns=['x','y','w','h','time'])
cv2.namedWindow("Trackbars")
cv2.moveWindow("Trackbars",1520,0)
cv2.createTrackbar("hueLower","Trackbars",0,179,nothing)
cv2.createTrackbar("hueHigher","Trackbars",26,179,nothing)
cv2.createTrackbar("hue2Lower","Trackbars",155,179,nothing)
cv2.createTrackbar("hue2Higher","Trackbars",179,179,nothing)




cv2.createTrackbar("satLow","Trackbars",7,255,nothing)
cv2.createTrackbar("satHigh","Trackbars",127,255,nothing)
cv2.createTrackbar("valLow","Trackbars",128,255,nothing)
cv2.createTrackbar("valHigh","Trackbars",255,255,nothing)




#print(cv2.__version__)
dispW=320
dispH=240
flip=2
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)

outVid=cv2.VideoWriter('Test1_4.avi',cv2.VideoWriter_fourcc(*'XVID'),120,(dispW,dispH))
outVid2=cv2.VideoWriter('Test1_5.avi',cv2.VideoWriter_fourcc(*'XVID'),120,(dispW,dispH))

while True:
    _,frame = cam.read()
    #frame=cv2.imread("smarties.png")
    cv2.imshow("nanoCam",frame)
    #frame[:,:110]=0
    #frame[:,550:]=0
    #frame[:170,:]=0
    
    #fore smaller window 
    frame[:,:55]=0
    frame[:,270:]=0
    frame[:75,:]=0
    
    cv2.moveWindow("nanoCam",0,0)
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    
    hueLow=cv2.getTrackbarPos('hueLower','Trackbars')
    hueUp=cv2.getTrackbarPos('hueHigher','Trackbars')
    
    hue2Low=cv2.getTrackbarPos('hue2Lower','Trackbars')
    hue2Up=cv2.getTrackbarPos('hue2Higher','Trackbars')
    
    Ls=cv2.getTrackbarPos('satLow','Trackbars')
    Us=cv2.getTrackbarPos('satHigh','Trackbars')
    
    Lv=cv2.getTrackbarPos('valLow','Trackbars')
    Uv=cv2.getTrackbarPos('valHigh','Trackbars')
    
    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])
    
    l_b2=np.array([hue2Low,Ls,Lv])
    u_b2=np.array([hue2Up,Us,Uv])
    
    
    FGmask=cv2.inRange(hsv,l_b,u_b)
    FGmask2=cv2.inRange(hsv,l_b2,u_b2)
    FGmaskComp=cv2.add(FGmask,FGmask2)
    #cv2.imshow('FGmaskComp',FGmaskComp)
    #cv2.moveWindow('FGmaskComp',0,410)
    
    
    FG=cv2.bitwise_and(frame,frame, mask=FGmaskComp)
    #cv2.imshow('FG',FG)
   # cv2.moveWindow("FG",480,0)
    
    bgMask=cv2.bitwise_not(FGmaskComp)
    #cv2.imshow('bgMask',bgMask)
    #cv2.moveWindow('bgMask',480,410)
    
    BG=cv2.cvtColor(bgMask,cv2.COLOR_GRAY2BGR)
   
    final=cv2.add(FG,BG)
    cv2.imshow('Final',final)
    cv2.moveWindow('Final',900,0)
    outVid2.write(final)
    
    contours,_=cv2.findContours(FGmaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=100:
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
            #print(x,y,w,h)
            data.loc[len(data.index)]=[x,y,w,h,time.clock()]
            #50,60,265,180
            if (y1==None and x+w>=110 and x+w!=550):
                y1=x+w
                t1=time.clock()
                print(x,w,y1,t1)
            if(y2==None and x+w>=500 and x+w!=510):
                y2=x+w
                t2=time.clock()
                print(y2,t2)
        if(y1!=None and y2!=None):        
            speed=((y2-y1)/(t2-t1))*(220/440)*.036
            fnt=cv2.FONT_HERSHEY_DUPLEX
            frame=cv2.putText(frame,str((speed))[:4]+"k/h",(70,120),fnt,1,(0,0,255),1)
            
            
            
    cv2.imshow("nanoCam",frame)
    cv2.moveWindow("nanoCam",0,0)
    outVid.write(frame)
    if cv2.waitKey(1)==ord('q'):
        break
#data.to_csv('data.csv',index=False)
cam.release()
outVid.release()
outVid2.write(frame)
cv2.destroyAllWindows()
