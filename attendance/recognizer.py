import cv2, numpy as np;
import xlwrite,firebase.firebase_ini as fire;
import time
import sys
from playsound import playsound
start=time.time()
period=8
face_cas = cv2.CascadeClassifier('haarcascade_profileface.xml')

face_cas = cv2.CascadeClassifier('C:/project/Face-Recognition-Attendance-System-master/Face-Recognition-Attendance-System-master/attendance/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('C:/project/Face-Recognition-Attendance-System-master/Face-Recognition-Attendance-System-master/attendance/trainer/trainer.yml');
flag = 0;
id=0;
filename='filename';
dict = {
            'item1': 1
}
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    print(img)
    print("ret:{0}".format(ret))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    print("gray:{0}".format(gray))
    try:
        faces = face_cas.detectMultiScale(gray, 1.3, 7);
    except Exception as e:
        print(str(e))
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        id,conf=recognizer.predict(roi_gray)
        if(conf < 50):
         if(id==1):
            id='Santhosh'
            if((str(id)) not in dict):
                filename=xlwrite.output('attendance','class1',1,id,'yes');
                dict[str(id)]=str(id);
                
         elif(id==2):
            id = 'Dinesh'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);

         elif(id==3):
            id = 'Raveen'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 3, id, 'yes');
                dict[str(id)] = str(id);

         elif(id==4):
            id = 'Sonu'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
                dict[str(id)] = str(id);

        else:
             id = 'Unknown, can not recognize'
             flag=flag+1
             break
        
        cv2.putText(img,str(id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame',img);
    #cv2.imshow('gray',gray);
    if flag == 10:
        playsound('transactionSound.mp3')
        print("Transaction Blocked")
        break;
    if time.time()>start+period:
        break;
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cap.release();
cv2.destroyAllWindows();
