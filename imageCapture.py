import cv2
import os
from PIL import Image
import numpy as np
import pickle

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

print ("Press SPACE to capture images for register!")
print ("Press Esc to exit!")

img_counter = 0
while True:
    ret, frame = cam.read()
    #turn to gray
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        #print (x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]

        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("test", gray)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, gray)
        print("{} registered!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()


#train
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,"")

current_id=0
label_ids={}
y_labels=[]
x_train=[]
for root,dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            
            #print (label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
              
            #y_labels.append(label)
            #x_train.append(path)

            pil_image=Image.open(path).convert("L")
            image_array=np.array(pil_image,"uint8")
            #print (image_array)
            faces= face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print (y_labels)
#print (x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")

#check images
recognizer.read("trainner.yml")

cap=cv2.VideoCapture(0);

while True:
    ret, frame=cap.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        print (x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]

        id_, conf=recognizer.predict(roi_gray)
        #print (conf)
        if conf<25:
            print("Not Matched!")
        else:
            print ("Matched!")
        img_item="my-image.png"
        
        cv2.imwrite(img_item,roi_gray)

        color=(255,0,0)
        stroke=2
        cv2.rectangle(gray,(x,y),(x+w,y+h),color,stroke)
        
    cv2.imshow("frame",gray)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.distroyAllWindows()
















