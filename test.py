import cv2, time
import os
from PIL import Image
camera = 0  
webcam= cv2.VideoCapture(camera, cv2.CAP_DSHOW)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#membuat variable untuk merekognasi wajah dari seseorang menggunakan lbph face recognizer
recog=cv2.face.LBPHFaceRecognizer_create()
#melakukan rekognasi dengan menggunakan data training yang sebelumnya sudah kita buat
recog.read('Data/training.xml')
i= 0
while True:
    i=i+1
    check, frame = webcam.read()
    grey= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in wajah:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
        id, conf = recog.predict(grey[y:y+h,x:x+w])
        if (id == 1):
            id='Dedi'
        else:
            'Siapa dah lu'
        #mengatur format dan tulisan apa yang akan muncul saat melakukan scan secara realtime
        cv2.putText(frame, str(id),(x+40,y-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    cv2.imshow("Pengenalan Wajah", frame)
    #membuat waitkey agar jika sudah selesai dalam melakukan face scanning secara realtime kita bisa klik q untuk keluar dari webcam
    wkey= cv2.waitKey(1)
    if wkey == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()