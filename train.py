import cv2, os
import numpy as np
from PIL import Image
#menginisiasi recognizer yang digunakan, disini menggunakan lbph recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
#dengan detector yang berisi haarcascade untuk mengenali bentuk wajah
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
#fungsi untuk memberikan label pada gambar gambar yang sudah direcord sebelumnya agar nanti dapat digunakan kedalam realtimescan
def getImagesWithLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples, Ids
faces, Ids = getImagesWithLabels('Data')
recognizer.train(faces, np.array(Ids))
recognizer.save('Data/training.xml')