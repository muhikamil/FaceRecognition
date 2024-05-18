#Library yang dibutuhkan yaitu opencv
import cv2

#menginisiasi camera
camera = 0  
#menginisiasi variable webcam  dengan menggunakan library dari cv2 untuk melakukan penangkapan video dengan output gambar
webcam= cv2.VideoCapture(camera, cv2.CAP_DSHOW)
#melakukan inisiasi untuk bentuk wajah dengan menggunakan haarcascade dan library dari opencv yaitu cascade classifier
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#membuat variable id untuk memasukkan id yang akan disave kedalam data nanti
id= input('Masukkan Id: ')
i= 0
while True:
    #membuat variable i yang akan selalu bertambah 1 jika ada inputan tambahan
    i=i+1
    #untuk membuat sebuah frame 
    check, frame = webcam.read()
    #merubah hasil tangkapan dengan frame dari blue green red menjadi gray/abu
    grey= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #melakukan deteksi multi skala dengan memanggil variable grey 
    wajah = face.detectMultiScale(grey,1.3,5)
    #membuat looping untuk menambahkan data wajah kedalam folder yang telah ditentukan dengan format User.'id'.jpg
    for (x,y,w,h) in wajah:
        cv2.imwrite(r'Data\User.'+str(id)+'.'+str(i)+'.jpg', grey[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
    #membuat frame untuk melakukan penangkapan gambar dengan ketentuan jika i sudah lebih dari 29 maka akan langsung berhenti menangkap gambar
    cv2.imshow("Pengenalan Wajah", frame)
    if (i>29):
        break

webcam.release()
cv2.destroyAllWindows()