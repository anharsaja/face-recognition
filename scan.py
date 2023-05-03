import cv2, time
from PIL import Image

camera = 0

video = cv2.VideoCapture(camera)
faceDeteksi = cv2.CascadeClassifier('haarcascade_face_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('dataset/training.xml')

a = 0
while True:
    a +=1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
    for (x,y,w,h) in wajah:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        id, conf = recognizer.predict(abu[y:y+h, x:x+w])
        if id == 1:
            id = 'Anhar'
        elif id == 2 == 2:
            id = 'Danu'
        cv2.putText(frame,str(id), (x+30, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()