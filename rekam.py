import cv2, time

camera = 0

video = cv2.VideoCapture(camera)
faceDeteksi = cv2.CascadeClassifier('haarcascade_face_default.xml')
id = input('Masukkan id : ')
a = 0
while True:
    a +=1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
    for (x,y,w,h) in wajah:
        cv2.imwrite('dataset/User.'+str(id)+'.'+str(a)+'.jpg',abu[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Face Recognition", frame)
    if (a>29):
        break

video.release()
cv2.destroyAllWindows()