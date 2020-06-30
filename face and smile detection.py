import cv2
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile =cv2.CascadeClassifier('haarcascade_smile.xml')
def det(gray,frame):
    faces = face.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),((x+w),(y+h)),(255,0,0),2)
        rgray = gray[y:y+h,x:x+w]
        rcolor = frame[y:y+h,x:x+w]
        smiles = smile.detectMultiScale(rgray,1.8,20)

        for(sx,sy,sw,sh) in smiles:
            cv2.rectangle(rcolor,(sx,sy),((sx+sw),(sy+sh)),(0,0,255),2)
    return frame

video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = det(gray,frame)
    cv2.imshow('vid',canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
video.release()
cv2.destroyAllWindows()