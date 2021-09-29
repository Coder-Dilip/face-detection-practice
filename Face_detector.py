import cv2

# import all the faces data trained for program
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img=cv2.imread('man.jpg')
# 0 means webcam, 'video.mp4' can also be written instead of 0 for detecting video in realtimee
webcam=cv2.VideoCapture(0)
while(True):
    successful_frame_read,frame=webcam.read()
    img=frame
# converting to black and white
    grayscale_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face_coordinates= trained_face_data.detectMultiScale(grayscale_img)

# draw rectangle using coordinates
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(img,(x,y),(x+w, y+w),(0,255,0),1)

    cv2.imshow('Face Detector', img)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()