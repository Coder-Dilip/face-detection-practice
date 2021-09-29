import cv2

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('smile.xml')

webcam=cv2.VideoCapture(0)


while True:
    successful_frame_read,frame=webcam.read()
    if not successful_frame_read:
        break

    frame_grayscale=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinates=face_detector.detectMultiScale(frame_grayscale)
    

    for x,y,w,h in face_coordinates:

        cv2.rectangle(frame, (x,y), (x+w,y+w), (0, 255,255),2)
        #  crop the image by slicing image
        the_face=frame[y:y+h, x:x+w]

        face_grayscale=cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinates=smile_detector.detectMultiScale(face_grayscale,1.7,20)

        # draw rectangle
        # for X,Y,W,H in smile_coordinates:
        #     cv2.rectangle(the_face, (X,Y), (X+W,Y+W), (100, 100,100),2)

        if(len(smile_coordinates)>0):
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    cv2.imshow('why so serious?', frame)
    key = cv2.waitKey(1)
    if(key==81 or key==113):
        break

webcam.release()
cv2.destroyAllWindows()
