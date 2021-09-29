import cv2

imgfile='cars-and-pedestrains.jpg'

classifier_file='car_detector.xml'

pedestrain_file='pedestrains_detector.xml'

car_tracker=cv2.CascadeClassifier(classifier_file)
pedestrain_tracker=cv2.CascadeClassifier(pedestrain_file)
# start video or recording
webcam=cv2.VideoCapture('cars-and-pedestrains.mp4')
while(True):
    successful_frame_read,frame=webcam.read()
    img=frame
    if not successful_frame_read:
        break
# convert to black and white
    black_n_white=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect coordinates
    cars=car_tracker.detectMultiScale(black_n_white)
    pedestrains=pedestrain_tracker.detectMultiScale(black_n_white)
    
# Draw rectangles around the cars
    for(x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w, y+h),(253,253,0),2)

# Draw rectangles around the pedestrains
    for(x,y,w,h) in pedestrains:
        cv2.rectangle(img, (x,y), (x+w, y+h),(255,255,255),2)

# display image/video file
    cv2.imshow('cars detection', img)

# wait key is important otherwise image will close in miliseconds
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break

webcam.release()


