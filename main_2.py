#Facial detection
import cv2

video = cv2.VideoCapture(0)

#Add in face detection classifiers
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


#Facial Detection
while True:
    #return each frame
    ret, frame = video.read()
    #make the video feed gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces_detected = body_classifier.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #DRAW RECTANGLES
    # Draw a rectangle around the faces
    for (column, row, width, height) in faces_detected:
        cv2.rectangle(frame, (column, row), (column + width, row + height), (0, 255, 0), 2)

    # Show the video feed (WHY WON"T IT BE GRAY)
    cv2.imshow('gray_frame', frame)


    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
video.release()
cv2.destroyAllWindows()